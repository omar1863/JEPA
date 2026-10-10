"""Pretraining loop: cosine LR + warmup, EMA target schedule, AMP, collapse logging."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import ExperimentConfig, OptimConfig
from .diagnostics import CollapseMonitor


def set_seed(seed: int, deterministic: Optional[bool] = None):
    """Seed python/numpy/torch. ``deterministic=True`` (v0.2) also pins cuDNN.

    Note: full bit-for-bit determinism on GPU additionally needs
    ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` set before CUDA initialises (the notebook does
    this in its first cell); SDPA's memory-efficient backward is deterministic.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic is not None:
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = not bool(deterministic)


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cosine_lr(step: int, total: int, warmup: int, base_lr: float,
              min_lr: float) -> float:
    if total <= 0:
        return base_lr
    if warmup > 0 and step < warmup:
        return min_lr + (base_lr - min_lr) * step / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    p = min(max(p, 0.0), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * p))


def ema_momentum(step: int, total: int, start: float, end: float) -> float:
    if total <= 0:
        return start
    return start + (end - start) * min(step / total, 1.0)


def param_groups(model: torch.nn.Module, weight_decay: float) -> List[dict]:
    """No weight decay on biases, norms and other 1-D parameters (I-JEPA convention)."""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if (p.ndim <= 1 or n.endswith(".bias")) else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


class InterleavedLoader:
    """Round-robin over several loaders, keeping each *batch* homogeneous.

    Short 10 s clips and long-recording forward pairs need different loss branches,
    so mixing them inside one batch would mean two forward passes per step. Instead
    we alternate whole batches in proportion to ``weights``.
    """

    def __init__(self, loaders: Sequence[DataLoader], weights: Optional[Sequence[float]] = None,
                 steps_per_epoch: Optional[int] = None):
        self.loaders = [l for l in loaders if l is not None and len(l) > 0]
        if not self.loaders:
            raise ValueError("InterleavedLoader needs at least one non-empty loader")
        w = list(weights) if weights else [1.0] * len(self.loaders)
        w = w[:len(self.loaders)]
        total = sum(w) or 1.0
        self.weights = [x / total for x in w]
        self.steps = steps_per_epoch or sum(len(l) for l in self.loaders)
        self._epoch = 0

    def reset(self):
        """v0.2: restart the interleaving schedule.

        The schedule RNG is seeded by an epoch counter that v0.1 never reset, so the
        arms trained one after another on the same loader object saw *different*
        short/long batch orders (masked_only used schedules 0-29, forward_only 30-59,
        ...). ``pretrain`` now calls this, so every arm sees the same schedule.
        """
        self._epoch = 0

    def __len__(self) -> int:
        return self.steps

    def __iter__(self) -> Iterator[dict]:
        iters = [iter(l) for l in self.loaders]
        # vary the interleaving pattern across epochs (each sub-loader still shuffles
        # internally, but a fixed pattern would pair the same short/long batches up)
        rng = np.random.default_rng(self._epoch)
        self._epoch += 1
        for _ in range(self.steps):
            k = int(rng.choice(len(iters), p=self.weights))
            try:
                yield next(iters[k])
            except StopIteration:
                iters[k] = iter(self.loaders[k])
                yield next(iters[k])


def _to_device(batch: Dict[str, torch.Tensor], device) -> Dict[str, torch.Tensor]:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


@dataclass
class TrainResult:
    monitor: CollapseMonitor
    history: List[Dict[str, float]]
    seconds: float
    checkpoint: Optional[str] = None
    collapsed: bool = False
    collapse_step: Optional[int] = None
    collapse_reason: str = ""
    stopped_early: bool = False


@torch.no_grad()
def check_collapse(model, probe_batch, device, loss_now: float, loss_ref: float,
                   cfg: OptimConfig) -> Optional[str]:
    """Return a reason string if the run has collapsed, else None.

    **The loss is the primary signal.** Targets are LayerNormed encoder outputs, so a
    latent-prediction error at ~2 % of where it started is the constant solution: the
    encoder emits one vector, the predictor memorises it, and the loss goes to zero
    carrying no information. Measured: an EMA-only forward-only run went 0.39 -> 2e-5,
    while the same setup with a variance floor sat at 0.12.

    Embedding statistics are deliberately *not* used as the main gate. On this
    architecture the pooled representation is offset-dominated whether or not it has
    collapsed -- healthy and collapsed runs both sit at cosine similarity ~0.9999, and
    their covariance effective ranks (0.29 vs 0.20 of full) overlap. Gating on those
    would fire on healthy models. The embedding trigger below is kept only for the
    unambiguous case where the representation is numerically dead, which the loss
    signal can miss if training never had a meaningful starting reference.
    """
    if loss_ref > 0 and loss_now < cfg.collapse_loss_frac * loss_ref:
        return (f"loss {loss_now:.6f} fell to "
                f"{100.0 * loss_now / loss_ref:.2f}% of its initial value "
                f"({loss_ref:.6f}) - trivial constant solution")

    if probe_batch is None:
        return None
    z = _probe_embeddings(model, probe_batch, device)
    if z is None or z.size(0) < 4:
        return None
    from .diagnostics import embedding_report
    rep = embedding_report(z)
    # Numerically dead: one direction, no spread. Uses the *absolute* effective rank
    # rather than the ratio -- the ratio is normalised by min(probe_batch, dim), so a
    # ratio threshold silently changes meaning with the probe batch size.
    if (rep["cos_sim"] > 0.99999 and rep["std_mean"] < 1e-5
            and rep["effective_rank"] < 1.5):
        return (f"embeddings numerically dead: cos_sim={rep['cos_sim']:.6f}, "
                f"std={rep['std_mean']:.2e}, erank={rep['effective_rank']:.2f}")
    return None


@torch.no_grad()
def estimate_runtime(model, loader, cfg: ExperimentConfig, device=None,
                     n_steps: int = 10, n_models: int = 1) -> Dict[str, float]:
    """Time a few real steps and project the total, before committing a GPU session.

    Reports seconds/step, per-model and total projections. A Kaggle session is capped
    at ~12 h, so a projection past that is the signal to cut `STAGES`,
    `optim.epochs`, or `optim.max_steps_per_epoch` *before* starting rather than
    discovering it eight hours in.
    """
    device = device or pick_device()
    model.to(device).train()
    opt = torch.optim.AdamW(param_groups(model, cfg.optim.weight_decay), lr=cfg.optim.lr)
    use_amp = bool(cfg.optim.amp and device.type == "cuda")

    it = iter(loader)
    timings: List[float] = []
    for i in range(n_steps + 2):                    # first two steps warm up kernels
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        batch = _to_device(batch, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.enable_grad():
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                has_future = bool(batch.get("has_future", torch.tensor(False)).any())
                if hasattr(model, "forward_predictor"):
                    out = model(batch["ctx"],
                                future_window=batch.get("future") if has_future else None,
                                horizon_s=batch.get("horizon_s") if has_future else None,
                                l2_future=batch.get("l2_future"),
                                l2_horizon_s=batch.get("l2_horizon_s"))
                else:
                    out = model(batch["ctx"])
            out["loss"].backward()
            opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        if i >= 2:
            timings.append(time.time() - t0)

    s_per_step = float(np.median(timings)) if timings else float("nan")
    steps_per_epoch = max(1, len(loader))
    # v0.2 FIX: honour the step cap (v0.1 projected the uncapped 1818 steps/epoch
    # even after applying max_steps_per_epoch, so the "fixed" estimate never changed)
    if cfg.optim.max_steps_per_epoch:
        steps_per_epoch = min(steps_per_epoch, int(cfg.optim.max_steps_per_epoch))
    per_model = s_per_step * steps_per_epoch * cfg.optim.epochs
    if hasattr(loader, "reset"):
        loader.reset()
    return {"s_per_step": s_per_step, "steps_per_epoch": float(steps_per_epoch),
            "steps_per_model": float(steps_per_epoch * cfg.optim.epochs),
            "hours_per_model": per_model / 3600.0,
            "hours_total": per_model * n_models / 3600.0,
            "n_models": float(n_models)}


def budget_steps_per_epoch(s_per_step: float, epochs: int, n_models: int,
                           budget_hours: float = 11.0, minimum: int = 25) -> int:
    """Largest steps/epoch that keeps `n_models` runs inside `budget_hours`."""
    if s_per_step <= 0 or epochs <= 0 or n_models <= 0:
        return minimum
    total_steps = (budget_hours * 3600.0) / s_per_step
    return max(minimum, int(total_steps / (epochs * n_models)))


def report_runtime(est: Dict[str, float], budget_hours: float = 11.0) -> str:
    lines = [
        f"  {est['s_per_step']:.3f} s/step x {est['steps_per_epoch']:.0f} steps/epoch "
        f"x epochs = {est['steps_per_model']:.0f} steps/model",
        f"  ~{est['hours_per_model']:.2f} h per model, "
        f"~{est['hours_total']:.2f} h for {est['n_models']:.0f} model(s)",
    ]
    if est["hours_total"] > budget_hours:
        over = est["hours_total"] / budget_hours
        lines.append(
            f"  >>> {over:.1f}x over a ~{budget_hours:.0f} h session. This will not "
            f"finish; reduce the budget before starting, not eight hours in.")
    return "\n".join(lines)


def pretrain(model: torch.nn.Module,
             loader,
             cfg: ExperimentConfig,
             device: Optional[torch.device] = None,
             arm_name: str = "ema",
             log_every: int = 50,
             probe_batch: Optional[torch.Tensor] = None,
             save_path: Optional[str] = None,
             verbose: bool = True) -> TrainResult:
    """Train any model exposing ``forward(...) -> {'loss': ...}`` and ``update_target``."""
    device = device or pick_device()
    opt_cfg: OptimConfig = cfg.optim
    set_seed(opt_cfg.seed, getattr(opt_cfg, "deterministic", None))
    if hasattr(loader, "reset"):
        loader.reset()                       # v0.2: same batch schedule for every arm
    model.to(device)

    steps_per_epoch = max(1, len(loader))
    if opt_cfg.max_steps_per_epoch:
        steps_per_epoch = min(steps_per_epoch, opt_cfg.max_steps_per_epoch)
    total_steps = steps_per_epoch * opt_cfg.epochs
    warmup_steps = steps_per_epoch * opt_cfg.warmup_epochs

    optimizer = torch.optim.AdamW(param_groups(model, opt_cfg.weight_decay),
                                  lr=opt_cfg.lr, betas=(0.9, 0.95))
    use_amp = bool(opt_cfg.amp and device.type == "cuda")
    try:                                             # torch >= 2.3
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):              # pragma: no cover - older torch
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    monitor = CollapseMonitor(arm=arm_name)
    history: List[Dict[str, float]] = []
    step = 0
    lr = opt_cfg.lr
    t_start = time.time()

    loss_ref: Optional[float] = None          # smoothed loss over the first few steps
    collapsed = False
    collapse_step: Optional[int] = None
    collapse_reason = ""
    guard_on = (opt_cfg.on_collapse in ("stop", "warn")
                and opt_cfg.collapse_check_every > 0)

    for epoch in range(opt_cfg.epochs):
        if collapsed and opt_cfg.on_collapse == "stop":
            break
        model.train()
        agg: Dict[str, float] = {}
        n_batches = 0
        t_epoch = time.time()

        for batch in loader:
            lr = cosine_lr(step, total_steps, warmup_steps, opt_cfg.lr, opt_cfg.min_lr)
            for g in optimizer.param_groups:
                g["lr"] = lr

            batch = _to_device(batch, device)
            ctx = batch["ctx"]
            has_future = bool(batch.get("has_future", torch.tensor(False)).any())
            future = batch.get("future") if has_future else None
            horizon = batch.get("horizon_s") if has_future else None
            l2_future = batch.get("l2_future")
            l2_h = batch.get("l2_horizon_s")

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                if hasattr(model, "forward_predictor"):
                    out = model(ctx, future_window=future, horizon_s=horizon,
                                l2_future=l2_future, l2_horizon_s=l2_h)
                else:                                    # ST-MEM style baseline
                    out = model(ctx)
                loss = out["loss"]

            if not torch.isfinite(loss):
                if verbose:
                    print(f"  [warn] non-finite loss at step {step}, skipping batch")
                step += 1
                continue

            scaler.scale(loss).backward()
            if opt_cfg.grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt_cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            m = ema_momentum(step, total_steps, opt_cfg.ema_start, opt_cfg.ema_end)
            if hasattr(model, "update_target"):
                model.update_target(m)

            for k, v in out.items():
                if k in ("z_ctx",):
                    continue
                val = float(v.detach()) if torch.is_tensor(v) else float(v)
                agg[k] = agg.get(k, 0.0) + val
            n_batches += 1
            step += 1

            # reference loss from the first few steps, once warmup has settled
            if loss_ref is None and step >= 5:
                loss_ref = float(loss.detach())
            elif step < 5:
                loss_ref = None

            if verbose and log_every and step % log_every == 0:
                print(f"  step {step:6d}/{total_steps}  loss={float(loss):.4f}  "
                      f"lr={lr:.2e}  ema={m:.5f}")

            # ---- collapse guard ----------------------------------------------------
            if (guard_on and not collapsed and loss_ref is not None
                    and step % opt_cfg.collapse_check_every == 0):
                reason = check_collapse(model, probe_batch, device, float(loss),
                                        loss_ref, opt_cfg)
                if reason:
                    collapsed = True
                    collapse_step = step
                    collapse_reason = reason
                    print(f"\n  *** COLLAPSE DETECTED at step {step} [{arm_name}] ***")
                    print(f"      {reason}")
                    if opt_cfg.on_collapse == "stop":
                        print(f"      Stopping this run. A collapsed encoder cannot "
                              f"support any downstream probe, so the remaining "
                              f"{total_steps - step} steps would be wasted.")
                        print(f"      If the objective includes L_fwd, it has a trivial "
                              f"constant solution unless L_reg constrains the *pooled* "
                              f"representation: set loss.anti_collapse='vicreg' with "
                              f"loss.reg_target='pooled'.")
                        print(f"      To observe collapse instead of aborting (RQ3), "
                              f"set optim.on_collapse='warn'.\n")
                        break
                    print("      Continuing (on_collapse='warn').\n")

            if opt_cfg.max_steps_per_epoch and n_batches >= opt_cfg.max_steps_per_epoch:
                break

        means = {k: v / max(n_batches, 1) for k, v in agg.items()}
        means["epoch"] = epoch
        means["lr"] = lr
        means["seconds"] = time.time() - t_epoch
        history.append(means)

        z = _probe_embeddings(model, probe_batch, device) if probe_batch is not None else None
        rec = monitor.log(epoch, means.get("loss", float("nan")), z,
                          extra={k: v for k, v in means.items()
                                 if k.startswith(("loss_", "vicreg", "vq_", "sigreg"))})
        if verbose:
            er = rec.get("effective_rank", float("nan"))
            print(f"epoch {epoch:03d}/{opt_cfg.epochs}  loss={means.get('loss', 0):.4f}  "
                  f"fwd={means.get('loss_fwd', 0):.4f}  mask={means.get('loss_mask', 0):.4f}  "
                  f"reg={means.get('loss_reg', 0):.4f}  erank={er:.1f}  "
                  f"{means['seconds']:.1f}s")

    ckpt = None
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.save({"model": model.state_dict(),
                    "config": cfg.to_dict(),
                    "arm": arm_name,
                    "history": history}, save_path)
        ckpt = save_path
        with open(os.path.splitext(save_path)[0] + "_diagnostics.json", "w") as fh:
            json.dump({"history": monitor.history, "summary": monitor.summary(),
                       "train_history": history, "arm": arm_name,
                       "seconds": time.time() - t_start,
                       "collapsed": collapsed, "collapse_step": collapse_step,
                       "collapse_reason": collapse_reason},
                      fh, indent=2, default=float)

    return TrainResult(monitor=monitor, history=history,
                       seconds=time.time() - t_start, checkpoint=ckpt,
                       collapsed=collapsed, collapse_step=collapse_step,
                       collapse_reason=collapse_reason,
                       stopped_early=bool(collapsed and opt_cfg.on_collapse == "stop"))


@torch.no_grad()
def _probe_embeddings(model, probe_batch: torch.Tensor, device) -> Optional[torch.Tensor]:
    """Encoder outputs on a fixed held-out batch -- the substrate for RQ3 diagnostics."""
    if probe_batch is None:
        return None
    was_training = model.training
    model.eval()
    x = probe_batch.to(device)
    if x.dim() == 4:
        x = x[:, -1]
    try:
        z = model.representation(x)
    except Exception:
        z = None
    if was_training:
        model.train()
    return None if z is None else z.detach().float().cpu()


def load_train_result(checkpoint_path: str) -> TrainResult:
    """v0.2: rebuild a ``TrainResult`` from the ``*_diagnostics.json`` written next to
    a checkpoint, so a later Kaggle session can reuse a model trained in an earlier
    one (multi-session runs) without losing its collapse diagnostics."""
    diag = os.path.splitext(checkpoint_path)[0] + "_diagnostics.json"
    with open(diag) as fh:
        d = json.load(fh)
    mon = CollapseMonitor(arm=d.get("arm", "?"))
    mon.history = list(d.get("history", []))
    return TrainResult(monitor=mon, history=list(d.get("train_history", [])),
                       seconds=float(d.get("seconds", 0.0)),
                       checkpoint=checkpoint_path,
                       collapsed=bool(d.get("collapsed", False)),
                       collapse_step=d.get("collapse_step"),
                       collapse_reason=d.get("collapse_reason", ""),
                       stopped_early=False)


def free_cuda(*models):
    """Move models to CPU and release cached GPU memory (between stages)."""
    import gc
    for m in models:
        try:
            m.to("cpu")
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
