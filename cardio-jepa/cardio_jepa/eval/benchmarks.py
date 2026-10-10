"""Downstream benchmarks from Section 6.2.

  1. ``run_classification_probe``  -- PTB-XL / CPSC2018 multi-label (sanity check)
  2. ``run_anticipation_probe``    -- near-term rhythm/arrhythmia onset (H1)
  3. ``run_burden_probe``          -- AF-burden regression (H2)
  4. ``forecast_skill``            -- is the forward predictor forecasting at all?

Every probe runs on a **frozen** encoder. For Cardio-JEPA the feature vector can
optionally be extended with the predictor's forecast latents
(``feature_mode="world"``), which is the operational form of the world-model
claim: the anticipation head sees not only the current cardiac state but where
the model thinks that state is heading.

v0.2 changes are marked inline; the important ones:
  * anticipation CIs are a **record-level cluster bootstrap** (windows from one Holter
    record are not independent);
  * the probe's learning rate is chosen on a record-level inner validation split;
  * ``forecast_skill`` measures prediction and persistence in the *same* target space
    the model was trained on, adds a horizon-shuffle control, and must be run on
    records held out from pretraining.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import EvalConfig, SignalConfig
from ..data.longform import AnticipationDataset, LongRecord, WindowSample
from ..data.preprocess import expand_to_n_leads, prepare_window
from .linear_probe import macro_auroc, probe_encoder, sweep_fractions, train_linear_probe
from .stats import bootstrap_auroc


def make_feature_fn(model, horizons_s: Sequence[float] = (),
                    mode: str = "context") -> Callable[[torch.Tensor], torch.Tensor]:
    """Frozen-encoder feature extractor.

    mode='context': mean-pooled encoder embedding only (works for every model,
                    including the ST-MEM, SimCLR, CPC and supervised baselines).
                    For multi-window inputs (B, W, C, T) v0.2 pools *all* context
                    windows (v0.1 used only the last one, so 'context' saw 10 s while
                    'world' saw 20 s -- an unfair comparison).
    mode='world':   [context embedding | predicted future latent per horizon],
                    available only for models with a forward predictor.
    """
    if mode == "world" and hasattr(model, "anticipation_features") and horizons_s:
        def fn(x: torch.Tensor) -> torch.Tensor:
            return model.anticipation_features(x, horizons_s)
        return fn

    def fn(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            B, W, C, T = x.shape
            return model.representation(x.reshape(B * W, C, T)).reshape(B, W, -1).mean(1)
        return model.representation(x)
    return fn


@torch.no_grad()
def _extract(feature_fn, ds, device, batch_size: int = 64, num_workers: int = 0
             ) -> Tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    feats, labels = [], []
    for x, y in loader:
        with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
            f = feature_fn(x.to(dev))
        feats.append(f.detach().float().cpu())
        labels.append(y.detach().cpu())
    if not feats:
        return torch.zeros(0), torch.zeros(0)
    return torch.cat(feats), torch.cat(labels)


# --------------------------------------------------------------------------------------
# 1. standard diagnostic classification
# --------------------------------------------------------------------------------------
def run_classification_probe(model, splits: Dict[str, object], device,
                             eval_cfg: EvalConfig, batch_size: int = 64,
                             num_workers: int = 0, verbose: bool = True,
                             return_features: bool = False) -> Dict[str, object]:
    model.eval()
    fn = make_feature_fn(model, mode="context")
    groups = (splits.get("groups") or {}).get("train") if isinstance(
        splits.get("groups"), dict) else None
    res = probe_encoder(fn, splits["train"], splits["test"],
                        n_classes=int(splits["n_classes"]), device=device,
                        batch_size=batch_size, epochs=eval_cfg.linear_probe_epochs,
                        lr=eval_cfg.linear_probe_lr, multilabel=True,
                        label_fractions=eval_cfg.label_fractions,
                        num_workers=num_workers, verbose=verbose,
                        val_ds=splits.get("val"),
                        lr_grid=getattr(eval_cfg, "linear_probe_lr_grid", None),
                        patience=getattr(eval_cfg, "linear_probe_patience", 8),
                        repeats=getattr(eval_cfg, "label_fraction_repeats", 1),
                        train_groups=groups, return_features=return_features)
    res["label_names"] = splits.get("label_names")
    return res


# --------------------------------------------------------------------------------------
# 2. anticipation
# --------------------------------------------------------------------------------------
def run_anticipation_probe(model, records: Sequence[LongRecord],
                           train_samples: Sequence[WindowSample],
                           test_samples: Sequence[WindowSample],
                           signal: SignalConfig, device, eval_cfg: EvalConfig,
                           context_windows: int = 1,
                           feature_mode: str = "world",
                           horizons_s: Sequence[float] = (),
                           batch_size: int = 64, num_workers: int = 0,
                           verbose: bool = True,
                           train_ds=None, test_ds=None) -> Dict[str, object]:
    """Binary anticipation probe on a frozen encoder.

    v0.2: returns ``groups`` (record id per test window) and a record-level cluster
    bootstrap CI; the head's lr is chosen on a record-level inner validation split.
    ``train_ds`` / ``test_ds`` may be passed pre-cached (see ``data.cache``).
    """
    model.eval()
    if not train_samples or not test_samples:
        return {"auroc": float("nan"), "note": "insufficient samples"}

    tr_ds = train_ds or AnticipationDataset(records, train_samples, signal, context_windows)
    te_ds = test_ds or AnticipationDataset(records, test_samples, signal, context_windows)
    fn = make_feature_fn(model, horizons_s, mode=feature_mode)

    Ftr, Ytr = _extract(fn, tr_ds, device, batch_size, num_workers)
    Fte, Yte = _extract(fn, te_ds, device, batch_size, num_workers)
    if len(Ftr) == 0 or len(Fte) == 0 or len(np.unique(Yte.numpy())) < 2:
        return {"auroc": float("nan"), "note": "degenerate split (single-class test set)"}

    tr_groups = np.asarray([s.record for s in train_samples])
    te_groups = np.asarray([s.record for s in test_samples])
    by = sweep_fractions(Ftr, Ytr.reshape(-1, 1), Fte, Yte.reshape(-1, 1), 1, device,
                         eval_cfg.label_fractions,
                         repeats=getattr(eval_cfg, "label_fraction_repeats", 1),
                         multilabel=True, epochs=eval_cfg.linear_probe_epochs,
                         lr=eval_cfg.linear_probe_lr,
                         lr_grid=getattr(eval_cfg, "linear_probe_lr_grid", None),
                         patience=getattr(eval_cfg, "linear_probe_patience", 8),
                         train_groups=tr_groups)
    best = by.get(1.0) or list(by.values())[-1]
    by_frac = {f: r["macro_auroc"] for f, r in by.items()}

    y_true = Yte.numpy().reshape(-1)
    y_score = np.asarray(best["y_score"]).reshape(-1)
    ci = bootstrap_auroc(y_true, y_score, n=eval_cfg.bootstrap_n, groups=te_groups)
    try:
        from sklearn.metrics import average_precision_score
        auprc = float(average_precision_score(y_true, y_score))
    except Exception:
        auprc = float("nan")

    out = {"auroc": ci["auroc"], "ci_lo": ci["lo"], "ci_hi": ci["hi"],
           "auprc": auprc, "by_fraction": by_frac,
           "by_fraction_std": {f: r.get("macro_auroc_std", 0.0) for f, r in by.items()},
           "y_true": y_true, "y_score": y_score, "groups": te_groups,
           "n_train": len(train_samples), "n_test": len(test_samples),
           "n_test_records": int(len(np.unique(te_groups))),
           "prevalence": float(y_true.mean()), "feature_mode": feature_mode,
           "feature_dim": int(Ftr.shape[1]), "best_lr": best.get("best_lr")}
    if verbose:
        print(f"    anticipation [{feature_mode}]: AUROC = {ci['auroc']:.4f} "
              f"[{ci['lo']:.4f}, {ci['hi']:.4f}] (record-level CI)  "
              f"AUPRC = {auprc:.4f}  n_test={len(test_samples)} "
              f"({out['n_test_records']} records)  prev={out['prevalence']:.3f}")
    return out


# --------------------------------------------------------------------------------------
# 3. AF-burden regression
# --------------------------------------------------------------------------------------
def run_burden_probe(model, records: Sequence[LongRecord],
                     train_samples: Sequence[WindowSample],
                     test_samples: Sequence[WindowSample],
                     signal: SignalConfig, device, context_windows: int = 6,
                     feature_mode: str = "world", horizons_s: Sequence[float] = (),
                     batch_size: int = 32, num_workers: int = 0,
                     alpha: float = 1.0, verbose: bool = True) -> Dict[str, object]:
    """Ridge regression from a frozen embedding to whole-record AF burden.

    v0.2: alpha chosen by grouped (record-level) cross-validation on the training
    records; metrics also reported at the record level (mean prediction per record),
    since burden is a per-record target.
    """
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import GroupKFold

    model.eval()
    if not train_samples or not test_samples:
        return {"r2": float("nan"), "note": "insufficient samples"}

    tr_ds = AnticipationDataset(records, train_samples, signal, context_windows)
    te_ds = AnticipationDataset(records, test_samples, signal, context_windows)
    fn = make_feature_fn(model, horizons_s, mode=feature_mode)

    Ftr, Ytr = _extract(fn, tr_ds, device, batch_size, num_workers)
    Fte, Yte = _extract(fn, te_ds, device, batch_size, num_workers)
    if len(Ftr) == 0 or len(Fte) == 0:
        return {"r2": float("nan"), "note": "no features"}

    Xtr, ytr = Ftr.numpy(), Ytr.numpy()
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Fte.numpy() - mu) / sd
    gtr = np.asarray([s.record for s in train_samples])
    best_alpha, best_err = alpha, np.inf
    n_splits = min(5, len(np.unique(gtr)))
    if n_splits >= 2:
        for a in (0.1, 1.0, 10.0, 100.0, 1000.0):
            errs = []
            for tri, vai in GroupKFold(n_splits=n_splits).split(Xtr, ytr, gtr):
                m = Ridge(alpha=a).fit(Xtr[tri], ytr[tri])
                errs.append(mean_absolute_error(ytr[vai], np.clip(m.predict(Xtr[vai]), 0, 1)))
            if np.mean(errs) < best_err:
                best_alpha, best_err = a, float(np.mean(errs))
    reg = Ridge(alpha=best_alpha).fit(Xtr, ytr)
    pred = np.clip(reg.predict(Xte), 0.0, 1.0)
    y = Yte.numpy()
    gte = np.asarray([s.record for s in test_samples])
    rec_ids = np.unique(gte)
    y_rec = np.array([y[gte == r].mean() for r in rec_ids])
    p_rec = np.array([pred[gte == r].mean() for r in rec_ids])

    out: Dict[str, object] = {
        "r2": float(r2_score(y, pred)) if len(np.unique(y)) > 1 else float("nan"),
        "mae": float(mean_absolute_error(y, pred)),
        "r2_record": float(r2_score(y_rec, p_rec)) if len(rec_ids) > 2 else float("nan"),
        "mae_record": float(mean_absolute_error(y_rec, p_rec)),
        "n_test_records": int(len(rec_ids)), "alpha": best_alpha,
        "y_true": y, "y_pred": pred, "feature_mode": feature_mode,
        "n_train": len(train_samples), "n_test": len(test_samples)}
    try:
        from scipy.stats import spearmanr
        out["spearman"] = float(spearmanr(y, pred).correlation)
        out["spearman_record"] = (float(spearmanr(y_rec, p_rec).correlation)
                                  if len(rec_ids) > 2 else float("nan"))
    except Exception:
        out["spearman"] = out["spearman_record"] = float("nan")
    if verbose:
        print(f"    AF burden [{feature_mode}]: R2 = {out['r2']:.3f}  "
              f"MAE = {out['mae']:.3f}  rho = {out['spearman']:.3f}  "
              f"(record-level R2 {out['r2_record']:.3f}, n={len(rec_ids)})")
    return out


# --------------------------------------------------------------------------------------
# 4. forecast skill: is the predictor doing anything beyond copying the present?
# --------------------------------------------------------------------------------------
@torch.no_grad()
def forecast_skill(model, records: Sequence[LongRecord], signal: SignalConfig, device,
                   horizons_s: Sequence[float] = (10.0, 60.0, 300.0, 600.0),
                   n_samples: int = 256, context_windows: int = 1, seed: int = 0,
                   verbose: bool = True, n_boot: int = 500):
    """Compare the predictor against persistence and a horizon-shuffle control.

    For each (context, true future) pair, all in the *target space L_fwd is trained
    on* (LayerNormed target-encoder tokens, mean-pooled; v0.1 compared raw target
    tokens against raw online-encoder tokens, i.e. two different spaces):

      * ``cos_pred``        cos( g_phi(z_ctx, h),            target(x_future) )
      * ``cos_persistence`` cos( target(x_last_context_window), target(x_future) )
      * ``cos_shuffled``    cos( g_phi(z_ctx, h_wrong),      target(x_future) )

    ``skill = cos_pred - cos_persistence``; ``horizon_gain = cos_pred - cos_shuffled``.
    skill <= 0 means the predictor adds nothing over copying the present;
    horizon_gain ~ 0 means the horizon token is ignored (one forecast for every H).
    Embeddings are batch-centred (a shared offset otherwise pins every cosine near 1).

    **Pass only records held out from pretraining** -- on training records this
    measures memorisation, not forecasting (the v0.1 notebook used all records).
    """
    import pandas as pd
    from ..models.forward_predictor import horizon_to_bucket

    model.eval()
    if not hasattr(model, "forward_predictor") or not hasattr(model, "target_embedding"):
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    usable = [r for r in records if r.duration_s > 0]
    if not usable:
        return pd.DataFrame()
    trained = set(model.trained_horizons()) if hasattr(model, "trained_horizons") else set()

    def centred(a):
        return (a - a.mean(0, keepdim=True)).float()

    def cosv(a, b):
        return torch.nn.functional.cosine_similarity(centred(a), centred(b), dim=-1)

    rows = []
    for H in horizons_s:
        ctxs, futs = [], []
        need_s = context_windows * signal.window_s + H + signal.window_s
        cand = [r for r in usable if r.duration_s > need_s]
        if not cand:
            if verbose:
                print(f"  horizon {H:.0f}s: no record long enough, skipping")
            continue
        for _ in range(n_samples):
            rec = cand[int(rng.integers(len(cand)))]
            t0 = float(rng.uniform(0.0, rec.duration_s - need_s))
            start = int(t0 * rec.fs)
            win_n = int(round(signal.window_s * rec.fs))

            def prep(s):
                raw = rec.read(s, win_n)
                x = prepare_window(raw, rec.fs, signal.fs, signal.window_samples,
                                   reduced=(rec.n_channels >= 12))
                return expand_to_n_leads(x, signal.n_leads)

            ctxs.append(np.stack([prep(start + w * win_n) for w in range(context_windows)]))
            futs.append(prep(start + context_windows * win_n + int(round(H * rec.fs))))
        if len(ctxs) < 8:
            continue

        ctx = torch.from_numpy(np.stack(ctxs)).float().to(device)
        fut = torch.from_numpy(np.stack(futs)).float().to(device)

        pred = model.forecast_latent(ctx, H).float()
        true = model.target_embedding(fut)
        persist = model.target_embedding(ctx[:, -1])
        # horizon-shuffle control: the trained bucket farthest (in log space) from H
        others = [h for h in model.horizon_buckets
                  if (not trained or h in trained)
                  and int(horizon_to_bucket(torch.tensor([h]), model.horizon_buckets)[0])
                  != int(horizon_to_bucket(torch.tensor([H]), model.horizon_buckets)[0])]
        if others:
            h_wrong = max(others, key=lambda h: abs(np.log(h) - np.log(H)))
            shuffled = model.forecast_latent(ctx, h_wrong).float()
        else:
            h_wrong, shuffled = float("nan"), pred

        cp, cq, cs = cosv(pred, true), cosv(persist, true), cosv(shuffled, true)
        skill = (cp - cq).cpu().numpy()
        gain = (cp - cs).cpu().numpy()
        b_rng = np.random.default_rng(seed + 7)
        boots = [skill[b_rng.integers(0, len(skill), len(skill))].mean()
                 for _ in range(n_boot)]
        rows.append({"horizon_s": H, "cos_pred": float(cp.mean()),
                     "cos_persistence": float(cq.mean()),
                     "cos_shuffled": float(cs.mean()), "h_wrong_s": h_wrong,
                     "skill": float(skill.mean()),
                     "skill_lo": float(np.percentile(boots, 2.5)),
                     "skill_hi": float(np.percentile(boots, 97.5)),
                     "horizon_gain": float(gain.mean()),
                     "horizon_trained": (not trained) or any(
                         int(horizon_to_bucket(torch.tensor([H]), model.horizon_buckets)[0])
                         == int(horizon_to_bucket(torch.tensor([t]), model.horizon_buckets)[0])
                         for t in trained),
                     "n": len(ctxs)})
        if verbose:
            r = rows[-1]
            print(f"  horizon {H:6.0f}s  cos_pred={r['cos_pred']:+.4f}  "
                  f"persistence={r['cos_persistence']:+.4f}  "
                  f"skill={r['skill']:+.4f} [{r['skill_lo']:+.4f},{r['skill_hi']:+.4f}]  "
                  f"horizon_gain={r['horizon_gain']:+.4f}"
                  + ("" if r["horizon_trained"] else "  (UNTRAINED BUCKET)"))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# horizon sweep (H2)
# --------------------------------------------------------------------------------------
def horizon_sweep(model_fns: Dict[str, object], records: Sequence[LongRecord],
                  signal: SignalConfig, device, eval_cfg: EvalConfig,
                  horizons_s: Sequence[float], builder, context_windows: int = 1,
                  test_frac: float = 0.3, seed: int = 0, verbose: bool = True,
                  test_ids: Optional[Sequence[int]] = None):
    """Run an anticipation benchmark at several horizons for several models.

    H2 predicts the hierarchical-vs-single-scale gap *widens monotonically* with
    the horizon, so the sweep is the actual test, not a nice-to-have.

    v0.2: pass ``test_ids`` (the fixed pre-pretraining holdout) so every horizon uses
    the same test records, none of which were seen in pretraining.
    """
    import pandas as pd
    from ..data.longform import split_by_ids, split_by_record

    rows = []
    for H in horizons_s:
        samples, stats = builder(records, signal, horizon_s=H,
                                 context_windows=context_windows)
        if stats["n_positive"] < 5 or stats["n_negative"] < 5:
            if verbose:
                print(f"  horizon {H:.0f}s: too few samples "
                      f"(pos={stats['n_positive']:.0f}, neg={stats['n_negative']:.0f}), skipping")
            continue
        if test_ids is not None:
            tr, te = split_by_ids(samples, test_ids)
        else:
            tr, te = split_by_record(samples, len(records), test_frac=test_frac, seed=seed)
        for name, spec in model_fns.items():
            model = spec["model"] if isinstance(spec, dict) else spec
            mode = spec.get("feature_mode", "context") if isinstance(spec, dict) else "context"
            r = run_anticipation_probe(model, records, tr, te, signal, device, eval_cfg,
                                       context_windows=context_windows,
                                       feature_mode=mode, horizons_s=[H],
                                       verbose=False)
            rows.append({"horizon_s": H, "model": name, "auroc": r.get("auroc"),
                         "ci_lo": r.get("ci_lo"), "ci_hi": r.get("ci_hi"),
                         "auprc": r.get("auprc"),
                         "prevalence": r.get("prevalence"), "n_test": r.get("n_test")})
            if verbose:
                print(f"  horizon {H:6.0f}s  {name:28s} AUROC={r.get('auroc', float('nan')):.4f}")
    return pd.DataFrame(rows)
