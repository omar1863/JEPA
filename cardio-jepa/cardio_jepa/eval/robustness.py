"""v0.2: robustness and supervised reference evaluations.

* ``run_robustness_probe`` -- the linear head is trained on *clean* features, then
  evaluated on test ECGs corrupted with **real** noise from the MIT-BIH Noise Stress
  Test database (baseline wander ``bw``, electrode motion ``em``, muscle artefact
  ``ma``) at fixed SNRs, and on reduced lead sets (limb leads I+II; single lead II,
  the wearable / patch setting). Falls back to synthetic noise when NSTDB is absent.
* ``train_supervised`` -- the same CroPA encoder trained from scratch with labels:
  the reference every SSL method must be compared with at low label fractions.
"""

from __future__ import annotations

import copy
import os
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ..data import wfdb_lite
from ..data.preprocess import bandpass, resample_signal
from .linear_probe import macro_auroc, precompute_features, train_linear_probe

#: lead indices inside the 8 reduced leads (I, II, V1..V6)
LEAD_SETS: Dict[str, Optional[Sequence[int]]] = {
    "all8": None, "limb_I_II": (0, 1), "single_II": (1,)}


# --------------------------------------------------------------------------------------
# noise bank
# --------------------------------------------------------------------------------------
def load_nstdb_noise(root: Optional[str], fs: float, filter_like_pipeline: bool = True,
                     verbose: bool = True) -> Optional[Dict[str, np.ndarray]]:
    """Read the three pure-noise NSTDB records (bw/em/ma) -> {name: (C, N) at fs}."""
    if not root or not os.path.isdir(root):
        return None
    paths = wfdb_lite.list_records(root, ("hea",))
    bank: Dict[str, np.ndarray] = {}
    for p in paths:
        name = os.path.basename(p).lower()
        if name not in ("bw", "em", "ma"):
            continue
        try:
            rec = wfdb_lite.read_record(p)
        except Exception as exc:
            if verbose:
                print(f"[robustness] could not read {p}: {exc}")
            continue
        x = np.asarray(rec.p_signal, dtype=np.float32).T            # (C, N)
        x = np.nan_to_num(x)
        if filter_like_pipeline:
            x = bandpass(x, rec.fs)
        x = resample_signal(x, rec.fs, fs)
        x = x - x.mean(axis=1, keepdims=True)
        bank[name] = x.astype(np.float32)
    if verbose and bank:
        print(f"[robustness] NSTDB noise records: {sorted(bank)} "
              f"({min(v.shape[1] for v in bank.values()) / fs / 60:.0f} min each)")
    return bank or None


def synthetic_noise_bank(fs: float, minutes: float = 10.0, seed: int = 0
                         ) -> Dict[str, np.ndarray]:
    """Fallback noise: baseline wander, electrode-motion-like bursts, EMG-like noise."""
    rng = np.random.default_rng(seed)
    n = int(minutes * 60 * fs)
    t = np.arange(n) / fs
    bw = sum(rng.uniform(0.3, 1.0) * np.sin(2 * np.pi * rng.uniform(0.05, 0.5) * t
                                            + rng.uniform(0, 6.28)) for _ in range(4))
    em = np.cumsum(rng.normal(0, 1, n)) * 0.02
    em = em - np.convolve(em, np.ones(int(fs)) / fs, mode="same")
    em = em + (rng.random(n) < 2.0 / fs) * rng.normal(0, 3, n)
    ma = rng.normal(0, 1, n)
    out = {}
    for k, v in (("bw", bw), ("em", em), ("ma", ma)):
        v = np.stack([v, np.roll(v, n // 3)]).astype(np.float32)
        out[k] = bandpass(v, fs) if k != "bw" else v
    return out


class CorruptedDataset(Dataset):
    """Wrap an (x, y) dataset; add noise at ``snr_db`` and/or keep only ``lead_keep``.

    Deterministic per item (seeded by index), so every model sees the *same*
    corrupted test set.
    """

    def __init__(self, base: Dataset, noise: Optional[np.ndarray] = None,
                 snr_db: Optional[float] = None,
                 lead_keep: Optional[Sequence[int]] = None, seed: int = 0):
        self.base = base
        self.noise = noise
        self.snr_db = snr_db
        self.lead_keep = None if lead_keep is None else list(lead_keep)
        self.seed = seed

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y = self.base[idx]
        x = x.clone().float()
        C, T = x.shape[-2], x.shape[-1]
        if self.noise is not None and self.snr_db is not None:
            rng = np.random.default_rng(self.seed * 1_000_003 + idx)
            nz = np.zeros((C, T), dtype=np.float32)
            N = self.noise.shape[1]
            for c in range(C):
                ch = c % self.noise.shape[0]
                off = int(rng.integers(0, max(N - T, 1)))
                seg = self.noise[ch, off:off + T]
                if seg.shape[0] < T:
                    seg = np.pad(seg, (0, T - seg.shape[0]))
                nz[c] = seg
            nz = torch.from_numpy(nz)
            ps = (x ** 2).mean(dim=-1, keepdim=True).clamp_min(1e-8)
            pn = (nz ** 2).mean(dim=-1, keepdim=True).clamp_min(1e-8)
            scale = torch.sqrt(ps / (pn * (10.0 ** (self.snr_db / 10.0))))
            x = x + nz * scale
        if self.lead_keep is not None:
            mask = torch.zeros(C, 1)
            mask[self.lead_keep] = 1.0
            x = x * mask
        return x, y


def run_robustness_probe(model, splits: Dict[str, object], device, eval_cfg,
                         noise_bank: Dict[str, np.ndarray],
                         snrs_db: Sequence[float] = (24.0, 12.0, 6.0, 0.0),
                         lead_sets: Optional[Dict[str, Optional[Sequence[int]]]] = None,
                         batch_size: int = 64, num_workers: int = 0, seed: int = 0,
                         verbose: bool = True):
    """Clean-trained linear head, evaluated under noise and missing leads."""
    import pandas as pd
    lead_sets = LEAD_SETS if lead_sets is None else lead_sets
    model.eval()
    enc = model.representation
    n_cls = int(splits["n_classes"])

    def feats(ds):
        return precompute_features(enc, DataLoader(ds, batch_size=batch_size,
                                                   shuffle=False,
                                                   num_workers=num_workers), device)

    Ftr, Ytr = feats(splits["train"])
    Fva, Yva = feats(splits["val"]) if splits.get("val") is not None else (None, None)
    Fte, Yte = feats(splits["test"])
    head_res = train_linear_probe(Ftr, Ytr, Fte, Yte, n_classes=n_cls,
                                  epochs=eval_cfg.linear_probe_epochs,
                                  lr=eval_cfg.linear_probe_lr, device=device,
                                  val_feats=Fva, val_labels=Yva,
                                  lr_grid=getattr(eval_cfg, "linear_probe_lr_grid", None),
                                  patience=getattr(eval_cfg, "linear_probe_patience", 8),
                                  seed=seed)
    # the head is fit once on clean features and then only *applied* to corrupted ones
    head = head_res["head"].to(device).eval()
    rows = [{"condition": "clean", "noise": "none", "snr_db": np.inf, "leads": "all8",
             "macro_auroc": head_res["macro_auroc"]}]

    def eval_on(ds):
        Fc, Yc = feats(ds)
        with torch.no_grad():
            s = torch.sigmoid(head(Fc.to(device)).float()).cpu().numpy()
        return macro_auroc(Yc.numpy(), s)

    for nname, arr in noise_bank.items():
        for snr in snrs_db:
            ds = CorruptedDataset(splits["test"], arr, snr, None, seed=seed)
            a = eval_on(ds)
            rows.append({"condition": f"{nname}@{snr:g}dB", "noise": nname,
                         "snr_db": float(snr), "leads": "all8", "macro_auroc": a})
            if verbose:
                print(f"    {nname:3s} {snr:5.1f} dB  AUROC={a:.4f}")
    for lname, keep in lead_sets.items():
        if keep is None:
            continue
        ds = CorruptedDataset(splits["test"], None, None, keep, seed=seed)
        a = eval_on(ds)
        rows.append({"condition": f"leads:{lname}", "noise": "none", "snr_db": np.inf,
                     "leads": lname, "macro_auroc": a})
        if verbose:
            print(f"    leads {lname:10s} AUROC={a:.4f}")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# supervised-from-scratch reference
# --------------------------------------------------------------------------------------
class SupervisedEncoder(nn.Module):
    def __init__(self, encoder: nn.Module, n_classes: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.embed_dim, n_classes)

    def representation(self, x):
        if x.dim() == 4:
            x = x[:, -1]
        return self.encoder.representation(x)

    def forward(self, x):
        return self.head(self.representation(x))


def train_supervised(signal, model_cfg, splits: Dict[str, object], device,
                     label_fraction: float = 1.0, max_epochs: int = 20,
                     min_steps: int = 300, lr: float = 3e-4, batch_size: int = 64,
                     num_workers: int = 2, seed: int = 0, patience: int = 4,
                     verbose: bool = True) -> Dict[str, object]:
    """Train CroPA encoder + linear head end-to-end on (a fraction of) the labels.

    Model selection on the validation split (best epoch); test AUROC of that epoch.
    """
    from ..models.ssl_baselines import _encoder_from
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    n_cls = int(splits["n_classes"])
    model = SupervisedEncoder(_encoder_from(signal, model_cfg), n_cls).to(dev)

    tr = splits["train"]
    if label_fraction < 1.0:
        g = np.random.default_rng(seed)
        n = max(n_cls * 4, int(round(len(tr) * label_fraction)))
        idx = np.sort(g.choice(len(tr), size=min(n, len(tr)), replace=False))
        tr = torch.utils.data.Subset(tr, idx.tolist())
    steps_per_epoch = max(1, len(tr) // batch_size)
    epochs = int(min(max_epochs * 10, max(max_epochs, np.ceil(min_steps / steps_per_epoch))))
    dl = DataLoader(tr, batch_size=batch_size, shuffle=True, drop_last=len(tr) > batch_size,
                    num_workers=num_workers,
                    generator=torch.Generator().manual_seed(seed))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    total = epochs * max(1, len(dl))
    warm = max(1, int(0.1 * total))

    def _lr_lambda(step):
        if step < warm:
            return (step + 1) / warm
        p = (step - warm) / max(1, total - warm)
        return 0.5 * (1.0 + np.cos(np.pi * min(p, 1.0)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    crit = nn.BCEWithLogitsLoss()

    def scores(ds):
        model.eval()
        out_s, out_y = [], []
        with torch.no_grad():
            for x, y in DataLoader(ds, batch_size=128, shuffle=False,
                                   num_workers=num_workers):
                with torch.autocast(device_type=dev.type, enabled=use_amp):
                    out_s.append(torch.sigmoid(model(x.to(dev)).float()).cpu())
                out_y.append(y)
        return torch.cat(out_y).numpy(), torch.cat(out_s).numpy()

    best_auc, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        for x, y in dl:
            x, y = x.to(dev), y.float().to(dev)
            with torch.autocast(device_type=dev.type, enabled=use_amp):
                loss = crit(model(x).float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
        if splits.get("val") is not None:
            yv, sv = scores(splits["val"])
            auc = macro_auroc(yv, sv)
            if auc > best_auc + 1e-4:
                best_auc, best_state, bad = auc, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
            if verbose:
                print(f"    [supervised {label_fraction:.0%}] epoch {ep:02d} val AUROC={auc:.4f}")
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    yt, st = scores(splits["test"])
    return {"model": model, "macro_auroc": macro_auroc(yt, st), "y_true": yt,
            "y_score": st, "val_auroc": best_auc, "label_fraction": label_fraction,
            "n_train": len(tr), "epochs_run": ep + 1}
