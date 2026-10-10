"""Linear probing on a frozen encoder (the primary comparison, Section 6.4).

Linear probing removes fine-tuning confounds, so every method is compared on the
representation it actually learned. Adapted from ECG-JEPA's ``linear_probe_utils.py``
(precompute features once, then train a cheap linear head).

v0.2 changes
------------
* The head's learning rate is **selected on a validation split** (PTB-XL fold 9, or a
  record/patient-level inner split of the training data) with early stopping, as in
  Kim's protocol. v0.1 trained a fixed 40 epochs at one learning rate and never used
  the validation fold.
* Sub-100 % label fractions are **repeated** with different random subsets and
  reported as mean +/- std (v0.1 drew one subset; its 1 % AUROC was sometimes higher
  than its 10 % AUROC, i.e. noise).
* Feature extraction runs in eval mode under autocast on GPU (faster, same numbers
  to ~1e-3).
"""

from __future__ import annotations

import copy
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@torch.no_grad()
def precompute_features(encode: Callable[[torch.Tensor], torch.Tensor],
                        loader: DataLoader, device, verbose: bool = False
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
    feats: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    use_amp = dev.type == "cuda"
    for i, batch in enumerate(loader):
        x, y = batch if isinstance(batch, (tuple, list)) else (batch, None)
        x = x.to(dev, non_blocking=True)
        with torch.autocast(device_type=dev.type, enabled=use_amp):
            f = encode(x)
        feats.append(f.detach().float().cpu())
        if y is not None:
            labels.append(y.detach().cpu())
        if verbose and i % 20 == 0:
            print(f"    features {i * loader.batch_size}...")
    F = torch.cat(feats) if feats else torch.zeros(0)
    Y = torch.cat(labels) if labels else torch.zeros(0)
    return F, Y


class LinearHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, batch_norm: bool = True):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_dim, affine=False, eps=1e-6) if batch_norm else nn.Identity()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(self.bn(x))


def macro_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Macro AUROC over label columns, skipping single-class columns."""
    from sklearn.metrics import roc_auc_score
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.ndim == 1 or (y_true.ndim == 2 and y_true.shape[1] == 1):
        yt = y_true.reshape(-1)
        if len(np.unique(yt)) < 2:
            return float("nan")
        return float(roc_auc_score(yt, y_score.reshape(-1)))
    scores = []
    for j in range(y_true.shape[1]):
        if len(np.unique(y_true[:, j])) < 2:
            continue
        scores.append(roc_auc_score(y_true[:, j], y_score[:, j]))
    return float(np.mean(scores)) if scores else float("nan")


def _inner_split(n: int, groups: Optional[np.ndarray], frac: float, seed: int
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Validation carve-out from the training rows -- by group when groups exist."""
    rng = np.random.default_rng(seed + 991)
    if groups is not None:
        g = np.asarray(groups)
        uniq = np.unique(g)
        n_val = max(1, int(round(len(uniq) * frac)))
        val_g = set(rng.permutation(uniq)[:n_val].tolist())
        is_val = np.array([x in val_g for x in g])
    else:
        is_val = np.zeros(n, dtype=bool)
        is_val[rng.permutation(n)[:max(1, int(round(n * frac)))]] = True
    return np.flatnonzero(~is_val), np.flatnonzero(is_val)


def _scores(head, feats, device, multilabel: bool, n_classes: int) -> np.ndarray:
    head.eval()
    with torch.no_grad():
        logits = head(feats.to(device)).float().cpu().numpy()
    if multilabel:
        return 1.0 / (1.0 + np.exp(-logits))
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _auroc_of(y, s, multilabel: bool, n_classes: int) -> float:
    y = np.asarray(y)
    if multilabel:
        return macro_auroc(y, s)
    return macro_auroc(np.eye(n_classes)[y.astype(int).reshape(-1)], s)


def _fit(Ftr, Ytr, Fva, Yva, n_classes, epochs, lr, weight_decay, batch_size, device,
         multilabel, seed, patience):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    ds = TensorDataset(Ftr, Ytr.float())
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=g,
                    drop_last=len(ds) > batch_size)
    head = LinearHead(Ftr.shape[1], n_classes).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    crit = nn.BCEWithLogitsLoss() if multilabel else nn.CrossEntropyLoss()

    best_auc, best_state, best_ep, bad = -1.0, None, -1, 0
    for ep in range(epochs):
        head.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            if xb.size(0) < 2:
                continue
            out = head(xb)
            loss = crit(out, yb if multilabel else yb.long().reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()
        if Fva is not None:
            auc = _auroc_of(Yva.numpy(), _scores(head, Fva, device, multilabel, n_classes),
                            multilabel, n_classes)
            if np.isfinite(auc) and auc > best_auc + 1e-5:
                best_auc, best_state, best_ep, bad = auc, copy.deepcopy(head.state_dict()), ep, 0
            else:
                bad += 1
                if patience and bad >= patience:
                    break
    if best_state is not None:
        head.load_state_dict(best_state)
    return head, best_auc, best_ep


def train_linear_probe(train_feats: torch.Tensor, train_labels: torch.Tensor,
                       test_feats: torch.Tensor, test_labels: torch.Tensor,
                       n_classes: int, epochs: int = 40, lr: float = 1e-3,
                       weight_decay: float = 1e-4, batch_size: int = 128,
                       device=None, multilabel: bool = True,
                       label_fraction: float = 1.0, seed: int = 0,
                       verbose: bool = False,
                       val_feats: Optional[torch.Tensor] = None,
                       val_labels: Optional[torch.Tensor] = None,
                       lr_grid: Optional[Sequence[float]] = None,
                       patience: Optional[int] = 8,
                       train_groups: Optional[np.ndarray] = None,
                       inner_val_frac: float = 0.15) -> Dict[str, object]:
    """Fit a linear head on frozen features; returns metrics and test scores.

    With ``lr_grid`` (v0.2), every learning rate is trained with early stopping on the
    validation AUROC and the best one is used for the test set. If no validation
    features are passed, a validation split is carved out of the (sub-sampled)
    training rows -- by ``train_groups`` (patients / records) when given.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = torch.Generator().manual_seed(seed)
    groups = None if train_groups is None else np.asarray(train_groups)

    if label_fraction < 1.0:
        n = max(n_classes * 2, int(round(len(train_feats) * label_fraction)))
        n = min(n, len(train_feats))
        idx = torch.randperm(len(train_feats), generator=g)[:n]
        train_feats, train_labels = train_feats[idx], train_labels[idx]
        if groups is not None:
            groups = groups[idx.numpy()]

    if lr_grid and val_feats is None and len(train_feats) >= 20:
        tr_i, va_i = _inner_split(len(train_feats), groups, inner_val_frac, seed)
        if len(va_i) and len(tr_i):
            val_feats, val_labels = train_feats[va_i], train_labels[va_i]
            train_feats, train_labels = train_feats[tr_i], train_labels[tr_i]

    use_val = lr_grid and val_feats is not None and len(val_feats) > 1
    grid = list(lr_grid) if use_val else [lr]
    best = (None, -1.0, -1, lr)
    for cand in grid:
        head, vauc, vep = _fit(train_feats, train_labels,
                               val_feats if use_val else None,
                               val_labels if use_val else None,
                               n_classes, epochs, cand, weight_decay, batch_size,
                               device, multilabel, seed, patience if use_val else None)
        if best[0] is None or vauc > best[1]:
            best = (head, vauc, vep, cand)
    head, val_auc, best_ep, best_lr = best

    scores = _scores(head, test_feats, device, multilabel, n_classes)
    y_true = test_labels.numpy()
    auroc = _auroc_of(y_true, scores, multilabel, n_classes)

    out: Dict[str, object] = {"macro_auroc": auroc, "y_true": y_true,
                              "y_score": scores, "n_train": len(train_feats),
                              "label_fraction": label_fraction,
                              "best_lr": best_lr, "val_auroc": val_auc,
                              "best_epoch": best_ep, "head": head.cpu()}
    if multilabel and y_true.ndim == 2 and y_true.shape[1] > 1:
        from sklearn.metrics import f1_score
        out["macro_f1"] = float(f1_score(y_true, (scores >= 0.5).astype(int),
                                         average="macro", zero_division=0))
        from sklearn.metrics import roc_auc_score
        out["per_class_auroc"] = [
            float(roc_auc_score(y_true[:, j], scores[:, j]))
            if len(np.unique(y_true[:, j])) > 1 else float("nan")
            for j in range(y_true.shape[1])]
    if verbose:
        print(f"    linear probe: macro AUROC = {auroc:.4f} "
              f"(n_train={len(train_feats)}, frac={label_fraction}, lr={best_lr:g})")
    return out


def sweep_fractions(Ftr, Ytr, Fte, Yte, n_classes: int, device, fractions,
                    repeats: int = 3, Fva=None, Yva=None, multilabel: bool = True,
                    epochs: int = 40, lr: float = 1e-3, lr_grid=None, patience=8,
                    train_groups=None, seed: int = 0, verbose: bool = False
                    ) -> Dict[float, Dict[str, object]]:
    """Label-efficiency sweep: every fraction < 1 repeated ``repeats`` times."""
    out: Dict[float, Dict[str, object]] = {}
    for frac in fractions:
        reps = 1 if frac >= 1.0 else max(1, repeats)
        runs = [train_linear_probe(Ftr, Ytr, Fte, Yte, n_classes=n_classes,
                                   epochs=epochs, lr=lr, device=device,
                                   multilabel=multilabel, label_fraction=float(frac),
                                   seed=seed + 1000 * r, verbose=verbose,
                                   val_feats=Fva, val_labels=Yva, lr_grid=lr_grid,
                                   patience=patience, train_groups=train_groups)
                for r in range(reps)]
        vals = [r["macro_auroc"] for r in runs]
        rec = dict(runs[0])
        rec["macro_auroc"] = float(np.nanmean(vals)) if vals else float("nan")
        rec["macro_auroc_std"] = float(np.nanstd(vals)) if len(vals) > 1 else 0.0
        rec["macro_auroc_runs"] = [float(v) for v in vals]
        out[float(frac)] = rec
    return out


def probe_encoder(encode: Callable[[torch.Tensor], torch.Tensor],
                  train_ds, test_ds, n_classes: int, device,
                  batch_size: int = 64, epochs: int = 40, lr: float = 1e-3,
                  multilabel: bool = True,
                  label_fractions: Sequence[float] = (1.0,),
                  num_workers: int = 0, seed: int = 0,
                  verbose: bool = True, val_ds=None,
                  lr_grid: Optional[Sequence[float]] = None, patience: int = 8,
                  repeats: int = 1, train_groups=None,
                  return_features: bool = False) -> Dict[str, object]:
    """Precompute features once, then sweep label fractions (label-efficiency curve)."""
    tr_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers)
    te_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers)
    if verbose:
        print(f"    extracting features: {len(train_ds)} train / {len(test_ds)} test"
              + (f" / {len(val_ds)} val" if val_ds is not None else ""))
    Ftr, Ytr = precompute_features(encode, tr_loader, device)
    Fte, Yte = precompute_features(encode, te_loader, device)
    Fva = Yva = None
    if val_ds is not None:
        va_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers)
        Fva, Yva = precompute_features(encode, va_loader, device)

    results: Dict[str, object] = {"feature_dim": int(Ftr.shape[1]) if len(Ftr) else 0}
    results["by_fraction"] = sweep_fractions(
        Ftr, Ytr, Fte, Yte, n_classes, device, label_fractions, repeats=repeats,
        Fva=Fva, Yva=Yva, multilabel=multilabel, epochs=epochs, lr=lr,
        lr_grid=lr_grid, patience=patience, train_groups=train_groups, seed=seed,
        verbose=verbose)
    full = results["by_fraction"].get(1.0) or list(results["by_fraction"].values())[-1]
    results["macro_auroc"] = full["macro_auroc"]
    results["y_true"] = full["y_true"]
    results["y_score"] = full["y_score"]
    results["macro_f1"] = full.get("macro_f1")
    results["per_class_auroc"] = full.get("per_class_auroc")
    results["best_lr"] = full.get("best_lr")
    if return_features:
        results["features"] = {"train": (Ftr, Ytr), "test": (Fte, Yte),
                               "val": (Fva, Yva)}
    return results
