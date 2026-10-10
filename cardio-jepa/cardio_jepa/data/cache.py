"""v0.2: one-time preprocessing cache for evaluation datasets.

Every probe in v0.1 re-read and re-filtered every PTB-XL / anticipation window from
Kaggle's network file system *for every model* (Stage 1 probing alone took ~19 min
for four models, Stage 2 took 54 min). The preprocessing is deterministic for
evaluation datasets (centre crop, no augmentation), so it is done once and stored as
a float16 memmap under ``cfg.data.cache_dir`` (default ``/tmp``, which does **not**
count against Kaggle's 20 GB output limit).

The cache pass also *audits* the data: windows that came back all-zero (a failed
read or an invalid/flat recording silently zero-filled by the loader) are counted
across all DataLoader workers and reported, so a broken reader can no longer hide.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class MemmapWindowDataset(Dataset):
    """(x, y) pairs backed by float16 / float32 memmaps."""

    def __init__(self, x_path: str, y_path: Optional[str], meta: Dict[str, object]):
        self.meta = meta
        self.X = np.load(x_path, mmap_mode="r")
        self.Y = np.load(y_path, mmap_mode="r") if y_path else None

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        x = torch.from_numpy(np.asarray(self.X[idx], dtype=np.float32))
        if self.Y is None:
            return x
        return x, torch.from_numpy(np.asarray(self.Y[idx]))


def _key(name: str, n: int, extra: str = "") -> str:
    h = hashlib.md5(f"{name}|{n}|{extra}".encode()).hexdigest()[:10]
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    return f"{safe}_{h}"


def cache_dataset(ds: Dataset, name: str, cache_dir: str, batch_size: int = 64,
                  num_workers: int = 2, extra_key: str = "", verbose: bool = True
                  ) -> MemmapWindowDataset:
    """Materialise ``ds`` (yielding ``x`` or ``(x, y)``) once; reuse afterwards."""
    os.makedirs(cache_dir, exist_ok=True)
    key = _key(name, len(ds), extra_key)
    x_path = os.path.join(cache_dir, key + "_x.npy")
    y_path = os.path.join(cache_dir, key + "_y.npy")
    meta_path = os.path.join(cache_dir, key + "_meta.json")

    if os.path.exists(meta_path) and os.path.exists(x_path):
        with open(meta_path) as fh:
            meta = json.load(fh)
        if verbose:
            print(f"[cache] {name}: reusing {meta['n']} windows "
                  f"(zero-filled: {meta['zero_frac']:.2%})")
        return MemmapWindowDataset(x_path, y_path if meta.get("has_y") else None, meta)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers)
    X = Y = None
    n_zero = 0
    pos = 0
    has_y = False
    for batch in loader:
        if isinstance(batch, (tuple, list)):
            xb, yb = batch[0], batch[1]
            has_y = True
        else:
            xb, yb = batch, None
        if X is None:
            X = np.lib.format.open_memmap(x_path + ".tmp.npy", mode="w+",
                                          dtype=np.float16,
                                          shape=(len(ds),) + tuple(xb.shape[1:]))
            if yb is not None:
                Y = np.lib.format.open_memmap(y_path + ".tmp.npy", mode="w+",
                                              dtype=np.float32,
                                              shape=(len(ds),) + tuple(yb.shape[1:]))
        b = xb.shape[0]
        flat = xb.reshape(b, -1)
        n_zero += int((flat.abs().sum(dim=1) == 0).sum())
        X[pos:pos + b] = np.clip(xb.numpy(), -6e4, 6e4).astype(np.float16)
        if Y is not None:
            Y[pos:pos + b] = yb.numpy().astype(np.float32)
        pos += b
    if X is None:
        raise RuntimeError(f"[cache] {name}: dataset is empty")
    X.flush()
    del X
    os.replace(x_path + ".tmp.npy", x_path)
    if Y is not None:
        Y.flush()
        del Y
        os.replace(y_path + ".tmp.npy", y_path)
    meta = {"name": name, "n": len(ds), "has_y": has_y,
            "zero_frac": n_zero / max(len(ds), 1), "n_zero": n_zero}
    with open(meta_path, "w") as fh:
        json.dump(meta, fh)
    if verbose:
        flag = "  <-- CHECK THE READER" if meta["zero_frac"] > 0.01 else ""
        print(f"[cache] {name}: cached {len(ds)} windows, zero-filled (failed or "
              f"invalid reads): {n_zero} ({meta['zero_frac']:.2%}){flag}")
    return MemmapWindowDataset(x_path, y_path if has_y else None, meta)


def cache_probe_splits(splits: Dict[str, object], task: str, cache_dir: str,
                       num_workers: int = 2, verbose: bool = True) -> Dict[str, object]:
    """Cache the train/val/test datasets of one probe task in place (returns a copy)."""
    out = dict(splits)
    for part in ("train", "val", "test"):
        ds = splits.get(part)
        if ds is None:
            continue
        out[part] = cache_dataset(ds, f"{task}_{part}", cache_dir,
                                  num_workers=num_workers, verbose=verbose)
    return out
