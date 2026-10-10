"""v0.2: self-supervised baselines that share Cardio-JEPA's exact CroPA encoder.

A Q1 reviewer will ask how a forward-predicting JEPA compares with

  * **SimCLR** -- instance discrimination with ECG-specific augmentations (the
    contrastive family; Kim 2024 reports it at 0.875 PTB-XL linear AUROC), and
  * **CPC**    -- contrastive *forward* latent prediction (van den Oord 2018; Mehari &
    Strodthoff 2022 found it the strongest SSL objective on PTB-XL). CPC is the closest
    prior art to L_fwd: it also predicts future latents, but contrastively and without
    an EMA target. Cardio-JEPA's novelty claim is only defensible against it.

Both are architecture-matched (same ``CroPAEncoder``, same width/depth, same patching),
expose ``representation(x)`` like ``CardioJEPA``, and train with ``engine.pretrain``
on the same interleaved loader, so the comparison isolates the objective.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import CroPAEncoder
from .transformer import init_weights_


def _encoder_from(signal, model) -> CroPAEncoder:
    return CroPAEncoder(
        embed_dim=model.encoder_embed_dim, depth=model.encoder_depth,
        num_heads=model.encoder_num_heads, mlp_ratio=model.mlp_ratio,
        qkv_bias=model.qkv_bias, drop_rate=model.drop_rate,
        attn_drop_rate=model.attn_drop_rate, drop_path_rate=model.drop_path_rate,
        init_std=model.init_std, pos_type=model.pos_type, use_cropa=model.use_cropa,
        n_leads=signal.n_leads, n_time_patches=signal.n_time_patches,
        patch_samples=signal.patch_samples)


# --------------------------------------------------------------------------------------
# ECG augmentations (operate on per-lead z-scored (B, C, T) tensors)
# --------------------------------------------------------------------------------------
def ecg_augment(x: torch.Tensor, fs: float = 250.0,
                generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Physiologically plausible random transforms (Mehari & Strodthoff 2022 style).

    amplitude scaling per lead, additive Gaussian noise, baseline wander, a random
    circular time shift, random zeroing of one time segment, random lead dropout.
    """
    B, C, T = x.shape
    dev, dt = x.device, x.dtype
    g = generator

    def rand(*shape):
        return torch.rand(*shape, generator=g, device="cpu").to(dev, dt)

    out = x * (0.8 + 0.4 * rand(B, C, 1))                         # amplitude
    out = out + 0.05 * torch.randn(B, C, T, generator=g, device="cpu").to(dev, dt)
    t = torch.arange(T, device=dev, dtype=dt) / fs                # baseline wander
    f = 0.05 + 0.45 * rand(B, 1, 1)
    ph = 2 * math.pi * rand(B, C, 1)
    out = out + (0.1 + 0.2 * rand(B, C, 1)) * torch.sin(2 * math.pi * f * t + ph)

    shifts = (rand(B) * T).long()                                 # circular shift
    idx = (torch.arange(T, device=dev).unsqueeze(0) + shifts.unsqueeze(1)) % T
    out = torch.gather(out, 2, idx.unsqueeze(1).expand(B, C, T))

    seg = int(0.5 * fs)                                           # time cut-out
    starts = (rand(B) * max(T - seg, 1)).long()
    ar = torch.arange(T, device=dev).unsqueeze(0)
    cut = (ar >= starts.unsqueeze(1)) & (ar < (starts + seg).unsqueeze(1))
    keep_cut = rand(B, 1) < 0.5
    out = out.masked_fill((cut & keep_cut).unsqueeze(1), 0.0)

    drop = rand(B, C, 1) < 0.15                                   # lead dropout
    all_dropped = drop.all(dim=1, keepdim=True)                   # keep >= 1 lead
    drop[:, :1] = drop[:, :1] & ~all_dropped
    out = out.masked_fill(drop, 0.0)
    return out


def _no_autocast(device: torch.device):
    """Contrastive logits must be fp32: 1/temperature-scaled fp16 logits overflow."""
    return torch.autocast(device_type=device.type, enabled=False)


def _last_window(x: torch.Tensor) -> torch.Tensor:
    return x[:, -1] if x.dim() == 4 else x


# --------------------------------------------------------------------------------------
# SimCLR
# --------------------------------------------------------------------------------------
class SimCLRBaseline(nn.Module):
    """NT-Xent over two augmented views; projector discarded for evaluation."""

    def __init__(self, signal, model, proj_dim: int = 128, temperature: float = 0.1):
        super().__init__()
        self.signal = signal
        self.temperature = temperature
        self.encoder = _encoder_from(signal, model)
        D = model.encoder_embed_dim
        self.projector = nn.Sequential(nn.Linear(D, D), nn.BatchNorm1d(D),
                                       nn.ReLU(inplace=True), nn.Linear(D, proj_dim))
        self.projector.apply(lambda m: init_weights_(m, model.init_std))

    def representation(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder.representation(_last_window(x))

    def update_target(self, momentum: float):      # API parity with CardioJEPA
        return

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = _last_window(x)
        v1 = ecg_augment(x, self.signal.fs)
        v2 = ecg_augment(x, self.signal.fs)
        z = self.projector(self.encoder.representation(torch.cat([v1, v2])))
        n = x.size(0)
        with _no_autocast(z.device):
            z = F.normalize(z.float(), dim=-1)
            sim = (z @ z.T) / self.temperature
            sim = sim.masked_fill(torch.eye(2 * n, dtype=torch.bool, device=z.device),
                                  -1e9)
            target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
            loss = F.cross_entropy(sim, target)
        with torch.no_grad():
            acc = (sim.argmax(dim=1) == target).float().mean()
        return {"loss": loss, "loss_nce": loss.detach(), "nce_acc": acc}


# --------------------------------------------------------------------------------------
# CPC (intra-window, causal) -- same split as Cardio-JEPA's intra-window L_fwd
# --------------------------------------------------------------------------------------
class CPCBaseline(nn.Module):
    """Contrastive predictive coding inside a 10 s window.

    Context = the first ``ctx_frac`` of the time patches, encoded with the future
    patches *removed* (the CroPA encoder's time mask), so the context never sees the
    future. Targets = per-time-patch latents of the future part, encoded on their own
    (lead-averaged). One prediction head per future offset; InfoNCE with the other
    samples in the batch as negatives. This mirrors Cardio-JEPA's intra-window
    forward objective exactly, except contrastive instead of regression-to-EMA.
    """

    def __init__(self, signal, model, ctx_frac: float = 0.6, temperature: float = 0.1,
                 proj_dim: int = 128):
        super().__init__()
        self.signal = signal
        self.temperature = temperature
        self.encoder = _encoder_from(signal, model)
        P = signal.n_time_patches
        self.k = max(1, min(int(round(P * ctx_frac)), P - 1))
        n_future = P - self.k
        D = model.encoder_embed_dim
        self.ctx_head = nn.Linear(D, proj_dim)
        self.offset_heads = nn.Parameter(torch.randn(n_future, proj_dim, proj_dim) * 0.02)
        self.tgt_head = nn.Linear(D, proj_dim)
        self.ctx_head.apply(lambda m: init_weights_(m, model.init_std))
        self.tgt_head.apply(lambda m: init_weights_(m, model.init_std))

    def representation(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder.representation(_last_window(x))

    def update_target(self, momentum: float):
        return

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = _last_window(x)
        B, C, T = x.shape
        P = self.signal.n_time_patches
        fut_mask = torch.zeros(P, dtype=torch.bool)
        fut_mask[self.k:] = True                                   # True = hidden

        ctx = self.encoder(x, time_mask=fut_mask)                  # (B, C*k, D)
        c = self.ctx_head(ctx.mean(dim=1))                         # (B, d)

        fut = self.encoder(x, time_mask=~fut_mask)                 # (B, C*(P-k), D)
        fut = fut.reshape(B, C, P - self.k, -1).mean(dim=1)        # (B, P-k, D)
        z = self.tgt_head(fut)                                     # (B, P-k, d)

        with _no_autocast(x.device):
            z = F.normalize(z.float(), dim=-1)
            pred = torch.einsum("bd,kde->bke", c.float(), self.offset_heads.float())
            pred = F.normalize(pred, dim=-1)                       # (B, P-k, d)
            logits = torch.einsum("bkd,nkd->kbn", pred, z) / self.temperature
            target = torch.arange(B, device=x.device).unsqueeze(0).expand(P - self.k, B)
            loss = F.cross_entropy(logits.reshape(-1, B), target.reshape(-1))
        with torch.no_grad():
            acc = (logits.argmax(dim=-1) == target).float().mean()
        return {"loss": loss, "loss_nce": loss.detach(), "nce_acc": acc}
