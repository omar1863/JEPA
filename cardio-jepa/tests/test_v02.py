"""Regression tests for every v0.2 fix. Run with: python -m pytest tests/ -q"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import pytest
import torch

import cardio_jepa as cj
from cardio_jepa.config import smoke_config
from cardio_jepa.data.build import build_pretrain_loaders, build_probe_splits, probe_batch_from_loader
from cardio_jepa.data.cache import cache_dataset
from cardio_jepa.data.discovery import discover
from cardio_jepa.data.longform import (LongRecord, build_onset_dataset, split_by_ids,
                                       split_long_records, synthetic_long_records)
from cardio_jepa.data.short import ArrayWindowDataset
from cardio_jepa.data.synthetic import synth_short_corpus
from cardio_jepa.eval.stats import bootstrap_auroc, paired_seed_test, seed_level_summary
from cardio_jepa.models import CardioJEPA, CPCBaseline, SimCLRBaseline, STMEMBaseline
from cardio_jepa.models import transformer as tr
from cardio_jepa.regularizers import SoftVQ


def _cfg():
    cfg = smoke_config()
    cfg.data.allow_synthetic_fallback = True
    cfg.data.input_root = "/nonexistent-input"
    return cfg


def _model(cfg):
    return CardioJEPA(cfg.signal, cfg.model, cfg.loss, cfg.hierarchical, cfg.forward)


# ------------------------------------------------------------------- memory / numerics
def test_sdpa_attention_matches_explicit_softmax():
    torch.manual_seed(0)
    att = tr.Attention(32, num_heads=4, qkv_bias=True).eval()
    x = torch.randn(3, 20, 32)
    mask = (torch.rand(20, 20) > 0.5).float()
    mask.fill_diagonal_(1.0)
    old = tr.USE_SDPA
    try:
        tr.USE_SDPA = True
        a = att(x, mask)
        tr.USE_SDPA = False
        b = att(x, mask)
    finally:
        tr.USE_SDPA = old
    assert torch.allclose(a, b, atol=1e-5)


# ------------------------------------------------------------------- leakage guards
def test_probe_dataset_in_pretrain_raises():
    cfg = _cfg()
    cfg.data.pretrain_datasets = ("chapman", "ptbxl")
    with pytest.raises(ValueError, match="probe datasets"):
        build_pretrain_loaders({}, cfg, verbose=False)


def test_long_holdout_is_disjoint_and_used_by_split_by_ids():
    cfg = _cfg()
    recs = synthetic_long_records(10, 30.0, cfg.signal.fs, cfg.signal.n_leads, seed=0)
    sp = split_long_records(recs, test_frac=0.3, seed=0)
    assert set(sp.train_ids).isdisjoint(sp.test_ids)
    assert len(sp.test_ids) == 3
    assert sorted(sp.train_ids + sp.test_ids) == list(range(10))
    samples, _ = build_onset_dataset(recs, cfg.signal, horizon_s=60.0, stride_s=30.0,
                                     max_per_record=40)
    trs, tes = split_by_ids(samples, sp.test_ids)
    assert {s.record for s in tes} <= set(sp.test_ids)
    assert not ({s.record for s in trs} & set(sp.test_ids))


def test_pretraining_never_sees_holdout_records():
    cfg = _cfg()
    recs = synthetic_long_records(6, 30.0, cfg.signal.fs, cfg.signal.n_leads, seed=0)
    sp = split_long_records(recs, test_frac=0.34, seed=0)
    _, info = build_pretrain_loaders({}, cfg, long_records=sp.train_records(recs),
                                     samples_per_record=8, verbose=False)
    assert not (set(info["pretrain_long_record_names"]) & set(sp.test_names))


# ------------------------------------------------------------------- EMA / target
def test_target_encoder_stays_in_eval_mode():
    cfg = _cfg()
    cfg.model.drop_path_rate = 0.2
    m = _model(cfg).train()
    assert m.encoder.training
    assert not m.target_encoder.training


def test_horizon_counts_track_training_and_warn_on_untrained_bucket():
    cfg = _cfg()
    m = _model(cfg).train()
    B, W = 2, cfg.forward.context_windows
    ctx = torch.randn(B, W, cfg.signal.n_leads, cfg.signal.window_samples)
    fut = torch.randn(B, cfg.signal.n_leads, cfg.signal.window_samples)
    m(ctx, future_window=fut, horizon_s=torch.tensor([60.0, 60.0]))
    assert 60.0 in m.trained_horizons()
    with pytest.warns(RuntimeWarning, match="never trained"):
        m.forecast_latent(ctx, 1800.0)


def test_cross_window_target_is_pooled_by_default():
    cfg = _cfg()
    m = _model(cfg).eval()
    ctx = torch.randn(2, 1, cfg.signal.n_leads, cfg.signal.window_samples)
    z = m.forecast_latent(ctx, 60.0)
    assert z.shape == (2, cfg.model.encoder_embed_dim)


def test_level2_mask_predictor_gets_positions():
    cfg = _cfg()
    cfg.hierarchical.enabled = True
    m = _model(cfg).train()
    W = cfg.hierarchical.l2_context_windows
    ctx = torch.randn(2, W, cfg.signal.n_leads, cfg.signal.window_samples)
    fut = torch.randn(2, cfg.signal.n_leads, cfg.signal.window_samples)
    out = m(ctx, future_window=fut, horizon_s=torch.tensor([60.0, 60.0]),
            l2_future=fut, l2_horizon_s=torch.tensor([300.0, 300.0]))
    assert torch.isfinite(out["loss"])
    out["loss"].backward()


# ------------------------------------------------------------------- collapse arms
def test_soft_vq_targets_keep_their_spread():
    """v0.1 settings (T=0.5, random codebook) squash distinct d=384 targets onto
    almost one vector (the VQ-arm collapse); v0.2 (T=0.1, data-seeded codes) keeps
    their spread."""
    import torch.nn.functional as F
    torch.manual_seed(0)
    cent = torch.randn(16, 384)
    h = F.layer_norm(cent[torch.randint(0, 16, (2048,))] + 0.5 * torch.randn(2048, 384),
                     (384,))
    true_spread = float(h.std(0).mean())
    old = SoftVQ(384, 512, temperature=0.5).train()
    old.initialized.fill_(1)                              # v0.1: random codebook
    new = SoftVQ(384, 512, temperature=0.1).train()
    assert float(old.transform_target(h).std(0).mean()) < 0.3 * true_spread
    assert float(new.transform_target(h).std(0).mean()) > 0.8 * true_spread
    loss, stats = new(h)
    assert torch.isfinite(loss) and "vq_perplexity" in stats


def test_collapse_verdict_flags_a_collapsed_embedding():
    mon = cj.CollapseMonitor(arm="x")
    base = torch.randn(1, 64)
    # everyone points the same way and the residual lives in 2 directions (the v0.1
    # VQ-arm picture: cos 0.94, low covariance rank)
    z = base + 0.05 * torch.randn(512, 2) @ torch.randn(2, 64)
    mon.log(0, 1.0, z)
    mon.log(1, 0.5, z)
    mon.log(2, 0.4, z)
    s = mon.summary()
    assert s["representation_collapse"] and s["verdict"] == "collapsed"
    mon2 = cj.CollapseMonitor(arm="y")
    for e in range(3):
        mon2.log(e, 1.0 - 0.1 * e, torch.randn(512, 64))
    assert mon2.summary()["verdict"] == "healthy"


# ------------------------------------------------------------------- baselines
def _xbatch(cfg, n=16, seed=1):
    from cardio_jepa.data.preprocess import prepare_window
    w, _ = synth_short_corpus(n, cfg.signal.fs, cfg.signal.window_s, cfg.signal.n_leads,
                              seed=seed)
    w = np.stack([prepare_window(x, cfg.signal.fs, cfg.signal.fs,
                                 cfg.signal.window_samples, reduced=False) for x in w])
    return torch.from_numpy(w).float()


def test_stmem_fixed_target_learns_while_patch_norm_does_not():
    cfg = _cfg()
    X = _xbatch(cfg, 32)
    res = {}
    for norm in ("patch", "none"):
        torch.manual_seed(0)
        m = STMEMBaseline(cfg.signal, cfg.model, target_norm=norm)
        opt = torch.optim.AdamW(m.parameters(), lr=5e-4)
        ls = []
        for _ in range(40):
            idx = torch.randint(0, len(X), (8,))
            loss = m(X[idx])["loss"]
            opt.zero_grad(); loss.backward(); opt.step()
            ls.append(float(loss.detach()))
        res[norm] = (np.mean(ls[:5]), np.mean(ls[-5:]))
    assert res["none"][1] < 0.95 * res["none"][0]
    assert res["patch"][0] > 0.9                         # ~white-noise target


@pytest.mark.parametrize("cls", [SimCLRBaseline, CPCBaseline])
def test_ssl_baselines_train(cls):
    cfg = _cfg()
    torch.manual_seed(0)
    m = cls(cfg.signal, cfg.model)
    X = _xbatch(cfg, 16)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    first = last = None
    for i in range(15):
        out = m(X.unsqueeze(1))                           # (B, 1, C, T) like the loader
        opt.zero_grad(); out["loss"].backward(); opt.step()
        first = first if first is not None else float(out["loss"].detach())
        last = float(out["loss"].detach())
    assert np.isfinite(last) and last < first
    assert m.representation(X).shape == (16, cfg.model.encoder_embed_dim)


# ------------------------------------------------------------------- data
def test_mitbih_csv_sampling_rate_is_inferred(tmp_path):
    from cardio_jepa.data.discovery import FoundDataset
    from cardio_jepa.data.longform import load_long_records
    d = tmp_path / "mit-bih-long-term-ecg-database"
    d.mkdir()
    n = 128 * 60
    t_ms = np.arange(n) * (1000.0 / 128.0)
    pd.DataFrame({"time_ms": t_ms, "ECG1": np.random.randn(n),
                  "ECG2": np.random.randn(n)}).to_csv(d / "14046_ekg.csv", index=False)
    pd.DataFrame({"ECG1": np.random.randn(n), "ECG2": np.random.randn(n)}
                 ).to_csv(d / "14134_ekg.csv", index=False)
    ds = FoundDataset(key="mitbih_lt", label="MIT-BIH Long-Term ECG", kind="long",
                      root=str(d), data_root=str(d), fmt="mitbih_csv")
    recs = load_long_records(ds, limit=10)
    fs = {r.name: r.fs for r in recs}
    assert abs(fs["14046"] - 128.0) < 0.5            # from the time column
    assert fs["14134"] == 128.0                       # corpus default, not 360
    assert all(r.n_channels == 2 for r in recs)       # time column not read as a lead


def test_discovery_separates_mitbih_long_term(tmp_path):
    root = tmp_path / "input"
    lt = root / "datasets/protobioengineering/mit-bih-long-term-ecg-database"
    lt.mkdir(parents=True)
    (lt / "14046_ekg.csv").write_text("a,b\n1,2\n")
    found = discover(str(root), verbose=False)
    assert "mitbih_lt" in found and "mitbih" not in found


def test_discovery_skip_list(tmp_path):
    root = tmp_path / "input"
    m = root / "datasets/x/meeti-final-dataset"
    m.mkdir(parents=True)
    (m / "a.hea").write_text("")
    assert "meeti" in discover(str(root), verbose=False)
    assert "meeti" not in discover(str(root), verbose=False, skip=("meeti",))


def test_cache_roundtrip_and_zero_audit(tmp_path):
    w = np.random.randn(10, 8, 2500).astype(np.float32)
    w[3] = 0.0
    y = np.eye(2)[np.arange(10) % 2].astype(np.int64)
    ds = ArrayWindowDataset(w, y, return_dict=False)
    c = cache_dataset(ds, "t", str(tmp_path), num_workers=0, verbose=False)
    assert len(c) == 10 and c.meta["n_zero"] == 1
    x0, y0 = c[0]
    assert torch.allclose(x0, torch.from_numpy(w[0]), atol=1e-2)
    c2 = cache_dataset(ds, "t", str(tmp_path), num_workers=0, verbose=False)
    assert c2.meta == c.meta


def test_interleaved_loader_reset_gives_identical_schedule():
    cfg = _cfg()
    loader, _ = build_pretrain_loaders({}, cfg, samples_per_record=4, verbose=False)

    def schedule():
        loader.reset()
        return [tuple(b["ctx"].shape) for b in loader]
    assert schedule() == schedule()


def test_probe_batch_is_short_and_large():
    cfg = _cfg()
    loader, _ = build_pretrain_loaders({}, cfg, samples_per_record=4, verbose=False)
    pb = probe_batch_from_loader(loader, n=40)
    assert pb.shape[0] == 40 and pb.dim() == 3


# ------------------------------------------------------------------- statistics
def test_cluster_bootstrap_is_wider_than_iid():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(20), 50)
    eff = rng.normal(0, 1.0, 20)[groups]                 # strong within-record correlation
    y = (eff + rng.normal(0, 0.5, len(groups)) > 0).astype(int)
    s = eff + rng.normal(0, 1.0, len(groups))
    iid = bootstrap_auroc(y, s, n=300)
    clu = bootstrap_auroc(y, s, n=300, groups=groups)
    assert (clu["hi"] - clu["lo"]) > (iid["hi"] - iid["lo"])


def test_seed_level_statistics():
    s = seed_level_summary({0: 0.80, 1: 0.82, 2: 0.81})
    assert s["n"] == 3 and s["ci_lo"] < 0.81 < s["ci_hi"]
    t = paired_seed_test({0: 0.82, 1: 0.83, 2: 0.84}, {0: 0.80, 1: 0.80, 2: 0.81})
    assert t["delta"] > 0 and t["n_seeds"] == 3


# ------------------------------------------------------------------- evaluation
def test_probe_uses_validation_and_repeats():
    cfg = _cfg()
    splits = build_probe_splits({}, cfg, verbose=False)["synthetic"]
    m = _model(cfg)
    from cardio_jepa.eval import run_classification_probe
    res = run_classification_probe(m, splits, "cpu", cfg.evaluation, batch_size=32,
                                   verbose=False)
    assert res["best_lr"] in cfg.evaluation.linear_probe_lr_grid
    small = res["by_fraction"][0.1]
    assert len(small["macro_auroc_runs"]) == cfg.evaluation.label_fraction_repeats


def test_robustness_and_supervised_reference_run():
    from cardio_jepa.eval import run_robustness_probe, synthetic_noise_bank, train_supervised
    cfg = _cfg()
    splits = build_probe_splits({}, cfg, verbose=False)["synthetic"]
    m = _model(cfg)
    bank = synthetic_noise_bank(cfg.signal.fs, minutes=1.0)
    df = run_robustness_probe(m, splits, "cpu", cfg.evaluation, bank, snrs_db=(6.0,),
                              batch_size=32, verbose=False)
    assert {"clean", "bw@6dB", "leads:single_II"} <= set(df.condition)
    sup = train_supervised(cfg.signal, cfg.model, splits, "cpu", max_epochs=1,
                           min_steps=2, batch_size=32, num_workers=0, verbose=False)
    assert 0.0 <= sup["macro_auroc"] <= 1.0


def test_forecast_skill_reports_controls():
    from cardio_jepa.eval import forecast_skill
    cfg = _cfg()
    recs = synthetic_long_records(3, 30.0, cfg.signal.fs, cfg.signal.n_leads, seed=0)
    m = _model(cfg)
    df = forecast_skill(m, recs, cfg.signal, "cpu", horizons_s=(10.0, 60.0),
                        n_samples=12, verbose=False, n_boot=20)
    assert {"skill", "skill_lo", "horizon_gain", "cos_shuffled"} <= set(df.columns)
