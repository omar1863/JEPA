# Changelog

## 0.2.0 — corrected and extended framework (v3 notebook)

Bug IDs refer to `docs/BUG_REPORT.md`.

### Fixed — leakage and validity
- PTB-XL / CPSC2018 removed from pretraining; guard raises if a probe dataset is pretrained on (B01).
- Long-record holdout fixed before pretraining (`split_long_records`, `split_by_ids`); forward pairs only from training records (B02).
- Forward horizons now cover every evaluated horizon; `horizon_counts` buffer + warning on untrained buckets (B03).
- MIT-BIH Long-Term ECG: own corpus key, sampling rate inferred (128 Hz, was read as 360 Hz) (B04).
- RQ3: a real EMA-only arm; Stage-1 reuse only for an identical VICReg config (B05).
- ST-MEM target fixed (it never trained) (B06).
- Soft-VQ: temperature, data-seeded codebook, commitment weight, all assignments regularised (B07).
- Cluster (record-level) bootstrap; multi-seed paired tests (B10); random-predictor control for "world" features (B11).

### Fixed — training and data
- Fused SDPA attention (fixes Stage-4 OOM, faster); GPU freed between stages (B08).
- Holter share of batches 67 % → 33 % (configurable) (B12); identical batch schedule for every arm (B13).
- Diagnostics on 512 short 12-lead windows (B14); stricter collapse verdict (B15).
- PTB-XL: full dataset, 500 Hz, unlabelled rows dropped, patient-overlap check (B18).
- Level-2 masked predictor gets positions; Level-2 target consistent (B19, B20).
- Target encoder always in eval mode (B21); runtime estimate honours the step cap (B22).
- Failed reads counted and audited (B23); discovery 20 k-file cap, probe cache, skip list (B24).
- Cross-window forward target pooled (phase-invariant) by default (B25).
- Deterministic seeding (cuDNN flags, cuBLAS workspace) (B29).

### Added
- `SimCLRBaseline` and `CPCBaseline` (same CroPA encoder) — CPC is the closest prior art to `L_fwd`.
- `train_supervised` (supervised-from-scratch reference, incl. label fractions) and a random-init floor.
- `run_robustness_probe`: real NSTDB noise (bw / em / ma) at 24 / 12 / 6 / 0 dB, and lead subsets (I+II, single lead II).
- Linear probe: validation-selected learning rate + early stopping; repeated label fractions.
- `forecast_skill`: same-space persistence, horizon-shuffle control, bootstrap CI.
- Burden probe: grouped-CV ridge alpha, record-level metrics.
- `data.cache`: one-time float16 memmap cache of evaluation sets in `/tmp` (+ zero-fill audit).
- `load_train_result`, `free_cuda`; `seed_level_summary`, `paired_seed_test`.
- 24 new regression tests (`tests/test_v02.py`); all 59 tests pass on CPU.

### Notebook (v3)
- Self-contained: writes the corrected package from the notebook (no model re-upload needed).
- Four-session plan (A–D) with automatic reuse of checkpoints from earlier sessions.
- Results ledger saved after every stage; exit-criteria checker; loop log; comparison with v0.1.
- Paper figures (PNG + PDF) and all docs written to `/kaggle/working`, zipped at the end.
- Removed `CUDA_LAUNCH_BLOCKING`, `pip install` of flash-attn/xformers/triton, warm-start-by-default.

### Changed defaults (may change numbers vs v0.1)
`pretrain_datasets`, `long_loader_weight=0.5`, `forward.horizons_s`, `forward_target_long="pooled"`,
`vq_temperature=0.1`, `vq_commit_weight=1.0`, `anticipation_horizons_s=(300, 600, 1800)`,
`max_records_per_dataset=46000` (all of Chapman+Ningbo), `ptbxl_use_hr=True`, `cache_dir=/tmp/...`.

## 0.1.0
Initial implementation (see the v0.1 Kaggle run for its results).
