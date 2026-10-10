# Experiment and ablation plan (v3 notebook)

All models: CroPA encoder 384-d × 8 layers, predictor 192-d × 4, batch 32, 30 epochs ×
200 steps (6 000 steps per model; same budget for every method), AdamW + cosine, AMP.
Pretraining: Chapman + Ningbo (+ CODE-15 / MIMIC if attached) short clips + **training**
LTAFDB / MIT-BIH-LT records (holdout fixed first). Probes: PTB-XL superdiagnostic
(folds 1–8 / 9 / 10) and CPSC2018 if attached.

## Session schedule (one Kaggle T4 session ≈ 9–11 h each)

| Session | Trains | Evaluates | ≈ GPU h |
|---|---|---|---|
| **A** | seed 0: masked_only, forward_only, dual, SimCLR, CPC; supervised (100/10/1 %) | Stage 1 probes, robustness, Stage 2 anticipation + forecast skill | 8–10 |
| **B** | seeds 1, 2: masked_only, dual; seed 0: ST-MEM, masked_only_cm (compute-matched) | Stage 1 seed table + go/no-go | 7–9 |
| **C** | RQ3: ema, vq, sigreg (seed 0; vicreg = Stage-1 dual); seeds 1, 2: SimCLR, CPC | Stage 3 Pareto, Stage 1 baselines with seeds | 8–10 |
| **D** | Stage 4: hierarchical, single_scale; seeds 1, 2: forward_only | Stage 4 sweep + burden, final tables, exit criteria | 6–9 |

Attach the previous session's notebook output as an input before starting the next session;
the notebook finds the checkpoints and the results ledger automatically. Weekly Kaggle GPU
quota is ~30 h, so A–D take about one week.

## Main ablation matrix

| Ablation | Arms | Seeds | Answers |
|---|---|---|---|
| Objective | masked_only / forward_only / dual | 3 | Does `L_fwd` help, hurt or add nothing? |
| Compute match | masked_only_cm (≈1.5× steps) vs dual | 1→3 | Is the gain just more gradient signal per step? |
| SSL family | SimCLR / CPC / ST-MEM / dual | 3 | JEPA+dynamics vs contrastive vs contrastive-forward vs reconstruction |
| Supervision | supervised (1/10/100 %) / random-init | 1–3 | Floor and label-efficiency reference |
| Anti-collapse (RQ3) | ema / vicreg / vq / sigreg | 1–2 | Which mechanism keeps the forward objective healthy? |
| Hierarchy (H2) | hierarchical / single_scale (same 12-window loader) | 1 | Does a rhythm-level stack help at long horizons? |
| Optional (config flags, 1 seed each) | `forward_target_long` pooled vs tokens; `long_loader_weight` 0.25 / 0.5 / 1.0; `intra_window_frac` 0.5 / 0.6 / 0.75; `mask_type` block vs random; horizon-shuffle at train time | 1 | Design choices reviewers ask about |

## Tables for the paper

| Table | Content | Produced by |
|---|---|---|
| T1 | Datasets: role, records, patients, fs, leads, split, leakage controls | `results/data_report.csv` |
| T2 | PTB-XL (and CPSC2018) linear probe, macro AUROC mean ± std (3 seeds) and macro-F1, all methods; published ECG-JEPA 0.906 as an external reference (different budget) | `results/table_linear_probe.csv` |
| T3 | Label efficiency 1 / 10 / 50 / 100 % (mean ± std over repeats and seeds) incl. supervised | `results/table_label_efficiency.csv` |
| T4 | Robustness: AUROC per noise type × SNR, and lead subsets | `results/table_robustness.csv` |
| T5 | Anticipation at 5 / 10 / 30 min: context vs world vs random-predictor control, AUROC + AUPRC with record-level CIs | `results/table_anticipation.csv` |
| T6 | Forecast skill per horizon: skill [CI], horizon gain | `results/table_forecast_skill.csv` |
| T7 | RQ3: verdict, effective rank, cov-rank ratio, std, AUROC | `results/table_rq3.csv` |
| T8 | Stage 4: AUROC by horizon (hierarchical vs single-scale) + AF-burden R² / ρ | `results/table_stage4.csv` |
| T9 | Statistical tests: paired seed t-tests and bootstrap deltas | `results/table_tests.csv` |

## Figures for the paper

| Figure | Content | File |
|---|---|---|
| F1 | Method overview (masked branch + forward branch + horizon token + regulariser) — draw by hand from `models/cardio_jepa.py` | — |
| F2 | Training curves: total / L_mask / L_fwd / L_reg and effective rank per arm | `figures/fig_training_curves.pdf` |
| F3 | Linear-probe AUROC bars with seed error bars | `figures/fig_linear_probe.pdf` |
| F4 | Label-efficiency curves | `figures/fig_label_efficiency.pdf` |
| F5 | Robustness: AUROC vs SNR per noise type + lead-subset bars | `figures/fig_robustness.pdf` |
| F6 | Anticipation AUROC: context / world / control per horizon | `figures/fig_anticipation.pdf` |
| F7 | Forecast skill vs horizon with CI, and horizon gain | `figures/fig_forecast_skill.pdf` |
| F8 | RQ3 stability–accuracy scatter + effective-rank trajectories | `figures/fig_rq3.pdf` |
| F9 | Stage-4 horizon sweep | `figures/fig_stage4.pdf` |

## Exit criteria (checked automatically by the notebook)

1. Leakage checks pass (no probe dataset in pretraining; no holdout record in forward pairs; PTB-XL patient overlap = 0).
2. No collapse in any reported main or baseline arm (verdict ≠ collapsed).
3. Every baseline trained (loss fell ≥ 10 % from the first epoch).
4. ≥ 3 seeds for masked_only and dual, seed std of PTB-XL AUROC ≤ 0.01.
5. Determinism on (fixed seeds, deterministic cuDNN) and a repeat-run check passes.
6. dual ≥ masked_only (paired seed test) and dual ≥ the best SSL baseline (mean AUROC).

Criteria 1–5 are about correctness and must pass. Criterion 6 is the scientific result: if it
fails after 3 seeds, the result is negative and should be reported as such (see the
research plan, section 4).
