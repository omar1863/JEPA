# Research plan — making Cardio-JEPA a defensible Q1 paper

## 1. What the v0.1 run actually showed (after reading the logs)

| Question | v0.1 result | Can it be trusted? |
|---|---|---|
| Does `L_fwd` hurt classification? (Stage 1) | dual 0.818 > masked-only 0.768 > forward-only 0.718 > ST-MEM 0.650 (PTB-XL macro AUROC) | **No.** PTB-XL test fold was in pretraining (B01); one seed (B10); ST-MEM never trained (B06); 2/3 of batches were tiled Holter (B12). All numbers are far below the published ECG-JEPA linear probe (0.906, Kim 2024), so the models are under-trained. The *direction* (dual ≥ masked) is promising and must be re-tested. |
| Does forward pretraining help anticipation? (H1) | No: dual 0.525 / masked-only 0.622 AUROC at 15 min | **No** in either direction: tested on records the forward predictor was trained on (B02), with an untrained 600 s horizon embedding (B03), window-level CIs (B10). |
| Is the predictor forecasting? | skill ≈ +0.05 at every horizon (flat) | Flat skill is the signature of an ignored / untrained horizon token (B03) measured in-sample (B02, B16). |
| What prevents collapse? (RQ3) | "ema" ≈ vicreg > vq ≈ sigreg | **No:** there was no EMA-only arm (B05); the VQ collapse was a bug (B07); ranks capped at 128 (B14). |

**Bottom line:** v0.1 does not yet support any claim. v0.2 removes the leakage and bugs, so the
next run gives the first valid numbers. Plan the paper around what v0.2 shows, not around
the v0.1 table.

## 2. Related work and the real research gap

* **Masked latent prediction for ECG.** Kim's ECG-JEPA ([arXiv:2410.08559](https://arxiv.org/abs/2410.08559)), Weimann & Conrad's I-JEPA for ECG ([arXiv:2410.13867](https://arxiv.org/abs/2410.13867)) and the hierarchical ER-JEPA ([arXiv:2607.01145](https://arxiv.org/abs/2607.01145)) all predict *masked* latents inside one ~10 s recording.
* **Forward latent prediction already exists — contrastively.** CPC predicts future latents; on 12-lead ECG it was the strongest self-supervised objective in Mehari & Strodthoff ([arXiv:2103.12676](https://arxiv.org/abs/2103.12676), PTB-XL linear-probe macro AUROC 0.927 vs SimCLR 0.883), and a 2026 systematic study found CPC slightly ahead of JEPA for transfer ([arXiv:2605.12241](https://arxiv.org/abs/2605.12241)).
* **Non-contrastive forward prediction in time series.** Eidos ([arXiv:2602.14024](https://arxiv.org/abs/2602.14024)) predicts future latents autoregressively with a stop-gradient target; SC-JEPA/MTS-JEPA ([arXiv:2602.04643](https://arxiv.org/abs/2602.04643)) forecasts the next window in latent space for anomaly *anticipation* with a soft codebook + EMA (close to our VQ arm); TS-JEPA ([arXiv:2509.25449](https://arxiv.org/abs/2509.25449)). None is on multi-lead ECG or clinical rhythm anticipation.
* **World models / anti-collapse.** V-JEPA 2 ([arXiv:2506.09985](https://arxiv.org/abs/2506.09985)) uses latent future prediction as a world model; LeJEPA/SIGReg ([arXiv:2511.08544](https://arxiv.org/abs/2511.08544)) removes the EMA teacher.
* **Clinical anticipation.** AF-onset prediction from long RR series reaches high AUROC with ~1 h of context (AF-Mamba, [arXiv:2609.06984](https://arxiv.org/abs/2609.06984)); IRIDIA-AF ([Sci Data 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10584890)) provides 167 Holter records (152 patients, 2 leads, 200 Hz, 388 AF episodes, CC-BY 4.0).

**So the honest gap is narrower than "nobody predicts forward in time":**
1. Nobody has *combined* masked infilling (morphology) with **horizon-conditioned, non-contrastive** forward prediction (dynamics) for **multi-lead ECG**, and tested whether the forward term is what makes CPC strong.
2. ECG "world-model" claims are not evaluated with a **leakage-safe forecasting protocol** (held-out records, persistence and horizon-shuffle controls, record-level CIs).
3. No ECG-JEPA reports **robustness to real ambulatory noise and missing leads**.

Frame the paper as: *"Do JEPAs need dynamics? Unifying masked and forward latent prediction for
ECG, with a leakage-safe anticipation benchmark."* That is defensible even if the anticipation
result is modest, because the controls make it a clean answer.

## 3. Proposed contributions (realistic on Kaggle T4 sessions)

| # | Contribution | What to do | Real-world impact |
|---|---|---|---|
| C1 | **Dual objective vs strong SSL baselines** | dual vs masked-only vs forward-only vs CPC vs SimCLR vs ST-MEM vs supervised vs random-init; same encoder; 3 seeds; compute-matched masked-only control | Shows whether adding dynamics gives better general ECG features → fewer labels needed for diagnosis models |
| C2 | **Leakage-safe anticipation + forecast-skill protocol** | pre-pretraining record holdout, persistence & horizon-shuffle controls, random-predictor control, record-level cluster bootstrap, AUROC + AUPRC | A reusable evaluation standard; prevents inflated "early warning" claims in AF screening |
| C3 | **Label efficiency** | 1 / 10 / 50 / 100 % labels, 3 repeats, vs supervised from scratch | Hospitals have few expert-labelled ECGs; quantifies annotation savings |
| C4 | **Robustness to real noise and missing leads** | NSTDB baseline wander / electrode motion / muscle noise at 24→0 dB; I+II and single lead II | Holter, patch and smartwatch ECGs are noisy and have 1–2 leads; tells clinicians when outputs stay reliable |
| C5 | **Cross-dataset generalisation** | pretrain Chapman+Ningbo (+CODE-15) → probe PTB-XL (Germany) and CPSC2018 (China) as external sites; leave-one-dataset-out pretraining; harmonise SNOMED↔SCP labels for shared classes (AF, RBBB, LBBB, 1AVB, PVC) | Models must work in a different hospital and country than they were trained in |
| C6 | **ECG-specific masking** | random vs temporal-block (Kim) vs lead-wise vs **beat-aligned** (mask whole beats from R-peaks) | Learns beat-level morphology instead of interpolating smooth segments |
| C7 | **Anti-collapse recipe for forward prediction (RQ3)** | EMA-only vs VICReg vs VQ vs SIGReg, with stability–accuracy Pareto, 2 seeds | A practical recipe others can reuse for JEPA-style forecasting |
| C8 | **Clinical horizons (stretch)** | external AF-onset test on IRIDIA-AF (upload as a Kaggle dataset); hierarchical stack with ≥ 30 min context; compare with an RR/HRV logistic-regression baseline | Early warning of paroxysmal AF from Holter/patch monitoring |

**Priority.** Must have for Q1: C1, C2, C3, C4, plus C5 on PTB-XL + CPSC2018. Strongly
recommended: C6, C7. Stretch: C8 (needs IRIDIA-AF on Kaggle and longer context).

## 4. Expected outcomes and how to write them

* If **dual > masked-only (3 seeds, p < 0.05) and ≥ CPC**: headline claim "forward dynamics improve JEPA ECG representations"; C1 + C3 + C4 carry the paper.
* If **dual ≈ masked-only but forecast skill > 0 with positive horizon gain**: claim a *free* world-model capability (forecasting without losing diagnostic quality); C2 carries the paper.
* If **forecast skill ≤ 0 on held-out records**: report it — the controls are the contribution; pivot the paper to C2 + C4 + C5 (benchmark + robustness study of ECG JEPAs).

## 5. Suggested venues (Q1)
IEEE Journal of Biomedical and Health Informatics; Computers in Biology and Medicine;
Biomedical Signal Processing and Control; Artificial Intelligence in Medicine. Check each
journal's current quartile in your target index before submitting.

## 6. Risks
* **Compute:** a T4 session gives ~6 000 steps per model. Report this budget honestly and compare all methods at equal steps; do not compare raw numbers with papers trained 100 epochs on 180 k ECGs.
* **2-lead Holter vs 12-lead:** tiling is a compromise. Ablate `long_loader_weight` (0.25 / 0.5 / 1.0) and report it.
* **MEETI** has no reader yet (no readable files found). Add one only if its format is waveforms; otherwise detach it.
