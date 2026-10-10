"""Driver-cell sources for the v3 Kaggle notebook (imported by build_notebook_v3.py)."""

TITLE = r"""# Cardio-JEPA v3 — corrected and extended framework (package v0.2.0)

**Latent forward-prediction JEPA for ECG**: masked latent infilling (morphology, Kim's
ECG-JEPA) + horizon-conditioned forward latent prediction (dynamics), compared against
SimCLR, CPC, ST-MEM, supervised and random-init baselines on the same encoder.

### What changed from the v0.1 run
The v0.1 run had test-set leakage (PTB-XL was pretrained on), forecasting leakage (the
forward predictor was trained on the records it was tested on), untrained horizon
embeddings at test time, a mis-read 128 Hz corpus, a missing EMA-only arm, a baseline that
never trained, a buggy VQ arm, over-confident statistics and an OOM that lost all results.
All 30 issues are listed in the **Appendix (bug report)** at the end, with the fix and the
regression test for each. **v0.1 numbers are not valid evidence; re-run with this notebook.**

### How to run (multi-session — one Kaggle T4 session cannot fit the study)
1. Attach datasets (table below). **Detach** MEETI (no readable files, cost ~most of the
   51-min discovery) and *ECG Heartbeat Categorization* (single beats; leakage risk).
2. Set `SESSION = "A"` in the configuration cell → **Save Version → Run All**.
3. For session B (then C, D): **Add Input → your previous notebook version's output**, set
   `SESSION = "B"`, run again. Checkpoints and results from earlier sessions are found and
   reused automatically.
4. Every session writes `/kaggle/working/cardio_jepa_v3_outputs.zip` (results ledger,
   tables, figures, docs, package source). The last session has the complete paper tables.

`RUN_MODE = "smoke"` runs every code path on synthetic data in a few minutes (CPU ok).

The framework source is **written by this notebook** (section 1) — no model upload needed.
"""

DATA_MD = r"""## Datasets

| Role | Kaggle dataset | Notes |
|---|---|---|
| Pretraining (12-lead) | `bjoernjostein/shaoxing-and-ningbo-first-hospital-database` | all 45,152 recordings (cap raised) |
| Pretraining (optional) | CODE-15 (`ccf002/code-15-ecg-complete`) | needs `h5py` (present on Kaggle) |
| **Probe only** | PTB-XL (`omar1863/ptb-xl` or `khyeh0719/ptb-xl-dataset`) | official folds 1–8 / 9 / 10; **never pretrained on** |
| Probe only (external site) | CPSC2018 (`physionet/china-physiological-signal-challenge-in-2018`) | recommended for cross-dataset results |
| Forward pretraining + anticipation | Long-Term AF (`sahilshahare12/long-term-af-database`) | 30 % of records held out before pretraining |
| Forward pretraining | MIT-BIH Long-Term ECG (`protobioengineering/mit-bih-long-term-ecg-database`) | now read at 128 Hz (was 360 Hz) |
| Robustness noise | MIT-BIH Noise Stress Test (ships inside `omar1863/ptb-xl`) | `bw` / `em` / `ma` noise records |

Not used: **MEETI** (no reader for its format — detach or add a reader), **ECG Heartbeat
Categorization** (single 125 Hz beats derived from MIT-BIH/PTB — detach). The separate
Chapman-Shaoxing mirror is redundant with the Shaoxing+Ningbo mirror (indexed once).
"""

SRC_MD = r"""## 1. Framework source (cardio_jepa v0.2.0)

The next cells write the corrected package to `/kaggle/working/cardio_jepa_v3_src/`. They
are the single source of truth used by the rest of the notebook (and they are saved in the
output zip, so every result is traceable to the exact code). Collapse this section.
"""

ENV = r"""# --- 0. environment: must run before torch initialises CUDA -----------------------------
import os, sys, json, time, math, glob, shutil, warnings, platform, datetime, copy, gc, traceback
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")       # fragmentation fix
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # older torch name
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")                   # deterministic cuBLAS
# NOTE (v0.1 notebook bug): do NOT set CUDA_LAUNCH_BLOCKING=1 -- it serialises every kernel.

import numpy as np
warnings.filterwarnings("ignore", category=FutureWarning)

WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.abspath(os.getcwd())
os.chdir(WORK)
RUNS_DIR = os.path.join(WORK, "runs")
RESULTS_DIR = os.path.join(WORK, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
SCORE_DIR = os.path.join(RESULTS_DIR, "scores")
for d in (RUNS_DIR, RESULTS_DIR, FIG_DIR, SCORE_DIR, "cardio_jepa_v3_src/cardio_jepa/models",
          "cardio_jepa_v3_src/cardio_jepa/data", "cardio_jepa_v3_src/cardio_jepa/eval",
          "cardio_jepa_v3_src/tests", "cardio_jepa_v3_src/docs"):
    os.makedirs(d, exist_ok=True)

import torch
print("python ", sys.version.split()[0], "| torch", torch.__version__,
      "| cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "(CPU only)")
print("workdir", WORK)
"""

IMPORT = r"""# --- 2. import the package written above -------------------------------------------------
SRC_ROOT = os.path.join(WORK, "cardio_jepa_v3_src")
sys.path.insert(0, SRC_ROOT)
for _m in [m for m in list(sys.modules) if m == "cardio_jepa" or m.startswith("cardio_jepa.")]:
    del sys.modules[_m]                 # never pick up an older attached cardio_jepa package
import cardio_jepa as cj
assert cj.__version__ == "0.2.0", f"wrong package version {cj.__version__} at {cj.__file__}"
print(f"cardio_jepa {cj.__version__} from {os.path.dirname(cj.__file__)}")

from cardio_jepa.config import configure_arm, kaggle_config, smoke_config, full_config
from cardio_jepa.data import (build_onset_dataset, build_burden_dataset, discover, summarize,
                              split_long_records, split_by_ids, cache_dataset,
                              cache_probe_splits, AnticipationDataset)
from cardio_jepa.data.build import (build_pretrain_loaders, build_probe_splits,
                                    load_long_corpus, probe_batch_from_loader)
from cardio_jepa.eval import (run_classification_probe, run_anticipation_probe,
                              run_burden_probe, forecast_skill, horizon_sweep,
                              run_robustness_probe, load_nstdb_noise, synthetic_noise_bank,
                              train_supervised, bootstrap_auroc_delta, seed_level_summary,
                              paired_seed_test)
from cardio_jepa.models import CardioJEPA, STMEMBaseline, SimCLRBaseline, CPCBaseline
from cardio_jepa.engine import estimate_runtime, load_train_result, free_cuda
from cardio_jepa.models import transformer as _tr

import pandas as pd
pd.set_option("display.width", 180); pd.set_option("display.max_columns", 60)
DEVICE = cj.pick_device()
print("device:", DEVICE, "| fused SDPA attention:", _tr.USE_SDPA)
"""

CONFIG_MD = r"""## 3. Configuration

* `SESSION` picks which models this run **trains** (see the plan below). Models from earlier
  sessions are **loaded** from the attached previous output and are not retrained.
* The step budget (`EPOCHS × MAX_STEPS`) is **fixed for every model in every session**.
  v0.1 auto-shrank it to fit one session, which would make arms trained in different
  sessions incomparable; if a session does not fit, move jobs to another session instead.
"""

CONFIG = r"""# --- 3. configuration ----------------------------------------------------------------
RUN_MODE = "kaggle"        # "kaggle" (T4/P100) | "smoke" (synthetic, minutes, CPU ok)
SESSION = "A"              # "A" | "B" | "C" | "D" | "ALL"  (ALL is meant for smoke mode)
VERSION_TAG = "v3"
INPUT_ROOT = "/kaggle/input"
SKIP_DATASETS = ("meeti",)  # not even walked by discovery (no reader for MEETI's format)
SPLIT_SEED = 0              # fixes the long-record holdout for ALL sessions -- do not change
BUDGET_HOURS = 11.0
FORCE_REEVAL = False        # re-run evaluations already present in the results ledger
ZIP_CHECKPOINTS = False     # .pth files stay in /kaggle/working/runs (saved by Kaggle anyway)
PREVIOUS_OUTPUT_DIRS = "auto"   # or a list of directories holding runs/ and results/

CFG = {"kaggle": kaggle_config, "smoke": smoke_config, "full": full_config}[RUN_MODE]()
CFG.data.input_root = INPUT_ROOT
CFG.data.skip_datasets = SKIP_DATASETS
CFG.out_dir = RUNS_DIR
CFG.forward.context_windows = 2           # 20 s of context for the forward objective
CFG.hierarchical.enabled = False          # switched on for Stage 4 only
if RUN_MODE == "kaggle":
    CFG.optim.epochs = 30
    CFG.optim.max_steps_per_epoch = 200   # 6 000 steps per model, every model, every session
    CFG.optim.num_workers = 2
    SAMPLES_PER_RECORD = 200
    MAX_PER_RECORD = 300                  # anticipation windows per record
    N_SKILL = 256
    PROBE_N = 512
    SUP_MAX_EPOCHS = 20
    RQ3_SEEDS = (0,)
else:                                     # smoke
    CFG.optim.epochs = 2
    CFG.optim.max_steps_per_epoch = 6
    CFG.optim.num_workers = 0
    SAMPLES_PER_RECORD = 8
    MAX_PER_RECORD = 40
    N_SKILL = 16
    PROBE_N = 48
    SUP_MAX_EPOCHS = 1
    RQ3_SEEDS = (0,)

CM_FACTOR = 1.5    # masked_only_cm gets 1.5x steps: dual costs ~1.5x per step (compute match)

# Which models each session TRAINS. (arm, seed). Everything else is loaded if available.
SESSION_PLAN = {
    "A": {"train": [("masked_only", 0), ("forward_only", 0), ("dual", 0),
                    ("simclr", 0), ("cpc", 0)],
          "supervised": [(1.0, 0), (0.1, 0), (0.01, 0)],
          "stages": ["s1", "robust", "s2"]},
    "B": {"train": [("masked_only", 1), ("dual", 1), ("masked_only", 2), ("dual", 2),
                    ("st_mem", 0), ("masked_only_cm", 0)],
          "supervised": [],
          "stages": ["s1", "s2"]},
    "C": {"train": [("rq3_ema", 0), ("rq3_vq", 0), ("rq3_sigreg", 0),
                    ("simclr", 1), ("simclr", 2), ("cpc", 1), ("cpc", 2)],
          "supervised": [(1.0, 1), (1.0, 2)],
          "stages": ["s1", "s3"]},
    "D": {"train": [("stage4_hier", 0), ("stage4_single", 0),
                    ("forward_only", 1), ("forward_only", 2)],
          "supervised": [],
          "stages": ["s1", "s3", "s4"]},
}
SESSION_PLAN["ALL"] = {
    "train": sum((SESSION_PLAN[k]["train"] for k in "ABCD"), []),
    "supervised": sum((SESSION_PLAN[k]["supervised"] for k in "ABCD"), []),
    "stages": ["s1", "robust", "s2", "s3", "s4"]}
if RUN_MODE == "smoke" and SESSION == "ALL":   # keep the smoke run short: one seed per arm
    SESSION_PLAN["ALL"]["train"] = [j for j in SESSION_PLAN["ALL"]["train"] if j[1] == 0]
    SESSION_PLAN["ALL"]["supervised"] = [(1.0, 0), (0.1, 0)]
PLAN = SESSION_PLAN[SESSION]

cj.set_seed(CFG.optim.seed, CFG.optim.deterministic)
print(f"mode={RUN_MODE} session={SESSION} | encoder {CFG.model.encoder_embed_dim}d x "
      f"{CFG.model.encoder_depth}L | {CFG.optim.epochs} epochs x {CFG.optim.max_steps_per_epoch} "
      f"steps, batch {CFG.optim.batch_size}")
print("trains:", [f"{a}_s{s}" for a, s in PLAN["train"]],
      "| supervised:", PLAN["supervised"], "| stages:", PLAN["stages"])
print("forward horizons (s):", CFG.forward.horizons_s,
      "| anticipation horizons (s):", CFG.evaluation.anticipation_horizons_s)
"""

HELPERS = r"""# --- 4. results ledger, checkpoint reuse, export -------------------------------------
LEDGER_PATH = os.path.join(RESULTS_DIR, "ledger.csv")
LEDGER_COLS = ["version", "session", "time", "model_id", "arm", "family", "seed", "task",
               "setting", "metric", "value", "extra"]
_ROWS = []

def family_of(arm):
    if arm.startswith("rq3_"): return "rq3"
    if arm.startswith("stage4_"): return "stage4"
    if arm in ("masked_only", "forward_only", "dual", "masked_only_cm"): return "jepa"
    if arm.startswith("supervised"): return "supervised"
    return "baseline"

def log(model_id, arm, seed, task, metric, value, setting="", extra=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = float("nan")
    _ROWS.append({"version": VERSION_TAG, "session": SESSION,
                  "time": datetime.datetime.utcnow().isoformat(timespec="seconds"),
                  "model_id": model_id, "arm": arm, "family": family_of(arm),
                  "seed": seed, "task": task, "setting": str(setting), "metric": metric,
                  "value": value, "extra": json.dumps(extra, default=str) if extra else ""})

def _bfs_dirs(root, max_depth, skip):
    out, stack = [], [(root, 0)]
    while stack:
        p, d = stack.pop()
        if d > max_depth or any(os.path.abspath(p).startswith(s) for s in skip):
            continue
        out.append(p)
        try:
            ents = [e for e in os.scandir(p) if e.is_dir(follow_symlinks=False)]
        except OSError:
            continue
        if len(ents) > 2000:
            continue
        stack.extend((e.path, d + 1) for e in ents if not e.name.startswith("."))
    return out

def find_previous_outputs():
    if PREVIOUS_OUTPUT_DIRS != "auto":
        return list(PREVIOUS_OUTPUT_DIRS)
    if not os.path.isdir(INPUT_ROOT):
        return []
    skip = [os.path.abspath(d.root) for d in globals().get("FOUND", {}).values()]
    hits = []
    for p in _bfs_dirs(INPUT_ROOT, 6, skip):
        if (os.path.isfile(os.path.join(p, "results", "ledger.csv"))
                or glob.glob(os.path.join(p, "runs", "*_diagnostics.json"))):
            hits.append(p)
    return hits

def previous_ledgers():
    frames = []
    for d in PREV_DIRS:
        f = os.path.join(d, "results", "ledger.csv")
        if os.path.isfile(f):
            frames.append(pd.read_csv(f, keep_default_na=False, na_values=[""]))
    return frames

def ledger_df():
    frames = previous_ledgers() + [pd.DataFrame(_ROWS, columns=LEDGER_COLS)]
    df = pd.concat([f for f in frames if len(f)], ignore_index=True) if any(
        len(f) for f in frames) else pd.DataFrame(columns=LEDGER_COLS)
    if len(df):
        df["setting"] = df["setting"].fillna("").astype(str)
        df = (df.sort_values("time")
                .drop_duplicates(["version", "model_id", "task", "setting", "metric"], keep="last"))
    return df

def save_ledger():
    df = ledger_df()
    df.to_csv(LEDGER_PATH, index=False)
    return df

def already_done(model_id, task, metric="macro_auroc"):
    if FORCE_REEVAL:
        return False
    df = ledger_df()
    if not len(df):
        return False
    m = (df.version == VERSION_TAG) & (df.model_id == model_id) & (df.task == task) & (df.metric == metric)
    return bool(m.any())

def checkpoint_index():
    idx = {}
    for d in PREV_DIRS:                         # earlier sessions (first match wins)
        for p in glob.glob(os.path.join(d, "runs", "*.pth")):
            idx.setdefault(os.path.basename(p), p)
    for p in glob.glob(os.path.join(RUNS_DIR, "*.pth")):   # this session overrides
        idx[os.path.basename(p)] = p
    return idx

def save_scores(name, **arrays):
    np.savez_compressed(os.path.join(SCORE_DIR, name + ".npz"), **arrays)

def load_scores(name):
    for d in [WORK] + PREV_DIRS:
        f = os.path.join(d, "results", "scores", name + ".npz")
        if os.path.isfile(f):
            return dict(np.load(f, allow_pickle=False))
    return None

PREV_DIRS = []           # filled after discovery (needs the dataset roots to skip them)
REGISTRY = {}            # model_id -> {arm, seed, model, res, cfg}
print("ledger helpers ready")
"""

DISCOVERY = r"""# --- 5. discover datasets ----------------------------------------------------------------
t0 = time.time()
FOUND = discover(INPUT_ROOT, verbose=True, skip=CFG.data.skip_datasets)
print(); print(summarize(FOUND))
print(f"discovery took {time.time() - t0:.0f}s (v0.1: 3066 s)")
for k in ("ptbxl", "cpsc"):
    if k in FOUND and k in CFG.data.pretrain_datasets:
        raise RuntimeError(f"{k} must not be a pretraining corpus")
PREV_DIRS = find_previous_outputs()
CKPTS = checkpoint_index()
print(f"previous outputs found: {PREV_DIRS or 'none'} | checkpoints available: {len(CKPTS)}")
"""

DATA = r"""# --- 6. data: fixed long-record holdout FIRST, then loaders, probe splits, caches ----------
t0 = time.time()
LONG_ALL = load_long_corpus(FOUND, CFG, verbose=True)
HOLDOUT = split_long_records(LONG_ALL, test_frac=CFG.evaluation.long_test_frac, seed=SPLIT_SEED)
LONG_TRAIN = HOLDOUT.train_records(LONG_ALL)
LONG_TEST = HOLDOUT.test_records(LONG_ALL)
print(f"long records: {len(LONG_ALL)} | pretrain {len(LONG_TRAIN)} | held out {len(LONG_TEST)} "
      f"(annotated, burden-stratified, fixed by SPLIT_SEED={SPLIT_SEED})")

LOADER, DATA_INFO = build_pretrain_loaders(FOUND, CFG, long_records=LONG_TRAIN,
                                           samples_per_record=SAMPLES_PER_RECORD, verbose=True)
PROBE_BATCH = probe_batch_from_loader(LOADER, n=PROBE_N)

# ---- leakage checks (all must pass; recorded in the ledger) ----
CHECKS = {}
CHECKS["probe_sets_not_pretrained"] = not (set(DATA_INFO.get("short_sources", {})) &
                                           set(CFG.data.probe_datasets))
CHECKS["holdout_not_in_forward_pairs"] = not (set(DATA_INFO.get("pretrain_long_record_names", []))
                                              & set(HOLDOUT.test_names))
from cardio_jepa.models.forward_predictor import horizon_to_bucket
fh = DATA_INFO.get("forward_horizon_counts", {})
_tb = {int(horizon_to_bucket([t], cj.HORIZON_BUCKETS_S)[0]) for t, n in fh.items() if n > 0}
_nb = {int(horizon_to_bucket([h], cj.HORIZON_BUCKETS_S)[0]) for h in CFG.evaluation.anticipation_horizons_s}
CHECKS["eval_horizons_are_trained_buckets"] = (not fh) or _nb <= _tb

RAW_SPLITS = build_probe_splits(FOUND, CFG, verbose=True)
PROBE_SPLITS = {}
for task, sp in RAW_SPLITS.items():
    PROBE_SPLITS[task] = cache_probe_splits(sp, task, CFG.data.cache_dir,
                                            num_workers=CFG.optim.num_workers)
    if "patient_overlap" in sp:
        CHECKS[f"{task}_patient_disjoint"] = sp["patient_overlap"] == 0
    zf = max(getattr(PROBE_SPLITS[task][p], "meta", {}).get("zero_frac", 0.0)
             for p in ("train", "val", "test") if PROBE_SPLITS[task].get(p) is not None)
    CHECKS[f"{task}_reads_ok(<1% zero-filled)"] = zf < 0.01

short_ds = getattr(LOADER.loaders[0], "dataset", None)
if hasattr(short_ds, "audit_reads"):
    aud = short_ds.audit_reads(n=200 if RUN_MODE == "kaggle" else 20)
    CHECKS["pretrain_reads_ok(<2% failed)"] = aud["failed_frac"] < 0.02
    print(f"pretraining read audit: {aud}")

for k, v in CHECKS.items():
    log("data", "data", -1, "checks", k, float(bool(v)))
print("\nLEAKAGE / DATA CHECKS")
for k, v in CHECKS.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}")

rows = [{"corpus": k, "label": d.label, "kind": d.kind, "fmt": d.fmt, "files>=": d.n_files,
         "role": ("probe" if k in CFG.data.probe_datasets else
                  "pretrain" if k in CFG.data.pretrain_datasets else
                  "long (pretrain/holdout)" if d.kind == "long" else "noise / robustness")}
        for k, d in FOUND.items()]
DATA_REPORT = pd.DataFrame(rows)
DATA_REPORT.to_csv(os.path.join(RESULTS_DIR, "data_report.csv"), index=False)
with open(os.path.join(RESULTS_DIR, "holdout_records.json"), "w") as fh_:
    json.dump({"test": HOLDOUT.test_names, "train": HOLDOUT.train_names}, fh_, indent=1)
print(f"\nshort clips {DATA_INFO.get('n_short', 0)} | forward pairs {DATA_INFO.get('n_forward_pairs', 0)}"
      f" | batch share {DATA_INFO.get('loader_weights')} | probe tasks {list(PROBE_SPLITS)}"
      f" | ready in {time.time() - t0:.0f}s")
save_ledger()
"""

MODELS = r"""# --- 7. arms, models, train-or-load ---------------------------------------------------
def arm_cfg(arm, seed):
    c = copy.deepcopy(CFG)
    c.optim.seed = seed
    c.name = f"{arm}_s{seed}"
    if arm in ("masked_only", "masked_only_cm"):
        c.loss.weight_forward, c.loss.lambda_mask = 0.0, 1.0
        if arm == "masked_only_cm":
            c.optim.max_steps_per_epoch = int(round(c.optim.max_steps_per_epoch * CM_FACTOR))
    elif arm == "forward_only":
        c.loss.weight_forward, c.loss.lambda_mask = 1.0, 0.0
    elif arm == "dual":
        c.loss.weight_forward, c.loss.lambda_mask = 1.0, 1.0
    elif arm.startswith("rq3_"):
        c = configure_arm(c, arm[4:])
        c.optim.seed, c.name = seed, f"{arm}_s{seed}"
        c.loss.weight_forward, c.loss.lambda_mask = 1.0, 1.0
        c.optim.on_collapse = "warn"            # RQ3 observes collapse instead of aborting
    elif arm.startswith("stage4_"):
        c.hierarchical.enabled = arm == "stage4_hier"
        c.forward.context_windows = c.hierarchical.l2_context_windows
        c.loss.weight_forward, c.loss.lambda_mask = 1.0, 1.0
    return c

def make_model(arm, c):
    if arm == "simclr": return SimCLRBaseline(c.signal, c.model)
    if arm == "cpc":    return CPCBaseline(c.signal, c.model)
    if arm == "st_mem": return STMEMBaseline(c.signal, c.model)
    return CardioJEPA(c.signal, c.model, c.loss, c.hierarchical, c.forward)

def log_training(model_id, arm, seed, res):
    s = res.monitor.summary()
    losses = [h.get("loss") for h in res.history if h.get("loss") is not None]
    first, last = (losses[0], losses[-1]) if losses else (float("nan"), float("nan"))
    drop = (first - last) / abs(first) if losses and first else float("nan")
    for k, v in {"first_loss": first, "final_loss": last, "loss_drop": drop,
                 "effective_rank": s.get("final_effective_rank"),
                 "cov_erank_ratio": s.get("final_cov_erank_ratio"),
                 "std_mean": s.get("final_std_mean"), "cos_sim": s.get("final_cos_sim"),
                 "collapsed_flag": float(bool(res.collapsed)),
                 "loss_to_zero": float(bool(s.get("collapsed_to_zero"))),
                 "seconds": res.seconds}.items():
        log(model_id, arm, seed, "train", k, v)
    log(model_id, arm, seed, "train", "verdict_collapsed",
        float(s.get("verdict") == "collapsed"), extra={"verdict": s.get("verdict")})

def get_model(arm, seed, loader=None, allow_train=False):
    model_id = f"{arm}_s{seed}"
    if model_id in REGISTRY:
        return REGISTRY[model_id]
    c = arm_cfg(arm, seed)
    path = CKPTS.get(model_id + ".pth")
    if path and os.path.isfile(os.path.splitext(path)[0] + "_diagnostics.json"):
        m = make_model(arm, c)
        rep = cj.load_weights(m, path, strict=False, verbose=False)
        if not rep.shape_mismatch and len(rep.missing) == 0:
            res = load_train_result(path)
            print(f"[{model_id}] loaded from {path}")
            REGISTRY[model_id] = {"arm": arm, "seed": seed, "model": m, "res": res, "cfg": c}
            return REGISTRY[model_id]
        print(f"[{model_id}] checkpoint does not match this config ({len(rep.missing)} missing, "
              f"{len(rep.shape_mismatch)} mismatched) -> " + ("retraining" if allow_train else "skipped"))
    if not allow_train:
        return None
    print(f"\n=== training [{model_id}] ===")
    cj.set_seed(seed, c.optim.deterministic)
    m = make_model(arm, c)
    n_par = sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6
    print(f"    {n_par:.1f}M trainable parameters | budget {c.optim.epochs} x {c.optim.max_steps_per_epoch} steps")
    res = cj.pretrain(m, loader or LOADER, c, device=DEVICE, arm_name=model_id,
                      probe_batch=PROBE_BATCH, save_path=os.path.join(RUNS_DIR, model_id + ".pth"),
                      log_every=0 if RUN_MODE == "smoke" else 200)
    log_training(model_id, arm, seed, res)
    CKPTS[model_id + ".pth"] = os.path.join(RUNS_DIR, model_id + ".pth")
    m.to("cpu"); free_cuda()
    REGISTRY[model_id] = {"arm": arm, "seed": seed, "model": m, "res": res, "cfg": c}
    save_ledger()
    return REGISTRY[model_id]

class RandomInit(torch.nn.Module):
    # floor: the same encoder, never trained
    def __init__(self):
        super().__init__()
        cj.set_seed(0)
        self.inner = CardioJEPA(CFG.signal, CFG.model, CFG.loss, CFG.hierarchical, CFG.forward)
    def representation(self, x):
        return self.inner.representation(x)
print("model helpers ready")
"""

RUNTIME = r"""# --- 8. runtime projection (informational; the budget is NOT changed automatically) ----
jobs = [j for j in PLAN["train"] if f"{j[0]}_s{j[1]}.pth" not in CKPTS
        and not j[0].startswith("stage4_")]
if jobs and RUN_MODE == "kaggle":
    _m = CardioJEPA(CFG.signal, CFG.model, CFG.loss, CFG.hierarchical, CFG.forward)
    c_ = arm_cfg("dual", 0)
    EST = estimate_runtime(_m, LOADER, c_, DEVICE, n_steps=8)
    del _m; free_cuda()
    cost = {"masked_only": 0.7, "forward_only": 0.75, "dual": 1.0, "masked_only_cm": 1.05,
            "simclr": 0.8, "cpc": 0.7, "st_mem": 0.6, "rq3_ema": 1.0, "rq3_vq": 1.0,
            "rq3_sigreg": 1.0}
    hours = sum(cost.get(a, 1.0) for a, _ in jobs) * EST["s_per_step"] * \
        CFG.optim.epochs * CFG.optim.max_steps_per_epoch / 3600
    hours += 0.6 * len(PLAN["supervised"]) + 1.5      # supervised runs + evaluation
    print(f"dual: {EST['s_per_step']:.3f} s/step -> this session ~{hours:.1f} h "
          f"for {len(jobs)} pretraining job(s) + evaluation (budget {BUDGET_HOURS} h)")
    if hours > BUDGET_HOURS:
        print(">>> over budget: move some jobs from PLAN['train'] to a later session "
              "(do not shrink MAX_STEPS -- all arms must share one budget)")
    log("session", "session", -1, "runtime", "projected_hours", hours)
else:
    print("nothing new to pretrain in this session (or smoke mode)")
"""

STAGE1_MD = r"""## Stage 1 — objective ablation + baselines (PTB-XL / CPSC2018 linear probe)

| arm | `w_fwd` | `λ` | role |
|---|---|---|---|
| `masked_only` | 0 | 1 | Kim-style masked latent infilling (same encoder, our budget) |
| `forward_only` | 1 | 0 | forward objective alone |
| `dual` | 1 | 1 | **Cardio-JEPA** |
| `masked_only_cm` | 0 | 1 | compute-matched (1.5× steps) |
| `simclr`, `cpc`, `st_mem` | — | — | contrastive / contrastive-forward / reconstruction baselines |
| `supervised_f*` | — | — | same encoder, trained with labels (1 / 10 / 100 %) |
| `random_init` | — | — | untrained encoder (floor) |

All arms use VICReg on the pooled representation (β = 0.1) except the RQ3 arms. Linear
probe: frozen encoder, head learning rate chosen on fold 9, test on fold 10.
"""

STAGE1_TRAIN = r"""# --- Stage 1a: train (or load) this session's models --------------------------------------
for arm, seed in PLAN["train"]:
    if arm.startswith("stage4_"):
        continue                      # Stage 4 needs its own wide-context loader (below)
    try:
        get_model(arm, seed, allow_train=True)
    except torch.cuda.OutOfMemoryError as exc:
        print(f"[{arm}_s{seed}] OOM: {exc}"); free_cuda()
    save_ledger()

# load every model that earlier sessions trained (for summaries / reuse), without training
for sess in ("A", "B", "C", "D"):
    for arm, seed in SESSION_PLAN[sess]["train"]:
        if not arm.startswith("stage4_"):
            get_model(arm, seed, allow_train=False)
print("\navailable models:", sorted(REGISTRY))
"""

STAGE1_EVAL = r"""# --- Stage 1b: linear probes (+ supervised references) -----------------------------------
def eval_probe(model_id, arm, seed, model):
    for task, splits in PROBE_SPLITS.items():
        if already_done(model_id, f"probe_{task}"):
            continue
        out = run_classification_probe(model.to(DEVICE), splits, DEVICE, CFG.evaluation,
                                       batch_size=64, num_workers=CFG.optim.num_workers,
                                       verbose=False)
        log(model_id, arm, seed, f"probe_{task}", "macro_auroc", out["macro_auroc"], "1.0",
            extra={"best_lr": out.get("best_lr")})
        log(model_id, arm, seed, f"probe_{task}", "macro_f1", out.get("macro_f1"), "1.0")
        for j, a in enumerate(out.get("per_class_auroc") or []):
            log(model_id, arm, seed, f"probe_{task}", "class_auroc",
                a, (splits.get("label_names") or [str(j)] * 99)[j])
        for frac, r in out["by_fraction"].items():
            log(model_id, arm, seed, f"labeleff_{task}", "macro_auroc", r["macro_auroc"], frac)
            log(model_id, arm, seed, f"labeleff_{task}", "macro_auroc_std",
                r.get("macro_auroc_std", 0.0), frac)
        save_scores(f"{model_id}__{task}", y_true=np.asarray(out["y_true"]),
                    y_score=np.asarray(out["y_score"]))
        print(f"  {model_id:22s} {task:8s} macro AUROC={out['macro_auroc']:.4f} "
              f"(lr {out.get('best_lr')})")
    model.to("cpu"); free_cuda()

if "s1" in PLAN["stages"] and PROBE_SPLITS:
    print("### linear probes")
    if SESSION in ("A", "ALL") and not already_done("random_init_s0", f"probe_{next(iter(PROBE_SPLITS))}"):
        eval_probe("random_init_s0", "random_init", 0, RandomInit())
    for mid, e in sorted(REGISTRY.items()):
        if not e["arm"].startswith("stage4_"):
            eval_probe(mid, e["arm"], e["seed"], e["model"])
    for frac, seed in PLAN["supervised"]:
        mid = f"supervised_f{frac:g}_s{seed}"
        for task, splits in PROBE_SPLITS.items():
            if already_done(mid, f"probe_{task}"):
                continue
            sup = train_supervised(CFG.signal, CFG.model, splits, DEVICE, label_fraction=frac,
                                   max_epochs=SUP_MAX_EPOCHS, seed=seed,
                                   num_workers=CFG.optim.num_workers, verbose=False)
            log(mid, f"supervised_f{frac:g}", seed, f"probe_{task}", "macro_auroc",
                sup["macro_auroc"], "1.0", extra={"epochs": sup["epochs_run"]})
            log(mid, f"supervised_f{frac:g}", seed, f"labeleff_{task}", "macro_auroc",
                sup["macro_auroc"], frac)
            save_scores(f"{mid}__{task}", y_true=sup["y_true"], y_score=sup["y_score"])
            print(f"  {mid:22s} {task:8s} macro AUROC={sup['macro_auroc']:.4f} "
                  f"({sup['n_train']} labelled ECGs)")
            if frac == 1.0 and seed == 0 and task == next(iter(PROBE_SPLITS)):
                REGISTRY[mid] = {"arm": "supervised_f1", "seed": 0, "model": sup["model"].cpu(),
                                 "res": None, "cfg": CFG}
            free_cuda()
    save_ledger()
else:
    print("Stage 1 probes skipped in this session")
"""

STAGE1_SUMMARY = r"""# --- Stage 1c: seed-level tables and the go/no-go test ------------------------------------
def probe_table(task):
    df = ledger_df()
    df = df[(df.version == VERSION_TAG) & (df.task == f"probe_{task}")]
    if not len(df):
        return pd.DataFrame()
    au = df[df.metric == "macro_auroc"]; f1 = df[df.metric == "macro_f1"]
    rows = []
    for arm, g in au.groupby("arm"):
        s = seed_level_summary(dict(zip(g.seed.astype(int), g.value)))
        f = f1[f1.arm == arm].value
        rows.append({"arm": arm, "family": g.family.iloc[0], "n_seeds": s["n"],
                     "macro_auroc_mean": s["mean"], "macro_auroc_std": s["std"],
                     "ci_lo": s["ci_lo"], "ci_hi": s["ci_hi"],
                     "macro_f1_mean": f.mean() if len(f) else np.nan})
    return pd.DataFrame(rows).sort_values("macro_auroc_mean", ascending=False)

TESTS = []
for task in PROBE_SPLITS:
    t = probe_table(task)
    if not len(t):
        continue
    t.to_csv(os.path.join(RESULTS_DIR, f"table_linear_probe_{task}.csv"), index=False)
    print(f"\n### {task}: linear probe (mean over seeds)"); display(t.round(4))
    df = ledger_df()
    au = df[(df.version == VERSION_TAG) & (df.task == f"probe_{task}") & (df.metric == "macro_auroc")]
    by = {a: dict(zip(g.seed.astype(int), g.value)) for a, g in au.groupby("arm")}
    for a, b in (("dual", "masked_only"), ("dual", "masked_only_cm"), ("dual", "forward_only"),
                 ("dual", "cpc"), ("dual", "simclr"), ("dual", "st_mem")):
        if a in by and b in by:
            p = paired_seed_test(by[a], by[b])
            sa, sb = load_scores(f"{a}_s0__{task}"), load_scores(f"{b}_s0__{task}")
            boot = (bootstrap_auroc_delta(sa["y_true"], sa["y_score"], sb["y_score"],
                                          n=CFG.evaluation.bootstrap_n)
                    if sa is not None and sb is not None else {})
            TESTS.append({"task": task, "a": a, "b": b, "seed_delta": p["delta"],
                          "n_seeds": p["n_seeds"], "seed_p": p["p_value"],
                          "boot_delta_s0": boot.get("delta"), "boot_lo": boot.get("lo"),
                          "boot_hi": boot.get("hi")})
if TESTS:
    TESTS_DF = pd.DataFrame(TESTS)
    TESTS_DF.to_csv(os.path.join(RESULTS_DIR, "table_tests.csv"), index=False)
    print("\n### paired tests (seed-level t-test; bootstrap on seed 0 test set)")
    display(TESTS_DF.round(4))
    g = TESTS_DF[(TESTS_DF.a == "dual") & (TESTS_DF.b == "masked_only")]
    if len(g):
        r = g.iloc[0]
        if r.n_seeds >= 3:
            verdict = ("GO: dual better" if r.seed_delta > 0 and r.seed_p < 0.05 else
                       "NO-GO: dual significantly worse" if r.seed_delta < 0 and r.seed_p < 0.05 else
                       "TIE: no significant difference over seeds")
        else:
            verdict = f"PRELIMINARY ({int(r.n_seeds)} seed(s)) -- run sessions B-D for 3 seeds"
        print(f"\nGo/no-go (dual vs masked_only): delta={r.seed_delta:+.4f} -> {verdict}")
"""

ROBUST_MD = r"""## Stage 1d — robustness to real noise and missing leads

The linear head is trained on clean PTB-XL features and applied to test ECGs with NSTDB
noise (baseline wander, electrode motion, muscle artefact) at 24 / 12 / 6 / 0 dB SNR, and to
reduced lead sets (I+II; single lead II, as in patch/wearable ECG). The noise is band-passed
like the ECG pipeline (0.5–45 Hz), i.e. it is the part of the noise that survives standard
filtering.
"""

ROBUST = r"""# --- Stage 1d: robustness ------------------------------------------------------------------
if "robust" in PLAN["stages"] and PROBE_SPLITS:
    task = "ptbxl" if "ptbxl" in PROBE_SPLITS else next(iter(PROBE_SPLITS))
    nst = FOUND.get("mitbih_nst")
    BANK = load_nstdb_noise(nst.data_root if nst else None, CFG.signal.fs) or \
        synthetic_noise_bank(CFG.signal.fs, minutes=10.0 if RUN_MODE == "kaggle" else 1.0)
    snrs = (24.0, 12.0, 6.0, 0.0) if RUN_MODE == "kaggle" else (6.0,)
    targets = [m for m in sorted(REGISTRY) if m.endswith("_s0") and
               REGISTRY[m]["arm"] in ("masked_only", "forward_only", "dual", "simclr", "cpc",
                                      "st_mem", "supervised_f1")]
    for mid in targets:
        e = REGISTRY[mid]
        if already_done(mid, f"robust_{task}", "macro_auroc_clean"):
            continue
        print(f"[{mid}]")
        df_r = run_robustness_probe(e["model"].to(DEVICE), PROBE_SPLITS[task], DEVICE,
                                    CFG.evaluation, BANK, snrs_db=snrs, batch_size=64,
                                    num_workers=CFG.optim.num_workers, verbose=True)
        for _, r in df_r.iterrows():
            log(mid, e["arm"], e["seed"], f"robust_{task}",
                "macro_auroc_clean" if r.condition == "clean" else "macro_auroc",
                r.macro_auroc, r.condition)
        e["model"].to("cpu"); free_cuda(); save_ledger()
else:
    print("robustness skipped in this session")
"""

STAGE2_MD = r"""## Stage 2 — H1: AF-onset anticipation on **held-out** records + forecast skill

The test records were removed from pretraining (section 6), so the forward predictor never
saw their futures. Features: `context` (pooled encoder embedding of the 20 s context) and
`world` (`context` ‖ predicted future latent at H). **Read `world` gains against the
`masked_only/world` control**: its forward predictor is never trained, so it adds a random
projection of the context of the same size. CIs are record-level cluster bootstraps.

Forecast skill = cos(prediction, true future) − cos(present, true future) in the trained
target space; `horizon_gain` = cos(prediction at H) − cos(prediction at a wrong horizon).
"""

STAGE2 = r"""# --- Stage 2: anticipation + forecast skill -------------------------------------------------
CW = CFG.forward.context_windows
annotated_test = [r for r in LONG_TEST if r.episodes]
if "s2" in PLAN["stages"] and annotated_test:
    jepa = [m for m in sorted(REGISTRY) if REGISTRY[m]["arm"] in
            ("masked_only", "forward_only", "dual", "masked_only_cm")]
    others = [m for m in sorted(REGISTRY) if REGISTRY[m]["arm"] in ("simclr", "cpc", "st_mem")
              and m.endswith("_s0")]
    for H in CFG.evaluation.anticipation_horizons_s:
        samples, stats = build_onset_dataset(
            LONG_ALL, CFG.signal, horizon_s=H, context_windows=CW,
            stride_s=CFG.evaluation.anticipation_stride_s,
            guard_s=CFG.evaluation.anticipation_guard_s, max_per_record=MAX_PER_RECORD)
        tr, te = split_by_ids(samples, HOLDOUT.test_ids)
        n_pos_te = sum(s.label for s in te)
        print(f"\nH={H:.0f}s: {stats['n_positive']:.0f} pos / {stats['n_negative']:.0f} neg | "
              f"train {len(tr)} / test {len(te)} windows ({n_pos_te:.0f} test positives)")
        if n_pos_te < 5 or len(te) - n_pos_te < 5 or not tr:
            print("  too few test events at this horizon -> skipped"); continue
        tr_ds = cache_dataset(AnticipationDataset(LONG_ALL, tr, CFG.signal, CW), f"onset{H:.0f}_tr",
                              CFG.data.cache_dir, num_workers=CFG.optim.num_workers)
        te_ds = cache_dataset(AnticipationDataset(LONG_ALL, te, CFG.signal, CW), f"onset{H:.0f}_te",
                              CFG.data.cache_dir, num_workers=CFG.optim.num_workers)
        for mid in jepa + others:
            e = REGISTRY[mid]
            modes = ("context", "world") if hasattr(e["model"], "forward_predictor") else ("context",)
            for mode in modes:
                key = f"onset_{mode}"
                if already_done(mid, key, f"auroc@{H:.0f}"):
                    continue
                r = run_anticipation_probe(e["model"].to(DEVICE), LONG_ALL, tr, te, CFG.signal,
                                           DEVICE, CFG.evaluation, context_windows=CW,
                                           feature_mode=mode, horizons_s=[H], batch_size=64,
                                           train_ds=tr_ds, test_ds=te_ds, verbose=False)
                if not np.isfinite(r.get("auroc", np.nan)):
                    print(f"  {mid}/{mode}: {r.get('note')}"); continue
                for k in ("auroc", "ci_lo", "ci_hi", "auprc", "prevalence"):
                    log(mid, e["arm"], e["seed"], key, f"{k}@{H:.0f}", r[k], H)
                for f, v in r["by_fraction"].items():
                    log(mid, e["arm"], e["seed"], f"onset_labeleff_{mode}", f"auroc@{H:.0f}", v, f)
                save_scores(f"{mid}__onset{H:.0f}_{mode}", y_true=r["y_true"],
                            y_score=r["y_score"], groups=r["groups"])
                print(f"  {mid:20s} {mode:7s} AUROC={r['auroc']:.4f} [{r['ci_lo']:.4f},{r['ci_hi']:.4f}]"
                      f"  AUPRC={r['auprc']:.4f}")
            e["model"].to("cpu"); free_cuda()
        save_ledger()

    # forecast skill on held-out records only
    HS = tuple(h for h in (10.0, 60.0, 300.0, 600.0, 1800.0) if RUN_MODE == "kaggle" or h <= 300)
    for mid in jepa:
        e = REGISTRY[mid]
        if e["arm"] == "masked_only_cm" or already_done(mid, "forecast_skill", f"skill@{HS[0]:.0f}"):
            continue
        print(f"[{mid}] forecast skill on {len(LONG_TEST)} held-out records")
        df_s = forecast_skill(e["model"].to(DEVICE), LONG_TEST, CFG.signal, DEVICE, horizons_s=HS,
                              n_samples=N_SKILL, context_windows=CW, seed=0, verbose=True)
        for _, r in df_s.iterrows():
            for k in ("skill", "skill_lo", "skill_hi", "horizon_gain", "cos_pred", "cos_persistence"):
                log(mid, e["arm"], e["seed"], "forecast_skill", f"{k}@{r.horizon_s:.0f}", r[k],
                    r.horizon_s)
        e["model"].to("cpu"); free_cuda()
    save_ledger()
else:
    print("Stage 2 skipped (not in this session, or no annotated held-out records)")
"""

STAGE2_SUMMARY = r"""# --- Stage 2 summary: world gain vs the random-predictor control --------------------------
df = ledger_df()
an = df[(df.version == VERSION_TAG) & df.task.isin(["onset_context", "onset_world"])
        & df.metric.str.startswith("auroc@")]
if len(an):
    an = an.assign(H=an.metric.str.split("@").str[1].astype(float),
                   mode=an.task.str.replace("onset_", ""))
    tab = an.pivot_table(index=["arm", "H"], columns="mode", values="value", aggfunc="mean")
    if "world" in tab.columns and "context" in tab.columns:
        tab["world_gain"] = tab["world"] - tab["context"]
        ctrl = tab.xs("masked_only", level="arm")["world_gain"] if "masked_only" in tab.index.get_level_values(0) else None
        if ctrl is not None:
            tab["gain_over_random_predictor"] = [
                g - ctrl.get(h, np.nan) for (a, h), g in tab["world_gain"].items()]
    tab = tab.reset_index()
    tab.to_csv(os.path.join(RESULTS_DIR, "table_anticipation.csv"), index=False)
    print("AUROC (mean over seeds); 'gain_over_random_predictor' is the honest world-model effect")
    display(tab.round(4))
sk = df[(df.version == VERSION_TAG) & (df.task == "forecast_skill")]
if len(sk):
    sk = sk.assign(H=sk.metric.str.split("@").str[1].astype(float), m=sk.metric.str.split("@").str[0])
    skt = sk.pivot_table(index=["arm", "H"], columns="m", values="value", aggfunc="mean").reset_index()
    skt.to_csv(os.path.join(RESULTS_DIR, "table_forecast_skill.csv"), index=False)
    display(skt.round(4))
"""

STAGE3_MD = r"""## Stage 3 — RQ3: anti-collapse mechanisms for the dual objective

`ema` (β = 0, EMA + stop-gradient only — now actually trained), `vicreg` (= Stage-1 `dual`,
reused only if its loss config is identical), `vq` (fixed soft codebook), `sigreg`
(LeJEPA, no EMA teacher). Verdict per arm: healthy / dimensional / collapsed, on a
512-window 12-lead probe batch.
"""

STAGE3 = r"""# --- Stage 3: RQ3 -------------------------------------------------------------------------
if "s3" in PLAN["stages"]:
    task = next(iter(PROBE_SPLITS)) if PROBE_SPLITS else None
    rows = []
    for arm in ("rq3_ema", "rq3_vicreg", "rq3_vq", "rq3_sigreg"):
        for seed in RQ3_SEEDS:
            if arm == "rq3_vicreg":
                _wo = lambda c: {k: v for k, v in vars(c.loss).items()
                                 if k not in ("weight_forward", "lambda_mask")}
                same = _wo(configure_arm(CFG, "vicreg")) == _wo(arm_cfg("dual", seed))
                e = get_model("dual", seed) if same else get_model(arm, seed, allow_train=True)
                if same:
                    print(f"rq3_vicreg_s{seed}: identical loss config to dual_s{seed} -> reused")
            else:
                e = get_model(arm, seed, allow_train=(arm, seed) in PLAN["train"])
            if e is None or e["res"] is None:
                continue
            s = e["res"].monitor.summary()
            row = {"arm": arm, "seed": seed, "verdict": s.get("verdict"),
                   "final_loss": s.get("final_loss"), "loss_to_zero": s.get("collapsed_to_zero"),
                   "erank": s.get("final_effective_rank"), "cov_erank_ratio": s.get("final_cov_erank_ratio"),
                   "std_mean": s.get("final_std_mean"), "cos_sim": s.get("final_cos_sim")}
            mid = f"{e['arm']}_s{seed}"
            if task:
                if not already_done(mid, f"probe_{task}"):
                    eval_probe(mid, e["arm"], seed, e["model"])
                d = ledger_df()
                v = d[(d.model_id == mid) & (d.task == f"probe_{task}") & (d.metric == "macro_auroc")].value
                row["macro_auroc"] = float(v.iloc[-1]) if len(v) else np.nan
            rows.append(row)
    if rows:
        RQ3 = pd.DataFrame(rows)
        RQ3.to_csv(os.path.join(RESULTS_DIR, "table_rq3.csv"), index=False)
        display(RQ3.round(4))
        print("Pareto rule: prefer the highest AUROC among arms whose verdict is 'healthy'.")
    save_ledger()
else:
    print("Stage 3 skipped in this session")
"""

STAGE4_MD = r"""## Stage 4 — H2: hierarchical vs single-scale across horizons

Both arms train on the same 12-window (2 min) context loader and differ only in the
Level-2 stack. The batch size is reduced automatically if the GPU runs out of memory (the
v0.1 crash); both arms always use the same batch and learning rate.
"""

STAGE4 = r"""# --- Stage 4: hierarchical stack -----------------------------------------------------------
if "s4" in PLAN["stages"] and any(r.episodes for r in LONG_TRAIN):
    L2W = CFG.hierarchical.l2_context_windows
    want = [(a, s) for a, s in PLAN["train"] if a.startswith("stage4_")]
    hcfg = arm_cfg("stage4_hier", 0)
    S4_LOADER = None
    need_train = [j for j in want if f"{j[0]}_s{j[1]}.pth" not in CKPTS]
    bs = hcfg.optim.batch_size
    while need_train and bs >= 2:
        hcfg.optim.batch_size = bs
        S4_LOADER, _ = build_pretrain_loaders(FOUND, hcfg, long_records=LONG_TRAIN,
                                              samples_per_record=SAMPLES_PER_RECORD, verbose=False)
        probe_m, fits = None, True
        try:
            probe_m = CardioJEPA(hcfg.signal, hcfg.model, hcfg.loss, hcfg.hierarchical, hcfg.forward)
            estimate_runtime(probe_m, S4_LOADER, hcfg, DEVICE, n_steps=2)
        except torch.cuda.OutOfMemoryError:
            fits = False
        del probe_m; free_cuda()
        if fits:
            break
        print(f"  batch {bs} does not fit -> {bs // 2}"); bs //= 2
    S4_BATCH = bs
    print(f"Stage 4: context {L2W} x {CFG.signal.window_s:.0f}s, batch {S4_BATCH}")
    for arm, seed in want:
        c = arm_cfg(arm, seed)
        c.optim.batch_size = S4_BATCH
        c.optim.lr = CFG.optim.lr * S4_BATCH / CFG.optim.batch_size   # linear lr scaling
        mid = f"{arm}_s{seed}"
        if mid not in REGISTRY and mid + ".pth" not in CKPTS:
            print(f"\n=== training [{mid}] ===")
            cj.set_seed(seed, c.optim.deterministic)
            m = make_model(arm, c)
            res = cj.pretrain(m, S4_LOADER, c, device=DEVICE, arm_name=mid, probe_batch=PROBE_BATCH,
                              save_path=os.path.join(RUNS_DIR, mid + ".pth"),
                              log_every=0 if RUN_MODE == "smoke" else 200)
            log_training(mid, arm, seed, res); m.to("cpu"); free_cuda()
            REGISTRY[mid] = {"arm": arm, "seed": seed, "model": m, "res": res, "cfg": c}
            CKPTS[mid + ".pth"] = os.path.join(RUNS_DIR, mid + ".pth")
        else:
            get_model(arm, seed)
    s4 = {k: {"model": REGISTRY[k]["model"].to(DEVICE), "feature_mode": "world"}
          for k in REGISTRY if REGISTRY[k]["arm"].startswith("stage4_")}
    if len(s4) == 2:
        HS4 = (300.0, 600.0, 1800.0) if RUN_MODE == "kaggle" else (60.0, 300.0)
        def _builder(records, signal, horizon_s, context_windows):
            return build_onset_dataset(records, signal, horizon_s=horizon_s,
                                       context_windows=context_windows,
                                       stride_s=CFG.evaluation.anticipation_stride_s,
                                       guard_s=CFG.evaluation.anticipation_guard_s,
                                       max_per_record=MAX_PER_RECORD)
        sweep = horizon_sweep(s4, LONG_ALL, CFG.signal, DEVICE, CFG.evaluation, horizons_s=HS4,
                              builder=_builder, context_windows=L2W, test_ids=HOLDOUT.test_ids)
        for _, r in sweep.iterrows():
            e = REGISTRY[r.model]
            log(r.model, e["arm"], e["seed"], "stage4_sweep", f"auroc@{r.horizon_s:.0f}", r.auroc,
                r.horizon_s, extra={"ci": [r.ci_lo, r.ci_hi]})
        bs_, bst = build_burden_dataset(LONG_ALL, CFG.signal, context_windows=L2W,
                                        from_first_minutes=30.0, samples_per_record=8, seed=0)
        btr, bte = split_by_ids(bs_, HOLDOUT.test_ids)
        for k, spec in s4.items():
            r = run_burden_probe(spec["model"], LONG_ALL, btr, bte, CFG.signal, DEVICE,
                                 context_windows=L2W, feature_mode="world",
                                 horizons_s=[1800.0 if RUN_MODE == "kaggle" else 300.0],
                                 batch_size=16, verbose=True)
            for m_ in ("r2", "mae", "spearman", "r2_record", "spearman_record"):
                log(k, REGISTRY[k]["arm"], 0, "stage4_burden", m_, r.get(m_))
        if len(sweep):
            piv = sweep.pivot(index="horizon_s", columns="model", values="auroc")
            display(piv.round(4)); piv.to_csv(os.path.join(RESULTS_DIR, "table_stage4.csv"))
    for k in s4: REGISTRY[k]["model"].to("cpu")
    free_cuda(); save_ledger()
else:
    print("Stage 4 skipped in this session")
"""

FIGURES = r"""# --- figures for the paper (PNG + PDF; built from the merged ledger) -------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.3, "font.size": 9})

def savefig(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"), bbox_inches="tight")
    plt.show(); plt.close(fig)

DF = ledger_df()
DF = DF[DF.version == VERSION_TAG] if len(DF) else DF
ORDER = ["random_init", "st_mem", "simclr", "cpc", "masked_only", "masked_only_cm",
         "forward_only", "dual", "supervised_f0.01", "supervised_f0.1", "supervised_f1"]

# F2 training curves
hist = {m: e["res"] for m, e in REGISTRY.items() if e.get("res") is not None and e["res"].history}
if hist:
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.4))
    for m, r in sorted(hist.items()):
        h = pd.DataFrame(r.history); mon = pd.DataFrame(r.monitor.history)
        ax[0].plot(h.epoch, h.loss, label=m)
        if "loss_fwd" in h: ax[1].plot(h.epoch, h.loss_fwd, label=m)
        if "cov_effective_rank" in mon: ax[2].plot(mon.epoch, mon.cov_effective_rank, label=m)
    for a, t in zip(ax, ("total loss", "L_fwd", "covariance effective rank (512 windows)")):
        a.set_title(t); a.set_xlabel("epoch")
    ax[2].legend(fontsize=6, ncol=2); savefig(fig, "fig_training_curves")

# F3 linear probe bars
for task in PROBE_SPLITS:
    t = probe_table(task)
    if len(t):
        t = (t.assign(_o=t.arm.map({a: i for i, a in enumerate(ORDER)}).fillna(99))
              .sort_values("_o").drop(columns="_o"))
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.bar(t.arm, t.macro_auroc_mean, yerr=t.macro_auroc_std.fillna(0), capsize=3,
               color=["#c44e52" if a == "dual" else "#4c72b0" for a in t.arm])
        ax.set_ylim(max(0.5, t.macro_auroc_mean.min() - 0.05), 1.0)
        ax.set_ylabel("macro AUROC"); ax.set_title(f"{task}: frozen-encoder linear probe (mean ± sd over seeds)")
        plt.xticks(rotation=35, ha="right"); savefig(fig, f"fig_linear_probe_{task}")

# F4 label efficiency
for task in PROBE_SPLITS:
    le = DF[(DF.task == f"labeleff_{task}") & (DF.metric == "macro_auroc")] if len(DF) else DF
    if len(le):
        le = le.assign(frac=le.setting.astype(float))
        g = le.groupby(["arm", "frac"]).value.agg(["mean", "std"]).reset_index()
        g.to_csv(os.path.join(RESULTS_DIR, f"table_label_efficiency_{task}.csv"), index=False)
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        for arm, gg in g.groupby("arm"):
            if arm.startswith("supervised"):
                continue
            ax.errorbar(gg.frac * 100, gg["mean"], yerr=gg["std"].fillna(0), marker="o", label=arm, capsize=2)
        sup = g[g.arm.str.startswith("supervised")]
        if len(sup):
            ax.plot(sup.frac * 100, sup["mean"], "k--", marker="s", label="supervised")
        ax.set_xscale("log"); ax.set_xlabel("% labelled training ECGs"); ax.set_ylabel("macro AUROC")
        ax.legend(fontsize=7); ax.set_title(f"label efficiency ({task})"); savefig(fig, f"fig_label_efficiency_{task}")

# F5 robustness
rb = DF[DF.task.str.startswith("robust_")] if len(DF) else DF
if len(rb):
    rb.to_csv(os.path.join(RESULTS_DIR, "table_robustness.csv"), index=False)
    noise = rb[rb.setting.str.contains("dB")]
    if len(noise):
        noise = noise.assign(kind=noise.setting.str.split("@").str[0],
                             snr=noise.setting.str.extract(r"@([-\d.]+)dB")[0].astype(float))
        kinds = sorted(noise.kind.unique())
        fig, ax = plt.subplots(1, len(kinds), figsize=(4 * len(kinds), 3), squeeze=False)
        for a, k in zip(ax[0], kinds):
            for arm, g in noise[noise.kind == k].groupby("arm"):
                g = g.sort_values("snr", ascending=False)
                a.plot(g.snr, g.value, marker="o", label=arm)
            a.invert_xaxis(); a.set_title(f"noise: {k}"); a.set_xlabel("SNR (dB)"); a.set_ylabel("macro AUROC")
        ax[0][-1].legend(fontsize=6); savefig(fig, "fig_robustness_noise")

# F6/F7 anticipation + forecast skill
ant = DF[DF.task.isin(["onset_context", "onset_world"]) & DF.metric.str.startswith("auroc@")] if len(DF) else DF
if len(ant):
    ant = ant.assign(H=ant.metric.str.split("@").str[1].astype(float))
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for (arm, task), g in ant.groupby(["arm", "task"]):
        g = g.groupby("H").value.mean()
        ax.plot(g.index / 60, g.values, marker="o", ls="-" if task.endswith("world") else ":",
                label=f"{arm}/{task.split('_')[1]}")
    ax.set_xlabel("horizon (min)"); ax.set_ylabel("AUROC (held-out records)"); ax.legend(fontsize=6, ncol=2)
    savefig(fig, "fig_anticipation")
sk = DF[(DF.task == "forecast_skill")] if len(DF) else DF
if len(sk):
    sk = sk.assign(H=sk.metric.str.split("@").str[1].astype(float), m=sk.metric.str.split("@").str[0])
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    for arm, g in sk.groupby("arm"):
        p = g.pivot_table(index="H", columns="m", values="value", aggfunc="mean")
        if "skill" in p:
            lo = np.abs(p["skill"] - p.get("skill_lo", p["skill"]))
            hi = np.abs(p.get("skill_hi", p["skill"]) - p["skill"])
            ax[0].errorbar(p.index, p["skill"], yerr=[lo, hi], marker="o", label=arm, capsize=2)
        if "horizon_gain" in p:
            ax[1].plot(p.index, p["horizon_gain"], marker="o", label=arm)
    for a, t in zip(ax, ("forecast skill vs persistence", "horizon gain (right vs wrong horizon)")):
        a.axhline(0, color="k", lw=0.8); a.set_xscale("log"); a.set_xlabel("horizon (s)"); a.set_title(t)
    ax[0].legend(fontsize=7); savefig(fig, "fig_forecast_skill")

# F8 RQ3
rq = [e for m, e in REGISTRY.items() if (e["arm"].startswith("rq3_") or m == "dual_s0") and e.get("res")]
if rq:
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    for e in rq:
        mon = pd.DataFrame(e["res"].monitor.history)
        lab = "rq3_vicreg(=dual)" if e["arm"] == "dual" else e["arm"]
        if "cov_effective_rank" in mon: ax[0].plot(mon.epoch, mon.cov_effective_rank, label=lab)
        ax[1].plot(mon.epoch, mon.loss, label=lab)
    ax[0].set_title("covariance effective rank"); ax[1].set_title("training loss"); ax[1].set_yscale("log")
    ax[0].legend(fontsize=7); savefig(fig, "fig_rq3")
print("figures written to", FIG_DIR, ":", sorted(os.listdir(FIG_DIR)))
"""

EXIT = r"""# --- exit criteria, loop log, comparison with v0.1 and the previous version ----------------
def repro_check(steps=6):
    # same seed twice -> same losses (tiny model, real loader)
    c = smoke_config(); c.signal = CFG.signal; c.optim.max_steps_per_epoch = steps
    c.optim.epochs = 1; c.optim.amp = CFG.optim.amp; c.optim.num_workers = 0
    c.loss.anti_collapse, c.loss.beta_reg = "vicreg", 0.1
    c.optim.deterministic = True      # keep the global cuDNN flags deterministic
    sub_loader, _ = build_pretrain_loaders({}, c, samples_per_record=4, verbose=False)
    losses = []
    for _ in range(2):
        cj.set_seed(123, True)
        m = CardioJEPA(c.signal, c.model, c.loss, c.hierarchical, c.forward)
        r = cj.pretrain(m, sub_loader, c, device=DEVICE, arm_name="repro", verbose=False)
        losses.append(r.history[-1]["loss"])
    return abs(losses[0] - losses[1]) <= 1e-3 * max(1.0, abs(losses[0])), losses

DF = ledger_df(); DFv = DF[DF.version == VERSION_TAG] if len(DF) else DF
crit = {}
chk = DFv[DFv.task == "checks"] if len(DFv) else DFv
crit["1_leakage_and_data_checks"] = bool(len(chk)) and bool((chk.value == 1).all())
tr = DFv[DFv.task == "train"] if len(DFv) else DFv
main = tr[~tr.arm.str.startswith("rq3_")] if len(tr) else tr
coll = main[main.metric == "verdict_collapsed"] if len(main) else main
crit["2_no_collapse_in_reported_arms"] = bool(len(coll)) and bool((coll.value == 0).all())
# fixed-target baselines must reduce their loss; JEPA losses may legitimately rise as the
# EMA target becomes more informative, so JEPA arms are judged by the collapse verdict (2)
drops = tr[(tr.metric == "loss_drop") & tr.arm.isin(["st_mem", "simclr", "cpc"])] if len(tr) else tr
crit["3_every_baseline_trained(loss_drop>=10%)"] = (bool((drops.value >= 0.10).all())
                                                    if len(drops) else None)
task0 = next(iter(PROBE_SPLITS), None)
seeds_ok = std_ok = False
if task0:
    t = probe_table(task0)
    if len(t) and {"dual", "masked_only"} <= set(t.arm):
        sub = t[t.arm.isin(["dual", "masked_only"])]
        seeds_ok = bool((sub.n_seeds >= 3).all())
        std_ok = bool((sub.macro_auroc_std.fillna(1) <= 0.01).all())
crit["4_three_seeds_and_stable(std<=0.01)"] = seeds_ok and std_ok
ok, rl = repro_check() if RUN_MODE == "kaggle" or SESSION == "ALL" else (None, [])
crit["5_reproducible(same_seed_same_loss)"] = None if ok is None else bool(ok)
better = None
if task0 and len(TESTS):
    T = pd.DataFrame(TESTS); T = T[T.task == task0]
    dm = T[(T.a == "dual") & (T.b == "masked_only")]
    if len(dm) and dm.iloc[0].n_seeds >= 3:
        tab = probe_table(task0).set_index("arm").macro_auroc_mean
        best_ssl = max([tab.get(a, -1) for a in ("simclr", "cpc", "st_mem")])
        better = bool(dm.iloc[0].seed_delta > 0 and dm.iloc[0].seed_p < 0.05 and tab.get("dual", 0) >= best_ssl)
crit["6_dual_beats_masked_and_ssl_baselines"] = better
EXIT_CRITERIA = crit
print("EXIT CRITERIA (1-5 = correctness, must pass; 6 = the scientific result)")
for k, v in crit.items():
    print(f"  {'PASS' if v else ('n/a ' if v is None else 'FAIL')}  {k}")
if rl: print(f"  repro losses: {rl}")
done = all(crit[k] is not False for k in list(crit)[:5]) and crit["1_leakage_and_data_checks"]
print("\nLOOP STATUS:", "exit criteria 1-5 met" if done else
      "continue the loop: run the next session / fix the failing checks")

# v0.1 reference numbers (leaky, single seed -- for orientation only)
V01 = {"probe_ptbxl": {"dual": 0.8177, "masked_only": 0.7675, "forward_only": 0.7176, "st_mem": 0.6500}}
if task0 == "ptbxl":
    t = probe_table("ptbxl")
    if len(t):
        cmp_ = t[["arm", "macro_auroc_mean"]].assign(v01=t.arm.map(V01["probe_ptbxl"]))
        cmp_["delta_vs_v01"] = cmp_.macro_auroc_mean - cmp_.v01
        print("\nvs v0.1 (v0.1 pretrained on PTB-XL -> NOT comparable; shown only to track the loop)")
        display(cmp_.round(4))
prev = previous_ledgers()
if prev:
    p = pd.concat(prev); p = p[(p.metric == "macro_auroc") & p.task.str.startswith("probe_")]
    q = DFv[(DFv.metric == "macro_auroc") & DFv.task.str.startswith("probe_")] if len(DFv) else DFv
    if len(p) and len(q):
        j = q.merge(p, on=["model_id", "task", "setting"], suffixes=("", "_prev"))
        j = j[j.version_prev != j.version]
        if len(j):
            print("\nchange vs previous ledger version:"); display(j[["model_id", "task", "value", "value_prev", "version_prev"]])

with open(os.path.join(RESULTS_DIR, "exit_criteria.json"), "w") as fh:
    json.dump({k: (None if v is None else bool(v)) for k, v in crit.items()}, fh, indent=1)
with open(os.path.join(RESULTS_DIR, "loop_log.jsonl"), "a") as fh:
    fh.write(json.dumps({"time": datetime.datetime.utcnow().isoformat(timespec="seconds"),
                         "version": VERSION_TAG, "session": SESSION, "run_mode": RUN_MODE,
                         "models": sorted(REGISTRY), "criteria": {k: (None if v is None else bool(v))
                                                                   for k, v in crit.items()}}) + "\n")
save_ledger()
"""

EXPORT = r"""# --- export: docs, config, source, ledger, tables, figures -> one zip ----------------------
import zipfile
with open(os.path.join(RESULTS_DIR, "config.json"), "w") as fh:
    json.dump({"run_mode": RUN_MODE, "session": SESSION, "plan": PLAN, "config": CFG.to_dict(),
               "datasets": {k: v.label for k, v in FOUND.items()},
               "holdout_test_records": HOLDOUT.test_names,
               "torch": torch.__version__, "package": cj.__version__}, fh, indent=1, default=str)
for name, text in DOCS.items():
    with open(os.path.join(WORK, name), "w") as fh:
        fh.write(text)
save_ledger()
ZIP = os.path.join(WORK, "cardio_jepa_v3_outputs.zip")
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for base in ("results", "cardio_jepa_v3_src"):
        for root, _d, files in os.walk(os.path.join(WORK, base)):
            for f in files:
                p = os.path.join(root, f); z.write(p, os.path.relpath(p, WORK))
    for f in glob.glob(os.path.join(RUNS_DIR, "*")):
        if f.endswith(".pth") and not ZIP_CHECKPOINTS:
            continue
        z.write(f, os.path.relpath(f, WORK))
    for name in DOCS:
        z.write(os.path.join(WORK, name), name)
print(f"wrote {ZIP} ({os.path.getsize(ZIP) / 1e6:.1f} MB)")
print("next step:", "attach this version's output and run the next SESSION" if SESSION in "ABC"
      else "all sessions done -- tables are in results/, figures in results/figures/")
"""
