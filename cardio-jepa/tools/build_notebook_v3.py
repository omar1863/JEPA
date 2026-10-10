#!/usr/bin/env python3
"""Build the self-contained v3 Kaggle notebook (package source + driver + docs).

Usage:  python tools/build_notebook_v3.py [-o notebooks/cardio_jepa_v3_kaggle.ipynb]
                                          [--mode kaggle|smoke] [--session A]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
import nb_cells_v3 as C  # noqa: E402

_N = [0]


def _id():
    _N[0] += 1
    return f"cell-{_N[0]:03d}"


def md(src):
    return {"cell_type": "markdown", "id": _id(), "metadata": {}, "source": src.splitlines(True)}


def code(src, collapsed=False):
    meta = {"jupyter": {"source_hidden": True}, "collapsed": True} if collapsed else {}
    return {"cell_type": "code", "id": _id(), "execution_count": None, "metadata": meta,
            "outputs": [], "source": src.rstrip("\n").splitlines(True)}


PKG_FILES = [
    "cardio_jepa/__init__.py", "cardio_jepa/config.py", "cardio_jepa/regularizers.py",
    "cardio_jepa/diagnostics.py", "cardio_jepa/engine.py", "cardio_jepa/checkpoint.py",
    "cardio_jepa/models/__init__.py", "cardio_jepa/models/pos_encoding.py",
    "cardio_jepa/models/transformer.py", "cardio_jepa/models/encoder.py",
    "cardio_jepa/models/forward_predictor.py", "cardio_jepa/models/cardio_jepa.py",
    "cardio_jepa/models/baselines.py", "cardio_jepa/models/ssl_baselines.py",
    "cardio_jepa/data/__init__.py", "cardio_jepa/data/wfdb_lite.py",
    "cardio_jepa/data/preprocess.py", "cardio_jepa/data/discovery.py",
    "cardio_jepa/data/synthetic.py", "cardio_jepa/data/short.py",
    "cardio_jepa/data/longform.py", "cardio_jepa/data/cache.py", "cardio_jepa/data/build.py",
    "cardio_jepa/eval/__init__.py", "cardio_jepa/eval/stats.py",
    "cardio_jepa/eval/linear_probe.py", "cardio_jepa/eval/benchmarks.py",
    "cardio_jepa/eval/robustness.py", "tests/test_smoke.py", "tests/test_v02.py",
]
DOC_FILES = ["BUG_REPORT.md", "CHANGELOG.md", "RESEARCH_PLAN.md", "EXPERIMENT_PLAN.md"]


def read(rel):
    with open(os.path.join(REPO, rel), encoding="utf-8") as fh:
        return fh.read()


def build(mode="kaggle", session="A"):
    cells = [md(C.TITLE), md(C.DATA_MD), code(C.ENV), md(C.SRC_MD)]
    for rel in PKG_FILES:
        src = read(rel)
        cells.append(code(f"%%writefile cardio_jepa_v3_src/{rel}\n{src}", collapsed=True))
    cells.append(code(C.IMPORT))
    cfg = C.CONFIG.replace('RUN_MODE = "kaggle"', f'RUN_MODE = "{mode}"', 1) \
                  .replace('SESSION = "A"', f'SESSION = "{session}"', 1)
    cells += [md(C.CONFIG_MD), code(cfg), code(C.HELPERS),
              md("## 5–6. Discovery, fixed holdout, loaders, probe caches, leakage checks"),
              code(C.DISCOVERY), code(C.DATA),
              md("## 7–8. Arms, train-or-load, runtime projection"),
              code(C.MODELS), code(C.RUNTIME),
              md(C.STAGE1_MD), code(C.STAGE1_TRAIN), code(C.STAGE1_EVAL), code(C.STAGE1_SUMMARY),
              md(C.ROBUST_MD), code(C.ROBUST),
              md(C.STAGE2_MD), code(C.STAGE2), code(C.STAGE2_SUMMARY),
              md(C.STAGE3_MD), code(C.STAGE3),
              md(C.STAGE4_MD), code(C.STAGE4),
              md("## Figures for the paper\nBuilt from the merged results ledger, so the "
                 "last session draws every figure."),
              code(C.FIGURES),
              md("## Loop engineering: exit criteria, loop log, comparison with earlier versions\n"
                 "Each session is one loop iteration: *analyse → fix → run → evaluate → compare → "
                 "log*. Criteria 1–5 check correctness; criterion 6 is the scientific result."),
              code(C.EXIT)]
    docs = {name: read(os.path.join("docs", name)) for name in DOC_FILES}
    cells.append(code("# --- documents written into the output zip (also shown in the Appendix) ---\n"
                      "DOCS = " + json.dumps(docs, indent=0, ensure_ascii=False) + "\n"
                      "print('docs:', list(DOCS))", collapsed=True))
    cells += [md("## Export"), code(C.EXPORT)]
    cells.append(md("# Appendix"))
    for name in DOC_FILES:
        cells.append(md(docs[name]))
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                        "name": "python3"},
                         "language_info": {"name": "python", "version": "3.12"},
                         "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": False,
                                    "language": "python", "sourceType": "notebook",
                                    "isGpuEnabled": True}}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default=os.path.join(REPO, "notebooks",
                                                        "cardio_jepa_v3_kaggle.ipynb"))
    ap.add_argument("--mode", default="kaggle")
    ap.add_argument("--session", default="A")
    a = ap.parse_args()
    nb = build(a.mode, a.session)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(nb, fh, indent=1, ensure_ascii=False)
    print(f"wrote {a.out}: {len(nb['cells'])} cells")


if __name__ == "__main__":
    main()
