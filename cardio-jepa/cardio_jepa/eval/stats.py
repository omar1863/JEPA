"""Statistical testing for Section 6.4: bootstrap CIs on AUROC differences."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np


def _auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if y_true.ndim == 1:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_score.reshape(-1)))
    per = []
    for j in range(y_true.shape[1]):
        if len(np.unique(y_true[:, j])) < 2:
            continue
        per.append(roc_auc_score(y_true[:, j], y_score[:, j]))
    return float(np.mean(per)) if per else float("nan")


def _resampler(N: int, groups: Optional[np.ndarray], rng):
    """v0.2: index generator for the (cluster) bootstrap.

    Without ``groups`` samples are resampled i.i.d. (v0.1 behaviour). With ``groups``
    (record / patient ids) whole groups are resampled -- windows from one Holter
    record are strongly correlated, so an i.i.d. window bootstrap gives CIs that are
    far too narrow and p-values of 0.000 that the data cannot support.
    """
    if groups is None:
        return lambda: rng.integers(0, N, N)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    members = [np.flatnonzero(groups == g) for g in uniq]
    G = len(uniq)

    def draw():
        pick = rng.integers(0, G, G)
        return np.concatenate([members[i] for i in pick])
    return draw


def bootstrap_auroc(y_true: np.ndarray, y_score: np.ndarray, n: int = 2000,
                    alpha: float = 0.05, seed: int = 0,
                    groups: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Percentile bootstrap CI for a single model's macro AUROC (cluster bootstrap
    over ``groups`` when given)."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    rng = np.random.default_rng(seed)
    N = len(y_true)
    draw = _resampler(N, groups, rng)
    vals = []
    for _ in range(n):
        idx = draw()
        v = _auroc(y_true[idx], y_score[idx])
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return {"auroc": float("nan"), "lo": float("nan"), "hi": float("nan")}
    vals = np.asarray(vals)
    return {"auroc": _auroc(y_true, y_score),
            "lo": float(np.percentile(vals, 100 * alpha / 2)),
            "hi": float(np.percentile(vals, 100 * (1 - alpha / 2))),
            "std": float(vals.std())}


def bootstrap_auroc_delta(y_true: np.ndarray, score_a: np.ndarray, score_b: np.ndarray,
                          n: int = 2000, alpha: float = 0.05, seed: int = 0,
                          groups: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Paired bootstrap on AUROC(a) - AUROC(b).

    Resampling the *same* indices for both models keeps the comparison paired,
    which is what Section 6.4 asks for. ``p_value`` is the two-sided proportion of
    bootstrap deltas on the opposite side of zero (an achieved significance level),
    not a parametric test.
    """
    y_true = np.asarray(y_true)
    a = np.asarray(score_a)
    b = np.asarray(score_b)
    rng = np.random.default_rng(seed)
    N = len(y_true)
    draw = _resampler(N, groups, rng)
    deltas = []
    for _ in range(n):
        idx = draw()
        va = _auroc(y_true[idx], a[idx])
        vb = _auroc(y_true[idx], b[idx])
        if not (np.isnan(va) or np.isnan(vb)):
            deltas.append(va - vb)
    if not deltas:
        return {"delta": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "p_value": float("nan"), "significant": False}
    d = np.asarray(deltas)
    obs = _auroc(y_true, a) - _auroc(y_true, b)
    lo = float(np.percentile(d, 100 * alpha / 2))
    hi = float(np.percentile(d, 100 * (1 - alpha / 2)))
    p = 2.0 * min(float((d <= 0).mean()), float((d >= 0).mean()))
    return {"delta": float(obs), "lo": lo, "hi": hi, "p_value": min(p, 1.0),
            "significant": bool(lo > 0 or hi < 0)}


def compare_models(y_true: np.ndarray, scores: Dict[str, np.ndarray],
                   reference: str, n: int = 2000, seed: int = 0,
                   groups: Optional[np.ndarray] = None):
    """Table of every model vs. ``reference`` with paired-bootstrap CIs."""
    import pandas as pd
    rows = []
    for name, s in scores.items():
        ci = bootstrap_auroc(y_true, s, n=n, seed=seed, groups=groups)
        row = {"model": name, "auroc": ci["auroc"],
               "ci_lo": ci["lo"], "ci_hi": ci["hi"]}
        if name != reference and reference in scores:
            d = bootstrap_auroc_delta(y_true, s, scores[reference], n=n, seed=seed,
                                      groups=groups)
            row.update({"delta_vs_ref": d["delta"], "delta_lo": d["lo"],
                        "delta_hi": d["hi"], "p_value": d["p_value"],
                        "significant": d["significant"]})
        rows.append(row)
    return pd.DataFrame(rows).sort_values("auroc", ascending=False).reset_index(drop=True)


def label_efficiency_table(results_by_model: Dict[str, Dict[float, float]]):
    """Rows = model, columns = label fraction, values = AUROC (tests H1's
    'margin grows as labels shrink' claim)."""
    import pandas as pd
    df = pd.DataFrame(results_by_model).T
    df.columns = [f"{int(c * 100)}%" for c in df.columns]
    return df


def seed_level_summary(values_by_seed: Dict[int, float]) -> Dict[str, float]:
    """v0.2: mean / std / 95% t-interval across *training seeds* (the variance the
    test-set bootstrap cannot see)."""
    v = np.asarray([x for x in values_by_seed.values() if np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0,
                "ci_lo": float("nan"), "ci_hi": float("nan")}
    if len(v) == 1:
        return {"mean": float(v[0]), "std": float("nan"), "n": 1,
                "ci_lo": float("nan"), "ci_hi": float("nan")}
    from scipy import stats as _st
    m, sd = float(v.mean()), float(v.std(ddof=1))
    h = float(_st.t.ppf(0.975, len(v) - 1) * sd / np.sqrt(len(v)))
    return {"mean": m, "std": sd, "n": int(len(v)), "ci_lo": m - h, "ci_hi": m + h}


def paired_seed_test(a: Dict[int, float], b: Dict[int, float]) -> Dict[str, float]:
    """v0.2: paired t-test over seeds shared by both arms (a - b)."""
    common = sorted(set(a) & set(b))
    d = np.asarray([a[k] - b[k] for k in common], dtype=float)
    if len(d) < 2:
        return {"delta": float(d.mean()) if len(d) else float("nan"),
                "n_seeds": int(len(d)), "p_value": float("nan")}
    from scipy import stats as _st
    t = _st.ttest_1samp(d, 0.0)
    return {"delta": float(d.mean()), "delta_std": float(d.std(ddof=1)),
            "n_seeds": int(len(d)), "p_value": float(t.pvalue)}
