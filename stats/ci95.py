"""
stats/ci95.py
-------------
95% confidence intervals and significance tests used in the paper.

1. Paired comparison of two pipelines (AUPRC and ROC-AUC), from out-of-fold predictions:
   - fold-wise paired t-test on the per-fold differences (df = k - 1): mean difference, two-sided 95% CI, p-value;
   - patient-clustered paired bootstrap: within each fold, patients are resampled with replacement (a patient keeps
     all its vertebrae); both pipelines are scored on the same resample, the metric is computed per fold and averaged
     over folds, and the difference is taken. Percentile 95% CI (default 2,000 replicates, seed 42). A replicate in
     which some fold has only one class is redrawn.
   Scopes: all vertebrae, patient level (patient score = maximum vertebra score), and optionally each vertebra level.

2. Difference between two subgroups of a per-instance score (e.g. segmentation Dice of fractured vs non-fractured
   vertebrae), with the same patient-clustered bootstrap. Scopes: each level (with at least `min_n` instances in
   group 1), mean of the per-level values, all instances pooled, and fold-wise (pooled within fold, mean over folds).
   If a view column is given (e.g. sagittal / coronal), values are averaged within each view and then over views.

Usage
-----
    python -m stats.ci95 paired  predictions.csv --score-a proposed --score-b baseline --per-level
    python -m stats.ci95 subgroup dice.csv --value dice --group fractured --view view

Input columns (names configurable): patient_id, fold, label, vertebra, plus the score / value columns.
A vertebra without a prediction should be given score 0 (scored as non-fractured).
Metrics and differences are reported in percentage points.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats

__all__ = ["paired_comparison", "subgroup_difference"]


# --------------------------------------------------------------------------------------------- AP / ROC-AUC
def _prep(y: np.ndarray, s: np.ndarray):
    order = np.argsort(-s, kind="stable")
    ss = s[order]
    ends = np.r_[np.nonzero(np.diff(ss))[0], len(ss) - 1]        # last index of each tied-score group
    return order, y[order].astype(float), ends


def _ap_auc(w: np.ndarray, ys: np.ndarray, ends: np.ndarray):
    """Weighted average precision and ROC-AUC (ties grouped as in scikit-learn). None if a class is empty."""
    tp, fp = np.cumsum(w * ys)[ends], np.cumsum(w * (1 - ys))[ends]
    p_tot, n_tot = tp[-1], fp[-1]
    if p_tot <= 0 or n_tot <= 0:
        return None
    rec = tp / p_tot
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    ap = np.sum(np.diff(np.r_[0.0, rec]) * prec)
    fpr = fp / n_tot
    auc = np.sum(np.diff(np.r_[0.0, fpr]) * (np.r_[0.0, rec][:-1] + rec) / 2)
    return ap, auc


def _t_ci(d: np.ndarray, level: float = 0.95):
    k = len(d)
    m, s = d.mean(), d.std(ddof=1)
    se = s / np.sqrt(k)
    h = stats.t.ppf(0.5 + level / 2, k - 1) * se
    p = 2 * stats.t.sf(abs(m / se), k - 1) if se > 0 else np.nan
    return m, m - h, m + h, p


def _strata(patients: pd.Series, fold_of: pd.Series):
    folds = sorted(fold_of.unique())
    return folds, [np.nonzero(fold_of.reindex(patients).values == f)[0] for f in folds]


# --------------------------------------------------------------------------------------------- 1. paired
def paired_comparison(df: pd.DataFrame, score_a: str, score_b: str, *, patient: str = "patient_id", fold: str = "fold",
                      label: str = "label", vertebra: str | None = "vertebra", per_level: bool = False,
                      n_reps: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Paired comparison A - B of AUPRC and ROC-AUC; see module docstring."""
    df = df.copy()
    if df.groupby(patient)[fold].nunique().max() > 1:
        raise ValueError("Each patient must belong to exactly one test fold.")
    fold_of = df.groupby(patient)[fold].first()
    pids = pd.Series(sorted(fold_of.index))
    pidx = {p: i for i, p in enumerate(pids)}
    folds, strata = _strata(pids, fold_of)

    pat = df.groupby(patient).agg(**{label: (label, "max"), score_a: (score_a, "max"), score_b: (score_b, "max"), fold: (fold, "first")}).reset_index()
    scopes = {"Vertebra (all levels)": df, "Patient": pat}
    if per_level and vertebra:
        for lv in sorted(df[vertebra].unique()):
            scopes[str(lv)] = df[df[vertebra] == lv]

    groups = []                                                     # (scope, model, fold, patient index, ys, ends)
    for name, d in scopes.items():
        for m in (score_a, score_b):
            for f in folds:
                g = d[d[fold] == f]
                order, ys, ends = _prep(g[label].values, g[m].values.astype(float))
                groups.append((name, m, f, g[patient].map(pidx).values[order], ys, ends))

    def evaluate(w):
        acc = {}
        for name, m, f, pi, ys, ends in groups:
            r = _ap_auc(w[pi], ys, ends)
            if r is None:
                return None
            acc.setdefault((name, m), []).append(r)
        return {k: np.array(v) for k, v in acc.items()}               # (n_folds, 2) per scope and model

    point = evaluate(np.ones(len(pids)))
    rng = np.random.default_rng(seed)
    boot, redraws = [], 0
    while len(boot) < n_reps:
        w = np.zeros(len(pids))
        for s in strata:
            np.add.at(w, rng.choice(s, size=len(s), replace=True), 1)
        e = evaluate(w)
        if e is None:
            redraws += 1
            continue
        boot.append({k: v.mean(0) for k, v in e.items()})

    rows = []
    for name in scopes:
        for j, metric in ((0, "AUPRC"), (1, "ROC-AUC")):
            fa, fb = 100 * point[(name, score_a)][:, j], 100 * point[(name, score_b)][:, j]
            d, lo, hi, p = _t_ci(fa - fb)
            bd = 100 * np.array([b[(name, score_a)][j] - b[(name, score_b)][j] for b in boot])
            rows.append({"scope": name, "metric": metric, "mean_A": fa.mean(), "mean_B": fb.mean(), "delta": d,
                         "foldwise_ci95_low": lo, "foldwise_ci95_high": hi, "foldwise_p": p,
                         "bootstrap_ci95_low": np.percentile(bd, 2.5), "bootstrap_ci95_high": np.percentile(bd, 97.5),
                         "n_folds": len(folds), "bootstrap_redraws": redraws})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------- 2. subgroup
def subgroup_difference(df: pd.DataFrame, value: str, group: str, *, patient: str = "patient_id", fold: str = "fold",
                        level: str | None = "vertebra", view: str | None = None, min_n: int = 5,
                        n_reps: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Difference group1 - group0 of the mean of `value`; see module docstring."""
    df = df.copy()
    if df.groupby(patient)[fold].nunique().max() > 1:
        raise ValueError("Each patient must belong to exactly one test fold.")
    fold_of = df.groupby(patient)[fold].first()
    pids = pd.Series(sorted(fold_of.index))
    folds, strata = _strata(pids, fold_of)
    levels = sorted(df[level].unique()) if level else ["all"]
    views = sorted(df[view].unique()) if view else ["all"]
    L, V, F = len(levels), len(views), len(folds)

    pi = df[patient].map({p: i for i, p in enumerate(pids)}).values
    g = df[group].astype(int).values
    li = df[level].map({l: i for i, l in enumerate(levels)}).values if level else np.zeros(len(df), int)
    vi = df[view].map({v: i for i, v in enumerate(views)}).values if view else np.zeros(len(df), int)
    fi = df[fold].map({f: i for i, f in enumerate(folds)}).values
    y = df[value].values.astype(float)
    first_view = vi == 0
    n1 = pd.Series(li[(g == 1) & first_view]).value_counts()
    tested = [i for i in range(L) if n1.get(i, 0) >= min_n]

    def means(w, key, n):
        ww = w[pi]
        cnt = np.bincount(key, ww, n)
        return np.bincount(key, ww * y, n) / np.maximum(cnt, 1e-12), cnt

    def compute(w):
        m, c = means(w, (g * L + li) * V + vi, 2 * L * V)
        m, c = m.reshape(2, L, V), c.reshape(2, L, V)
        if (c[:, tested, :] == 0).any():
            return None
        lv = m.mean(2)
        out = {str(levels[k]): (lv[1, k], lv[0, k]) for k in tested}
        out["Mean of levels"] = (lv[1, tested].mean(), lv[0, tested].mean())
        ma, _ = means(w, g * V + vi, 2 * V)
        ma = ma.reshape(2, V).mean(1)
        out["All levels pooled"] = (ma[1], ma[0])
        mf, cf = means(w, (g * V + vi) * F + fi, 2 * V * F)
        if (cf == 0).any():
            return None
        mf = mf.reshape(2, V, F).mean(1)
        out["All levels, fold-wise"] = (mf[1].mean(), mf[0].mean())
        return out

    point = compute(np.ones(len(pids)))
    if point is None:
        raise ValueError("A required subgroup cell is empty on the original data.")
    rng = np.random.default_rng(seed)
    boot, redraws = {k: [] for k in point}, 0
    while len(boot["Mean of levels"]) < n_reps:
        w = np.zeros(len(pids))
        for s in strata:
            np.add.at(w, rng.choice(s, size=len(s), replace=True), 1)
        e = compute(w)
        if e is None:
            redraws += 1
            continue
        for k, v in e.items():
            boot[k].append(v)

    rows = []
    for k, (a, b) in point.items():
        bb = np.asarray(boot[k])
        d = bb[:, 0] - bb[:, 1]
        if k in point and k not in ("Mean of levels", "All levels pooled", "All levels, fold-wise"):
            sel = (df[level].astype(str) == k) & first_view if level else first_view
        else:
            sel = first_view & (np.isin(li, tested) if k == "Mean of levels" else True)
        rows.append({"scope": k, "n_group1": int(((g == 1) & sel).sum()), "n_group0": int(((g == 0) & sel).sum()),
                     "group1": a, "group1_ci95_low": np.percentile(bb[:, 0], 2.5), "group1_ci95_high": np.percentile(bb[:, 0], 97.5),
                     "group0": b, "group0_ci95_low": np.percentile(bb[:, 1], 2.5), "group0_ci95_high": np.percentile(bb[:, 1], 97.5),
                     "delta": a - b, "delta_ci95_low": np.percentile(d, 2.5), "delta_ci95_high": np.percentile(d, 97.5),
                     "bootstrap_redraws": redraws})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------- CLI
def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("Usage")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("paired", "subgroup"):
        p = sub.add_parser(name)
        p.add_argument("csv")
        p.add_argument("--patient", default="patient_id")
        p.add_argument("--fold", default="fold")
        p.add_argument("--n-reps", type=int, default=2000)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--out", help="write the result table to this CSV")
    sp = sub.choices["paired"]
    sp.add_argument("--score-a", required=True); sp.add_argument("--score-b", required=True)
    sp.add_argument("--label", default="label"); sp.add_argument("--vertebra", default="vertebra")
    sp.add_argument("--per-level", action="store_true")
    sg = sub.choices["subgroup"]
    sg.add_argument("--value", required=True); sg.add_argument("--group", required=True)
    sg.add_argument("--level", default="vertebra"); sg.add_argument("--view", default=None)
    sg.add_argument("--min-n", type=int, default=5)
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    if a.cmd == "paired":
        res = paired_comparison(df, a.score_a, a.score_b, patient=a.patient, fold=a.fold, label=a.label,
                                vertebra=a.vertebra, per_level=a.per_level, n_reps=a.n_reps, seed=a.seed)
    else:
        res = subgroup_difference(df, a.value, a.group, patient=a.patient, fold=a.fold, level=a.level, view=a.view,
                                  min_n=a.min_n, n_reps=a.n_reps, seed=a.seed)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)
    print(res.round(3).to_string(index=False))
    if a.out:
        res.to_csv(a.out, index=False)


if __name__ == "__main__":
    _main()
