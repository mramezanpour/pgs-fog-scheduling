"""Summary statistics shared by the experiment scripts."""

import os
import numpy as np
from scipy import stats

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def results_path(name):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return os.path.join(RESULTS_DIR, name)


def ci95(x):
    x = np.asarray(x, float)
    return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


def wilcoxon_p(a, b):
    """Two-sided Wilcoxon signed-rank test on paired seeds (1.0 if all differences are zero)."""
    try:
        return stats.wilcoxon(a, b).pvalue
    except ValueError:
        return 1.0


def improvement(baseline_mean, ours_mean):
    return 100 * (baseline_mean - ours_mean) / baseline_mean


def summarize(res, ours="PGS (ours)", no_test=("PGS (ours)", "PGS-oracle")):
    """res: {method: array of (mean, p99, unfinished) per seed} -> list of row dicts."""
    o = res[ours][:, 0]; rows = []
    for name, r in res.items():
        m = r[:, 0]
        rows.append({"method": name, "mean_latency_s": m.mean(), "ci95": ci95(m), "p99_s": r[:, 1].mean(),
                     "unfinished_requests": r[:, 2].mean(),
                     "PGS_improvement_%": improvement(m.mean(), o.mean()),
                     "wilcoxon_p": np.nan if name in no_test else wilcoxon_p(o, m)})
    return rows
