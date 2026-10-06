"""Figures 2, 3 and 4 from the CSV files in results/ (PNG 300 dpi and PDF)."""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, NullFormatter
from matplotlib.lines import Line2D
from plotstyle import STYLE, INK2, panel_labels, save
from stats import RESULTS_DIR

EXP1_CSV = os.path.join(RESULTS_DIR, "prompt_sweep.csv")
EXP2_CSV = os.path.join(RESULTS_DIR, "sensitivity.csv")
LAT_CSV = os.path.join(RESULTS_DIR, "routing_latencies.csv")
SHARE_CSV = os.path.join(RESULTS_DIR, "routing_numbers.csv")


def plain_log(ax):
    ax.yaxis.set_major_formatter(ScalarFormatter()); ax.yaxis.set_minor_formatter(NullFormatter())


def line(ax, x, y, m, err=None):
    c, mk, ls = STYLE[m]
    ax.plot(x, y, color=c, marker=mk, linestyle=ls, label=m, markeredgecolor="white", markeredgewidth=0.8)
    if err is not None:
        ax.fill_between(x, y - err, y + err, color=c, alpha=0.12, linewidth=0)


def top_legend(fig, names, y=1.0):
    hs = [Line2D([], [], color=STYLE[m][0], marker=STYLE[m][1], linestyle=STYLE[m][2], markeredgecolor="white")
          for m in names]
    fig.legend(hs, names, loc="lower center", ncol=len(names), fontsize=7, bbox_to_anchor=(0.5, y))


# ---------------------------------------------------------------- Figure 2: latency distribution and routing
if os.path.exists(LAT_CSV) and os.path.exists(SHARE_CSV):
    lat = pd.read_csv(LAT_CSV); sh = pd.read_csv(SHARE_CSV).set_index("method")
    methods = list(dict.fromkeys(lat.method))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for m in methods:
        x = np.sort(lat[lat.method == m].latency_s.values); y = np.arange(1, len(x) + 1) / len(x)
        a1.plot(x, y, color=STYLE[m][0], linestyle=STYLE[m][2], label=m)
    a1.set_xscale("log"); a1.xaxis.set_major_formatter(ScalarFormatter()); a1.xaxis.set_minor_formatter(NullFormatter())
    a1.set_xticks([0.5, 1, 2, 5, 10, 20, 50]); a1.set_xlabel("End-to-end latency (s, log)"); a1.set_ylabel("CDF")
    a1.legend(fontsize=7, loc="lower right")
    types = [c.split("_")[1] for c in sh.columns if c.startswith("share_")]
    groups = ["prefill capacity"] + methods; x = np.arange(len(groups)); w = 0.26
    tcol = {"3080": "#9db7d8", "3090": "#4f7fb8", "4090": "#1f3f6e"}
    for i, k in enumerate(types):
        vals = [sh.loc[g, f"share_{k}_%"] for g in groups]
        a2.bar(x + (i - 1) * w, vals, w, color=tcol.get(k), label=f"RTX {k}")
    top = max(sh.loc[g, f"share_{k}_%"] for g in groups for k in types)
    a2.set_ylim(0, top * 1.25)
    a2.set_xticks(x); a2.set_xticklabels(["capacity"] + [m.replace(" (ours)", "") for m in methods], rotation=25, fontsize=7)
    a2.set_ylabel("Share of prompt tokens (%)"); a2.legend(fontsize=7, loc="upper center", ncol=3)
    fig.tight_layout()
    panel_labels(fig, [a1, a2])
    save(fig, os.path.join(RESULTS_DIR, "fig2_routing"))

# ---------------------------------------------------------------- Figure 3: prompt length
if os.path.exists(EXP1_CSV):
    d = pd.read_csv(EXP1_CSV)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.6))
    methods = ["PGS (ours)", "DRL-Q", "SAS", "LL", "JSQ-speed", "RR"]
    for m in methods:
        s = d[d.method == m].sort_values("prompt_len")
        if len(s):
            line(a1, s.prompt_len, s.mean_latency_s, m, s.ci95)
    a1.set_yscale("log"); a1.set_yticks([1, 2, 3, 5, 10, 20, 50]); plain_log(a1)
    a1.set_xlabel("Mean prompt length (tokens)"); a1.set_ylabel("Mean latency (s, log)")

    best_h, drl = [], []; L = sorted(d.prompt_len.unique())
    for p in L:
        s = d[d.prompt_len == p]; ours = s[s.method == "PGS (ours)"].mean_latency_s.iloc[0]
        h = s[s.method.isin(["RR", "LL", "SAS", "JSQ-speed", "Random"])].mean_latency_s.min()
        best_h.append(100 * (h - ours) / h)
        q = s[s.method == "DRL-Q"].mean_latency_s
        drl.append(100 * (q.iloc[0] - ours) / q.iloc[0] if len(q) else np.nan)
    x = np.arange(len(L)); w = 0.38
    a2.bar(x - w / 2 - 0.01, best_h, w, color=STYLE["SAS"][0], label="vs. best heuristic")
    a2.bar(x + w / 2 + 0.01, drl, w, color=STYLE["DRL-Q"][0], label="vs. DRL-Q")
    for xi, v in zip(x - w / 2, best_h):
        a2.text(xi, v + 0.8, f"{v:.0f}%", ha="center", fontsize=7, color=INK2)
    a2.set_xticks(x); a2.set_xticklabels(L); a2.set_xlabel("Mean prompt length (tokens)")
    a2.set_ylabel("Latency reduction by PGS (%)"); a2.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    top_legend(fig, methods)
    panel_labels(fig, [a1, a2])
    save(fig, os.path.join(RESULTS_DIR, "fig3_prompt_sweep"))

# ---------------------------------------------------------------- Figure 4: sensitivity
if os.path.exists(EXP2_CSV):
    d = pd.read_csv(EXP2_CSV)
    fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.3))
    s = d[d.sweep == "sigma"]; order = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0", "none"]
    for m in ["PGS (ours)", "SAS", "LL"]:
        t = s[s.method == m].set_index("value").reindex(order)
        line(ax[0], np.arange(len(order)), t.mean_latency_s.values, m)
    ax[0].set_xticks(range(len(order))); ax[0].set_xticklabels(["0", ".2", ".4", ".6", ".8", "1.0", "none"], rotation=35)
    ax[0].set_xlabel("Prediction error σ"); ax[0].set_ylabel("Mean latency (s)")

    s = d[d.sweep == "servers"]
    for m in ["PGS (ours)", "SAS", "LL", "RR"]:
        t = s[s.method == m]; line(ax[1], t.value.astype(int), t.mean_latency_s, m)
    ax[1].set_xlabel("Edge servers")

    s = d[d.sweep == "hardware"]; mixes = list(dict.fromkeys(s.value))
    ms = ["PGS (ours)", "SAS", "LL", "JSQ-speed", "RR"]; w = 0.16
    for i, m in enumerate(ms):
        t = s[s.method == m].set_index("value").reindex(mixes)
        ax[2].bar(np.arange(len(mixes)) + (i - (len(ms) - 1) / 2) * (w + 0.01), t.mean_latency_s, w,
                  color=STYLE[m][0], label=m)
    ax[2].set_xticks(range(len(mixes))); ax[2].set_xticklabels([x.split(" ")[0] for x in mixes])
    ax[2].set_xlabel("Hardware mix")

    s = d[d.sweep == "users"]
    for m in ["PGS (ours)", "SAS", "LL", "JSQ-speed"]:
        t = s[s.method == m]; line(ax[3], t.value.astype(int), t.mean_latency_s, m)
    ax[3].set_yscale("log"); ax[3].set_yticks([2, 3, 5, 10]); plain_log(ax[3])
    ax[3].set_xlabel("Requests per second")
    fig.tight_layout()
    top_legend(fig, ["PGS (ours)", "SAS", "LL", "JSQ-speed", "RR"])
    panel_labels(fig, list(ax))
    save(fig, os.path.join(RESULTS_DIR, "fig4_sensitivity"))

print("figures written")
