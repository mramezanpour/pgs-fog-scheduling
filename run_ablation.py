"""Table 6: ablation of PGS. Each component is switched off while the rest is kept.

"w/o C1" removes the prompts already committed to the server, the carry-over backlog and the
prefill interference charge, but keeps the request's own prefill time l_in / v_pre.
"""

import time
import numpy as np
import pandas as pd
import config as C
import simulator as SIM
import policies as P
from stats import wilcoxon_p, results_path

WORKLOADS = {   # name: (requests per slot, mean prompt, mean output, bursty, hardware)
    "Short prompt (70)":       (40, 70, 215, False, C.HW_1B),
    "Medium prompt (500)":     (50, 500, 215, False, C.HW_1B),
    "Long prompt (1500)":      (20, 1500, 215, False, C.HW_1B),
    "Memory-constrained (S5)": (*C.S5_SCENARIO, C.HW_8B),
}
VARIANTS = {
    "PGS (full)":                                   {},
    "w/o C1: batch-coupled prefill & interference": {"c1": False},
    "w/o carry-over backlog (part of C1)":          {"backlog": False},
    "w/o C2: virtual state rollout":                {"rollout": False},
    "w/o C2: footprint ordering":                   {"order": False},
    "w/o C1 and C2":                                {"c1": False, "rollout": False, "order": False},
    "w/o KV-growth externality":                    {"kv_ext": False},
    "w/o memory-commitment wait":                   {"mem_wait": False},
}
OUT = results_path("ablation.csv")


def cell(r):
    if r.variant == "PGS (full)":
        return f"{r.mean_latency_s:.2f}"
    mark = "" if r["significant_5%"] else "†"
    return f"{r.mean_latency_s:.2f} ({r['change_vs_full_%']:+.1f}%{mark})"


if __name__ == "__main__":
    rows = []
    for wname, (u, mi, mo, b, hw) in WORKLOADS.items():
        env = SIM.EdgeSimulator(u, mi, mo, b, hw=hw)
        t0 = time.time()
        res = {}
        for vname, flags in VARIANTS.items():
            fn = lambda e, fl=flags: P.pgs(e, True, **fl)
            res[vname] = np.array([SIM.run(env, fn, sd) for sd in C.EVAL_SEEDS])
        full = res["PGS (full)"][:, 0]
        for vname, r in res.items():
            m = r[:, 0]
            p = np.nan if vname == "PGS (full)" else wilcoxon_p(full, m)
            rows.append({"workload": wname, "variant": vname, "mean_latency_s": m.mean(),
                         "change_vs_full_%": 100 * (m.mean() - full.mean()) / full.mean(), "wilcoxon_p": p,
                         "significant_5%": (p < 0.05) if not np.isnan(p) else np.nan,
                         "unfinished_requests": r[:, 2].mean()})
        df = pd.DataFrame(rows); df.to_csv(OUT, index=False)
        print(f"\n=== {wname} ({time.time() - t0:.0f} s) ===")
        print(df[df.workload == wname].drop(columns="workload").round(4).to_string(index=False), flush=True)
    df["cell"] = df.apply(cell, axis=1)
    table = df.pivot(index="variant", columns="workload", values="cell").reindex(list(VARIANTS))[list(WORKLOADS)]
    table.to_csv(results_path("ablation_table6.csv"))
    print("\n" + table.to_string())
    print(f"\nSaved to {OUT}")
