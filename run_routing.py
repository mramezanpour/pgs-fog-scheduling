"""Figure 2 data: long-prompt workload S4 (mean prompt 1500 tokens, 20 requests/s), three seeds.
Saves every per-request latency and the share of prompt tokens routed to each GPU type;
make_figures.py draws the figure from these files.
"""

import numpy as np
import pandas as pd
import config as C
import simulator as SIM
import policies as P
import drl
from stats import results_path

WORKLOAD = C.SCENARIOS["S4-long-prompt"]
SEEDS = [1000, 1001, 1002]
METHODS = ["PGS (ours)", "DRL-Q", "SAS", "LL", "JSQ-speed"]

if __name__ == "__main__":
    env = SIM.EdgeSimulator(*WORKLOAD)
    fns = {m: P.POLICIES[m] for m in METHODS if m != "DRL-Q"}
    fns["DRL-Q"] = drl.policy(drl.train(env))
    types = list(dict.fromkeys(C.SERVERS))
    vpre = np.array([C.HW_1B["gpu"][k][1] for k in C.SERVERS], float)
    is_type = {k: np.array([s == k for s in C.SERVERS]) for k in types}

    lat_rows, rows = [], []
    for m in METHODS:
        ls, tok = [], np.zeros(len(C.SERVERS))
        for sd in SEEDS:
            l, t = SIM.run_record(env, fns[m], sd); ls.append(l); tok += t
        lat = np.concatenate(ls)
        lat_rows.append(pd.DataFrame({"method": m, "latency_s": lat}))
        rows.append({"method": m, "mean_s": lat.mean(), "median_s": np.median(lat),
                     "p90_s": np.percentile(lat, 90), "p99_s": np.percentile(lat, 99),
                     **{f"share_{k}_%": 100 * tok[is_type[k]].sum() / tok.sum() for k in types}})
    rows.append({"method": "prefill capacity", **{f"share_{k}_%": 100 * vpre[is_type[k]].sum() / vpre.sum()
                                                   for k in types}})
    pd.concat(lat_rows).to_csv(results_path("routing_latencies.csv"), index=False)
    df = pd.DataFrame(rows); df.to_csv(results_path("routing_numbers.csv"), index=False)
    print(df.round(2).to_string(index=False))
    print("Saved routing_latencies.csv and routing_numbers.csv; run make_figures.py to draw Figure 2.")
