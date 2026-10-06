"""Table 7: wall-clock scheduling time per slot (mean and 99th percentile over slots with arrivals)
at 20, 40 and 90 requests per slot with a mean prompt of 500 tokens, on one CPU core.
Also reports the training time of DRL-Q. Run on an otherwise idle machine.
"""

import time
import numpy as np
import pandas as pd
import torch
import simulator as SIM
import policies as P
import drl
from stats import results_path

RATES = [20, 40, 90]
MEAN_PROMPT = 500
SEED = 1000
METHODS = ["RR", "LL", "SAS", "JSQ-speed", "PGS (ours)"]
OUT = results_path("overhead.csv")

torch.set_num_threads(1)


def time_policy(env, fn, seed):
    env.reset(seed)
    done = False; ts = []
    while not done:
        if env.reqs:
            t = time.perf_counter(); a = fn(env); ts.append(time.perf_counter() - t)
        else:
            a = fn(env)
        done = env.step(a)
    ts = np.array(ts) * 1e3
    return ts.mean(), np.percentile(ts, 99)


if __name__ == "__main__":
    env = SIM.EdgeSimulator(RATES[1], MEAN_PROMPT, 215, False)
    t = time.perf_counter(); q = drl.train(env); train_s = time.perf_counter() - t
    print(f"DRL-Q training time: {train_s:.0f} s")
    rows = []
    for u in RATES:
        env = SIM.EdgeSimulator(u, MEAN_PROMPT, 215, False)
        fns = {m: P.POLICIES[m] for m in METHODS}
        fns["DRL-Q (inference)"] = drl.policy(q)
        for m, fn in fns.items():
            mean_ms, p99_ms = time_policy(env, fn, SEED)
            rows.append({"requests_per_slot": u, "method": m, "mean_ms": mean_ms, "p99_ms": p99_ms})
            print(f"{u:3d} req/slot  {m:18s} {mean_ms:7.2f} ms  (p99 {p99_ms:.2f})", flush=True)
    rows.append({"requests_per_slot": RATES[1], "method": "DRL-Q training (total, s)",
                 "mean_ms": train_s * 1e3, "p99_ms": np.nan})
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nSaved to {OUT}")
