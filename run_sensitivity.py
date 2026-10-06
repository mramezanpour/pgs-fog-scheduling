"""Figure 4: sensitivity to (a) prediction error, (b) fleet size, (c) hardware mix and (d) load.

Base workload: mean prompt 1000 tokens, 20 requests/s, moderate hardware mix, 5 seeds.
"""

import numpy as np
import pandas as pd
import config as C
import simulator as SIM
import policies as P
from stats import improvement, results_path

BASE_PROMPT = 1000
BASE_RATE = 20
SEEDS = [3000 + i for i in range(5)]
SIGMAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, "none"]
SERVER_COUNTS = [4, 8, 12, 16]
RATES = [10, 15, 20, 25, 30]
METHODS = ["RR", "LL", "SAS", "JSQ-speed", "PGS (ours)"]
OUT = results_path("sensitivity.csv")


def evaluate(rate, methods, servers, sigma=C.SIGMA):
    env = SIM.EdgeSimulator(rate, BASE_PROMPT, 215, False, servers=servers, sigma=sigma)
    return {n: np.array([SIM.run(env, P.POLICIES[n], sd) for sd in SEEDS]) for n in methods}


def record(rows, sweep, value, res):
    ours = res["PGS (ours)"][:, 0].mean()
    for n, r in res.items():
        rows.append({"sweep": sweep, "value": str(value), "method": n, "mean_latency_s": r[:, 0].mean(),
                     "p99_s": r[:, 1].mean(), "PGS_improvement_%": improvement(r[:, 0].mean(), ours)})
    print(f"[{sweep}={value}] " + " | ".join(f"{n} {r[:, 0].mean():.2f}s" for n, r in res.items()), flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)


if __name__ == "__main__":
    base = C.HW_MIXES["Mod (3x3080,3x3090,2x4090)"]
    rows = []
    for sg in SIGMAS:
        record(rows, "sigma", sg, evaluate(BASE_RATE, ["LL", "SAS", "PGS (ours)"], base, sigma=sg))
    for n in SERVER_COUNTS:
        mix = [base[i % len(base)] for i in range(n)]
        record(rows, "servers", n, evaluate(int(round(BASE_RATE * n / 8)), METHODS, mix))
    for name, mix in C.HW_MIXES.items():
        record(rows, "hardware", name, evaluate(BASE_RATE, METHODS, mix))
    for u in RATES:
        record(rows, "users", u, evaluate(u, METHODS, base))
    print(f"\nSaved to {OUT}")
