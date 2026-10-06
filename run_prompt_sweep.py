"""Figure 3: effect of the mean prompt length at 70% fleet-wide prefill utilization."""

import time
import numpy as np
import pandas as pd
import config as C
import simulator as SIM
import policies as P
import drl
from stats import ci95, wilcoxon_p, improvement, results_path

PROMPT_LENGTHS = [100, 250, 500, 1000, 1500, 2000]
PREFILL_UTIL = 0.7
MAX_RATE = 60
SEEDS = [2000 + i for i in range(10)]
OUT = results_path("prompt_sweep.csv")

if __name__ == "__main__":
    cap = sum(C.HW_1B["gpu"][k][1] for k in C.SERVERS)
    rows = []
    for L in PROMPT_LENGTHS:
        rate = int(min(MAX_RATE, round(PREFILL_UTIL * cap / L)))
        env = SIM.EdgeSimulator(rate, L, 215, False)
        t0 = time.time()
        res = {n: np.array([SIM.run(env, fn, sd) for sd in SEEDS]) for n, fn in P.POLICIES.items() if n != "PGS-oracle"}
        q = drl.train(env)
        res["DRL-Q"] = np.array([SIM.run(env, drl.policy(q), sd) for sd in SEEDS])
        ours = res["PGS (ours)"][:, 0]
        for n, r in res.items():
            m = r[:, 0]
            rows.append({"prompt_len": L, "users": rate, "method": n, "mean_latency_s": m.mean(), "ci95": ci95(m),
                         "p99_s": r[:, 1].mean(), "PGS_improvement_%": improvement(m.mean(), ours.mean()),
                         "wilcoxon_p": np.nan if n == "PGS (ours)" else wilcoxon_p(ours, m)})
        df = pd.DataFrame(rows); df.to_csv(OUT, index=False)
        d = df[df.prompt_len == L]
        print(f"prompt {L} (rate {rate}, {time.time() - t0:.0f} s): " +
              " | ".join(f"{a} {b:.2f}s" for a, b in zip(d.method, d.mean_latency_s)), flush=True)
    print(f"\nSaved to {OUT}")
