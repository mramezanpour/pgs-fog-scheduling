"""Tables 4 and 5, scenarios S1-S4: mean and P99 latency of all methods over 10 paired seeds."""

import time
import numpy as np
import pandas as pd
import config as C
import simulator as SIM
import policies as P
import drl
from stats import summarize, results_path

OUT = results_path("main_results.csv")

if __name__ == "__main__":
    rows = []
    for sc, wl in C.SCENARIOS.items():
        env = SIM.EdgeSimulator(*wl)
        t0 = time.time()
        res = {name: np.array([SIM.run(env, fn, sd) for sd in C.EVAL_SEEDS]) for name, fn in P.POLICIES.items()}
        q = drl.train(env)
        res["DRL-Q"] = np.array([SIM.run(env, drl.policy(q), sd) for sd in C.EVAL_SEEDS])
        part = [{"scenario": sc, **r} for r in summarize(res)]
        rows += part
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print(f"\n=== {sc} ({time.time() - t0:.0f} s) ===")
        print(pd.DataFrame(part).drop(columns="scenario").round(4).to_string(index=False), flush=True)
    print(f"\nSaved to {OUT}")
