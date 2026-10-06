"""Tables 4 and 5, scenario S5: memory-constrained deployment of an 8B-class model."""

import time
import numpy as np
import pandas as pd
import config as C
import simulator as SIM
import policies as P
import drl
from stats import summarize, results_path

OUT = results_path("s5_results.csv")

if __name__ == "__main__":
    env = SIM.EdgeSimulator(*C.S5_SCENARIO, hw=C.HW_8B)
    env.reset(0)
    print("KV capacity per server (tokens):", env.M.astype(int))
    t0 = time.time()
    res = {name: np.array([SIM.run(env, fn, sd) for sd in C.EVAL_SEEDS]) for name, fn in P.POLICIES.items()}
    q = drl.train(env)
    res["DRL-Q"] = np.array([SIM.run(env, drl.policy(q), sd) for sd in C.EVAL_SEEDS])
    df = pd.DataFrame([{"scenario": "S5-memory", **r} for r in summarize(res)])
    df.to_csv(OUT, index=False)
    print(f"\n=== S5 ({time.time() - t0:.0f} s) ===")
    print(df.drop(columns="scenario").round(4).to_string(index=False))
    print(f"\nSaved to {OUT}")
