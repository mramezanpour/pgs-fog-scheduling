"""Dispatching policies. Each policy maps the simulator state to one server index per request.

All policies that use output lengths see only the predicted lengths (env.view), except
PGS-oracle, which is given the true lengths as a reference.
"""

import numpy as np
import config as C


def random_policy(env):
    return list(env.rng.integers(0, env.S, len(env.reqs)))


def round_robin(env):
    out = [(env.rr + i) % env.S for i in range(len(env.reqs))]
    env.rr = (env.rr + len(env.reqs)) % env.S
    return out


def least_loaded(env):
    """Smallest committed plus queued footprint relative to the KV capacity."""
    load = [(env.committed_pred(s) + env.queued_pred(s)) / env.M[s] for s in range(env.S)]
    out = []
    for r in env.view:
        j = int(np.argmin(load)); out.append(j)
        load[j] += (r[1] + r[2]) / env.M[j]
    return out


def jsq_speed(env):
    """Shortest predicted decoding-iteration time."""
    kvs = [env.kv(s) + env.queued_pred(s) for s in range(env.S)]
    out = []
    for r in env.view:
        j = int(np.argmin([env.tau(k, kvs[k] + r[1]) for k in range(env.S)])); out.append(j)
        kvs[j] += r[1] + r[2] / 2
    return out


def sas(env):
    """Static-aware scheduling (Mou et al.): shortest requests first, each to the least utilized server."""
    util = [len(env.active[s]) + len(env.wait[s]) for s in range(env.S)]
    cap = [env.M[s] / 1e5 for s in range(env.S)]
    out = [0] * len(env.view)
    for k in sorted(range(len(env.view)), key=lambda k: env.view[k][1] + env.view[k][2]):
        j = int(np.argmin([util[s] / cap[s] for s in range(env.S)])); out[k] = j
        util[j] += 1
    return out


def pgs(env, use_pred=True, c1=True, rollout=True, order=True, kv_ext=True, mem_wait=True, backlog=True):
    """Prediction-guided scheduler (PGS): marginal-latency pricing on a virtual server state.

    c1        batch-coupled prefill and interference: prompts already committed to the server,
              the carry-over backlog and the interference charged to co-located requests
              (the request's own prefill time is always priced)
    rollout   update the virtual state after every commitment (C2)
    order     commit in non-increasing predicted footprint (C2)
    kv_ext    KV-growth externality
    mem_wait  memory-commitment wait
    backlog   carry-over backlog max(0, clock_s - t), part of c1
    """
    S = env.S
    reqs = env.view if use_pred else env.reqs
    rem_of = (lambda a: env.rem_pred(a)) if use_pred else (lambda a: a[1])
    com_of = (lambda s: env.committed_pred(s) + env.queued_pred(s)) if use_pred else \
             (lambda s: env.committed(s) + env.queued(s))
    W = C.W_PREFILL
    st = [{"kv": env.kv(s), "com": com_of(s), "rem": [rem_of(a) for a in env.active[s]],
           "pre": sum(r[1] for r in env.wait[s])} for s in range(S)]
    idx = range(len(reqs))
    if order:
        idx = sorted(idx, key=lambda k: -(reqs[k][1] + reqs[k][2]))
    out = [0] * len(reqs)
    for k in idx:
        r = reqs[k]; need = r[1] + r[2]; best = None
        for j in range(S):
            d = st[j]; growth = r[2] / 2
            tau_j = env.tau(j, d["kv"] + r[1] + growth)
            wait = 0.0
            if mem_wait:
                excess = d["com"] + need - env.M[j]
                if excess > 0:
                    mean_rem = (np.mean(d["rem"]) if d["rem"] else r[2]) * tau_j
                    wait = excess / (max(d["com"], 1.0) / max(mean_rem, 1e-3))
            bl = max(0.0, env.clock[j] - env.t) if (c1 and backlog) else 0.0
            pre_q = d["pre"] if c1 else 0.0
            own = wait + bl + (pre_q + r[1]) / env.vpre[j] + r[2] * tau_j
            ext = 0.0
            if kv_ext:
                dtau = env.tau(j, d["kv"] + r[1] + growth) - env.tau(j, d["kv"])
                ext += dtau * sum(min(x, r[2]) for x in d["rem"])
            if c1:
                ext += len(d["rem"]) * r[1] / env.vpre[j] * W
            c = own + ext
            if best is None or c < best[0]:
                best = (c, j)
        j = best[1]; out[k] = j
        if rollout:
            d = st[j]
            d["kv"] += r[1] + r[2] / 2; d["com"] += need; d["rem"].append(r[2]); d["pre"] += r[1]
    return out


POLICIES = {
    "Random": random_policy,
    "RR": round_robin,
    "LL": least_loaded,
    "SAS": sas,
    "JSQ-speed": jsq_speed,
    "PGS (ours)": lambda e: pgs(e, True),
    "PGS-oracle": lambda e: pgs(e, False),
}
