"""Slotted simulator of iteration-level LLM serving on heterogeneous edge servers.

Each server runs FIFO KV admission on the full footprint (prompt + output), a batched prefill
whose unfinished work carries over to the next slot, and continuous batching with early exit
and late join. Decoding iterations that overlap a prefill batch are slowed by W_PREFILL.
The iteration time follows the memory-bound roofline
    tau_s(m) = OVERHEAD + weights / B_s + m * kv_bytes / B_s.
The scheduler sees each request's prompt length and a noisy prediction of its output length.
"""

import numpy as np
import config as C


def lognormal(rng, mean, n, sigma=C.LENGTH_SHAPE):
    mu = np.log(mean) - sigma ** 2 / 2
    return np.maximum(1, rng.lognormal(mu, sigma, n).astype(int))


class EdgeSimulator:
    def __init__(self, rate, mean_in, mean_out, bursty, servers=None, hw=None, sigma=C.SIGMA,
                 t_slots=C.T_SLOTS):
        self.rate, self.mean_in, self.mean_out, self.bursty = rate, mean_in, mean_out, bursty
        self.servers = list(servers or C.SERVERS)
        self.hw = hw or C.HW_1B
        self.sigma = sigma                       # float, or "none" for no predictor (population mean)
        self.t_slots = t_slots
        self.S = len(self.servers)
        gpu = self.hw["gpu"]
        self.bw = np.array([gpu[k][2] * 1e9 for k in self.servers])

    # ------------------------------------------------------------------ episode control
    def reset(self, seed):
        gpu, kvb, wb = self.hw["gpu"], self.hw["kv_bytes"], self.hw["weight_bytes"]
        self.pred = {}; self.lin = {}
        self.prng = np.random.default_rng(seed + 7777)
        self.rng = np.random.default_rng(seed)
        self.M = np.array([gpu[k][0] * 1e9 / kvb for k in self.servers])        # KV capacity (tokens)
        self.vpre = np.array([gpu[k][1] for k in self.servers], float)
        self.t0 = np.array([C.OVERHEAD_S + wb / (gpu[k][2] * 1e9) for k in self.servers])
        self.active = [[] for _ in range(self.S)]     # [kv tokens, remaining output, request id]
        self.wait = [[] for _ in range(self.S)]       # admission queues
        self.clock = np.zeros(self.S)                 # time each server is busy until
        self.t = 0; self.arrive = {}; self.finish = {}; self.rid = 0; self.high = False
        self.rr = 0
        self._new()

    def _new(self):
        if self.t >= self.t_slots:
            self.reqs = []
        else:
            n = self.rate
            if self.bursty:
                if self.rng.random() < C.BURST_SWITCH:
                    self.high = not self.high
                n = self.rng.poisson(self.rate * (C.BURST_HIGH if self.high else C.BURST_LOW))
            lin = lognormal(self.rng, self.mean_in, n)
            lout = lognormal(self.rng, self.mean_out, n)
            self.reqs = [(self.rid + i, int(lin[i]), int(lout[i])) for i in range(n)]
            for r in self.reqs:
                self.arrive[r[0]] = self.t
            self.rid += n
        for r in self.reqs:
            self.lin[r[0]] = r[1]
            if self.sigma == "none":
                self.pred[r[0]] = self.mean_out
            else:
                s = self.sigma
                self.pred[r[0]] = max(1, int(r[2] * np.exp(self.prng.normal(0, s) - s * s / 2)))
        self.view = [(r[0], r[1], self.pred[r[0]]) for r in self.reqs]   # what the scheduler sees

    def step(self, assign):
        for r, s in zip(self.reqs, assign):
            self.wait[s].append(r)
        t = self.t
        for s in range(self.S):
            adm = []; com = self.committed(s)
            while self.wait[s] and com + self.wait[s][0][1] + self.wait[s][0][2] <= self.M[s]:
                r = self.wait[s].pop(0); adm.append(r); com += r[1] + r[2]
            pre = sum(r[1] for r in adm) / self.vpre[s]
            tnow = max(float(t), self.clock[s]); pstart = tnow

            def iterate(slow):
                nonlocal tnow
                tnow += self.tau(s, self.kv(s)) / slow
                for a in self.active[s]:
                    a[0] += 1; a[1] -= 1
                for a in self.active[s]:
                    if a[1] <= 0:
                        self.finish[a[2]] = tnow
                self.active[s] = [a for a in self.active[s] if a[1] > 0]

            while self.active[s] and tnow < pstart + pre:
                iterate(1 - C.W_PREFILL)
            tnow = max(tnow, pstart + pre)
            for r in adm:
                self.active[s].append([r[1], r[2], r[0]])
            while self.active[s] and tnow < t + 1:
                iterate(1.0)
            self.clock[s] = tnow
        self.t += 1
        self._new()
        idle = not any(self.active[s] or self.wait[s] for s in range(self.S))
        return self.t >= self.t_slots + C.DRAIN_SLOTS or (self.t >= self.t_slots and idle)

    # ------------------------------------------------------------------ server state
    def tau(self, s, m):
        return C.OVERHEAD_S + self.hw["weight_bytes"] / self.bw[s] + m * self.hw["kv_bytes"] / self.bw[s]

    def kv(self, s):
        return sum(a[0] for a in self.active[s])

    def committed(self, s):
        return sum(a[0] + a[1] for a in self.active[s])

    def queued(self, s):
        return sum(r[1] + r[2] for r in self.wait[s])

    # state as seen through the output-length predictions
    def rem_pred(self, a):
        gen = a[0] - self.lin[a[2]]
        return max(1, self.pred[a[2]] - gen)

    def committed_pred(self, s):
        return sum(self.lin[a[2]] + max(self.pred[a[2]], a[0] - self.lin[a[2]] + 1) for a in self.active[s])

    def queued_pred(self, s):
        return sum(r[1] + self.pred[r[0]] for r in self.wait[s])

    def server_features(self):
        f = np.zeros((self.S, 7))
        for s in range(self.S):
            f[s] = [self.t0[s] * 100, self.vpre[s] / 8000, self.M[s] / 1.5e6, len(self.active[s]) / 50,
                    self.kv(s) / self.M[s], self.committed_pred(s) / self.M[s], self.queued_pred(s) / self.M[s]]
        return f


def run(env, policy, seed):
    """Runs one episode; returns (mean latency, P99 latency, number of unfinished requests)."""
    env.reset(seed)
    done = False
    while not done:
        done = env.step(policy(env))
    lat = np.array([env.finish[i] - env.arrive[i] for i in env.finish])
    return lat.mean(), np.percentile(lat, 99), len(env.arrive) - len(env.finish)


def run_record(env, policy, seed):
    """Runs one episode; returns per-request latencies and prompt tokens routed to each server."""
    env.reset(seed)
    done = False; tok = np.zeros(env.S)
    while not done:
        a = policy(env)
        for r, s in zip(env.reqs, a):
            tok[s] += r[1]
        done = env.step(a)
    lat = np.array([env.finish[i] - env.arrive[i] for i in env.finish])
    return lat, tok
