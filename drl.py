"""DRL-Q: learned dispatcher used as the learning-based baseline.

A contextual-bandit variant of deep Q-learning: a three-layer perceptron regresses the realized
end-to-end latency of a request from server-state and request features, and each request is sent
to the server with the lowest predicted latency. Features use predicted output lengths only.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import config as C

SEED = 1
EPS_START, EPS_END = 0.5, 0.02
REPLAY = 150000
BATCH = 2048
EPOCHS = 3
LR = 1e-3


class QNet(nn.Module):
    def __init__(self, n_servers):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(7 + 3 + 2 + n_servers, 128), nn.ReLU(),
                                 nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def features(env, r, cnt, tok):
    sf = env.server_features()
    rem = np.array([sum(env.rem_pred(a) for a in env.active[s]) for s in range(env.S)]) / 3000
    extra = np.stack([cnt / 10, tok / env.M * 10, rem], 1)
    req = np.tile([r[1] / env.mean_in, r[2] / env.mean_out], (env.S, 1))
    return np.concatenate([sf, extra, req, np.eye(env.S)], 1).astype(np.float32)


def _episode(env, q, seed, eps, rng, data=None):
    env.reset(seed)
    done = False
    while not done:
        cnt = np.zeros(env.S); tok = np.zeros(env.S); a = []
        for r in env.view:
            x = features(env, r, cnt, tok)
            if rng.random() < eps:
                j = int(rng.integers(env.S))
            else:
                with torch.no_grad():
                    j = int(torch.argmin(q(torch.from_numpy(x))))
            a.append(j); cnt[j] += 1; tok[j] += r[1] + r[2]
            if data is not None:
                data.append((r[0], x[j]))
        done = env.step(a)


def train(env, episodes=C.DRL_EPISODES, verbose=True):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    q = QNet(env.S)
    opt = torch.optim.Adam(q.parameters(), lr=LR)
    X, Y = [], []; t0 = time.time()
    for ep in range(episodes):
        eps = EPS_START + (EPS_END - EPS_START) * ep / (episodes - 1)
        data = []
        _episode(env, q, C.DRL_TRAIN_SEED_BASE + ep, eps, rng, data)
        for rid, x in data:
            if rid in env.finish:
                X.append(x); Y.append(env.finish[rid] - env.arrive[rid])
        X, Y = X[-REPLAY:], Y[-REPLAY:]
        Xt = torch.from_numpy(np.array(X)); Yt = torch.tensor(Y, dtype=torch.float32)
        for _ in range(EPOCHS):
            perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), BATCH):
                b = perm[i:i + BATCH]
                loss = ((q(Xt[b]) - Yt[b]) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        if verbose and (ep + 1) % 30 == 0:
            print(f"      DRL-Q episode {ep + 1}/{episodes} ({time.time() - t0:.0f} s)", flush=True)
    return q


def policy(q):
    """Greedy dispatching with a trained network."""
    def fn(env):
        cnt = np.zeros(env.S); tok = np.zeros(env.S); a = []
        for r in env.view:
            x = features(env, r, cnt, tok)
            with torch.no_grad():
                j = int(torch.argmin(q(torch.from_numpy(x))))
            a.append(j); cnt[j] += 1; tok[j] += r[1] + r[2]
        return a
    return fn
