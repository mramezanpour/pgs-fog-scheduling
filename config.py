"""Hardware, workload and evaluation settings shared by all experiments."""

# Edge fleet: 3 x RTX 3080, 3 x RTX 3090, 2 x RTX 4090
SERVERS = ["3080"] * 3 + ["3090"] * 3 + ["4090"] * 2

HW_MIXES = {
    "Hom (8x3090)": ["3090"] * 8,
    "Mod (3x3080,3x3090,2x4090)": ["3080"] * 3 + ["3090"] * 3 + ["4090"] * 2,
    "Het (4x3080,1x3090,3x4090)": ["3080"] * 4 + ["3090"] + ["4090"] * 3,
}

# Model and GPU parameters.
#   gpu[type] = (KV budget in GB, prefill throughput in tokens/s, memory bandwidth in GB/s)
#   KV budget = VRAM - model weights - about 1.5 GB of runtime overhead.
#   Prefill throughputs follow LYREO (Li et al., arXiv:2609.17193); bandwidths are vendor figures.
HW_1B = {   # Llama-3.2-1B, FP16 weights; KV per token = 2 x 16 layers x 8 KV heads x 64 dim x 2 B
    "weight_bytes": 2.47e9,
    "kv_bytes": 32 * 1024,
    "gpu": {"3080": (6, 2500, 760), "3090": (20, 4000, 936), "4090": (20, 8000, 1008)},
}
HW_8B = {   # 8B-class model, INT4 weights; KV per token = 2 x 32 layers x 8 KV heads x 128 dim x 2 B
    "weight_bytes": 5.7e9,
    "kv_bytes": 128 * 1024,
    "gpu": {"3080": (2.8, 2500, 760), "3090": (16.8, 4000, 936), "4090": (16.8, 8000, 1008)},
}

# Per-iteration overhead, calibrated on an RTX 3060 Laptop GPU (Qwen2.5-3B, single stream:
# 103.7 tokens/s = 9.64 ms/token; 1.93 GB of weights / 336 GB/s = 5.74 ms -> 3.9 ms overhead).
OVERHEAD_S = 3.9e-3

W_PREFILL = 0.3                                   # decoding slowdown while a prefill batch runs
BURST_HIGH, BURST_LOW, BURST_SWITCH = 1.8, 0.4, 0.1
LENGTH_SHAPE = 0.9                                # log-normal shape of prompt and output lengths
T_SLOTS = 150                                     # arrival slots per run (1 s each)
DRAIN_SLOTS = 400                                 # maximum number of slots after the last arrival
SIGMA = 0.3                                       # output-length prediction error

# Scenarios: (requests per slot, mean prompt tokens, mean output tokens, bursty arrivals)
SCENARIOS = {
    "S1-default":     (40, 70, 215, False),
    "S2-high-load":   (90, 70, 215, False),
    "S4-long-prompt": (20, 1500, 215, False),
    "S3-bursty":      (60, 70, 215, True),
}
S5_SCENARIO = (12, 1500, 300, True)               # run with HW_8B

EVAL_SEEDS = [1000 + i for i in range(10)]
DRL_EPISODES = 120
DRL_TRAIN_SEED_BASE = 900000                      # training runs never reuse an evaluation seed
