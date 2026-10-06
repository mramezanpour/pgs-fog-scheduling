# PGS: Prediction-Guided Scheduling for LLM Inference on Heterogeneous Edge Servers

This repository contains the simulation code and the experimental data for the paper

> **Pricing the Prefill: Training-Free Prediction-Guided Request Scheduling for Low-Latency LLM Inference on Heterogeneous Edge Servers**
> Mohammadreza Ramezanpour and Reihaneh Khorsand
> Department of Computer Engineering, Islamic Azad University, Isfahan, Iran

PGS is a training-free dispatcher that assigns LLM inference requests to heterogeneous edge GPUs.
It prices every candidate placement by its marginal latency: the latency of the request itself plus the delay it imposes on the requests that already share the server. The price has two main parts:

- **Batch-coupled prefill and interference pricing (C1).** A new prompt shares a prefill batch with the prompts already committed to the server and waits for the work the server carries over from earlier slots. Its prefill also slows down the decoding of co-located requests.
- **Virtual state rollout (C2).** The requests of a slot are committed one at a time, from the largest to the smallest predicted footprint, on a virtual copy of the fleet state. Later requests are therefore priced against the load created by earlier ones.

The scheduler uses only the prompt length, a noisy prediction of the output length and telemetry that iteration-level serving engines already expose.

## Repository structure

| File | Content |
|---|---|
| `config.py` | Hardware, model, workload and evaluation settings |
| `simulator.py` | Slotted simulator of iteration-level LLM serving: FIFO KV admission, batched prefill with carry-over, continuous batching, prefill-decode interference, noisy output-length predictions |
| `policies.py` | Random, round robin (RR), least loaded (LL), SAS, JSQ-speed and PGS |
| `drl.py` | DRL-Q, the learning-based baseline (value-based learned dispatcher) |
| `stats.py` | Confidence intervals, paired Wilcoxon tests and result paths |
| `run_main.py` | Tables 4 and 5, scenarios S1 to S4 |
| `run_s5.py` | Tables 4 and 5, scenario S5 (memory-constrained 8B-class model) |
| `run_prompt_sweep.py` | Figure 3, effect of the prompt length |
| `run_sensitivity.py` | Figure 4, sensitivity to prediction error, fleet size, hardware mix and load |
| `run_ablation.py` | Table 6, ablation of PGS |
| `run_overhead.py` | Table 7, scheduling time per slot and DRL-Q training time |
| `run_routing.py` | Figure 2 data: per-request latencies and routing per GPU type |
| `make_figures.py` | Figures 2, 3 and 4 from the CSV files in `results/` |
| `plotstyle.py` | Common figure style |
| `calibration/` | Ollama/nvidia-smi script used to calibrate the per-iteration overhead, and its measurements |
| `results/` | CSV files and figures reported in the paper |

## Requirements

Python 3.9 or later and

```
pip install -r requirements.txt
```

PyTorch is needed only for DRL-Q and runs on the CPU. `requests` is needed only for the calibration script.

## Reproducing the results

Run each script from the repository folder, for example `python run_main.py`. Every script writes its results to `results/`. `make_figures.py` redraws Figures 2 to 4 from the saved CSV files without rerunning any simulation.

| Paper item | Script | Output | Approximate runtime on a laptop CPU |
|---|---|---|---|
| Tables 4 and 5 (S1 to S4) | `run_main.py` | `main_results.csv` | 1 h |
| Tables 4 and 5 (S5) | `run_s5.py` | `s5_results.csv` | 30 to 60 min |
| Figure 2 | `run_routing.py`, then `make_figures.py` | `routing_latencies.csv`, `routing_numbers.csv`, `fig2_routing.png` | 5 to 10 min |
| Figure 3 | `run_prompt_sweep.py`, then `make_figures.py` | `prompt_sweep.csv`, `fig3_prompt_sweep.png` | 1 to 2 h |
| Figure 4 | `run_sensitivity.py`, then `make_figures.py` | `sensitivity.csv`, `fig4_sensitivity.png` | 30 to 60 min |
| Table 6 | `run_ablation.py` | `ablation.csv`, `ablation_table6.csv` | 30 to 60 min |
| Table 7 | `run_overhead.py` | `overhead.csv` | 10 min |

Run `run_overhead.py` on an otherwise idle machine, because it measures wall-clock time.

All methods are evaluated on the same 10 paired seeds (5 in the sensitivity study). DRL-Q is trained for 120 episodes on seeds that are disjoint from the evaluation seeds. The heuristics and PGS are fully deterministic for a given seed. DRL-Q results can differ slightly across PyTorch versions and hardware.

## Simulation setting

| Parameter | Value |
|---|---|
| Edge servers | 3 × RTX 3080, 3 × RTX 3090, 2 × RTX 4090 |
| Memory bandwidth (GB/s) | 760 / 936 / 1008 |
| KV budget (GB) | 6 / 20 / 20 (S1 to S4), 2.8 / 16.8 / 16.8 (S5) |
| Prefill throughput (tokens/s) | 2500 / 4000 / 8000 |
| Model, S1 to S4 | Llama-3.2-1B, FP16, 2.47 GB of weights, 32 KB of KV cache per token |
| Model, S5 | 8B-class, INT4, 5.7 GB of weights, 128 KB of KV cache per token |
| Per-iteration overhead | 3.9 ms, calibrated on an RTX 3060 Laptop GPU (see `calibration/`) |
| Prefill-decode interference | decoding slowed by 30% while a prefill batch runs |
| Slot length and horizon | 1 s; 150 arrival slots, then up to 400 drain slots |
| Prompt and output lengths | log-normal, shape 0.9 |
| Output-length prediction | multiplicative log-normal error, σ = 0.3 |

The scenarios are S1 (40 requests/s, 70-token prompts), S2 (90 requests/s), S3 (bursty, nominal 60 requests/s), S4 (20 requests/s, 1500-token prompts) and S5 (8B-class model, 1500-token prompts, 300-token outputs, bursty, nominal 12 requests/s). The mean output length is 215 tokens in S1 to S4.

## Citation

If you use this code, please cite the paper. The full reference will be added after publication.

## Contact

Mohammadreza Ramezanpour, Department of Computer Engineering, Mo. C., Islamic Azad University, Isfahan, Iran
