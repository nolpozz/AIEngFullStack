# Phase 3 — Kernels

Profile the serving run to find the bottleneck, write a custom Triton kernel, microbenchmark it against
vLLM's kernel, then plug it into vLLM and re-run the load test to see the end-to-end effect.

One kernel, for the throughput objective:

| Objective | Operating point | Kernel | File |
|---|---|---|---|
| **T** throughput: max tasks/min, p90 ≤ 97 s, 0 server errors | S=36, B=4096, **A=32** | paged GQA decode attention, split-KV | `decode_attention.py` |

The latency objective (L, p90 at A=8) was profiled too but gets no kernel: see "Why no L kernel" below.

Runs on 2× H100 80GB (`H100!`), TP=2, bf16 weights and KV, CUDA graphs on.
Model: K2-Horizon-32B, L=64, 64 query / 8 KV heads (32 / 4 per GPU), head dim 128, hidden 5120.

---

## How the kernel was selected

### 1. Define "best" (TODOChecklist F)
- **T:** maximize completed tasks per minute, with p90 task time ≤ 3× the concurrency-1 p90 (≈ 97 s) and
  no server errors (5xx / connection). `max_steps` and 4xx context overflows are workload limits and don't count.
  Tie-break: the lowest A within 5% of the best.
- **L:** minimize p90 task time at A = 8.

### 2. Sweep the two knobs that matter (Oct 30 – Nov 1)
`max_num_seqs` S ∈ {18, 36, 72} × `max_num_batched_tokens` B ∈ {2048, 4096, 8192, 16384}: 12 configs,
A = 16–64 each (plus A = 1, 4, 8 at S=36). Everything else fixed (see TODOChecklist "Settings").
Search results were replayed from the cache, so every run saw the same workload.

| Finding | Evidence |
|---|---|
| S=18 caps the batch below what fits in KV | plateau 31–34 tasks/min vs 37–42; batch stuck at 18 with KV ~45% |
| S=72 ties S=36 on tasks/min but churns | at A=40, B=4096: 13 preemptions vs 0, prefix hit 29% vs 36%, slower decode. Its better TTFT just moves the wait from the queue into decode (time to answer p50 the same) |
| B=4096 has the best plateau at both S=36 and S=72; 16384 the worst | S=36 plateau (A=32–64): 2048 37.0, **4096 42.4**, 16384 35.3; at A=40 only 4096 stays under the KV limit and the p90 cap |
| B doesn't matter at light load | at A=8, ITL is 21 ms and nothing waits for every B; the p90 spread is the task mix |

Two measurement fixes along the way, both reflected in the numbers above: `analyze.py` had counted only tasks
that *started and ended* in the window (undercounting tasks/min and shortening p90), and one config had been
silently upgraded to H200s by Modal (`H100!` now pins the GPU).

### 3. Confirm (Nov 3, `tracing/sweep_2026-11-03/`)
Full curve on 36/4096, A = 1–64 plus 36, 44 and repeats of 32 and 8 (`python -m Benchmarking.curve --dir tracing/sweep_2026-11-03`):

| A | 1 | 8 | 16 | 24 | **32** | 36 | 40 | 44 | 48 | 64 |
|---|---|---|---|---|---|---|---|---|---|---|
| tasks/min | 3.4 | 19.9 | 29.6 | 37.6 | **39.7** | 40.4 | 40.1 | 37.2 | 40.0 | 38.5 |
| p90 (s) | 37 | 45 | 48 | 66 | **75** | 82 | 88 | 131 | 110 | 154 |
| preemptions | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 7 | 1 | 8 |

The knee by the F rule is **A=32** (39.7, within 5% of the best 40.4 at A=36). That is the T operating point and
`target_concurrency`. Identical runs differ by ~20% (A=32: 36.0 and 43.3), so the plateau is the number to trust,
not single points.

### 4. Profile (Nov 3)
torch profiler through vLLM (`PROFILER = "torch"` in `server_config.py`, `python -m Benchmarking.capture_profile`),
20 s at steady state at A=32 and at A=8, broken down with `python -m Benchmarking.kernel_breakdown <trace> --top 16`.
The GPU was busy 99–100% of both windows; kernels inside CUDA graphs are traced individually.

| | A=32 (T) | A=8 |
|---|---|---|
| decode-only steps | 400 × 25.5 ms, batch ~31 | 904 × 16.8 ms, batch 8 |
| mixed (prefill chunk + decode) steps | 48 × 201 ms → **49%** of GPU time | 24 × 185 ms → 23% |

Inside one decode-only step:

| | A=32 ms | A=32 share | A=8 ms | A=8 share |
|---|---|---|---|---|
| weight GEMMs (QKV, O, MLP) | 11.8 | 46% | 11.7 | **70%** |
| decode attention (FA3) | 11.5 | **45%** | 3.4 | 20% |
| all-reduce + residual add + RMSNorm | 1.4 | 5% | 1.0 | 6% |
| other elementwise, KV write | 0.8 | 3% | 0.7 | 4% |

Achieved HBM bandwidth (peak 3.35 TB/s per GPU):

| | bytes per call | time per call | achieved |
|---|---|---|---|
| GEMMs | 34.8 GB of weights per step | ~11.8 ms per step | **~89%**: nothing big left to win without fewer bytes (Phase 4) |
| decode attention at A=32 | ~227k KV tokens × 2048 B ≈ 465 MB | 176 µs | **~79%** |
| decode attention at A=8 | ~55k tokens × 2048 B ≈ 112 MB | 49 µs | ~68% |

(KV token counts come from the scraper's KV usage during each capture, ±10%.)

### 5. Choose

| Candidate | Headroom | Decision |
|---|---|---|
| **Decode attention at A=32** (biggest kernel at T, 79% of peak) | 176 → ~155 µs at 90% → **~1.4 ms per decode step (~5.5%)** | **✔ the T kernel** |
| Overlap the prefill all-reduce with GEMMs (T) | ~24 ms per mixed step (~6% of T time) | later; more scheduling than kernel |
| Fused all-reduce + residual add + RMSNorm (L) | 7.86 µs/call × 129 calls = 1.0 ms/step today; a fused call still pays the ~4–5 µs cross-GPU handshake → ~0.4–0.5 ms/step (~2.5–3%) | ✘ dropped |
| Decode attention at A=8 (L) | 3.4 → ~2.4 ms at 90% → ~1 ms/step (~6%) | ✘ as a separate kernel; the T kernel is still tested at A=8 so it doesn't regress there |
| Custom GEMV for the weight matmuls | ≤ ~11% of 11.7 ms, against cuBLAS/nvjet at 89% of peak | ✘ |

### Why no L kernel
At A=8 the weight GEMMs are 70% of a decode step and already at ~89% of the bandwidth limit. What's left
for a kernel is a few percent of a step (0.4–1 ms of 16.8 ms), and the L objective can't see that: three
identical A=8 runs gave p90s of 41.5, 47.8 and 63.7 s (about ±20%), so a 2–6% change would take hundreds of
repeats to show. The lever that would move L is reading fewer weight bytes, i.e. quantized weights (FP8 roughly
halves the 11.7 ms GEMM time), which is Phase 4.

---

## Measuring the kernel

The agent benchmark is too noisy to show a ~5% kernel gain (identical A=32 runs differ by ~20% in tasks/min).
Judge the kernel at the levels that can see it, and use the task-level numbers to check nothing got worse:

1. **Microbenchmark** (`python phase3_kernels/decode_attention.py test`, then `bench`, on Modal `H100!`):
   µs/call and achieved TB/s against FA3 at the A32 shape (176 µs, ~79%), and no regression at A8 (49 µs).
2. **Per-step profile**: integrate into vLLM, profile A=32 again, and compare decode-only step time (25.5 ms) and the
   attention row with `python -m Benchmarking.kernel_breakdown <new trace> --top 16`.
3. **Engine metrics**: redeploy with `PROFILER = None`, run A=32 a few times, compare ITL and generation
   throughput (low noise) with `analyze.py --dir tracing/sweep_2026-11-03`.
4. **Task level**: tasks/min and p90 against the baseline curve (`python -m Benchmarking.curve --dir tracing/sweep_2026-11-03`),
   as a sanity check rather than proof. Re-check the knee: a faster kernel moves it, and with it `target_concurrency`.
