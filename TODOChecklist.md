# Checklist

## A. Fix things that break runs

- [x] Raise MAX_MODEL_LEN to 32k–64k (the recipe uses 131072). Make sure prompt + max_tokens fits.
- [x] Wrap chat() in try/except. Log failures as trace events instead of letting them kill the run.
- [x] Load .env (python-dotenv, or export the key) so tools.py doesn't raise a KeyError on import.
- [x] Set max_retries=0 on the OpenAI client.
- [x] Fix the stale paths in the docstrings and README.

## B. Fix the workload definition (decide once, then keep fixed)

- [x] Set reasoning effort explicitly through chat_template_kwargs (high, per the model card).
- [x] Use the temperature the model card recommends, not 0.
- [x] Choose the Brave token cap, max_tokens and max_steps, and write them down.
- [x] Build a search cache: record real Brave results once, then replay them during load tests. This avoids rate limits and makes runs repeatable.
- [x] Assemble a question set of ~30–50 questions: some need search, some don't, some need a second query.

## C. Tracing that can support the analysis

- [x] Stream model calls with include_usage and record TTFT and decode time separately.
- [x] Record a timestamp, step index and result size for each tool call (size is in chars).
- [x] Give each run a config_id/tag: server settings, concurrency, workload settings. Use one trace file per sweep point, or filter by tag.
- [x] Make writes safe under concurrency: one writer, or an asyncio queue (a lock, for threads).

## D. Benchmark harness (agent_benchmark.py)

- [x] Local harness: run_one, build_tasks (shuffled, seeded), run_point (thread pool, closed loop).
- [x] Server fixes so agent calls work: `--chat-template-content-format string`, `min_containers=1`, longer startup timeout.
- [x] Model fixes so agent calls work: send reasoning back on assistant messages, read the `reasoning` stream field.
- [x] Smoke test, then three full passes at concurrency 1 (45 runs each).
- [x] Stop retrying 4xx errors in run_agent; retry only 5xx and connection errors.
- [x] Populate the search cache with the three passes (0.90 similarity threshold).
- [x] Timestamped output per run (RUN_TAG): `traces_<tag>.jsonl` and `results_<tag>.json`, so passes never overwrite each other.
- [x] Log `cache_hit` on every search in the traces. (Passes 1-3 predate it; the hit rate shows from the next run.)
- [x] Replace the question that spiraled (CUDA version) with a stable-answer one.
- [x] Analysis script `Benchmarking/analyze.py`: compare runs by metric group, `--trim` for the steady-state window, `--worksheet` for the ranges inputs.
- [x] Capture per-agent end-to-end time, p50/p90/p99, tasks per minute and error count (in analyze.py).
- [x] Scrape vLLM's /metrics (`python -m tracing.vllm_metrics`), separate file joined to results by timestamp. Verified locally in passes 2-3.
- [ ] Measurement window in the harness: discarded warmup pass, n_tasks scaled with concurrency. (analyze.py already trims by time.)
- [ ] Check the search-cache hit rate from the `cache_hit` field after the next run.
- [ ] Move the harness and the scraper to Modal: image, secrets, search cache, results volume, one call per sweep point.
- [x] Delete benchmarking.py (mock runs; replaced by agent_benchmark.py). Archived in `archive/`.
- [x] Curve script `Benchmarking/curve.py`: the latest sweep's tasks/min, p90 and KV against A, with the knee from the F rule.

## E. Baseline and ranges

- [x] Concurrency-1 baseline: calls per run, tokens per call, tool latency, model vs. tool time share.
- [x] Rough KV math from real contexts (pass 1). Redo it by hand in the worksheet with all three passes.
- [x] Work through `Benchmarking/ranges_worksheet.ipynb` to set the hyperparameter ranges and the agent sweep points. Reference: `Benchmarking/parameter_relationships.md`.

## F. Define "best"

Two objectives, both read from the same sweep. Each one picks its own best config. Only T gets a kernel: the L kernel was dropped after profiling (see L below and `phase3_kernels/README.md`).

- [x] **Throughput (T):** maximize completed tasks per hour over all configs (S, B) and agent counts (A).
  - Constraints: 0 errors; p90 task time ≤ 3× the concurrency-1 p90.
  - Tie-break: among results within 5% of the best, take the lowest A.
  - Expected operating point: running batch ≈ N_kv (~36). Expected kernel target: decode attention.
- [x] **Latency (L):** minimize p90 task time at A = 8.
  - Constraint: 0 errors.
  - Expected operating point: small batch. Expected kernel target: weight matrix-vector products and the TP all-reduce.
  - **No L kernel (decided Nov 9).** Profile at A=8: weight GEMMs are 70% of a decode step and already at ~89% of HBM peak; the best kernel candidates (fused all-reduce + RMSNorm ~0.4-0.5 ms, small-batch attention ~1 ms of a 16.8 ms step) are 2-6%, while p90 at A=8 varies ~±20% between identical runs, so the objective couldn't show them. The lever for L is fewer weight bytes: Phase 4 quantization.

## G. Sweep

Knobs: S = `max_num_seqs`, B = `max_num_batched_tokens`. Everything else is fixed (table below).

- [ ] Before sweeping: harness on Modal, discarded warmup, n_tasks rule below, search cache replay so misses don't hit Brave live.
- [ ] Before sweeping: check that the CUDA graph max capture size is ≥ 72 (the largest S) on the vLLM image.
- [x] **Low-load points** (latency objective; S doesn't matter here because demand < every S): A = 1, 4, 8 for each B in {2048, 4096, 8192, 16384}, with S = 36. 4 configs × 3 points.
- [x] **Knee points** (ran with A = 16, 24, 32, 40, 48, 64; 96 dropped) (throughput objective): A = 16, 24, 32, 48, 64, 96 for every (S, B) in {18, 36, 72} × {2048, 4096, 8192, 16384}. 12 configs × 6 points.
- [x] Pick the T winner and the L winner from the objectives in F. **Frozen: S=36, B=4096 for both; T at A=32, L at A=8.**
  - T: S=36, B=4096 (48.4 tasks/min at A=32, best plateau 42.4, 0 preemptions there). S=72 ties on tasks/min but admits past N_kv and churns (A=40: 13 preemptions vs 0, lower prefix hit, slower decode), so it's out. S=18 is capped below N_kv.
  - T confirmation curve (Nov 3, H100!): flat at ~38-40 tasks/min from A=32 to 64; knee by the F rule A=32 (39.7, mean of 36.0 and 43.3), best A=36 (40.4). The grid's 48.4 at A=32 was a high draw: identical runs differ by ~20%, so read the plateau, not single points. Operating point and target_concurrency: A=32.
  - L: S=36, B=4096 at A=8. At A=8 the server metrics match for every B (ITL ~21 ms, nothing waiting); 36/4096's own A=8 p90s span 41-64 s across three runs, and 36/16384's single 36.5 s sits at the edge of that noise. Same config as T, so one server profiled at two operating points.
- [x] **Full curve** for each winner, to confirm it (done for 36/4096 with A=1-64 plus 36, 44 and repeats of 32 and 8): A = 1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128.
- n_tasks per point: `45 × ceil(4A / 45)`, minimum 90 (multiples of 45 keep the question mix balanced). Trim the warmup and the end drain with `--trim`.
- Redeploy the server for every config change. Total: ~84 grid points + 2 confirmation curves, 12 redeploys.

## H. Before kernels

- [x] Note the throughput winner's saturation point (A=32): it becomes `target_concurrency` if this ever scales past one container. Re-check it after the kernel, since a faster kernel moves the knee. (See the comment above `@app.server` in `phase1_serving/modalForvLLM.py`.)
- [x] Sweep data rotated to `tracing/sweep_2026-11-03/` (results, traces, the metrics file and the curve PNG). It is the pre-kernel baseline: `python -m Benchmarking.analyze --dir tracing/sweep_2026-11-03 ...` / `python -m Benchmarking.curve --dir tracing/sweep_2026-11-03`.
- [x] Profiler support: `PROFILER = "torch"` in server_config.py turns on `--profiler-config` (traces to the `vllm-profiles` Modal volume); `python -m Benchmarking.capture_profile` starts and stops a capture while the harness runs.
- [x] Freeze the T and L configs. Profile each at its operating point (torch profiler or nsys) to see whether prefill attention, decode attention, the MLP/projection matmuls or the all-reduce dominate. Let that decide each kernel.
  - Done Nov 3 (torch profiler, 20 s each, traces in `profiles/`). A=32: decode attention 45% of a decode-only step (FA3 at ~79% of HBM peak), GEMMs 46% (~89%), prefill-carrying steps 49% of GPU time. A=8: GEMMs 70%, attention 20%. Decision: one kernel, paged GQA decode attention for T (`phase3_kernels/decode_attention.py`); no L kernel.

## I. Kernel T: paged GQA decode attention (`phase3_kernels/decode_attention.py`)

Goal: FA3 decode attention at A=32 takes 176 µs/call (~79% of 3.35 TB/s); target ≤ 155 µs (~90%), ~1.4 ms off each 25.5 ms decode-only step.
The agent benchmark can't show a ~5% gain (±20% run to run), so judge it from the microbenchmark, the profile and engine metrics.

- [ ] Reference implementation + random paged inputs (uneven seq_lens, shuffled blocks).
- [ ] Split kernel + combine kernel; `test` passes against the reference at small, A32 and A8 shapes.
- [ ] `bench` on Modal (`H100!`): µs/call and TB/s vs FA3. A32 ≤ 155 µs; A8 no slower than 49 µs. Tune NUM_SPLITS and BLOCK_N.
- [ ] Integrate into vLLM (attention backend or patched decode call), CUDA-graph safe.
- [ ] Profile A=32 again: decode-only step time vs 25.5 ms, attention row vs 11.5 ms (`python -m Benchmarking.kernel_breakdown`).
- [ ] `PROFILER = None`, run A=32 a few times: ITL and generation throughput vs `tracing/sweep_2026-11-03`.
- [ ] Sanity check tasks/min and p90 against the baseline curve; re-check the knee and `target_concurrency`.

## Settings

Swept:

| Knob | What it trades | Values |
|---|---|---|
| max_num_seqs (S) | Cap on the running batch. Below N_kv (~36) KV sits idle; above it, short requests pack in but long ones get preempted | 18, 36, 72 |
| max_num_batched_tokens (B) | Chunked prefill per step: bigger means lower TTFT for long agent prompts but bigger decode stalls for everyone else | 2048, 4096, 8192, 16384 |
| agents (A) | The load | see G |

Fixed for this sweep:

| Setting | Value | Why fixed |
|---|---|---|
| kv_cache_dtype | auto (bf16) | Out of scope for now (would need a quality check) |
| gpu_memory_utilization | 0.90 | Small, predictable effect (~0.8 requests per +0.01) |
| GPUs / tensor parallel | 2 | Out of scope for now |
| max_model_len | 64000 | Sets a limit, not memory use |
| prefix caching | on | Turning it off only measures a known loss |
| speculative decoding | off | Deferred, to get to the kernels sooner |
| workload | reasoning high, MAX_TOKENS 4096, MAX_STEPS 6, cached search | Changing it changes the workload, not the server |
