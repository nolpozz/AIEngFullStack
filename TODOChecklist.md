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
- [ ] Delete benchmarking.py (mock runs; replaced by agent_benchmark.py).

## E. Baseline and ranges

- [x] Concurrency-1 baseline: calls per run, tokens per call, tool latency, model vs. tool time share.
- [x] Rough KV math from real contexts (pass 1). Redo it by hand in the worksheet with all three passes.
- [x] Work through `Benchmarking/ranges_worksheet.ipynb` to set the hyperparameter ranges and the agent sweep points. Reference: `Benchmarking/parameter_relationships.md`.

## F. Define "best"

Two objectives, both read from the same sweep. Each one picks its own best config and gets its own kernel in H.

- [x] **Throughput (T):** maximize completed tasks per hour over all configs (S, B) and agent counts (A).
  - Constraints: 0 errors; p90 task time ≤ 3× the concurrency-1 p90.
  - Tie-break: among results within 5% of the best, take the lowest A.
  - Expected operating point: running batch ≈ N_kv (~36). Expected kernel target: decode attention.
- [x] **Latency (L):** minimize p90 task time at A = 8.
  - Constraint: 0 errors.
  - Expected operating point: small batch. Expected kernel target: weight matrix-vector products and the TP all-reduce.

## G. Sweep

Knobs: S = `max_num_seqs`, B = `max_num_batched_tokens`. Everything else is fixed (table below).

- [ ] Before sweeping: harness on Modal, discarded warmup, n_tasks rule below, search cache replay so misses don't hit Brave live.
- [ ] Before sweeping: check that the CUDA graph max capture size is ≥ 72 (the largest S) on the vLLM image.
- [ ] **Low-load points** (latency objective; S doesn't matter here because demand < every S): A = 1, 4, 8 for each B in {2048, 4096, 8192, 16384}, with S = 36. 4 configs × 3 points.
- [ ] **Knee points** (throughput objective): A = 16, 24, 32, 48, 64, 96 for every (S, B) in {18, 36, 72} × {2048, 4096, 8192, 16384}. 12 configs × 6 points.
- [ ] Pick the T winner and the L winner from the objectives in F.
- [ ] **Full curve** for each winner, to confirm it: A = 1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128.
- n_tasks per point: `45 × ceil(4A / 45)`, minimum 90 (multiples of 45 keep the question mix balanced). Trim the warmup and the end drain with `--trim`.
- Redeploy the server for every config change. Total: ~84 grid points + 2 confirmation curves, 12 redeploys.

## H. Before kernels

- [ ] Note the throughput winner's saturation point: it becomes `target_concurrency` if this ever scales past one container. Re-check it after the kernel, since a faster kernel moves the knee. (See the comment above `@app.server` in `phase1_serving/modalForvLLM.py`.)
- [ ] Freeze the T and L configs. Profile each at its operating point (torch profiler or nsys) to see whether prefill attention, decode attention, the MLP/projection matmuls or the all-reduce dominate. Let that decide each kernel.

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
