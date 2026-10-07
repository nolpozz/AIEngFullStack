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
- [x] Smoke test, then one full pass at concurrency 1 (45 runs, all ok).
- [ ] Stop retrying 4xx errors in run_agent; retry only 5xx and connection errors.
- [ ] Populate the search cache: 2-3 passes at concurrency 1, then check the hit rate at the 0.90 threshold.
- [ ] Measurement window: warmup pass, count only steady-state tasks, n_tasks scaled with concurrency.
- [ ] Capture per-agent end-to-end time, p50/p90/p99, tasks per minute and error count.
- [ ] Scrape vLLM's /metrics during the sweep (running/waiting requests, KV usage, preemptions, prefix-cache hits). Separate file, joined to results by timestamp.
- [ ] Move the harness to Modal: image, secrets, search cache, results volume, one call per sweep point.
- [ ] Delete benchmarking.py (mock runs; replaced by agent_benchmark.py).

## E. Baseline and ranges

- [x] Concurrency-1 baseline: calls per run, tokens per call, tool latency, model vs. tool time share.
- [x] Rough KV math from real contexts (median peak 8.4k tokens, max 15.7k, about 19–35 sequences). Refresh after the extra passes.
- [ ] Set the hyperparameter ranges from those numbers (see the settings table below).

## F. Define "best"

- [ ] Write the objective down before sweeping. For example: maximize agents per GPU-hour while keeping p90 task time ≤ X seconds and errors at 0.

## G. Sweep

- [ ] Run agents = 1, 2, 4, 8, 16, 32, 64, 96, 128 on the default config, with warmup and a populated cache. Find where it saturates: KV usage hits ~100%, preemptions start, and p90 bends upward.
- [ ] Change one setting at a time (list below) and re-run only the points around that saturation point. Redeploy the server for every config change.
- [ ] Re-run the best config across the full sweep to confirm it.

## H. Before kernels

- [ ] Freeze the best config. Profile it (torch profiler or nsys) to see whether prefill attention, decode attention or the MLP layers dominate. Let that decide what kernel you write.

## Settings you'll be checking

Main ones (vLLM):

| Knob | What it trades | Starting range |
|---|---|---|
| kv_cache_dtype | auto vs fp8: capacity vs a little accuracy | {auto, fp8} |
| max_num_batched_tokens | Chunked prefill per step: bigger means lower TTFT for long agent prompts but choppier decode for everyone else | 2k, 4k, 8k, 16k |
| max_num_seqs | Cap on batch size. Past KV capacity it only causes preemption | From E's math: ~0.5×, 1×, 2× of KV tokens ÷ avg context |
| gpu_memory_utilization | More KV vs OOM risk | 0.88–0.95 |
| max_model_len | Sets the longest allowed request, not memory use | Fixed at whatever your agents need |
