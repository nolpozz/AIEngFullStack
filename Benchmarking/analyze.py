"""
Compare benchmark runs side by side, grouped by what controls each metric.

A run is identified by its tag (RUN_TAG in tracing/tracer.py):
    tracing/results_<tag>.json   harness results (per-task time/status, t_start/t_end)
    tracing/traces_<tag>.jsonl   agent traces (per model call / tool call)
Server-side numbers come from tracing/vllm_metrics.jsonl (one continuous stream from the
scraper), sliced to each run's [t_start, t_end] window by timestamp.

Usage, from the repo root:
    python -m Benchmarking.analyze pass1 20261016_110431
    python -m Benchmarking.analyze --latest 3
    python -m Benchmarking.analyze --latest 2 --trim 30 --csv compare.csv

--trim S drops S seconds from both ends of each run (warmup / tail) for the server metrics
and for which tasks count. It is the steady-state window knob.

Sections, and what moves them:
    Settings          what differs between the runs (marked *)
    Workload shape    agent + prompts + question set. Should match across runs you compare.
    Objective         what the sweep optimizes (throughput, p90, errors)
    Client latency    what an agent experiences. Mix of hyperparameters and kernels.
    Scheduler/memory  vLLM hyperparameters: max_num_seqs, max_num_batched_tokens,
                      gpu_memory_utilization, kv_cache_dtype
    Engine speed      most sensitive to kernels. Compare at the same concurrency, because
                      batch size also changes them.
    Search tool       client-side; independent of vLLM and kernels
"""

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TRACING = ROOT / "tracing"

# Approximate H100 SXM peaks. Only used to turn vLLM's estimated FLOP/byte counters
# (they read 0 unless MFU metrics are enabled on the server) into a utilization %.
PEAK_TFLOPS = 989.0   # bf16 dense
PEAK_GBPS = 3350.0    # HBM3


# ---------- small helpers ----------

def read_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # a half-written last line while a run or the scraper is live
    return rows


def pct(vals, p):
    return float(np.percentile(vals, p)) if len(vals) else None


def mean(vals):
    return float(np.mean(vals)) if len(vals) else None


def ratio(a, b):
    return a / b if a is not None and b else None


def fmt(v, spec=".1f", scale=1.0):
    return "-" if v is None else f"{v * scale:{spec}}"


def val(metrics, name, contains=None):
    """Sum of every series of one metric (optionally only labels containing `contains`)."""
    total = None
    for key, v in metrics.items():
        if key.split("{", 1)[0] == name and (contains is None or contains in key):
            total = (total or 0.0) + v
    return total


def delta(a, b, name, contains=None):
    x, y = val(a, name, contains), val(b, name, contains)
    return None if x is None or y is None else y - x


def hist_mean(a, b, name):
    """Mean of a Prometheus histogram over the window: delta(sum) / delta(count)."""
    return ratio(delta(a, b, name + "_sum"), delta(a, b, name + "_count"))


def label(metrics, name, key):
    """One label's value on a metric, e.g. kv_cache_size_tokens on vllm:cache_config_info."""
    for k in metrics:
        if k.split("{", 1)[0] == name:
            m = re.search(rf'{key}="([^"]*)"', k)
            if m:
                return m.group(1)
    return None


# ---------- per-run computation ----------

def compute(run, trim):
    res, events = run["results"], run["events"]
    t0, t1 = res["t_start"] + trim, res["t_end"] - trim
    window = t1 - t0
    warn = []

    # --- settings, from the first run_start event ---
    start_ev = next((e for e in events if e["type"] == "run_start"), None)
    cfg = (start_ev or {}).get("config", {})
    srv = cfg.get("server", {})
    config_ids = sorted({e["config_id"] for e in events})
    if len(config_ids) > 1:
        warn.append(f"trace file mixes config_ids {config_ids}")
    settings = [
        ("config_id", ",".join(config_ids) or None),
        ("concurrency", res.get("concurrency", cfg.get("concurrency"))),
        ("reasoning effort", cfg.get("reasoning_effort")),
        ("max_tokens / max_steps", f"{cfg.get('max_tokens')}/{cfg.get('max_steps')}" if cfg else None),
        ("gpu", srv.get("gpu")),
        ("max_num_seqs", srv.get("max_num_seqs")),
        ("max_num_batched_tokens", srv.get("max_num_batched_tokens")),
        ("gpu_memory_utilization", srv.get("gpu_memory_utilization")),
        ("kv_cache_dtype", srv.get("kv_cache_dtype")),
        ("max_model_len", srv.get("max_model_len")),
        ("fast_boot (eager)", srv.get("fast_boot")),
    ]
    settings = [(k, "-" if v is None else str(v)) for k, v in settings]

    # --- harness tasks inside the window ---
    all_tasks = res["results"]
    tasks = [r for r in all_tasks if r["start_ts"] >= t0 and r["start_ts"] + r["duration"] <= t1]
    if len(tasks) < len(all_tasks):
        warn.append(f"window excludes {len(all_tasks) - len(tasks)} of {len(all_tasks)} tasks")
    ok = [r["duration"] for r in tasks if r["status"] == "ok"]
    bad = Counter(r["status"] for r in tasks if r["status"] != "ok")

    # --- trace runs inside the window (complete runs only) ---
    by_run = defaultdict(list)
    for e in events:
        by_run[e["run_id"]].append(e)
    runs = []
    for evs in by_run.values():
        s = next((e for e in evs if e["type"] == "run_start"), None)
        en = next((e for e in evs if e["type"] == "run_end"), None)
        if s and en and s["timestamp"] >= t0 and en["timestamp"] <= t1:
            runs.append(evs)
    if events and not runs:
        warn.append("no complete trace runs inside the window")
    model = [e for evs in runs for e in evs if e["type"] == "model"]
    tools = [e for evs in runs for e in evs if e["type"] == "tool"]
    lookups = [e for e in tools if e.get("cache_hit") is not None]  # runs from before cache_hit was logged have none
    ends = [next(e for e in evs if e["type"] == "run_end") for evs in runs]
    model_errors = sum(1 for evs in runs for e in evs if e["type"] == "model_error")

    prompts = [e["prompt_tokens"] for e in model if e.get("prompt_tokens") is not None]
    comps = [e["completion_tokens"] for e in model if e.get("completion_tokens") is not None]
    peaks = []
    for evs in runs:
        ms = [e for e in evs if e["type"] == "model" and e.get("prompt_tokens") is not None]
        if ms:
            last = max(ms, key=lambda e: e["call_index"])
            peaks.append(last["prompt_tokens"] + (last.get("completion_tokens") or 0))
    ttft_any = [e["ttft_any_ms"] for e in model if e.get("ttft_any_ms") is not None]
    ttft_vis = [e["ttft_visible_ms"] for e in model if e.get("ttft_visible_ms") is not None]
    decode_tps = [e["completion_tokens"] / (e["decode_ms"] / 1000) for e in model
                  if (e.get("completion_tokens") or 0) >= 50 and e.get("decode_ms")]
    run_ms = sum(e["duration_ms"] for e in ends)
    model_share = ratio(sum(e["latency_ms"] for e in model), run_ms)
    tool_share = ratio(sum(e["latency_ms"] for e in tools), run_ms)
    no_search = sum(1 for evs in runs if not any(e["type"] == "tool" for e in evs))

    # --- inputs for the ranges worksheet ---
    sized = [e for e in model if e.get("prompt_tokens") is not None]
    ctx_end = [e["prompt_tokens"] + (e.get("completion_tokens") or 0) for e in sized]  # context when the call finishes
    # KV a request holds while it runs: its prompt plus, on average, half of what it generates
    resident = [e["prompt_tokens"] + (e.get("completion_tokens") or 0) / 2 for e in sized]
    weights = [e["latency_ms"] for e in sized]  # weight by how long each call occupies the engine
    resident_mean = ratio(sum(w * r for w, r in zip(weights, resident)), sum(weights))
    growth = []  # tokens added to the prompt between consecutive calls of one run
    for evs in runs:
        ms = sorted((e for e in evs if e["type"] == "model" and e.get("prompt_tokens") is not None),
                    key=lambda e: e["call_index"])
        growth += [nxt["prompt_tokens"] - prev["prompt_tokens"] for prev, nxt in zip(ms, ms[1:])]
    near = min(run["samples"], key=lambda x: abs(x["timestamp"] - t0), default=None)
    kv_cfg = near["metrics"] if near and abs(near["timestamp"] - t0) < 120 else {}  # same server boot as the run

    def cfg_num(key):
        v = label(kv_cfg, "vllm:cache_config_info", key)
        return float(v) if v not in (None, "None") else None

    # --- server metrics inside the window ---
    samples = [x for x in run["samples"] if t0 <= x["timestamp"] <= t1]
    server = len(samples) >= 2
    if not server:
        warn.append("no server metrics in the window (scraper not running?)")
    else:
        a, b = samples[0]["metrics"], samples[-1]["metrics"]
        dt = samples[-1]["timestamp"] - samples[0]["timestamp"]
        if any((delta(a, b, n) or 0) < 0 for n in ("vllm:prompt_tokens_total", "vllm:generation_tokens_total")):
            warn.append("server counters went backwards (restart inside the window); server metrics skipped")
            server = False
    if server:
        def series(name):
            return [v for x in samples if (v := val(x["metrics"], name)) is not None]
        running, waiting, kv = series("vllm:num_requests_running"), series("vllm:num_requests_waiting"), series("vllm:kv_cache_usage_perc")
        hits, queries = delta(a, b, "vllm:prefix_cache_hits_total"), delta(a, b, "vllm:prefix_cache_queries_total")
        itl = hist_mean(a, b, "vllm:inter_token_latency_seconds")
        flops, rd, wr = (delta(a, b, f"vllm:estimated_{n}_per_gpu_total") for n in ("flops", "read_bytes", "write_bytes"))
        tflops = flops / dt / 1e12 if flops and dt > 0 else None
        gbps = (rd + (wr or 0)) / dt / 1e9 if rd and dt > 0 else None
        S = dict(
            run_mean=mean(running), run_max=max(running) if running else None,
            wait_mean=mean(waiting), wait_max=max(waiting) if waiting else None,
            kv_mean=mean(kv), kv_max=max(kv) if kv else None,
            preempt=delta(a, b, "vllm:num_preemptions_total"),
            queue=hist_mean(a, b, "vllm:request_queue_time_seconds"),
            hit=ratio(hits, queries),
            step_tokens=hist_mean(a, b, "vllm:iteration_tokens_total"),
            gen_tps=ratio(delta(a, b, "vllm:generation_tokens_total"), dt),
            itl=itl,
            prefill_tps=ratio(delta(a, b, "vllm:prompt_tokens_by_source_total", 'source="local_compute"'), dt),
            prefill_req_tps=ratio(delta(a, b, "vllm:request_prefill_kv_computed_tokens_sum"),
                                  delta(a, b, "vllm:request_prefill_time_seconds_sum")),
            computed=hist_mean(a, b, "vllm:request_prefill_kv_computed_tokens"),
            ttft=hist_mean(a, b, "vllm:time_to_first_token_seconds"),
            tflops=tflops, gbps=gbps,
        )
    else:
        S = {}
    g = S.get

    sections = [
        ("Settings  (* = differs between runs)", [(k, v) for k, v in settings]),
        ("Workload shape  (agent + prompts; should match across compared runs)", [
            ("runs in window", str(len(runs))),
            ("model calls / run", fmt(ratio(len(model), len(runs)), ".2f")),
            ("searches / run", fmt(ratio(len(tools), len(runs)), ".2f")),
            ("runs with no search (%)", fmt(ratio(no_search, len(runs)), ".0f", 100)),
            ("prompt tokens / call (mean)", fmt(mean(prompts), ".0f")),
            ("completion tokens / call (mean)", fmt(mean(comps), ".0f")),
            ("peak context per run (mean tokens)", fmt(mean(peaks), ".0f")),
            ("peak context per run (max tokens)", fmt(max(peaks) if peaks else None, ".0f")),
            ("calls cut off at max_tokens", str(sum(1 for e in model if e.get("finish_reason") == "length"))),
        ]),
        ("Objective  (what the sweep optimizes)", [
            ("tasks ok", str(len(ok))),
            ("tasks / min (ok, over window)", fmt(ratio(len(ok), window), ".2f", 60)),
            ("task time p50 (s)", fmt(pct(ok, 50))),
            ("task time p90 (s)", fmt(pct(ok, 90))),
            ("task time p99 (s)", fmt(pct(ok, 99))),
            ("failed tasks", str(sum(bad.values())) + (f" ({', '.join(f'{k}:{v}' for k, v in bad.items())})" if bad else "")),
            ("model errors (retried)", str(model_errors)),
        ]),
        ("Client latency  (what an agent feels; hyperparameters + kernels)", [
            ("TTFT, first token p50 (ms)", fmt(pct(ttft_any, 50), ".0f")),
            ("TTFT, first token p90 (ms)", fmt(pct(ttft_any, 90), ".0f")),
            ("time to answer/tool-call p50 (ms)", fmt(pct(ttft_vis, 50), ".0f")),
            ("decode speed per stream (tok/s)", fmt(mean(decode_tps), ".1f")),
            ("time in model (% of task)", fmt(model_share, ".0f", 100)),
            ("time in tools (% of task)", fmt(tool_share, ".0f", 100)),
        ]),
        ("Scheduler & memory  (vLLM hyperparameters)", [
            ("requests running (mean)", fmt(g("run_mean"), ".1f")),
            ("requests running (max)", fmt(g("run_max"), ".0f")),
            ("requests waiting (mean)", fmt(g("wait_mean"), ".2f")),
            ("requests waiting (max)", fmt(g("wait_max"), ".0f")),
            ("KV cache usage (mean %)", fmt(g("kv_mean"), ".1f", 100)),
            ("KV cache usage (max %)", fmt(g("kv_max"), ".1f", 100)),
            ("preemptions", fmt(g("preempt"), ".0f")),
            ("queue time before scheduling (ms)", fmt(g("queue"), ".0f", 1000)),
            ("prefix cache hit rate (% of prompt tokens)", fmt(g("hit"), ".1f", 100)),
            ("tokens per engine step (mean)", fmt(g("step_tokens"), ".0f")),
        ]),
        ("Engine speed  (kernel-sensitive; compare at equal concurrency)", [
            ("generation throughput (tok/s, whole server)", fmt(g("gen_tps"), ".0f")),
            ("inter-token latency (ms)", fmt(g("itl"), ".1f", 1000)),
            ("decode speed per request (tok/s)", fmt(ratio(1.0, g("itl")), ".1f")),
            ("prefill compute (tok/s, whole server)", fmt(g("prefill_tps"), ".0f")),
            ("prefill speed per request (tok/s)", fmt(g("prefill_req_tps"), ".0f")),
            ("server TTFT mean (ms)", fmt(g("ttft"), ".0f", 1000)),
            ("est. compute per GPU (TFLOP/s)", fmt(g("tflops"), ".0f")),
            ("est. compute (% of peak)", fmt(ratio(g("tflops"), PEAK_TFLOPS), ".0f", 100)),
            ("est. memory traffic per GPU (GB/s)", fmt(g("gbps"), ".0f")),
            ("est. memory traffic (% of peak)", fmt(ratio(g("gbps"), PEAK_GBPS), ".0f", 100)),
        ]),
        ("Search tool  (client-side; independent of vLLM and kernels)", [
            ("search cache hit rate (%)", fmt(ratio(sum(1 for e in lookups if e["cache_hit"]), len(lookups)), ".0f", 100)),
            ("tool latency mean (s)", fmt(mean([e["latency_ms"] for e in tools]), ".2f", 0.001)),
            ("tool latency p90 (s)", fmt(pct([e["latency_ms"] for e in tools], 90), ".2f", 0.001)),
            ("search result size (mean chars)", fmt(mean([e["result_chars"] for e in tools]), ".0f")),
            ("repeated identical searches", str(sum(1 for e in tools if e.get("repeated")))),
        ]),
        # only printed with --worksheet; the IDs are the ones used in Benchmarking/ranges_worksheet.ipynb
        ("Worksheet inputs  (IDs match Benchmarking/ranges_worksheet.ipynb)", [
            ("W1  KV cache capacity (tokens)", fmt(cfg_num("kv_cache_size_tokens"), ".0f")),
            ("W2  KV block size (tokens)", fmt(cfg_num("block_size"), ".0f")),
            ("W3  gpu_memory_utilization (server)", fmt(cfg_num("gpu_memory_utilization"), ".2f")),
            ("W4  mean resident context per request (tokens)", fmt(resident_mean, ".0f")),
            ("W5  context at end of call, p50 (tokens)", fmt(pct(ctx_end, 50), ".0f")),
            ("W6  context at end of call, p90 (tokens)", fmt(pct(ctx_end, 90), ".0f")),
            ("W7  context at end of call, max (tokens)", fmt(max(ctx_end) if ctx_end else None, ".0f")),
            ("W8  share of task time inside model calls (%)", fmt(model_share, ".0f", 100)),
            ("W9  mean task time (s)", fmt(mean(ok), ".1f")),
            ("W10 tasks / min (ok, over window)", fmt(ratio(len(ok), window), ".2f", 60)),
            ("W11 prompt growth per step, p50 (tokens)", fmt(pct(growth, 50), ".0f")),
            ("W12 prompt growth per step, p90 (tokens)", fmt(pct(growth, 90), ".0f")),
            ("W13 prompt growth per step, max (tokens)", fmt(max(growth) if growth else None, ".0f")),
            ("W14 prompt tokens computed per request (server mean)", fmt(g("computed"), ".0f")),
            ("W15 prefill speed per request (tok/s)", fmt(g("prefill_req_tps"), ".0f")),
            ("W16 inter-token latency (ms)", fmt(g("itl"), ".1f", 1000)),
            ("W17 model calls per run", fmt(ratio(len(model), len(runs)), ".2f")),
        ]),
    ]
    return sections, warn


# ---------- loading and output ----------

def discover(folder):
    files = sorted(folder.glob("results_*.json"), key=lambda p: p.stat().st_mtime)
    return [p.stem[len("results_"):] for p in files]


def load_run(tag, folder, samples):
    results = json.loads((folder / f"results_{tag}.json").read_text())
    trace = folder / f"traces_{tag}.jsonl"
    return {"tag": tag, "results": results, "events": read_jsonl(trace) if trace.exists() else [], "samples": samples}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tags", nargs="*", help="run tags (see tracing/results_<tag>.json)")
    p.add_argument("--latest", type=int, help="use the N most recent runs")
    p.add_argument("--trim", type=float, default=0.0, help="seconds to drop from both ends of each run")
    p.add_argument("--dir", type=Path, default=TRACING, help="folder with results_/traces_ files")
    p.add_argument("--metrics", type=Path, help="scraper file (default: <dir>/vllm_metrics.jsonl)")
    p.add_argument("--csv", type=Path, help="also write the table to this CSV")
    p.add_argument("--worksheet", action="store_true", help="print only the inputs for ranges_worksheet.ipynb")
    args = p.parse_args()

    tags = list(args.tags)
    if args.latest:
        tags += discover(args.dir)[-args.latest:]
    if not tags:
        print("available runs:", ", ".join(discover(args.dir)) or "(none)")
        return

    metrics_path = args.metrics or args.dir / "vllm_metrics.jsonl"
    samples = []
    if metrics_path.exists():
        samples = [{"timestamp": r["timestamp"], "metrics": r["metrics"]} for r in read_jsonl(metrics_path)]
        samples.sort(key=lambda r: r["timestamp"])

    computed, warnings = [], {}
    for tag in tags:
        sections, warn = compute(load_run(tag, args.dir, samples), args.trim)
        computed.append(sections)
        if warn:
            warnings[tag] = warn

    computed = [[s for s in c if s[0].startswith("Worksheet") == args.worksheet] for c in computed]
    label_w = max(len(name) for _, rows in computed[0] for name, _ in rows) + 4
    col_w = max(14, *(len(t) + 2 for t in tags))
    csv_rows = [["section", "metric"] + tags]
    for i, (title, rows) in enumerate(computed[0]):
        print(f"\n{title}")
        print("-" * (label_w + col_w * len(tags)))
        if i == 0:
            print(" " * label_w + "".join(t.rjust(col_w) for t in tags))
        for j, (name, _) in enumerate(rows):
            values = [c[i][1][j][1] for c in computed]
            mark = " *" if title.startswith("Settings") and len(set(values)) > 1 else ""
            print((name + mark).ljust(label_w) + "".join(v.rjust(col_w) for v in values))
            csv_rows.append([title.split("  ")[0], name] + values)

    if warnings:
        print("\nNotes")
        for tag, ws in warnings.items():
            for w in ws:
                print(f"  [{tag}] {w}")
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            csv.writer(f).writerows(csv_rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
