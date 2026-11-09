"""
Capture one torch profiler trace from the running vLLM server, at whatever load the harness is putting on it.

The server needs PROFILER = "torch" in server_config.py (then redeploy). Workflow, from the repo root:
    T1  modal serve phase1_serving/modalForvLLM.py         # boot with the profiler on
    T3  CONCURRENCY=32 python -m Benchmarking.agent_benchmark
    T4  (wait ~90 s for steady state, i.e. past the warmup the --trim drops)
        python -m Benchmarking.capture_profile --label T --concurrency 32
    then download:  modal volume get vllm-profiles / profiles/

What it does: POST /start_profile with a prefix and an iteration cap, wait, POST /stop_profile.
On vLLM 0.30.0 the server ignored both: files are named rank<i>.<ns timestamp>.pt.trace.json.gz (match them to
captures by time) and --seconds is what bounds the capture (20 s gave 448 steps at A=32, 928 at A=8, ~30-48 MB
per rank). Stopping flushes the trace files, one per TP rank; the stop call waits up to 30 min for it.
Break a trace down with: python -m Benchmarking.kernel_breakdown profiles/rank0.<id>.pt.trace.json.gz --top 16
Open the .pt.trace.json.gz files in https://ui.perfetto.dev (or chrome://tracing).
Runs on a profiled server are slower; their CONFIG_ID includes the profiler, so they never mix with the sweep.
"""

import argparse
import time

import requests

from server_config import PROFILER, SERVER_URL


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--label", default="profile", help="objective name in the file prefix, e.g. T or L")
    p.add_argument("--concurrency", type=int, required=True, help="the A the harness is running (goes in the prefix)")
    p.add_argument("--iterations", type=int, default=150,
                   help="engine steps to record (~6 s at A=32, ~3 s at A=8); bounds file size")
    p.add_argument("--seconds", type=float, default=20.0, help="how long to wait before stopping")
    args = p.parse_args()

    if PROFILER != "torch":
        print(f'server_config.PROFILER is {PROFILER!r}; set it to "torch" and redeploy first')
        return

    prefix = f"{args.label}_A{args.concurrency}_{time.strftime('%Y%m%d_%H%M%S')}"
    body = {"profile_prefix": prefix, "max_iterations": args.iterations}
    r = requests.post(f"{SERVER_URL}/start_profile", json=body, timeout=60)
    if r.status_code == 404:
        print("server has no /start_profile: it was booted without the profiler")
        return
    if r.status_code == 409:
        print(f"a profile is already running; stop it first:  curl -X POST {SERVER_URL}/stop_profile")
        return
    r.raise_for_status()
    print(f"[{prefix}] recording up to {args.iterations} engine steps; stopping in {args.seconds:.0f} s", flush=True)

    try:
        time.sleep(args.seconds)
    finally:  # stop even on Ctrl+C; a second start would get 409 until this happens
        print("stopping; the server is writing the trace files (can take minutes)...", flush=True)
        t0 = time.time()
        try:
            requests.post(f"{SERVER_URL}/stop_profile", timeout=30 * 60).raise_for_status()
            print(f"done in {time.time() - t0:.0f} s. download: modal volume get vllm-profiles / profiles/")
        except requests.Timeout:
            print("stop call timed out; the server may still be flushing. Check the volume in a few minutes.")


if __name__ == "__main__":
    main()
