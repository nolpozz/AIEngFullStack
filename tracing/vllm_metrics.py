# tracing/vllm_metrics.py
# Polls vLLM's Prometheus /metrics endpoint and appends one JSON line per poll.
# Run from the repo root, alongside your agents, with the same CONCURRENCY env var:
#   CONCURRENCY=8 .venv/bin/python -m tracing.vllm_metrics
# Stop with Ctrl+C.

import json
import sys
import time
from pathlib import Path

import requests

from phase2_agents.config import BASE_URL, CONFIG_ID

METRICS_URL = BASE_URL.removesuffix("/v1") + "/metrics"
OUT_FILE = Path(__file__).parent / "vllm_metrics.jsonl"
INTERVAL_S = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0


def scrape() -> dict:
    text = requests.get(METRICS_URL, timeout=5).text
    sample = {}
    for line in text.splitlines():
        # sample line looks like: vllm:num_requests_running{engine="0",model_name="..."} 3.0
        if line.startswith("vllm:") and "_bucket" not in line: # skip histogram buckets (huge)
            key, _, value = line.rpartition(" ")
            sample[key] = float(value)
    return sample


while True:
    try:
        event = {
            "timestamp": time.time(),
            "config_id": CONFIG_ID,
            "metrics": scrape(),
        }
        with OUT_FILE.open("a") as f:
            f.write(json.dumps(event) + "\n")
    except requests.RequestException as e:
        print(f"scrape failed: {e}") # server cold/restarting; keep polling
    time.sleep(INTERVAL_S)
