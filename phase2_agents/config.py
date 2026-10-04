# phase2_agents/config.py
# Single place for every setting that defines "the workload" and "the server".
# CONFIG_ID (a hash of all of it) is stamped on every trace event so you can
# tell sweep points apart later.

import hashlib
import json
import os

# --- model / endpoint / server knobs: shared with the server via server_config.py ---
from server_config import (
    FAST_BOOT,
    GPU_MEM_UTIL,
    GPU_TYPE,
    KV_CACHE_DTYPE,
    MAX_MODEL_LEN,
    MAX_NUM_BATCHED_TOKENS,
    MAX_NUM_SEQS,
    MODEL_NAME,
    N_GPU,
    SERVER_URL,
)

BASE_URL = SERVER_URL + "/v1"  # OpenAI client wants the /v1 suffix

# --- workload (keep fixed across a sweep) ---
TEMPERATURE = 1.0  # from the model card
TOP_P = 0.95  # from the model card
REASONING_EFFORT = "high"  # from the model card
MAX_TOKENS = 4096  # per model call; reasoning tokens count against this
MAX_STEPS = 6  # model calls per agent run
BRAVE_MAX_URLS = 3
BRAVE_MAX_TOKENS = 6000  # cap on the Brave result size

# --- server settings that get hashed into CONFIG_ID (values come from server_config.py) ---
SERVER = {
    "gpu": f"{GPU_TYPE}:{N_GPU}",
    "gpu_memory_utilization": GPU_MEM_UTIL,
    "max_model_len": MAX_MODEL_LEN,
    "max_num_seqs": MAX_NUM_SEQS,
    "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
    "kv_cache_dtype": KV_CACHE_DTYPE,
    "fast_boot": FAST_BOOT,  # enforce-eager vs CUDA graphs changes performance
}

# --- sweep point: how many agents run at once (set by the benchmark harness) ---
CONCURRENCY = int(os.environ.get("CONCURRENCY", "1"))

CONFIG = {
    "model": MODEL_NAME,
    "temperature": TEMPERATURE,
    "top_p": TOP_P,
    "reasoning_effort": REASONING_EFFORT,
    "max_tokens": MAX_TOKENS,
    "max_steps": MAX_STEPS,
    "brave_max_urls": BRAVE_MAX_URLS,
    "brave_max_tokens": BRAVE_MAX_TOKENS,
    "server": SERVER,
    "concurrency": CONCURRENCY,
}

CONFIG_ID = hashlib.sha1(
    json.dumps(CONFIG, sort_keys=True).encode()
).hexdigest()[:8]
