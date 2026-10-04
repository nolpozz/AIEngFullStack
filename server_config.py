# server_config.py
# Single source of truth for the vLLM server settings and endpoint.
# Imported by phase1_serving/modalForvLLM.py (launches the server),
# Benchmarking/benchmarking.py (synthetic sweep) and phase2_agents/config.py
# (stamps these into every trace via CONFIG_ID). Edit values HERE only.
# Pure Python on purpose: no modal import, so every consumer can load it.

MODEL_NAME = "IFM/K2-Horizon-32B"
SERVER_URL = (
    "https://nolpozz--vllm-inference-for-web-search-agent-server-dev."
    "us-east.modal.direct"
)  # no /v1 suffix; the OpenAI client appends it, vllm bench wants it bare

# -- hardware --
GPU_TYPE = "H100"
N_GPU = 2  # also used as --tensor-parallel-size. Might need 2xH100 unquantized; quantized should fit on one

# -- vLLM tuning knobs --
GPU_MEM_UTIL = 0.90
MAX_MODEL_LEN = 64000
MAX_NUM_SEQS = 135  # KV budget // per-sequence cost (approx 2200 tokens at .25 mb/tok)
MAX_NUM_BATCHED_TOKENS = 8192
KV_CACHE_DTYPE = "auto"

# -- set to False when benchmarking --
FAST_BOOT = False  # When False, vLLM will run JIT kernel optimization and CUDA graph capture
