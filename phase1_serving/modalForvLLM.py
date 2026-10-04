import subprocess
import modal

"""
For benchmarking, edit the vLLM knobs and endpoint in server_config.py (repo root)
(run modal from the repo root so it can be imported), and # bench_vLLM.sh knobs
I'm not sure what request-rate does
We may need N_GPU to be 2 if we start with the unquantized model because it prolly needs 2xH100 to run
quantized version should fit on one

When benchmarking, record each config as well as the throughput, TTFT, TPOT, and ITL

seed is 411 because that was my carpool number in elementary school

Everything should be good to run as soon as I get modal taken care of and up and running
it will return a URL that we plug in and run
"""

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.30.0",
        "huggingface_hub[hf_transfer]>=0.35.0",
        "flashinfer-python>=0.3.1",
        "torch>=2.8.0",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_python_source("server_config") # so the container can import it too
)

# Server settings live in server_config.py (repo root); run modal from the repo root
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
)


hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)


app = modal.App("vllm-inference-for-web-search-agent")

MINUTES = 60 # in seconds
VLLM_PORT = 8000
ROUTING_REGION = "us-east"

# layer 2 tuning knobs. What size/how many gpus, concurrency; controls container/serverless level
@app.server(
    image=vllm_image,
    gpu=f"{GPU_TYPE}:{N_GPU}", # What size is necessary? Prolly 2xH100 or we use quantized version
    startup_timeout=10*MINUTES, # How long should it take?
    target_concurrency=256, # How many requests modal aims to put in one container before spinning up another; doesn't matter unless max_containers > 1
    max_containers=1,
    min_containers=0, # If 1 or greater, ensure you do modal app stop or ctrl+c if serving
    scaledown_window=5*MINUTES, # How long idle before shutting off; higher is more cost but fewer cold starts
    port=VLLM_PORT,
    routing_region=ROUTING_REGION,
    unauthenticated=True,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
class Server:

    @modal.enter()
    def startup(self) -> None:
        cmd = [
            "vllm",
            "serve",
            MODEL_NAME,
            "--uvicorn-log-level=info",
            "--revision",
            "main",
            "--served-model-name",
            MODEL_NAME,
            "--host",
            "0.0.0.0",
            "--port",
            str(VLLM_PORT)
        ]
        cmd += [
            # --- model-specific ---
            "--trust-remote-code",
            "--dtype", "bfloat16",
            "--reasoning-parser", "k2_horizon",
            "--enable-auto-tool-choice",
            "--tool-call-parser", "k2_horizon",
        ]

        cmd += [
            # --- Layer 1 tuning knobs ---
            "--gpu-memory-utilization", str(GPU_MEM_UTIL),
            "--max-model-len", str(MAX_MODEL_LEN),
            "--max-num-seqs", str(MAX_NUM_SEQS),
            "--max-num-batched-tokens", str(MAX_NUM_BATCHED_TOKENS),
            "--kv-cache-dtype", KV_CACHE_DTYPE,
        ]

        cmd += ["--enforce-eager" if FAST_BOOT else "--no-enforce-eager"]
        cmd += ["--tensor-parallel-size", str(N_GPU)] # switch out for --pipeline-parallel-size to put whole layers on different GPUs
        print(cmd)

        self.process = subprocess.Popen(cmd)

    @modal.exit()
    def stop(self) -> None:
        self.process.terminate()
