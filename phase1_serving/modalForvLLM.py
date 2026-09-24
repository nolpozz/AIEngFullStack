import subprocess
import modal

"""
For benchmarking, edit:
# -- Tuning -- knobs
# --- Layer 1 tuning knobs ---
and # bench_vLLM.sh knobs
I'm not sure what request-rate does
We may ned N_GPU to be 2 if we start with the unquantized model because it prolly needs 2xH100 to run
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
        "vllm==0.30.0,",
        "huggingface_hub[hf_transfer]==0.35.0",
        "flashinfer-python==0.3.1",
        "torch==2.8.0",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

MODEL_NAME = "IFM/K2-Horizon-32B"



hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)


# --Tuning--
GPU_MEM_UTIL = "0.90"
MAX_MODEL_LEN = "32768"
MAX_NUM_SEQS = "256"
MAX_NUM_BATCHED_TOKENS = "8192"
KV_CACHE_DTYPE = "auto"

# -- set to False when benchmarking --
FAST_BOOT = True # When False, vLLM will run JIT kernel optimization and CUDA graph capture

app = modal.App("vllm-inference-for-web-search-agent")

N_GPU = 1
MINUTES = 60 # in seconds
VLLM_PORT = 8000
ROUTING_REGION = "us-east"

# layer 2 tuning knobs. What size/how many gpus, concurrency
@app.server(
    image=vllm_image,
    gpu=f"H100:{N_GPU}", # What size is necessary? Prolly 2xH100 or we use quantized version
    scaledown_window=900,
    startup_timeout=10*MINUTES, # How long should it take?
    target_concurrency=32, # what does this change?
    scaledown_window=1*MINUTES, # How long idle before shutting off; higher is more cost but fewer cold starts
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
            "--uvicorn-log-level=info",
            MODEL_NAME,
            "--revision", # don't need I think
            "main"
            "--served-model-name",
            MODEL_NAME,
            "--host",
            "0.0.0.0",
            "--port",
            str(VLLM_PORT)
        ]
        cmd = [
            # --- model-specific ---
            "--trust-remote-code",
            "--dtype", "bfloat16",
            "--reasoning-parser", "k2_horizon",
            "--enable-auto-tool-choice",
            "--tool-call-parser", "k2_horizon",
        ]

        cmd += [
            # --- Layer 1 tuning knobs ---
            "--gpu-memory-utilization", GPU_MEM_UTIL,
            "--max-model-len", MAX_MODEL_LEN,
            "--max-num-seqs", MAX_NUM_SEQS,
            "--max-num-batched-tokens", MAX_NUM_BATCHED_TOKENS,
            "--kv-cache-dtype", KV_CACHE_DTYPE,
        ]

        cmd += ["--enforce-eager" if FAST_BOOT else "--no-enforce-eager"]
        cmd += ["--tensor-parallel-size", str(N_GPU)]
        print(cmd)

        self.process = subprocess.Popen(cmd)

    @modal.exit()
    def stop(self) -> None:
        self.process.terminate()
