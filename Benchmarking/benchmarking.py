"""
Runs `vllm bench serve` against your already-running K2 server, FROM Modal
(same image, same region) so there are no local dependency problems and no
home-network latency in the numbers.

HOW TO USE:
  1. Make sure your server is up in another terminal:  modal serve modalForvLLM.py
  2. Paste that server's URL into SERVER_URL in server_config.py (repo root).
  3. Run:  modal run bench_modal.py
  4. Watch the stats tables stream back in your terminal.
  5. Download the saved JSONs:  modal volume get k2-bench-results / ./results
"""

import subprocess
import modal

# ==========================================================================
# EDIT THESE  ---  everything you'd normally change lives in this block
# ==========================================================================
from server_config import MODEL_NAME, SERVER_URL  # shared with the server; run modal from the repo root
INPUT_LEN   = 2000                     # tokens per prompt
OUTPUT_LEN   = 200                      # tokens generated per request
CONCURRENCIES = [1, 16, 64, 96, 128]    # the sweep — brackets the ~110 knee / ~135 ceiling
SEED         = 411
# ==========================================================================

# Same image family as your server, so vLLM + the K2 tokenizer load cleanly.
bench_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.30.0",
        "pandas",
        "huggingface_hub[hf_transfer]>=0.35.0",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_python_source("server_config") # so the container can import it too
)

app = modal.App("k2-benchmark", image=bench_image)
results_vol = modal.Volume.from_name("k2-bench-results", create_if_missing=True)


# No gpu= : the benchmark client is CPU-only. Don't pay for an H100 here.
@app.function(volumes={"/results": results_vol}, timeout=60 * 60)
def run_sweep():
    for c in CONCURRENCIES:
        cmd = [
            "vllm", "bench", "serve",
            "--backend", "openai",
            "--base-url", SERVER_URL,
            "--model", MODEL_NAME,
            "--trust-remote-code",
            "--dataset-name", "random",
            "--random-input-len", str(INPUT_LEN),
            "--random-output-len", str(OUTPUT_LEN),
            "--num-prompts", str(c * 4),        # keep the queue full
            "--max-concurrency", str(c),
            "--seed", str(SEED),
            "--save-result",
            "--ignore-eos",
            "--result-filename", f"/results/run_c{c}.json",
        ]
        print(f"\n===== concurrency {c} =====", flush=True)
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
    results_vol.commit()
    print("\nAll runs done. Fetch with:  modal volume get k2-bench-results / ./results")


@app.local_entrypoint()
def main():
    run_sweep.remote()