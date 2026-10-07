



# Later after single runs and rough numbers
# import subprocess
# import modal
# from server_config import MODEL_NAME, SERVER_URL  # shared with the server; run modal from the repo root
# # from agent import run_agent


# CONCURRENCIES = [1] #, 16, 64, 96, 128]    # the sweep
# SEED = 411

# # need to make sure that the agent, tools and model handler are all on modal

# bench_image = (
#     modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
#     .entrypoint([])
#     .uv_pip_install(
#         "vllm==0.30.0",
#         "pandas",
#         "huggingface_hub[hf_transfer]>=0.35.0",
#     )
#     .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
#     .add_local_python_source("server_config") # so the container can import it too
# )

# app = modal.App("k2-benchmark", image=bench_image)
# results_vol = modal.Volume.from_name("k2-bench-results", create_if_missing=True)

# @app.function(volumes={"/results": results_vol}, timeout=60 * 60)
# def run_sweep():
#     # how do I make sure the query cache, agent, model, and tools script are all on the modal volume?
#     queries = []
#     with open("datasets/search_agent_benchmark_queries.txt") as f:
#         for line in f:
#             queries.append(line)

#     for c in CONCURRENCIES:
#         # start vLLM data collection?
#         for query in queries:
#             # run agent cli call?
#             pass


import time
from concurrent.futures import ThreadPoolExecutor
from phase2_agents.agent import run_agent, MaxStepsExceeded
import random
import json

SEED=411

def load_queries(path="datasets/search_agent_benchmark_queries.txt") -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]

def run_one(query):
    start = time.perf_counter()
    start_ts = time.time() 
    status, error = "ok", None
    try:
        run_agent(query)
    except MaxStepsExceeded as e:
        status, error = "max_steps", str(e)
    except Exception as e:
        status, error = "error", f"{type(e).__name__}: {e}"
    return {
        "question": query,
        "start_ts": start_ts,
        "duration": time.perf_counter() - start,
        "status": status,
        "error": error,
    }

def build_tasks(queries, n_tasks, seed):
    rng = random.Random(seed)
    tasks = []
    while len(tasks) < n_tasks:
        batch = queries[:]
        rng.shuffle(batch)
        tasks.extend(batch)
    return tasks[:n_tasks]

def run_point(queries, concurrency, n_tasks):
    tasks = build_tasks(queries, n_tasks, SEED)   # new list, local RNG
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(run_one, tasks))
    t_end = time.time()
    return {"concurrency": concurrency, "t_start": t_start, "t_end": t_end, "results": results}


if __name__ == "__main__":
    queries = load_queries()
    res = run_point(queries, 1, len(queries))
    with open("tracing/first_run_benchmarks.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2) 