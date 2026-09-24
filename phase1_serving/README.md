# Phase 1 — Agents & Serving

Build serving from scratch to understand it — a batched generation loop, then a KV cache,
then continuous batching — then serve the model with vLLM on Modal and benchmark both against
each other. Then put a web-search agent on top: a simple retrieval agent first, then an
orchestrator that spins up content subagents whose concurrent calls load-test the serving
layer. Benchmark on AgentWebBench.
