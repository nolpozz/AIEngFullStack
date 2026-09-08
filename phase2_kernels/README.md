# Phase 2 — Kernels

Profile the Phase 1 serving run to find the bottleneck, write a custom Triton attention
kernel, microbenchmark it against a naive baseline and FlashAttention, then plug it into vLLM
and re-run the Phase 1 load test to see the end-to-end effect.
