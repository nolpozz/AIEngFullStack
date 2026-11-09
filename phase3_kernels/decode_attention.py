"""
Kernel T: paged GQA decode attention with split-KV (flash-decoding), in Triton.
Objective: throughput (TODOChecklist F). Operating point: S=36, B=4096, A=32, H100 x2 (TP=2).

---------------------------------------------------------------------------------------------------
Why this kernel (profile T_A32, 2026-11-03 11:47, rank 0; see phase3_kernels/README.md)
---------------------------------------------------------------------------------------------------
  Decode-only steps: 400 in 20 s, 25.5 ms each, batch p50 31.  Mixed (prefill) steps: 48 x 201 ms.
  GPU time in decode-only steps (per step):
      decode attention (FA3, FlashAttnFwdSm90)   11.26 ms   64 calls  176 us/call   45%   <- this kernel
      + FlashAttnFwdCombine (split-KV reduce)     0.24 ms   64 calls  3.8 us/call
      weight GEMMs (nvjet)                       11.82 ms                            46%   ~89% of HBM peak already
      all-reduce, RMSNorm, elementwise, KV write  ~2.1 ms                             8%
  Roofline for one call (one layer):
      bytes read  = total KV tokens x 2 (K,V) x 4 KV heads x 128 x 2 B = tokens x 2048 B
      at A=32     ~ 227k tokens (KV usage 74% of 307,200) -> ~465 MB per call
      floor       = 465 MB / 3.35 TB/s = 139 us        FA3 today: 176 us = ~79% of peak
  Target: <= 155 us/call (~90% of peak) -> ~1.4 ms off each decode-only step (~5.5% of that step).
  Decode steps are ~51% of T's GPU time, and mixed steps also run decode attention, so expect a few % on
  tasks/min end to end. The KV token count is from the scraper, +-10%.

---------------------------------------------------------------------------------------------------
What it computes (per GPU; TP=2 splits heads)
---------------------------------------------------------------------------------------------------
  Model: L=64 layers, QH=64 / KVH=8 heads total -> 32 query heads and 4 KV heads per GPU, HD=128,
         GQA group = 32/4 = 8 query heads share one KV head. bf16 everywhere, softmax in fp32.
  Inputs (vLLM's flash layout, block_size 16):
      q            [b, 32, 128]                       one new token per sequence
      k_cache      [num_blocks, 16, 4, 128]           kv_cache[0] in vLLM
      v_cache      [num_blocks, 16, 4, 128]           kv_cache[1]
      block_table  [b, max_blocks] int32              logical block i of seq s lives at block_table[s, i]
      seq_lens     [b] int32                          tokens in each sequence, including the new one
                                                      (vLLM writes the new K/V to the cache before attention)
      scale        1 / sqrt(128)
  Output:
      out          [b, 32, 128]   out[s, h] = softmax(scale * q[s,h] . K[s, kvh(h)]^T) @ V[s, kvh(h)],  kvh(h) = h // 8
  No causal mask is needed: the single decode query attends to every token in its sequence.

---------------------------------------------------------------------------------------------------
Design (the parts that make it fast)
---------------------------------------------------------------------------------------------------
  1. One program per (sequence, KV head, split), not per query head: load each KV block ONCE and use it
     for all 8 query heads of the group. Per-query-head programs read the same KV 8 times.
     The 8 heads form the M dimension of tl.dot; tl.dot needs M >= 16, so pad to 16 (mask the extra rows).
  2. Split-KV: b x 4 programs is 124 at A=32 and only 32 at A=8, under the 132 SMs, and one program
     walking a 7k-token sequence alone can't pull full bandwidth. Split each sequence's blocks into
     NUM_SPLITS chunks; each program writes a partial output and its log-sum-exp; a second small kernel
     combines the splits. (FA3's FlashAttnFwdCombine in the profile is exactly this second step.)
  3. Online softmax inside each split (running max m, running sum l, rescaled accumulator), fp32.
  4. Gather K/V through block_table: for each logical block, read its physical id, then load a
     [16, 128] tile of K and of V for this KV head. Process several blocks per iteration (BLOCK_N = 64
     or 128 tokens) so each load is big enough.
  5. CUDA graphs: vLLM replays captured graphs, so the launch must not sync with the host or depend on
     host-side values. Pick NUM_SPLITS from the batch size / max_seq_len known at capture, keep it fixed,
     and size the partial buffers for the max.

---------------------------------------------------------------------------------------------------
Plan (fill in the TODOs in this order)
---------------------------------------------------------------------------------------------------
  1. make_inputs + reference_attention: a slow, obviously-correct torch version. The test oracle.
  2. split kernel + combine kernel + decode_attention wrapper. Test against the reference
     (bf16: atol/rtol ~1e-2) on small shapes, then on the A32 and A8 shapes, with uneven seq_lens.
  3. Benchmark against vLLM's FA3 decode path with triton.testing.do_bench; report us/call and the
     achieved TB/s = bytes / time (bytes as above). Sweep NUM_SPLITS and BLOCK_N.
  4. Integrate into vLLM (an attention backend, or patching the decode call), re-run A=32, and compare with
     tracing/sweep_2026-11-03 (python -m Benchmarking.curve / analyze) and a new profile
     (python -m Benchmarking.kernel_breakdown).
  Needs an H100: run on Modal (gpu="H100!"), not on the Mac.

Shapes to test (per GPU):
  A32: b=31, total ~227k KV tokens (mean ~7.3k per sequence)   the operating point this kernel is tuned for
  A8:  b=8,  total ~55k KV tokens  (mean ~6.8k)                light load: vLLM runs every batch size through
                                                                this kernel, so it must not be slower there
  Real contexts are skewed (p50 ~3k, p90 ~12k, max ~40k tokens): draw uneven seq_lens, not all equal.
"""

import argparse

import torch
import triton
import triton.language as tl

NUM_Q_HEADS = 32   # per GPU at TP=2
NUM_KV_HEADS = 4   # per GPU at TP=2
HEAD_DIM = 128
GROUP = NUM_Q_HEADS // NUM_KV_HEADS  # 8 query heads per KV head
BLOCK_SIZE = 16    # vLLM KV block size (tokens)
SCALE = HEAD_DIM ** -0.5

SHAPES = {  # name: (batch, mean tokens per sequence)
    "A32": (31, 7300),  # the T operating point
    "A8": (8, 6800),    # light load: check it isn't slower than FA3 here
}


# ---------- inputs and reference ----------

def make_inputs(batch, mean_len, device="cuda", dtype=torch.bfloat16, seed=0):
    """Random paged KV cache in vLLM's layout, with uneven sequence lengths.

    Returns q, k_cache, v_cache, block_table, seq_lens.
    """
    # TODO: draw seq_lens (e.g. lognormal around mean_len, clipped to >= 1), sum the blocks each one needs,
    #       allocate k_cache/v_cache [num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM], and give each sequence
    #       a shuffled set of physical block ids in block_table (shuffled, so the gather is really tested).
    raise NotImplementedError


def reference_attention(q, k_cache, v_cache, block_table, seq_lens, scale=SCALE):
    """Slow torch version: per sequence, gather its K/V from the blocks, repeat the KV heads for GQA,
    softmax in fp32. Returns out [b, NUM_Q_HEADS, HEAD_DIM] in q's dtype."""
    # TODO
    raise NotImplementedError


# ---------- Triton kernels ----------

@triton.jit
def _split_kernel(
    # TODO: pointers (q, k_cache, v_cache, block_table, seq_lens, partial_out, partial_lse), strides,
    #       scale, and constexprs: GROUP, HEAD_DIM, BLOCK_SIZE, BLOCK_N, NUM_SPLITS, PAD_M (=16)
):
    # program ids: sequence, KV head, split
    # TODO:
    #   - this split's token range [start, end) within the sequence (skip if empty)
    #   - load the GROUP query heads of this KV head as a [PAD_M, HEAD_DIM] tile (mask rows >= GROUP)
    #   - loop over the range in BLOCK_N-token chunks:
    #       physical blocks from block_table -> K tile [BLOCK_N, HEAD_DIM], V tile [BLOCK_N, HEAD_DIM]
    #       scores = tl.dot(q, tl.trans(k)) * scale, masked past `end`
    #       online softmax update of m, l, acc (fp32)
    #   - write acc / l to partial_out[seq, head, split] and m + log(l) to partial_lse[seq, head, split]
    pass


@triton.jit
def _combine_kernel(
    # TODO: pointers (partial_out, partial_lse, out), strides, seq_lens, constexprs HEAD_DIM, NUM_SPLITS, ...
):
    # program ids: sequence, query head
    # TODO: weights w_i = exp(lse_i - max lse) over the splits that had tokens; out = sum w_i * out_i / sum w_i
    pass


def decode_attention(q, k_cache, v_cache, block_table, seq_lens, num_splits=8, block_n=64, scale=SCALE):
    """Launch the split and combine kernels. Returns out [b, NUM_Q_HEADS, HEAD_DIM]."""
    # TODO: allocate partial_out [b, NUM_Q_HEADS, num_splits, HEAD_DIM] fp32 and partial_lse [b, NUM_Q_HEADS, num_splits]
    #       fp32, launch _split_kernel on grid (b, NUM_KV_HEADS, num_splits), then _combine_kernel on (b, NUM_Q_HEADS).
    raise NotImplementedError


# ---------- test and benchmark ----------

def kv_bytes(seq_lens):
    """Bytes one call must read: every token's K and V for the 4 local KV heads."""
    return int(seq_lens.sum()) * 2 * NUM_KV_HEADS * HEAD_DIM * 2


def test():
    # TODO: for small shapes and both SHAPES, compare decode_attention with reference_attention
    #       (torch.testing.assert_close, atol=rtol=1e-2). Include a sequence shorter than one block and
    #       one that isn't a multiple of BLOCK_N.
    raise NotImplementedError


def bench():
    # TODO: for each of SHAPES, time decode_attention (and vLLM's FA3 path as the baseline) with
    #       triton.testing.do_bench; print us/call, achieved TB/s = kv_bytes / time, and % of 3.35 TB/s.
    #       Today's FA3 numbers from the profiles: A32 176 us/call (~79%), A8 49 us/call (~68%).
    raise NotImplementedError


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["test", "bench"])
    args = p.parse_args()
    test() if args.mode == "test" else bench()


if __name__ == "__main__":
    main()
