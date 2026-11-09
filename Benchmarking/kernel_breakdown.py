"""
Break GPU kernel time in a vLLM torch profiler trace down by kernel kind and step type.

vLLM labels every engine step on the GPU as execute_context_<prefill reqs>(<prefill tokens>)_generation_<decode reqs>(..),
so kernels split into decode-only steps and mixed (prefill + decode) steps. The trace is streamed line by line,
since one 20 s capture is ~1.7-2.6 GB of JSON.

Usage, from the repo root (one rank's trace is enough; TP ranks are near-identical):
    python -m Benchmarking.kernel_breakdown profiles/rank0.<id>.pt.trace.json.gz --label T
    python -m Benchmarking.kernel_breakdown profiles/rank0.<id>.pt.trace.json.gz --label T --top 16
--top N also lists the N biggest kernels inside decode-only steps (ms per step, calls per step, us per call).
"""
import argparse
import bisect
import gzip
import re
import sys
from collections import defaultdict

CATS = [  # first match wins; flashinfer:: comm/sampling kernels must not fall into "attention"
    ("all-reduce/comm", re.compile(r"allreduce|all_reduce|nccl|allgather", re.I)),
    ("sampling", re.compile(r"sampl|softmax|topk|argmax", re.I)),
    ("attention", re.compile(r"flash::|flashattn|fa3|paged|attn", re.I)),
    ("gemm (weights)", re.compile(r"nvjet|gemm|cutlass|splitkreduce|cublas|xmma", re.I)),
    ("rms_norm", re.compile(r"rms_norm", re.I)),
    ("kv cache write", re.compile(r"reshape_and_cache", re.I)),
    ("elementwise (triton fused)", re.compile(r"triton_", re.I)),
]
STEP = re.compile(r"execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)")


def cat(name):
    for c, rx in CATS:
        if rx.search(name):
            return c
    return "other"


def parse(path):
    kernels, steps = [], []
    cur = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            s = line.strip()
            if s.startswith('"cat":'):
                cur = {"cat": s[8:-2]}
            elif s.startswith('"name":') and cur:
                cur["name"] = s[9:-2]
            elif s.startswith('"ts":') and cur:
                cur["ts"] = float(s[6:].rstrip(","))
            elif s.startswith('"dur":') and cur:
                cur["dur"] = float(s[7:].rstrip(","))
                if cur["cat"] == "kernel":
                    kernels.append((cur["ts"], cur["dur"], cur["name"]))
                elif cur["cat"] == "gpu_user_annotation":
                    m = STEP.match(cur.get("name", ""))
                    if m:
                        steps.append((cur["ts"], cur["dur"], int(m[2]), int(m[4])))
                cur = {}
    return kernels, steps


def main(path, label):
    kernels, steps = parse(path)
    kernels.sort()
    steps.sort()
    t0 = kernels[0][0]
    t1 = max(t + d for t, d, _ in kernels)
    busy = sum(d for _, d, _ in kernels)
    print(f"\n=== {label}: {path.split('/')[-1]}")
    print(f"window {(t1 - t0) / 1e6:.1f} s, kernel time {busy / 1e6:.1f} s ({busy / (t1 - t0):.0%} of window), "
          f"{len(kernels)} kernels, {len(steps)} engine steps")

    # attribute each kernel to the step that contains it
    starts = [s[0] for s in steps]
    by = {"decode-only": defaultdict(float), "mixed (prefill+decode)": defaultdict(float), "outside steps": defaultdict(float)}
    for t, d, n in kernels:
        i = bisect.bisect_right(starts, t) - 1
        if i >= 0 and t <= steps[i][0] + steps[i][1]:
            kind = "decode-only" if steps[i][2] == 0 else "mixed (prefill+decode)"
        else:
            kind = "outside steps"
        by[kind][cat(n)] += d

    dec = [s for s in steps if s[2] == 0]
    mix = [s for s in steps if s[2] > 0]
    gen = sorted(s[3] for s in dec)
    print(f"decode-only steps: {len(dec)}, mean {sum(s[1] for s in dec) / max(len(dec), 1) / 1e3:.1f} ms, "
          f"batch p50 {gen[len(gen) // 2] if gen else '-'} (min {gen[0] if gen else '-'}, max {gen[-1] if gen else '-'})")
    if mix:
        print(f"mixed steps:       {len(mix)}, mean {sum(s[1] for s in mix) / len(mix) / 1e3:.1f} ms, "
              f"prefill tokens/step mean {sum(s[2] for s in mix) / len(mix):.0f}")
    step_time = sum(s[1] for s in steps)
    print(f"share of step time: decode-only {sum(s[1] for s in dec) / step_time:.0%}, mixed {sum(s[1] for s in mix) / step_time:.0%}")

    for kind, d in by.items():
        tot = sum(d.values())
        if tot < 0.005 * busy:
            continue
        print(f"\n  {kind}: {tot / 1e6:.2f} s of kernel time ({tot / busy:.0%})")
        for c, v in sorted(d.items(), key=lambda x: -x[1]):
            print(f"    {c:<28} {v / tot:6.1%}  {v / 1e6:7.2f} s")


def top_kernels(path, n):
    kernels, steps = parse(path)
    kernels.sort()
    dec = sorted(s for s in steps if s[2] == 0)
    starts = [s[0] for s in dec]
    tot, cnt = defaultdict(float), defaultdict(int)
    for t, d, name in kernels:
        i = bisect.bisect_right(starts, t) - 1
        if i >= 0 and t <= dec[i][0] + dec[i][1]:
            tot[name] += d
            cnt[name] += 1
    print(f"\n  biggest kernels in the {len(dec)} decode-only steps:")
    for name, v in sorted(tot.items(), key=lambda x: -x[1])[:n]:
        print(f"    {cat(name):<27} {v / len(dec) / 1e3:6.3f} ms/step {cnt[name] / len(dec):6.1f} calls/step "
              f"{v / cnt[name]:7.2f} us/call  {name[:80]}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trace", help="a rank's .pt.trace.json.gz")
    p.add_argument("--label", default="", help="name to print, e.g. T or L")
    p.add_argument("--top", type=int, default=0, help="also list the N biggest decode-step kernels")
    args = p.parse_args()
    main(args.trace, args.label)
    if args.top:
        top_kernels(args.trace, args.top)
