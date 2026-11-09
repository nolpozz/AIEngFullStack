"""
Throughput curve for one sweep: tasks/min, p90 and KV usage against agent count (A), with the knee.

By default it takes the most recent sweep: the newest results file plus every run before it with the
same config apart from concurrency (same server settings, GPU and workload). Pass tags to pick runs.
Every number comes from analyze.compute(), so it matches the analyze.py tables for the same --trim.

Usage, from the repo root:
    python -m Benchmarking.curve
    python -m Benchmarking.curve --trim 60 --p90-limit 97 --out tracing/curve.png
    python -m Benchmarking.curve 20261103_085654 20261103_092323 ...

The knee follows the throughput objective (TODOChecklist F):
    feasible  p90 task time <= --p90-limit and no server errors (5xx / connection) in the run.
              max_steps and 4xx errors (context overflow) are workload limits and don't disqualify.
    best      the highest tasks/min among feasible points.
    knee      the lowest feasible A within --tolerance (5%) of best. This is the operating point,
              and the value target_concurrency should get if the server ever scales past one container.
Repeated A values are averaged for the curve and the knee; each repeat is still drawn as its own dot.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # file output only; no window
import matplotlib.pyplot as plt

from Benchmarking.analyze import TRACING, compute, discover, load_run, read_jsonl

INK, MUTED, GRID = "#1f2328", "#6e7781", "#d8dee4"
SERIES, KNEE, LIMIT = "#2f6fdb", "#c2410c", "#8c959f"


# ---------- picking the sweep ----------

def sweep_key(tag, folder):
    """Everything in the run's config except concurrency, so points of one sweep compare equal."""
    trace = folder / f"traces_{tag}.jsonl"
    if not trace.exists():
        return None
    with open(trace) as f:
        for line in f:
            if '"run_start"' in line:
                cfg = dict(json.loads(line)["config"])
                cfg.pop("concurrency", None)
                return json.dumps(cfg, sort_keys=True)
    return None


def latest_sweep(folder):
    tags = discover(folder)  # oldest first
    if not tags:
        return []
    key = sweep_key(tags[-1], folder)
    picked = []
    for tag in reversed(tags):
        if sweep_key(tag, folder) != key:
            break  # stop at the first run from a different config
        picked.append(tag)
    return picked[::-1]


# ---------- per-point numbers ----------

def num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def server_errors(results):
    """Failed tasks the server caused. 4xx (the request itself was bad, e.g. context overflow) and
    max_steps are the workload's doing."""
    return sum(1 for r in results["results"]
               if r["status"] == "error" and "Error code: 4" not in (r["error"] or ""))


def point(tag, folder, samples, trim):
    run = load_run(tag, folder, samples)
    sections, _ = compute(run, trim)
    rows = {k: v for _, rs in sections for k, v in rs}
    return {
        "tag": tag,
        "A": run["results"]["concurrency"],
        "tpm": num(rows["tasks / min (ok, over window)"]),
        "p50": num(rows["task time p50 (s)"]),
        "p90": num(rows["task time p90 (s)"]),
        "running": num(rows["requests running (mean)"]),
        "kv": num(rows["KV cache usage (mean %)"]),
        "preempt": num(rows["preemptions"]),
        "ttft": num(rows["TTFT, first token p50 (ms)"]),
        "server_errors": server_errors(run["results"]),
        "other_failures": rows["failed tasks"],
        "S": rows["max_num_seqs"],
        "B": rows["max_num_batched_tokens"],
        "gpu": rows["gpu"],
    }


def by_concurrency(points):
    """One row per A: means over repeats; any server error in any repeat disqualifies that A."""
    groups = defaultdict(list)
    for p in points:
        groups[p["A"]].append(p)
    out = []
    for A in sorted(groups):
        ps = groups[A]

        def avg(k):
            vals = [p[k] for p in ps if p[k] is not None]
            return sum(vals) / len(vals) if vals else None

        out.append({"A": A, "n": len(ps), "tpm": avg("tpm"), "p90": avg("p90"), "p50": avg("p50"),
                    "running": avg("running"), "kv": avg("kv"), "preempt": avg("preempt"), "ttft": avg("ttft"),
                    "server_errors": sum(p["server_errors"] for p in ps)})
    return out


def find_knee(rows, p90_limit, tolerance):
    for r in rows:
        r["feasible"] = r["tpm"] is not None and r["p90"] is not None and r["p90"] <= p90_limit and r["server_errors"] == 0
    feasible = [r for r in rows if r["feasible"]]
    if not feasible:
        return None, None
    best = max(feasible, key=lambda r: r["tpm"])
    knee = min((r for r in feasible if r["tpm"] >= (1 - tolerance) * best["tpm"]), key=lambda r: r["A"])
    return best, knee


# ---------- output ----------

def print_table(rows, best, knee, p90_limit):
    f = lambda v, s=".1f": "-" if v is None else f"{v:{s}}"
    print(f"\n{'A':>4} {'runs':>4} {'tasks/min':>9} {'p50 s':>7} {'p90 s':>7} {'TTFT p50':>9} {'running':>8} {'KV %':>6} {'preempt':>8} {'srv err':>7}  ")
    for r in rows:
        mark = "  <- knee" if knee and r is knee else ("  <- best" if best and r is best else "")
        if not r["feasible"]:
            mark += "  (p90 over limit)" if r["p90"] is not None and r["p90"] > p90_limit else "  (server errors)"
        print(f"{r['A']:>4} {r['n']:>4} {f(r['tpm'], '.2f'):>9} {f(r['p50']):>7} {f(r['p90']):>7} {f(r['ttft'], '.0f'):>8}ms "
              f"{f(r['running']):>8} {f(r['kv']):>6} {f(r['preempt'], '.0f'):>8} {r['server_errors']:>7}{mark}")


def style(ax, ylabel):
    ax.set_ylabel(ylabel, color=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)


def plot(points, rows, best, knee, p90_limit, tolerance, title, out):
    xs = [r["A"] for r in rows]
    fig, axes = plt.subplots(3, 1, figsize=(8, 9.5), sharex=True, gridspec_kw={"height_ratios": [3, 2, 2]})

    # tasks/min: the objective
    ax = axes[0]
    ax.scatter([p["A"] for p in points], [p["tpm"] for p in points], s=22, color=SERIES, alpha=0.35, zorder=2,
               label="each run")
    ax.plot(xs, [r["tpm"] for r in rows], color=SERIES, linewidth=2, marker="o", markersize=7, zorder=3,
            label="mean per A")
    for r in rows:
        if not r["feasible"]:
            ax.scatter([r["A"]], [r["tpm"]], s=70, facecolor="white", edgecolor=SERIES, linewidth=1.5, zorder=4)
    if best:
        lo = (1 - tolerance) * best["tpm"]
        ax.axhspan(lo, best["tpm"], color=KNEE, alpha=0.08, zorder=1, label=f"within {tolerance:.0%} of best")
    if knee:
        ax.axvline(knee["A"], color=KNEE, linewidth=1.2, linestyle="--", zorder=1)
        ax.annotate(f"knee A={knee['A']}\n{knee['tpm']:.1f} tasks/min", (knee["A"], knee["tpm"]),
                    xytext=(10, -34), textcoords="offset points", color=INK, fontsize=9)
    style(ax, "tasks / min")
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=MUTED)

    # p90: the constraint
    ax = axes[1]
    ax.plot(xs, [r["p90"] for r in rows], color=SERIES, linewidth=2, marker="o", markersize=7, zorder=3)
    ax.axhline(p90_limit, color=LIMIT, linewidth=1.2, linestyle=":")
    ax.annotate(f"limit {p90_limit:.0f} s", (xs[0], p90_limit), xytext=(0, 4), textcoords="offset points",
                color=MUTED, fontsize=8)
    if knee:
        ax.axvline(knee["A"], color=KNEE, linewidth=1.2, linestyle="--", zorder=1)
    style(ax, "task time p90 (s)")

    # KV usage, with preemptions written at each point: why the curve bends
    ax = axes[2]
    ax.plot(xs, [r["kv"] for r in rows], color=SERIES, linewidth=2, marker="o", markersize=7, zorder=3)
    for r in rows:
        if r["kv"] is not None and r["preempt"]:
            ax.annotate(f"{r['preempt']:.0f} pre", (r["A"], r["kv"]), xytext=(0, 7), textcoords="offset points",
                        ha="center", color=MUTED, fontsize=8)
    if knee:
        ax.axvline(knee["A"], color=KNEE, linewidth=1.2, linestyle="--", zorder=1)
    ax.set_ylim(0, 105)
    style(ax, "KV cache usage (mean %)")
    ax.set_xlabel("agents (A)", color=INK)
    ax.set_xticks(xs)

    fig.text(0.01, 0.005, "open dots: infeasible (p90 over limit or server errors).  'pre' = preemptions in the window.",
             color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out, dpi=150)
    print(f"\nsaved {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tags", nargs="*", help="run tags (default: the most recent sweep)")
    p.add_argument("--trim", type=float, default=60.0, help="seconds to drop from both ends of each run")
    p.add_argument("--p90-limit", type=float, default=97.0, help="p90 task time ceiling in s (3x the baseline A=1 p90)")
    p.add_argument("--tolerance", type=float, default=0.05, help="knee: lowest A within this fraction of the best")
    p.add_argument("--dir", type=Path, default=TRACING, help="folder with results_/traces_ files")
    p.add_argument("--out", type=Path, help="PNG path (default: <dir>/curve_<last tag>.png)")
    args = p.parse_args()

    tags = args.tags or latest_sweep(args.dir)
    if not tags:
        print("no runs found")
        return
    metrics_path = args.dir / "vllm_metrics.jsonl"
    samples = []
    if metrics_path.exists():
        samples = [{"timestamp": r["timestamp"], "metrics": r["metrics"]} for r in read_jsonl(metrics_path)]
        samples.sort(key=lambda r: r["timestamp"])

    points = [point(t, args.dir, samples, args.trim) for t in tags]
    rows = by_concurrency(points)
    best, knee = find_knee(rows, args.p90_limit, args.tolerance)

    first = points[0]
    title = f"S={first['S']}  B={first['B']}  {first['gpu']}   ({len(points)} runs, trim {args.trim:.0f} s)"
    print(title)
    print("runs:", " ".join(tags))
    print_table(rows, best, knee, args.p90_limit)
    a1 = next((r for r in rows if r["A"] == 1), None)
    if a1 and a1["p90"]:
        print(f"\nthis sweep's A=1 p90 is {a1['p90']:.1f} s (3x = {3 * a1['p90']:.0f} s); limit used: {args.p90_limit:.0f} s")
    if knee:
        print(f"best: A={best['A']} at {best['tpm']:.2f} tasks/min")
        print(f"knee: A={knee['A']} at {knee['tpm']:.2f} tasks/min, p90 {knee['p90']:.1f} s  "
              f"(lowest A within {args.tolerance:.0%} of best) -> operating point / target_concurrency")
    else:
        print("no feasible point: every A is over the p90 limit or has server errors")

    out = args.out or args.dir / f"curve_{tags[-1]}.png"
    plot(points, rows, best, knee, args.p90_limit, args.tolerance, title, out)


if __name__ == "__main__":
    main()
