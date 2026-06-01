#!/usr/bin/env python3
"""Wilson score 95% CI for per-task TSR/CAR + LaTeX table + CSV output."""

import argparse
import csv
import json
import math
import os
import sys

DATA = {
    "vanilla_50r": {
        "s0": {"T0": {"TSR": 1.0, "CAR": 0.98}, "T1": {"TSR": 0.94, "CAR": 0.96}, "T2": {"TSR": 0.98, "CAR": 0.92}, "T3": {"TSR": 1.0, "CAR": 0.48}, "T4": {"TSR": 0.80, "CAR": 1.0}, "T5": {"TSR": 0.44, "CAR": 0.08}, "T6": {"TSR": 1.0, "CAR": 0.52}, "T7": {"TSR": 0.94, "CAR": 0.94}, "T8": {"TSR": 0.94, "CAR": 1.0}, "T9": {"TSR": 0.96, "CAR": 1.0}},
        "s1": {"T0": {"TSR": 1.0, "CAR": 0.98}, "T1": {"TSR": 0.94, "CAR": 0.90}, "T2": {"TSR": 0.94, "CAR": 0.90}, "T3": {"TSR": 1.0, "CAR": 0.26}, "T4": {"TSR": 0.92, "CAR": 0.98}, "T5": {"TSR": 0.46, "CAR": 0.06}, "T6": {"TSR": 1.0, "CAR": 0.58}, "T7": {"TSR": 0.94, "CAR": 1.0}, "T8": {"TSR": 1.0, "CAR": 1.0}, "T9": {"TSR": 0.98, "CAR": 1.0}},
        "s2": {"T0": {"TSR": 1.0, "CAR": 0.98}, "T1": {"TSR": 0.98, "CAR": 0.90}, "T2": {"TSR": 1.0, "CAR": 0.94}, "T3": {"TSR": 0.96, "CAR": 0.40}, "T4": {"TSR": 0.98, "CAR": 0.96}, "T5": {"TSR": 0.34, "CAR": 0.08}, "T6": {"TSR": 1.0, "CAR": 0.62}, "T7": {"TSR": 0.94, "CAR": 0.98}, "T8": {"TSR": 0.88, "CAR": 1.0}, "T9": {"TSR": 0.96, "CAR": 1.0}}
    },
    "occ_47500_50r": {
        "s0": {"T0": {"TSR": 0.98, "CAR": 0.98}, "T1": {"TSR": 0.86, "CAR": 0.86}, "T2": {"TSR": 0.88, "CAR": 0.88}, "T3": {"TSR": 0.94, "CAR": 0.20}, "T4": {"TSR": 0.78, "CAR": 0.98}, "T5": {"TSR": 0.60, "CAR": 0.26}, "T6": {"TSR": 0.96, "CAR": 0.36}, "T7": {"TSR": 0.98, "CAR": 0.98}, "T8": {"TSR": 0.96, "CAR": 1.0}, "T9": {"TSR": 0.98, "CAR": 1.0}},
        "s2": {"T0": {"TSR": 1.0, "CAR": 0.98}, "T1": {"TSR": 0.94, "CAR": 1.0}, "T2": {"TSR": 1.0, "CAR": 0.94}, "T3": {"TSR": 0.96, "CAR": 0.30}, "T4": {"TSR": 0.94, "CAR": 1.0}, "T5": {"TSR": 0.32, "CAR": 0.14}, "T6": {"TSR": 0.96, "CAR": 0.52}, "T7": {"TSR": 0.96, "CAR": 1.0}, "T8": {"TSR": 0.92, "CAR": 1.0}, "T9": {"TSR": 1.0, "CAR": 1.0}}
    },
    "occ_50k_50r": {
        "s0": {"T0": {"TSR": 1.0, "CAR": 0.98}, "T1": {"TSR": 0.96, "CAR": 0.82}, "T2": {"TSR": 0.96, "CAR": 0.94}, "T3": {"TSR": 0.96, "CAR": 0.80}, "T4": {"TSR": 0.90, "CAR": 0.98}, "T5": {"TSR": 0.38, "CAR": 0.20}, "T6": {"TSR": 0.98, "CAR": 0.62}, "T7": {"TSR": 0.96, "CAR": 0.98}, "T8": {"TSR": 0.94, "CAR": 1.0}, "T9": {"TSR": 1.0, "CAR": 1.0}},
        "s2": {"T0": {"TSR": 1.0, "CAR": 0.98}, "T1": {"TSR": 0.92, "CAR": 1.0}, "T2": {"TSR": 1.0, "CAR": 0.94}, "T3": {"TSR": 0.98, "CAR": 0.02}, "T4": {"TSR": 0.36, "CAR": 0.98}, "T5": {"TSR": 0.30, "CAR": 0.16}, "T6": {"TSR": 0.96, "CAR": 0.62}, "T7": {"TSR": 0.96, "CAR": 1.0}, "T8": {"TSR": 0.94, "CAR": 1.0}, "T9": {"TSR": 0.98, "CAR": 1.0}}
    }
}

TASKS = [f"T{i}" for i in range(10)]
COND_LABELS = {
    "vanilla_50r": "Vanilla",
    "occ_47500_50r": r"OccSafe@47.5K",
    "occ_50k_50r": r"OccSafe@50K"
}
TRIALS_PER_SEED = 50


def wilson_ci(successes, trials, z=1.96):
    if trials == 0:
        return (0.0, 0.0)
    p_hat = successes / trials
    denom = 1 + z ** 2 / trials
    center = (p_hat + z ** 2 / (2 * trials)) / denom
    margin = z * math.sqrt((p_hat * (1 - p_hat) + z ** 2 / (4 * trials)) / trials) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def load_data(extra_path=None):
    data = dict(DATA)
    if extra_path:
        with open(extra_path) as f:
            extra = json.load(f)
        for cond, seeds in extra.items():
            if cond not in data:
                data[cond] = {}
            for seed, tasks in seeds.items():
                data[cond][seed] = tasks
    return data


def compute_all(data):
    """Compute per-seed and pooled Wilson CIs.

    Returns list of dicts: condition, task, metric, mean, pooled_lo, pooled_hi,
    per_seed_details (list of (seed, val, lo, hi)).
    """
    rows = []
    for cond in sorted(data.keys()):
        seeds = sorted(data[cond].keys())
        n_seeds = len(seeds)
        for task in TASKS:
            for metric in ["TSR", "CAR"]:
                per_seed = []
                total_successes = 0
                total_trials = 0
                for s in seeds:
                    if task not in data[cond][s]:
                        continue
                    val = data[cond][s][task][metric]
                    succ = round(val * TRIALS_PER_SEED)
                    lo, hi = wilson_ci(succ, TRIALS_PER_SEED)
                    per_seed.append((s, val, lo, hi))
                    total_successes += succ
                    total_trials += TRIALS_PER_SEED

                pooled_lo, pooled_hi = wilson_ci(total_successes, total_trials)
                pooled_mean = total_successes / total_trials if total_trials > 0 else 0

                rows.append({
                    "condition": cond,
                    "task": task,
                    "metric": metric,
                    "n_seeds": n_seeds,
                    "pooled_mean": pooled_mean,
                    "pooled_lo": pooled_lo,
                    "pooled_hi": pooled_hi,
                    "per_seed": per_seed
                })
    return rows


def write_csv(rows, outpath):
    with open(outpath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "task", "metric", "n_seeds",
                          "pooled_mean", "pooled_ci_lo", "pooled_ci_hi"])
        for r in rows:
            writer.writerow([
                r["condition"], r["task"], r["metric"], r["n_seeds"],
                f"{r['pooled_mean']:.4f}",
                f"{r['pooled_lo']:.4f}",
                f"{r['pooled_hi']:.4f}"
            ])
    print(f"CSV saved: {outpath}")


def write_latex(rows, outpath):
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Per-task TSR and CAR with pooled Wilson 95\% CI}")
    lines.append(r"\label{tab:per_task_ci}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{ll" + "cc" * len(TASKS) + "}")
    lines.append(r"\toprule")

    header = r"Condition & Metric"
    for t in TASKS:
        header += f" & {t}"
    header += r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    conditions = sorted(set(r["condition"] for r in rows))
    for cond in conditions:
        label = COND_LABELS.get(cond, cond)
        cond_rows = [r for r in rows if r["condition"] == cond]
        for metric in ["TSR", "CAR"]:
            metric_rows = [r for r in cond_rows if r["metric"] == metric]
            metric_rows.sort(key=lambda r: TASKS.index(r["task"]))

            if metric == "TSR":
                line = f"\\multirow{{2}}{{*}}{{{label}}} & {metric}"
            else:
                line = f" & {metric}"

            for r in metric_rows:
                m = r["pooled_mean"]
                lo = r["pooled_lo"]
                hi = r["pooled_hi"]
                cell = f"{m:.2f} [{lo:.2f},{hi:.2f}]"
                line += f" & \\footnotesize{{{cell}}}"

            line += r" \\"
            lines.append(line)
        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table}")

    with open(outpath, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"LaTeX saved: {outpath}")


def print_summary(rows):
    print("\n=== Pooled Wilson 95% CI Summary ===\n")
    conditions = sorted(set(r["condition"] for r in rows))
    for cond in conditions:
        label = COND_LABELS.get(cond, cond)
        print(f"--- {label} ({cond}) ---")
        cond_rows = [r for r in rows if r["condition"] == cond]
        for metric in ["TSR", "CAR"]:
            metric_rows = sorted(
                [r for r in cond_rows if r["metric"] == metric],
                key=lambda r: TASKS.index(r["task"])
            )
            vals = []
            for r in metric_rows:
                vals.append(f"{r['task']}: {r['pooled_mean']:.3f} [{r['pooled_lo']:.3f}, {r['pooled_hi']:.3f}]")
            print(f"  {metric}: " + " | ".join(vals))
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extra-data", type=str, default=None,
                        help="Path to JSON with additional condition data")
    parser.add_argument("--outdir", type=str,
                        default=os.path.dirname(os.path.abspath(__file__)),
                        help="Output directory")
    args = parser.parse_args()

    data = load_data(args.extra_data)
    rows = compute_all(data)

    csv_path = os.path.join(args.outdir, "per_task_ci.csv")
    tex_path = os.path.join(args.outdir, "per_task_ci.tex")

    write_csv(rows, csv_path)
    write_latex(rows, tex_path)
    print_summary(rows)


if __name__ == "__main__":
    main()
