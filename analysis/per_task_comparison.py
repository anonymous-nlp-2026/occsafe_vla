#!/usr/bin/env python3
import argparse
import json
from itertools import combinations
from pathlib import Path
import numpy as np
from eval_utils import (
    TASK_SHORT, pool_binary, bootstrap_ci, bootstrap_diff_test,
    get_task_ids, parse_condition, setup_style,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", nargs="+", required=True,
                        help="label:file1,file2,...")
    parser.add_argument("--output_dir", default="./output/")
    parser.add_argument("--n_bootstrap", type=int, default=10000)
    args = parser.parse_args()

    plt = setup_style()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    conditions = [parse_condition(c) for c in args.conditions]
    all_paths = [p for _, paths in conditions for p in paths]
    task_ids = get_task_ids(all_paths)
    labels = [l for l, _ in conditions]
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(labels), 10)))[:len(labels)]

    results = {}
    for label, paths in conditions:
        results[label] = {}
        for tid in task_ids:
            safe = pool_binary(paths, tid, "safe_binary")
            succ = pool_binary(paths, tid, "success_binary")
            tsr_m, tsr_lo, tsr_hi = bootstrap_ci(succ, args.n_bootstrap)
            car_m, car_lo, car_hi = bootstrap_ci(safe, args.n_bootstrap)
            results[label][tid] = dict(
                tsr_mean=tsr_m, tsr_lo=tsr_lo, tsr_hi=tsr_hi,
                car_mean=car_m, car_lo=car_lo, car_hi=car_hi,
            )
        print(f"  [{label}] computed {len(task_ids)} tasks")

    def plot_grouped_bar(metric, ylabel, filename):
        fig, ax = plt.subplots(figsize=(14, 5))
        n_conds = len(labels)
        width = 0.8 / n_conds
        x = np.arange(len(task_ids))
        for i, label in enumerate(labels):
            means = [results[label][t][f"{metric}_mean"] for t in task_ids]
            lo = [results[label][t][f"{metric}_lo"] for t in task_ids]
            hi = [results[label][t][f"{metric}_hi"] for t in task_ids]
            yerr = [[m - l for m, l in zip(means, lo)],
                    [h - m for h, m in zip(hi, means)]]
            offset = (i - (n_conds - 1) / 2) * width
            ax.bar(x + offset, means, width, yerr=yerr, label=label,
                   color=colors[i], capsize=3, error_kw={"linewidth": 0.8})
        ax.set_xticks(x)
        ax.set_xticklabels([TASK_SHORT[t] for t in task_ids])
        ax.set_ylabel(ylabel)
        ax.set_title(f"Per-Task {ylabel}")
        ax.legend(frameon=False)
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            kw = {"dpi": 300} if ext == "png" else {}
            fig.savefig(outdir / f"{filename}.{ext}", bbox_inches="tight", **kw)
        plt.close(fig)
        print(f"  Saved {filename}.pdf/png")

    plot_grouped_bar("tsr", "TSR", "per_task_tsr")
    plot_grouped_bar("car", "CAR", "per_task_car")

    hard = [t for t in task_ids
            if any(results[l][t]["car_mean"] < 0.5 for l in labels)]
    if hard:
        fig, ax = plt.subplots(figsize=(max(6, len(hard) * 1.5), 5))
        n_conds = len(labels)
        width = 0.8 / n_conds
        x = np.arange(len(hard))
        for i, label in enumerate(labels):
            means = [results[label][t]["car_mean"] for t in hard]
            lo = [results[label][t]["car_lo"] for t in hard]
            hi = [results[label][t]["car_hi"] for t in hard]
            yerr = [[m - l for m, l in zip(means, lo)],
                    [h - m for h, m in zip(hi, means)]]
            offset = (i - (n_conds - 1) / 2) * width
            ax.bar(x + offset, means, width, yerr=yerr, label=label,
                   color=colors[i], capsize=3, error_kw={"linewidth": 0.8})
        ax.set_xticks(x)
        ax.set_xticklabels([TASK_SHORT[t] for t in hard])
        ax.set_ylabel("CAR")
        ax.set_title("Hard Tasks (CAR < 50%)")
        ax.axhline(0.5, color="gray", ls="--", lw=0.5)
        ax.legend(frameon=False)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            kw = {"dpi": 300} if ext == "png" else {}
            fig.savefig(outdir / f"hard_tasks.{ext}", bbox_inches="tight", **kw)
        plt.close(fig)
        print("  Saved hard_tasks.pdf/png")
    else:
        print("  No hard tasks (all CAR >= 50%)")

    stats = {}
    for tid in task_ids:
        tk = TASK_SHORT[tid]
        stats[tk] = {}
        for l1, l2 in combinations(labels, 2):
            paths1 = [p for l, ps in conditions if l == l1 for p in ps]
            paths2 = [p for l, ps in conditions if l == l2 for p in ps]
            succ1 = pool_binary(paths1, tid, "success_binary")
            succ2 = pool_binary(paths2, tid, "success_binary")
            td, tlo, thi, tp = bootstrap_diff_test(succ1, succ2, args.n_bootstrap)
            safe1 = pool_binary(paths1, tid, "safe_binary")
            safe2 = pool_binary(paths2, tid, "safe_binary")
            cd, clo, chi, cp = bootstrap_diff_test(safe1, safe2, args.n_bootstrap)
            stats[tk][f"{l1}_vs_{l2}"] = dict(
                tsr_diff=round(td, 4), tsr_ci=[round(tlo, 4), round(thi, 4)],
                tsr_p=round(tp, 4),
                car_diff=round(cd, 4), car_ci=[round(clo, 4), round(chi, 4)],
                car_p=round(cp, 4),
            )
    with open(outdir / "comparison_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("  Saved comparison_stats.json")


if __name__ == "__main__":
    main()
