#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import numpy as np
from eval_utils import (
    TASK_SHORT, pool_binary, bootstrap_ci, get_task_ids, parse_condition, setup_style,
)

CASE_COLORS = {"A": "#3498db", "B": "#2ecc71", "C": "#e74c3c", "D": "#9b59b6"}
CASE_LABELS = {
    "A": "Gradient interference",
    "B": "Spatial signal useless",
    "C": "Deeper issue",
    "D": "Mixed",
}
THRESHOLD = 0.05


def classify_case(delta_detach, delta_random, vanilla_car, full_occ_car):
    if delta_detach < -THRESHOLD:
        return "A"
    if abs(delta_random) < THRESHOLD:
        return "B"
    if full_occ_car < vanilla_car - THRESHOLD:
        return "C"
    return "D"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vanilla", required=True)
    parser.add_argument("--full_occ", required=True)
    parser.add_argument("--detach", required=True)
    parser.add_argument("--random", required=True)
    parser.add_argument("--output_dir", default="./output/")
    parser.add_argument("--n_bootstrap", type=int, default=10000)
    args = parser.parse_args()

    plt = setup_style()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    conds = {}
    for name, raw in [("vanilla", args.vanilla), ("full_occ", args.full_occ),
                       ("detach", args.detach), ("random", args.random)]:
        label, paths = parse_condition(raw)
        conds[name] = {"label": label, "paths": paths}

    all_paths = [p for c in conds.values() for p in c["paths"]]
    task_ids = get_task_ids(all_paths)
    cond_names = ["vanilla", "full_occ", "detach", "random"]
    cond_labels = [conds[n]["label"] for n in cond_names]
    colors = ["#95a5a6", "#e74c3c", "#3498db", "#f39c12"]

    results = {}
    for cn in cond_names:
        results[cn] = {}
        for tid in task_ids:
            safe = pool_binary(conds[cn]["paths"], tid, "safe_binary")
            succ = pool_binary(conds[cn]["paths"], tid, "success_binary")
            cm, clo, chi = bootstrap_ci(safe, args.n_bootstrap)
            tm, tlo, thi = bootstrap_ci(succ, args.n_bootstrap)
            results[cn][tid] = dict(
                car_mean=cm, car_lo=clo, car_hi=chi,
                tsr_mean=tm, tsr_lo=tlo, tsr_hi=thi,
            )
        print(f"  [{conds[cn]['label']}] computed {len(task_ids)} tasks")

    def plot_ablation_bar(metric, ylabel, filename):
        fig, ax = plt.subplots(figsize=(14, 5))
        n_c = len(cond_names)
        width = 0.8 / n_c
        x = np.arange(len(task_ids))
        for i, cn in enumerate(cond_names):
            means = [results[cn][t][f"{metric}_mean"] for t in task_ids]
            lo = [results[cn][t][f"{metric}_lo"] for t in task_ids]
            hi = [results[cn][t][f"{metric}_hi"] for t in task_ids]
            yerr = [[m - l for m, l in zip(means, lo)],
                    [h - m for h, m in zip(hi, means)]]
            offset = (i - (n_c - 1) / 2) * width
            ax.bar(x + offset, means, width, yerr=yerr, label=cond_labels[i],
                   color=colors[i], capsize=3, error_kw={"linewidth": 0.8})
        ax.set_xticks(x)
        ax.set_xticklabels([TASK_SHORT[t] for t in task_ids])
        ax.set_ylabel(ylabel)
        ax.set_title(f"Ablation Per-Task {ylabel}")
        ax.legend(frameon=False, ncol=2)
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            kw = {"dpi": 300} if ext == "png" else {}
            fig.savefig(outdir / f"{filename}.{ext}", bbox_inches="tight", **kw)
        plt.close(fig)
        print(f"  Saved {filename}.pdf/png")

    plot_ablation_bar("car", "CAR", "ablation_car_comparison")
    plot_ablation_bar("tsr", "TSR", "ablation_tsr_comparison")

    # Diagnostic scatter
    deltas_detach, deltas_random, case_list = [], [], []
    report = {}
    case_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    for tid in task_ids:
        fc = results["full_occ"][tid]["car_mean"]
        dc = results["detach"][tid]["car_mean"]
        rc = results["random"][tid]["car_mean"]
        vc = results["vanilla"][tid]["car_mean"]
        dd = fc - dc
        dr = fc - rc
        case = classify_case(dd, dr, vc, fc)
        deltas_detach.append(dd)
        deltas_random.append(dr)
        case_list.append(case)
        case_counts[case] += 1
        report[TASK_SHORT[tid]] = dict(
            full_occ_car=round(fc, 4), detach_car=round(dc, 4),
            random_car=round(rc, 4), vanilla_car=round(vc, 4),
            delta_detach=round(dd, 4), delta_random=round(dr, 4),
            case=case,
        )

    fig, ax = plt.subplots(figsize=(8, 8))
    for cc in ["A", "B", "C", "D"]:
        mask = [c == cc for c in case_list]
        if any(mask):
            xs = [deltas_detach[i] for i, m in enumerate(mask) if m]
            ys = [deltas_random[i] for i, m in enumerate(mask) if m]
            lbls = [TASK_SHORT[task_ids[i]] for i, m in enumerate(mask) if m]
            ax.scatter(xs, ys, c=CASE_COLORS[cc], s=120, zorder=3,
                       label=f"Case {cc}: {CASE_LABELS[cc]}")
            for xv, yv, lb in zip(xs, ys, lbls):
                ax.annotate(lb, (xv, yv), textcoords="offset points",
                            xytext=(6, 6), fontsize=9)
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel("Detach Effect (full_occ CAR - detach CAR)")
    ax.set_ylabel("Random Effect (full_occ CAR - random CAR)")
    ax.set_title("Ablation Diagnostic Matrix")
    ax.legend(loc="best", frameon=False, fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        kw = {"dpi": 300} if ext == "png" else {}
        fig.savefig(outdir / f"diagnostic_matrix.{ext}", bbox_inches="tight", **kw)
    plt.close(fig)
    print("  Saved diagnostic_matrix.pdf/png")

    majority = max(case_counts, key=case_counts.get)
    diag_output = {
        "per_task": report,
        "case_counts": case_counts,
        "majority_case": majority,
        "diagnosis": f"Majority of tasks ({case_counts[majority]}/{len(task_ids)}) fall into Case {majority}: {CASE_LABELS[majority]}",
    }
    with open(outdir / "diagnostic_report.json", "w") as f:
        json.dump(diag_output, f, indent=2)
    print("  Saved diagnostic_report.json")


if __name__ == "__main__":
    main()
