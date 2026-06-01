#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import numpy as np
from eval_utils import (
    TASK_SHORT, seed_level_metric, get_task_ids, setup_style,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vanilla", required=True, help="comma-separated JSON paths")
    parser.add_argument("--occ", required=True, help="comma-separated JSON paths")
    parser.add_argument("--output_dir", default="./output/")
    args = parser.parse_args()

    plt = setup_style()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    van_paths = [p.strip() for p in args.vanilla.split(",") if p.strip()]
    occ_paths = [p.strip() for p in args.occ.split(",") if p.strip()]
    task_ids = get_task_ids(van_paths + occ_paths)

    van_n = len(van_paths)
    occ_n = len(occ_paths)

    van_stds, occ_stds = [], []
    stats = {}
    for tid in task_ids:
        v = seed_level_metric(van_paths, tid, "car")
        o = seed_level_metric(occ_paths, tid, "car")
        vs = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
        os_ = float(np.std(o, ddof=1)) if len(o) > 1 else 0.0
        ratio = os_ / vs if vs > 1e-9 else (float("inf") if os_ > 1e-9 else float("nan"))
        van_stds.append(vs)
        occ_stds.append(os_)

        lev_p = float("nan")
        try:
            from scipy.stats import levene
            if len(v) >= 2 and len(o) >= 2:
                _, lev_p = levene(v, o)
                lev_p = float(lev_p)
        except ImportError:
            pass

        stats[TASK_SHORT[tid]] = dict(
            vanilla_std=round(vs, 4), occ_std=round(os_, 4),
            ratio=round(ratio, 4) if np.isfinite(ratio) else str(ratio),
            levene_p=round(lev_p, 4) if np.isfinite(lev_p) else None,
            vanilla_n=int(len(v)), occ_n=int(len(o)),
        )

    # Heatmap
    heatmap_data = np.array([van_stds, occ_stds]).T
    fig, ax = plt.subplots(figsize=(4, 8))
    im = ax.imshow(heatmap_data, cmap="YlOrRd", aspect="auto", vmin=0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"Vanilla\n(N={van_n})", f"OCC\n(N={occ_n})"])
    ax.set_yticks(range(len(task_ids)))
    ax.set_yticklabels([TASK_SHORT[t] for t in task_ids])
    for i in range(len(task_ids)):
        for j in range(2):
            ax.text(j, i, f"{heatmap_data[i, j]:.3f}", ha="center", va="center",
                    fontsize=10)
    ax.set_title("Cross-Seed CAR Std Dev")
    fig.colorbar(im, ax=ax, shrink=0.6)
    if occ_n <= 2:
        ax.annotate(f"OCC: N={occ_n}, interpret with caution",
                    xy=(0.5, -0.05), xycoords="axes fraction",
                    ha="center", fontsize=9, fontstyle="italic", color="red")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        kw = {"dpi": 300} if ext == "png" else {}
        fig.savefig(outdir / f"seed_variance_heatmap.{ext}", bbox_inches="tight", **kw)
    plt.close(fig)
    print("  Saved seed_variance_heatmap.pdf/png")

    # Variance ratio bar chart
    ratios = [occ_stds[i] / van_stds[i] if van_stds[i] > 1e-9
              else (float("inf") if occ_stds[i] > 1e-9 else float("nan"))
              for i in range(len(task_ids))]
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(task_ids))
    finite_ratios = [r if np.isfinite(r) else 0 for r in ratios]
    bar_colors = ["#e74c3c" if r > 1 else "#2ecc71" for r in finite_ratios]
    ax.bar(x, finite_ratios, color=bar_colors)
    ax.axhline(1.0, color="gray", ls="--", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_SHORT[t] for t in task_ids])
    ax.set_ylabel("OCC Std / Vanilla Std")
    ax.set_title("Variance Ratio (OCC / Vanilla)")
    for i, r in enumerate(ratios):
        if not np.isfinite(r):
            ax.annotate("inf" if r > 0 else "nan", (i, 0.1), ha="center", fontsize=9)
    if occ_n <= 2:
        ax.annotate(f"OCC: N={occ_n}, interpret with caution",
                    xy=(0.5, -0.08), xycoords="axes fraction",
                    ha="center", fontsize=9, fontstyle="italic", color="red")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        kw = {"dpi": 300} if ext == "png" else {}
        fig.savefig(outdir / f"variance_ratio.{ext}", bbox_inches="tight", **kw)
    plt.close(fig)
    print("  Saved variance_ratio.pdf/png")

    with open(outdir / "instability_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("  Saved instability_stats.json")


if __name__ == "__main__":
    main()
