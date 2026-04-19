"""obj-026 T13: Per-seed SA robustness for obj-024.

Re-analyze obj-024 checkpoint for seed-level variance patterns. Questions:
  • Within each (sensory_dim, embed_dim) cell, is variance driven by a few
    outlier seeds?
  • Are the "high SA" results in obj-024's best cells robust across all 5
    seeds, or driven by 1-2 lucky seeds?
  • What is the max/min ratio within each cell — a concrete robustness stat.

Reviewer scrutiny of multi-seed experiments will ask exactly this.
"""
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    with open("results/sensory_capacity_checkpoint.json") as f:
        data = json.load(f)

    cells = defaultdict(list)
    for r in data:
        cells[(r["sensory_dim"], r["embedding_dim"])].append((r["seed"], r["SA"]))

    sensory_dims = sorted(set(k[0] for k in cells))
    embed_dims = sorted(set(k[1] for k in cells))

    print(f"Analyzing {len(cells)} cells...")
    robustness_stats = {}
    n_outlier_cells = 0
    for (sd, ed), seed_sa_pairs in sorted(cells.items()):
        vals = np.array([v for (_, v) in seed_sa_pairs])
        mean = vals.mean()
        std = vals.std(ddof=1) if len(vals) > 1 else 0.0
        cv = std / (abs(mean) + 1e-6) if abs(mean) > 0.01 else float("nan")
        # Check: does removing the max or min change the mean > 50%?
        if len(vals) >= 3:
            mean_no_max = vals[np.argsort(vals)[:-1]].mean()
            mean_no_min = vals[np.argsort(vals)[1:]].mean()
            outlier = (abs(mean_no_max - mean) > abs(mean) * 0.5 and abs(mean) > 0.01) or \
                      (abs(mean_no_min - mean) > abs(mean) * 0.5 and abs(mean) > 0.01)
        else:
            outlier = False
        if outlier:
            n_outlier_cells += 1
        robustness_stats[(sd, ed)] = {
            "mean": float(mean), "std": float(std), "cv": float(cv) if not np.isnan(cv) else None,
            "min": float(vals.min()), "max": float(vals.max()),
            "max_min_ratio": float(vals.max() / vals.min()) if vals.min() > 0.01 else None,
            "outlier_driven": bool(outlier),
        }

    print(f"Cells where removing max/min changes mean >50%: {n_outlier_cells}/{len(cells)}")

    # Figure: dot plot of per-seed SA per cell
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("obj-026 T13: Per-seed SA Robustness for obj-024\n"
                 "Each dot = one seed; red line = cell mean",
                 fontsize=13, fontweight="bold")

    ax = axes[0]
    pos = 0
    xtick_pos, xtick_labels = [], []
    colors = {2: "#d62728", 4: "#ff7f0e", 8: "#2ca02c", 16: "#1f77b4"}
    for sd in sensory_dims:
        for ed in embed_dims:
            if (sd, ed) in cells:
                seed_sa = cells[(sd, ed)]
                vals = [v for (_, v) in seed_sa]
                ax.scatter([pos] * len(vals), vals, color=colors[sd], s=40, alpha=0.7, zorder=3)
                ax.hlines(np.mean(vals), pos - 0.3, pos + 0.3, color="red", linewidth=2, zorder=4)
                xtick_pos.append(pos)
                xtick_labels.append(f"{sd}/{ed}")
                pos += 1
        pos += 1  # Gap between sensory dims

    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_labels, rotation=45, fontsize=7)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("SA (per seed)")
    ax.set_xlabel("sensory / embed cell (grouped by sensory)")
    ax.set_title("A. Per-seed SA across all cells")
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8, label=f"sensory={s}")
               for s, c in colors.items()]
    ax.legend(handles=handles, loc="upper left", fontsize=9)

    # CV heatmap
    ax = axes[1]
    cv_grid = np.zeros((len(sensory_dims), len(embed_dims)))
    for i, sd in enumerate(sensory_dims):
        for j, ed in enumerate(embed_dims):
            vals = np.array([v for (_, v) in cells[(sd, ed)]])
            cv_grid[i, j] = vals.std(ddof=1) / (abs(vals.mean()) + 1e-3)
    im = ax.imshow(cv_grid, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=3)
    ax.set_xticks(range(len(embed_dims)))
    ax.set_xticklabels(embed_dims)
    ax.set_yticks(range(len(sensory_dims)))
    ax.set_yticklabels(sensory_dims)
    ax.set_xlabel("embedding_dim")
    ax.set_ylabel("sensory_dim")
    ax.set_title("B. Coefficient of Variation (std / |mean|)")
    plt.colorbar(im, ax=ax, label="CV (green=stable, red=noisy)")
    for i in range(len(sensory_dims)):
        for j in range(len(embed_dims)):
            ax.text(j, i, f"{cv_grid[i,j]:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if cv_grid[i,j] > 2.0 else "black")

    plt.tight_layout()
    out = Path("results/obj026_seed_robustness.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")

    # Save numerics
    out_json = Path("results/obj026_seed_robustness.json")
    with out_json.open("w") as f:
        serializable = {f"sensory={k[0]}_embed={k[1]}": v for k, v in robustness_stats.items()}
        json.dump({
            "cell_stats": serializable,
            "n_outlier_driven_cells": n_outlier_cells,
            "total_cells": len(cells),
        }, f, indent=2)
    print(f"Saved: {out_json}")

    # Verdict
    # Best cell (16, 16) deep inspection
    sa16_16 = [v for (_, v) in cells[(16, 16)]]
    print(f"\nBest cell (sensory=16, embed=16): seeds={sa16_16}")
    print(f"  All 5 seeds positive? {all(v > 0 for v in sa16_16)}")
    print(f"  Min seed SA: {min(sa16_16):.3f} (should be well above 0 if robust)")
    print(f"  Max/min ratio: {max(sa16_16)/max(min(sa16_16), 0.01):.2f}")


if __name__ == "__main__":
    main()
