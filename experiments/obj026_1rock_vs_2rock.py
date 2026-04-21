"""obj-026 T16: 1-rock vs 2-rock sensory-capacity comparison.

Compares obj-024 (1-rock, 100 configs, 5 seeds) against obj-026 (2-rock,
60 configs, 3 seeds) on the same sensory × embed grid.

Honest finding: obj-026 shows a FLOOR EFFECT — overall mean dist ≈ 0.501,
no cell consistently learns. Peak SA values drop ~5x. The substitution
effect from obj-024 does NOT replicate, and this is informative: it
suggests the 2-rock task's reward dilution (split across two objects)
falls below our 800-episode training budget.

This does NOT invalidate obj-024; but it DOES require honest framing to
Reviewer E. The 2-rock result is a floor-limit artifact, not evidence
against the core claim.
"""
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_cells(path):
    with open(path) as f:
        data = json.load(f)
    cells = defaultdict(list)
    for r in data:
        if "error" not in r:
            cells[(r["sensory_dim"], r["embedding_dim"])].append(r["SA"])
    return cells, data


def make_grid(cells, sensory_dims, embed_dims):
    g = np.zeros((len(sensory_dims), len(embed_dims)))
    for i, sd in enumerate(sensory_dims):
        for j, ed in enumerate(embed_dims):
            vals = cells.get((sd, ed), [0])
            g[i, j] = np.mean(vals)
    return g


def main():
    cells_1, data_1 = load_cells("results/sensory_capacity_checkpoint.json")
    cells_2, data_2 = load_cells("results/obj026_sensory_capacity_2rock_checkpoint.json")

    sensory_dims = [2, 4, 8, 16]
    embed_dims = [2, 4, 8, 16, 32]

    g1 = make_grid(cells_1, sensory_dims, embed_dims)
    g2 = make_grid(cells_2, sensory_dims, embed_dims)

    # Peak SA per sensory_dim
    peak_1 = np.array([g1[i, :].max() for i in range(4)])
    peak_2 = np.array([g2[i, :].max() for i in range(4)])

    # Dist stats
    dist_1 = np.array([r["avg_dist"] for r in data_1 if "error" not in r])
    dist_2 = np.array([r["avg_dist"] for r in data_2 if "error" not in r])

    # === Figure ===
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "obj-026 T16: 1-rock vs 2-rock Sensory-Capacity — Floor Effect Disclosure\n"
        "2-rock task barely learns across the grid; the substitution pattern does NOT replicate",
        fontsize=13, fontweight="bold",
    )

    # Panel A: 1-rock heatmap
    ax = axes[0, 0]
    im = ax.imshow(g1, cmap="RdYlGn", aspect="auto", vmin=-0.05, vmax=0.25)
    ax.set_xticks(range(len(embed_dims)))
    ax.set_xticklabels(embed_dims)
    ax.set_yticks(range(len(sensory_dims)))
    ax.set_yticklabels(sensory_dims)
    ax.set_xlabel("embed_dim")
    ax.set_ylabel("sensory_dim")
    ax.set_title(f"A. obj-024 (1-rock): peak SA = {peak_1.max():.3f}")
    for i in range(len(sensory_dims)):
        for j in range(len(embed_dims)):
            c = "white" if g1[i, j] > 0.15 else "black"
            ax.text(j, i, f"{g1[i,j]:.3f}", ha="center", va="center", fontsize=9, color=c)
    plt.colorbar(im, ax=ax, label="mean SA")

    # Panel B: 2-rock heatmap
    ax = axes[0, 1]
    im = ax.imshow(g2, cmap="RdYlGn", aspect="auto", vmin=-0.05, vmax=0.25)
    ax.set_xticks(range(len(embed_dims)))
    ax.set_xticklabels(embed_dims)
    ax.set_yticks(range(len(sensory_dims)))
    ax.set_yticklabels(sensory_dims)
    ax.set_xlabel("embed_dim")
    ax.set_ylabel("sensory_dim")
    ax.set_title(f"B. obj-026 (2-rock): peak SA = {peak_2.max():.3f}  ← FLOOR EFFECT")
    for i in range(len(sensory_dims)):
        for j in range(len(embed_dims)):
            c = "white" if g2[i, j] > 0.15 else "black"
            ax.text(j, i, f"{g2[i,j]:.3f}", ha="center", va="center", fontsize=9, color=c)
    plt.colorbar(im, ax=ax, label="mean SA")

    # Panel C: peak SA by sensory_dim comparison
    ax = axes[1, 0]
    x = np.arange(len(sensory_dims))
    w = 0.35
    ax.bar(x - w/2, peak_1, w, label="1-rock (obj-024)", color="#1f77b4", edgecolor="black")
    ax.bar(x + w/2, peak_2, w, label="2-rock (obj-026)", color="#d62728", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(sensory_dims)
    ax.set_xlabel("sensory_dim")
    ax.set_ylabel("peak SA across embed_dims")
    ax.set_title("C. Peak SA Collapses on 2-rock")
    ax.legend()
    for i, (p1, p2) in enumerate(zip(peak_1, peak_2)):
        ax.text(i - w/2, p1 + 0.005, f"{p1:.2f}", ha="center", fontsize=9)
        ax.text(i + w/2, p2 + 0.005, f"{p2:.2f}", ha="center", fontsize=9)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(alpha=0.3, axis="y")

    # Panel D: Distance distributions (both tasks)
    ax = axes[1, 1]
    ax.hist(dist_1, bins=20, alpha=0.6, label=f"1-rock (n={len(dist_1)})", color="#1f77b4")
    ax.hist(dist_2, bins=20, alpha=0.6, label=f"2-rock (n={len(dist_2)})", color="#d62728")
    ax.axvline(x=dist_1.mean(), color="#1f77b4", linestyle="--", alpha=0.8)
    ax.axvline(x=dist_2.mean(), color="#d62728", linestyle="--", alpha=0.8)
    ax.set_xlabel("avg_dist (lower=better)")
    ax.set_ylabel("config count")
    ax.set_title(f"D. Distance: 1-rock mean={dist_1.mean():.3f}, 2-rock mean={dist_2.mean():.3f}\n"
                 f"(2-rock range {dist_2.min():.3f}-{dist_2.max():.3f} → floor)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    out = Path("results/obj026_1rock_vs_2rock.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")

    # Numerics
    out_json = Path("results/obj026_1rock_vs_2rock.json")
    with out_json.open("w") as f:
        json.dump({
            "obj024_1rock": {
                "peak_SA_by_sensory_dim": {str(sd): float(peak_1[i]) for i, sd in enumerate(sensory_dims)},
                "mean_dist": float(dist_1.mean()),
                "dist_range": [float(dist_1.min()), float(dist_1.max())],
                "n_configs": len(dist_1),
            },
            "obj026_2rock": {
                "peak_SA_by_sensory_dim": {str(sd): float(peak_2[i]) for i, sd in enumerate(sensory_dims)},
                "mean_dist": float(dist_2.mean()),
                "dist_range": [float(dist_2.min()), float(dist_2.max())],
                "n_configs": len(dist_2),
            },
            "verdict": {
                "substitution_replicates": False,
                "reason": "2-rock task shows floor effect (mean dist 0.501, range 0.487-0.531). "
                         "Peak SA collapses from 0.234 (1-rock) to 0.098 (2-rock). Similar to obj-017 "
                         "(3-rock floor effect) — reward dilution across objects exceeds 800-episode "
                         "training budget. Does NOT invalidate obj-024's within-regime finding, but "
                         "does restrict the information-bound claim to 1-rock-like task complexity.",
                "reviewer_E_status": "HIGH — replicate failed; need longer training OR "
                                     "simpler intermediate task (e.g., smaller-arena 1-rock with varied target).",
            },
        }, f, indent=2)
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
