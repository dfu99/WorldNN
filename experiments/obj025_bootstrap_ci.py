"""
obj-025 T5: Bootstrap CIs and effect sizes for sensory-capacity tradeoff.

Discriminating analysis: is the obj-024 substitution effect statistically
distinguishable from zero?

Primary test: head-to-head on the key pair claimed to demonstrate the
substitution hypothesis:
  A = (sensory=16, embed=2)   — rich input, tiny model
  B = (sensory=2,  embed=32)  — poor input, max model

If hypothesis holds, SA_A > SA_B with non-trivial effect size.

Secondary test: per-cell 95% bootstrap CIs on mean SA.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def bootstrap_mean_ci(x: np.ndarray, n_boot: int = 10000, alpha: float = 0.05, rng=None):
    """Return (mean, lo, hi) 95% bootstrap CI."""
    rng = rng or np.random.default_rng(0)
    x = np.asarray(x, float)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boots = x[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled_sd = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                        / (len(a) + len(b) - 2))
    if pooled_sd < 1e-10:
        return 0.0
    return (a.mean() - b.mean()) / pooled_sd


def bootstrap_diff_ci(a, b, n_boot=10000, alpha=0.05, rng=None):
    """Bootstrap CI on mean_a - mean_b."""
    rng = rng or np.random.default_rng(1)
    a, b = np.asarray(a, float), np.asarray(b, float)
    diffs = np.array([
        a[rng.integers(0, len(a), len(a))].mean() -
        b[rng.integers(0, len(b), len(b))].mean()
        for _ in range(n_boot)
    ])
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return float((a.mean() - b.mean())), float(lo), float(hi), float((diffs > 0).mean())


def main():
    with open("results/sensory_capacity_checkpoint.json") as f:
        data = json.load(f)

    rng = np.random.default_rng(42)

    # Per-cell CIs
    cells = defaultdict(list)
    for r in data:
        cells[(r["sensory_dim"], r["embedding_dim"])].append(r["SA"])

    sensory_dims = sorted(set(k[0] for k in cells))
    embed_dims = sorted(set(k[1] for k in cells))

    print(f"Computing 95% bootstrap CIs for {len(cells)} cells (5 seeds each, n_boot=10000)...")
    mean_grid = np.zeros((len(sensory_dims), len(embed_dims)))
    lo_grid = np.zeros_like(mean_grid)
    hi_grid = np.zeros_like(mean_grid)
    for i, sd in enumerate(sensory_dims):
        for j, ed in enumerate(embed_dims):
            m, lo, hi = bootstrap_mean_ci(cells[(sd, ed)], 10000, 0.05, rng)
            mean_grid[i, j] = m
            lo_grid[i, j] = lo
            hi_grid[i, j] = hi

    # Key head-to-head: (16, 2) vs (2, 32)
    A = np.asarray(cells[(16, 2)])  # rich input, tiny model
    B = np.asarray(cells[(2, 32)])  # poor input, max model
    d = cohens_d(A, B)
    diff, dlo, dhi, p_a_gt_b = bootstrap_diff_ci(A, B, 10000, 0.05, rng)
    print(f"\nKey head-to-head: rich-min vs poor-max")
    print(f"  A (sensory=16, embed=2): mean={A.mean():.4f}, std={A.std(ddof=1):.4f}, n={len(A)}")
    print(f"  B (sensory=2,  embed=32): mean={B.mean():.4f}, std={B.std(ddof=1):.4f}, n={len(B)}")
    print(f"  Mean difference (A - B): {diff:+.4f} [95% CI: {dlo:+.4f}, {dhi:+.4f}]")
    print(f"  Cohen's d: {d:+.3f}")
    print(f"  P(A > B | bootstrap): {p_a_gt_b:.3f}")

    # Compare to peak cell (16, 16) vs random baseline (2, 4)
    C = np.asarray(cells[(16, 16)])
    D = np.asarray(cells[(2, 4)])
    d2 = cohens_d(C, D)
    diff2, dlo2, dhi2, p2 = bootstrap_diff_ci(C, D, 10000, 0.05, rng)
    print(f"\nCeiling vs floor: best cell vs worst cell")
    print(f"  C (sensory=16, embed=16): mean={C.mean():.4f}, std={C.std(ddof=1):.4f}")
    print(f"  D (sensory=2,  embed=4):  mean={D.mean():.4f}, std={D.std(ddof=1):.4f}")
    print(f"  Mean difference (C - D): {diff2:+.4f} [95% CI: {dlo2:+.4f}, {dhi2:+.4f}]")
    print(f"  Cohen's d: {d2:+.3f}")
    print(f"  P(C > D | bootstrap): {p2:.3f}")

    # === Figure ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        f"obj-025 T5: Statistical Rigor for Sensory-Capacity Tradeoff\n"
        f"95% bootstrap CIs (5 seeds/cell, n_boot=10k) and effect sizes",
        fontsize=13, fontweight="bold",
    )

    # Panel A: CIs grouped by sensory_dim
    ax = axes[0]
    colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]
    for i, sd in enumerate(sensory_dims):
        x_pos = np.arange(len(embed_dims)) + i * 0.18 - 0.27
        means = mean_grid[i, :]
        los = lo_grid[i, :]
        his = hi_grid[i, :]
        err_lo = means - los
        err_hi = his - means
        ax.errorbar(x_pos, means, yerr=[err_lo, err_hi], fmt="o-",
                    color=colors[i], label=f"sensory={sd}", capsize=4, linewidth=1.8, markersize=7)
    ax.set_xticks(range(len(embed_dims)))
    ax.set_xticklabels(embed_dims)
    ax.set_xlabel("Embedding Dim")
    ax.set_ylabel("Mean SA (95% bootstrap CI)")
    ax.set_title("A. Per-Cell CIs Across Grid")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    # Panel B: key head-to-head comparisons
    ax = axes[1]
    compare = [
        ("s=16,e=2\n(rich,tiny)", A, "#1f77b4"),
        ("s=2,e=32\n(poor,max)", B, "#d62728"),
        ("s=16,e=16\n(peak)",    C, "#2ca02c"),
        ("s=2,e=4\n(floor)",     D, "#ff7f0e"),
    ]
    xs = np.arange(len(compare))
    for i, (name, arr, col) in enumerate(compare):
        m, lo, hi = bootstrap_mean_ci(arr, 10000, 0.05, rng)
        ax.bar(i, m, yerr=[[m - lo], [hi - m]], color=col, edgecolor="black", capsize=6)
        ax.text(i, m + (hi - m) + 0.008, f"{m:+.3f}", ha="center", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels([c[0] for c in compare], fontsize=9)
    ax.set_ylabel("Mean SA (95% CI)")
    ax.set_title(f"B. Head-to-Head: d(rich-min,poor-max)={d:+.2f}, d(peak,floor)={d2:+.2f}")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    out = Path("results/obj025_bootstrap_ci.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nSaved: {out}")

    # Save numerics
    out_json = Path("results/obj025_bootstrap_ci.json")
    with out_json.open("w") as f:
        json.dump({
            "sensory_dims": sensory_dims,
            "embed_dims": embed_dims,
            "mean_grid": mean_grid.tolist(),
            "CI_lo": lo_grid.tolist(),
            "CI_hi": hi_grid.tolist(),
            "substitution_test_rich_min_vs_poor_max": {
                "A_label": "sensory=16, embed=2",
                "B_label": "sensory=2, embed=32",
                "mean_A": float(A.mean()),
                "mean_B": float(B.mean()),
                "diff": diff,
                "diff_CI_lo": dlo,
                "diff_CI_hi": dhi,
                "cohens_d": d,
                "P_A_gt_B_bootstrap": p_a_gt_b,
            },
            "ceiling_vs_floor": {
                "C_label": "sensory=16, embed=16",
                "D_label": "sensory=2, embed=4",
                "mean_C": float(C.mean()),
                "mean_D": float(D.mean()),
                "diff": diff2,
                "diff_CI_lo": dlo2,
                "diff_CI_hi": dhi2,
                "cohens_d": d2,
                "P_C_gt_D_bootstrap": p2,
            },
        }, f, indent=2)
    print(f"Saved: {out_json}")

    # Verdict
    print("\n=== VERDICT ===")
    if dlo > 0:
        print(f"Substitution effect CI [{dlo:+.4f}, {dhi:+.4f}] excludes zero → significant")
    else:
        print(f"Substitution effect CI [{dlo:+.4f}, {dhi:+.4f}] includes zero → NOT significant at α=0.05")
    print(f"Effect size d={d:+.2f} ({'negligible' if abs(d) < 0.2 else 'small' if abs(d) < 0.5 else 'medium' if abs(d) < 0.8 else 'large'})")


if __name__ == "__main__":
    main()
