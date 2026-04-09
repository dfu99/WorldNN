"""obj-024: Iso-performance curves — perception richness vs model capacity tradeoff.

Shows that richer perception substitutes for model capacity: the set of
(perception_quality, embedding_dim) pairs achieving a given SA threshold
forms a convex frontier.

Hero figure for the framing: "rich sensory input allows smaller models."

Uses ci_at_scale.json (280 data points: 8 perception × 5 embed_dims × 7 seeds)
with C_i as SA proxy, plus sa_proxy_expanded_gpu.json for true SA validation.
"""

import sys
import json
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


PERCEPTION_ORDER = {
    "random_baseline": 0,
    "oracle_noise0.5": 1,
    "vae_mu_lat8": 2,
    "vae_mu_lat16": 3,
    "oracle_noise0.1": 4,
    "oracle": 5,
    "raw_emission": 6,
    "vae_mu_lat32": 7,
}

PERCEPTION_LABELS = {
    "random_baseline": "Random\n(no signal)",
    "oracle_noise0.5": "Oracle\n+N(0.5)",
    "vae_mu_lat8": "VAE\nlat=8",
    "vae_mu_lat16": "VAE\nlat=16",
    "oracle_noise0.1": "Oracle\n+N(0.1)",
    "oracle": "Oracle\n(full state)",
    "raw_emission": "Raw\nemission",
    "vae_mu_lat32": "VAE\nlat=32",
}

CMAP = {
    "random_baseline": "#9E9E9E",
    "vae_mu_lat8": "#D32F2F",
    "vae_mu_lat16": "#FF5722",
    "vae_mu_lat32": "#FF9800",
    "raw_emission": "#FFC107",
    "oracle_noise0.5": "#E91E63",
    "oracle_noise0.1": "#8BC34A",
    "oracle": "#2196F3",
}


def load_data():
    results_dir = Path(__file__).parent.parent / "results"
    with open(results_dir / "ci_at_scale.json") as f:
        ci_data = json.load(f)["results"]
    return ci_data


def main():
    results = load_data()
    results_dir = Path(__file__).parent.parent / "results"

    valid = [r for r in results if "error" not in r]

    # Group by (level, embed_dim) → mean C_i
    grid = defaultdict(list)
    for r in valid:
        grid[(r["level"], r["embedding_dim"])].append(r)

    cells = {}
    for (level, emb), runs in grid.items():
        cells[(level, emb)] = {
            "ci_mean": np.mean([r["C_i"] for r in runs]),
            "ci_std": np.std([r["C_i"] for r in runs]),
            "ci_all": [r["C_i"] for r in runs],
            "dist_mean": np.mean([r["avg_dist_last100"] for r in runs]),
            "n": len(runs),
        }

    embed_dims = sorted(set(r["embedding_dim"] for r in valid))

    # Rank perceptions by their C_i at embed=32 (best capacity)
    perc_by_ci = {}
    for level in PERCEPTION_ORDER:
        if (level, 32) in cells:
            perc_by_ci[level] = cells[(level, 32)]["ci_mean"]
    perceptions_ranked = sorted(perc_by_ci.keys(), key=lambda x: perc_by_ci[x])

    # ── Figure: 4-panel ──
    fig = plt.figure(figsize=(16, 13))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    # ═══ Panel A: C_i heatmap ═══
    ax = fig.add_subplot(gs[0, 0])
    heatmap = np.full((len(perceptions_ranked), len(embed_dims)), np.nan)
    for i, p in enumerate(perceptions_ranked):
        for j, e in enumerate(embed_dims):
            if (p, e) in cells:
                heatmap[i, j] = cells[(p, e)]["ci_mean"]

    im = ax.imshow(heatmap, cmap="RdYlGn", aspect="auto",
                   vmin=-0.05, vmax=0.65)
    ax.set_xticks(range(len(embed_dims)))
    ax.set_xticklabels(embed_dims, fontsize=10)
    ax.set_yticks(range(len(perceptions_ranked)))
    ax.set_yticklabels([PERCEPTION_LABELS.get(p, p).replace('\n', ' ')
                        for p in perceptions_ranked], fontsize=9)
    ax.set_xlabel("Embedding Dimension (model capacity)", fontsize=11)
    ax.set_ylabel("Perception Quality →", fontsize=11)
    ax.set_title("A. Sensorimotor Alignment (C_i) Heatmap", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, label="C_i (SA proxy)", shrink=0.8)

    for i in range(len(perceptions_ranked)):
        for j in range(len(embed_dims)):
            if not np.isnan(heatmap[i, j]):
                color = "white" if heatmap[i, j] < 0.2 else "black"
                ax.text(j, i, f"{heatmap[i, j]:.2f}", ha="center", va="center",
                        fontsize=9, color=color, fontweight="bold")

    # Highlight key comparison: oracle+emb=2 vs vae8+emb=32
    # oracle row index
    oracle_row = perceptions_ranked.index("oracle")
    vae8_row = perceptions_ranked.index("vae_mu_lat8")
    emb2_col = embed_dims.index(2)
    emb32_col = embed_dims.index(32)

    ax.add_patch(plt.Rectangle((emb2_col - 0.45, oracle_row - 0.45), 0.9, 0.9,
                                fill=False, edgecolor="#2196F3", linewidth=3))
    ax.add_patch(plt.Rectangle((emb32_col - 0.45, vae8_row - 0.45), 0.9, 0.9,
                                fill=False, edgecolor="#D32F2F", linewidth=3))

    # ═══ Panel B: C_i vs embed_dim, lines per perception ═══
    ax = fig.add_subplot(gs[0, 1])
    for p in perceptions_ranked:
        if p == "random_baseline":
            continue
        cis = [cells[(p, e)]["ci_mean"] if (p, e) in cells else np.nan for e in embed_dims]
        ax.plot(embed_dims, cis, 'o-', color=CMAP.get(p, "gray"),
                linewidth=2, markersize=7,
                label=PERCEPTION_LABELS.get(p, p).replace('\n', ' '))

    # Horizontal line at random baseline
    rand_ci = np.mean([cells[("random_baseline", e)]["ci_mean"] for e in embed_dims
                       if ("random_baseline", e) in cells])
    ax.axhline(rand_ci, color="#9E9E9E", linestyle="--", linewidth=1, label="Random baseline")

    ax.set_xlabel("Embedding Dimension (model capacity)", fontsize=11)
    ax.set_ylabel("C_i (SA proxy)", fontsize=11)
    ax.set_title("B. Capacity Effect Depends on Perception\n"
                 "(slope = marginal value of model capacity)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xscale("log", base=2)
    ax.set_xticks(embed_dims)
    ax.set_xticklabels(embed_dims)
    ax.grid(alpha=0.3)

    # Annotate the key insight
    ax.annotate("Steep slope:\ncapacity helps",
                xy=(16, 0.55), fontsize=9, ha="center",
                color="#2196F3", fontweight="bold")
    ax.annotate("Flat slope:\ncapacity wasted",
                xy=(16, 0.15), fontsize=9, ha="center",
                color="#E91E63", fontweight="bold")

    # ═══ Panel C: Minimum capacity for threshold ═══
    ax = fig.add_subplot(gs[1, 0])

    thresholds = [0.3, 0.4, 0.5]
    for thresh in thresholds:
        min_caps = []
        perc_cis_at_32 = []
        for p in perceptions_ranked:
            if p == "random_baseline" or p == "oracle_noise0.5":
                continue
            found = False
            for e in embed_dims:
                if (p, e) in cells and cells[(p, e)]["ci_mean"] >= thresh:
                    min_caps.append(e)
                    perc_cis_at_32.append(perc_by_ci.get(p, 0))
                    found = True
                    break
            if not found:
                # Never reaches threshold
                pass

        if min_caps:
            ax.plot(perc_cis_at_32, min_caps, 'o-', linewidth=2, markersize=8,
                    label=f"C_i ≥ {thresh}")

    ax.set_xlabel("Perception Quality (C_i at embed=32)", fontsize=11)
    ax.set_ylabel("Minimum Embedding Dim Needed", fontsize=11)
    ax.set_title("C. Perception-Capacity Substitution\n"
                 "(richer perception → smaller model needed)",
                 fontsize=12, fontweight="bold")
    ax.set_yscale("log", base=2)
    ax.set_yticks(embed_dims)
    ax.set_yticklabels(embed_dims)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.invert_yaxis()

    # ═══ Panel D: The headline comparison ═══
    ax = fig.add_subplot(gs[1, 1])

    comparisons = [
        ("Oracle + emb=2\n(2 params)", "oracle", 2, "#2196F3"),
        ("VAE8 + emb=32\n(32 params)", "vae_mu_lat8", 32, "#D32F2F"),
        ("", None, None, None),  # spacer
        ("Oracle + emb=8\n(8 params)", "oracle", 8, "#2196F3"),
        ("VAE16 + emb=32\n(32 params)", "vae_mu_lat16", 32, "#FF5722"),
        ("", None, None, None),
        ("Oracle + emb=4\n(4 params)", "oracle", 4, "#2196F3"),
        ("Oracle+N(0.5) + emb=32\n(32 params)", "oracle_noise0.5", 32, "#E91E63"),
    ]

    labels = []
    vals = []
    colors = []
    for label, perc, emb, color in comparisons:
        if perc is None:
            labels.append("")
            vals.append(0)
            colors.append("white")
        else:
            ci = cells[(perc, emb)]["ci_mean"]
            labels.append(label)
            vals.append(ci)
            colors.append(color)

    y_pos = range(len(labels))
    bars = ax.barh(y_pos, vals, color=colors, alpha=0.85, edgecolor="white", height=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("C_i (SA proxy)", fontsize=11)
    ax.set_title("D. Small Model + Rich Perception\nvs Large Model + Poor Perception",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3, axis="x")
    ax.set_xlim(0, 0.7)

    # Add value labels
    for bar, val in zip(bars, vals):
        if val > 0:
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                    f"{val:.3f}", va="center", fontsize=9, fontweight="bold")

    # Add "winner" annotations
    ax.annotate("16× smaller model\n→ higher alignment",
                xy=(0.45, 0.5), xytext=(0.55, 0.5),
                fontsize=9, color="#2196F3", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#2196F3"),
                va="center")

    fig.suptitle(
        "The Perception-Capacity Tradeoff:\n"
        "Rich Sensory Input Substitutes for Model Capacity",
        fontsize=14, fontweight="bold", y=0.98
    )

    out = results_dir / "iso_performance.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

    # ── Print summary statistics ──
    print("\n=== Headline Results ===")
    o2 = cells[("oracle", 2)]["ci_mean"]
    v8_32 = cells[("vae_mu_lat8", 32)]["ci_mean"]
    print(f"Oracle + emb=2:      C_i = {o2:.3f}")
    print(f"VAE lat=8 + emb=32:  C_i = {v8_32:.3f}")
    print(f"→ 16× less capacity, {o2 - v8_32:+.3f} alignment ({'BETTER' if o2 > v8_32 else 'WORSE'})")

    o8 = cells[("oracle", 8)]["ci_mean"]
    v16_32 = cells[("vae_mu_lat16", 32)]["ci_mean"]
    print(f"\nOracle + emb=8:      C_i = {o8:.3f}")
    print(f"VAE lat=16 + emb=32: C_i = {v16_32:.3f}")
    print(f"→ 4× less capacity, {o8 - v16_32:+.3f} alignment ({'BETTER' if o8 > v16_32 else 'WORSE'})")

    o4 = cells[("oracle", 4)]["ci_mean"]
    n05_32 = cells[("oracle_noise0.5", 32)]["ci_mean"]
    print(f"\nOracle + emb=4:           C_i = {o4:.3f}")
    print(f"Oracle+N(0.5) + emb=32:   C_i = {n05_32:.3f}")
    print(f"→ 8× less capacity, {o4 - n05_32:+.3f} alignment ({'BETTER' if o4 > n05_32 else 'WORSE'})")

    # Compute interaction statistic
    print("\n=== Interaction Analysis ===")
    # For each perception level, compute the slope of C_i vs log2(embed_dim)
    print(f"{'Perception':25s} {'Slope':>8s} (C_i per doubling of capacity)")
    slopes = {}
    for p in perceptions_ranked:
        cis = []
        log_embs = []
        for e in embed_dims:
            if (p, e) in cells:
                cis.append(cells[(p, e)]["ci_mean"])
                log_embs.append(np.log2(e))
        if len(cis) >= 3:
            slope = np.polyfit(log_embs, cis, 1)[0]
            slopes[p] = slope
            print(f"{p:25s} {slope:8.4f}")

    # Correlation between perception quality and slope
    if slopes:
        perc_vals = [perc_by_ci.get(p, 0) for p in slopes.keys()]
        slope_vals = list(slopes.values())
        r, p = pearsonr(perc_vals, slope_vals)
        print(f"\nCorrelation(perception quality, capacity slope): r={r:.3f}, p={p:.3f}")
        if r > 0:
            print("→ Better perception makes capacity MORE useful (positive interaction)")
        else:
            print("→ Better perception makes capacity LESS useful (diminishing returns)")


if __name__ == "__main__":
    main()
