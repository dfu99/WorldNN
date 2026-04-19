"""
obj-025 T4: SA vs reconstruction-capability head-to-head.

Reviewer A (World Model) asks: "Doesn't Dreamer's reconstruction loss tell you
the same thing as SA?" This analysis tests whether SA captures task-relevant
structure beyond what simple input-information predictors offer.

We don't have trained-organism embeddings saved from obj-024. Instead we
compare SA (per-config, measured) against several input-only predictors
that serve as proxies for "information available to reconstruct state":
  • sensory_dim (how many channels observed)
  • embed_dim (model capacity)
  • linear-probe R²(S | obs[:sensory_dim]) — what a Dreamer-style decoder
    could recover in the best case from the input
  • Gaussian-MI I(S; obs) (from T3)

If SA adds predictive signal beyond these, it's not redundant with recon.
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def partial_corr(x, y, z):
    """Pearson r between x and y after regressing out z (linear)."""
    x, y, z = np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)
    if z.ndim == 1: z = z.reshape(-1, 1)
    z = np.concatenate([z, np.ones((z.shape[0], 1))], axis=1)
    bx, *_ = np.linalg.lstsq(z, x, rcond=None); rx = x - z @ bx
    by, *_ = np.linalg.lstsq(z, y, rcond=None); ry = y - z @ by
    if rx.std() < 1e-10 or ry.std() < 1e-10:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    # Load obj-024 per-config data
    with open("results/sensory_capacity_checkpoint.json") as f:
        data = json.load(f)

    # Load T3 rate-distortion numbers
    with open("results/obj025_mi_vs_sensory.json") as f:
        rd = json.load(f)

    r2_by_sd = {int(k): v for k, v in rd["linear_probe_R2"].items()}
    mi_by_sd = {int(k): v for k, v in rd["Gaussian_MI_nats"].items()}

    # Build per-config arrays
    sa = np.array([r["SA"] for r in data])
    dist = np.array([r["avg_dist"] for r in data])
    sensory = np.array([r["sensory_dim"] for r in data])
    embed = np.array([r["embedding_dim"] for r in data])
    log2e = np.log2(embed.astype(float))
    recon_r2 = np.array([r2_by_sd[sd] for sd in sensory])
    mi_in = np.array([mi_by_sd[sd] for sd in sensory])

    print(f"N configs: {len(data)}")
    print()

    # Raw correlations with dist (lower dist = better performance)
    def rr(x, y):
        if x.std() < 1e-10 or y.std() < 1e-10:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])

    rows = [
        ("SA",            rr(sa, dist)),
        ("sensory_dim",   rr(sensory, dist)),
        ("embed_dim",     rr(embed, dist)),
        ("log2(embed)",   rr(log2e, dist)),
        ("recon-R²(S|obs)", rr(recon_r2, dist)),
        ("Gauss-MI(S;obs)", rr(mi_in, dist)),
    ]

    print("Raw Pearson r with avg_dist (more negative = better predictor):")
    for name, r in rows:
        print(f"  {name:22s}  r={r:+.4f}")

    # Partial correlations: SA vs dist, controlling for input predictors
    z = np.stack([sensory, log2e, recon_r2], axis=1)
    pr_sa = partial_corr(sa, dist, z)
    print(f"\nPartial r(SA, dist | sensory_dim, log2(embed), recon-R²) = {pr_sa:+.4f}")

    # Also partial: recon-R² vs dist, controlling for SA
    pr_r2 = partial_corr(recon_r2, dist, sa.reshape(-1, 1))
    print(f"Partial r(recon-R², dist | SA) = {pr_r2:+.4f}")

    # Multiple regression: how much variance does SA explain beyond input vars?
    def r2_mult(X, y):
        X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ b
        ss_res = ((y - pred) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        return 1.0 - ss_res / ss_tot

    r2_input = r2_mult(np.stack([sensory, log2e, recon_r2], axis=1), dist)
    r2_input_plus_sa = r2_mult(np.stack([sensory, log2e, recon_r2, sa], axis=1), dist)
    delta = r2_input_plus_sa - r2_input
    print(f"\nR² predicting dist:")
    print(f"  input vars only (sensory_dim, log2(embed), recon-R²): R² = {r2_input:.4f}")
    print(f"  input vars + SA                                      : R² = {r2_input_plus_sa:.4f}")
    print(f"  Δ R² from adding SA                                  : {delta:+.4f}")

    # === Figure ===
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "obj-025 T4: SA vs Reconstruction-Capability as Performance Predictors\n"
        "Reviewer A question: does SA add signal beyond what recon loss captures?",
        fontsize=13, fontweight="bold",
    )

    # Panel A: bar chart of |r| with dist
    ax = axes[0]
    names = [r[0] for r in rows]
    rs = [abs(r[1]) for r in rows]
    colors = ["#d62728"] + ["#1f77b4"] * 5
    ax.barh(range(len(names)), rs, color=colors, edgecolor="black")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("|Pearson r| with avg_dist")
    ax.set_title("A. Predictor Strength (|r|)")
    for i, (n, r) in enumerate(rows):
        ax.text(abs(r) + 0.005, i, f"{r:+.3f}", va="center", fontsize=9)
    ax.invert_yaxis()
    ax.axvline(x=abs(rr(sa, dist)), color="red", linestyle=":", alpha=0.5)

    # Panel B: SA vs dist colored by recon-R²
    ax = axes[1]
    sc = ax.scatter(sa, dist, c=recon_r2, cmap="viridis", s=40, edgecolor="black", linewidth=0.5)
    plt.colorbar(sc, ax=ax, label="recon-R²(S|obs)")
    ax.set_xlabel("SA per config")
    ax.set_ylabel("avg_dist (lower is better)")
    ax.set_title(f"B. SA vs dist (r={rr(sa,dist):+.3f})")
    ax.grid(alpha=0.3)

    # Panel C: ΔR² / partial r summary
    ax = axes[2]
    summary = [
        ("r(SA, dist)", rr(sa, dist)),
        ("r(recon-R², dist)", rr(recon_r2, dist)),
        ("partial r(SA, dist | inputs)", pr_sa),
        ("partial r(recon-R², dist | SA)", pr_r2),
    ]
    names_s = [s[0] for s in summary]
    vals = [s[1] for s in summary]
    bcolors = ["#d62728", "#1f77b4", "#d62728", "#1f77b4"]
    ax.barh(range(len(summary)), vals, color=bcolors, edgecolor="black")
    ax.set_yticks(range(len(summary)))
    ax.set_yticklabels(names_s, fontsize=9)
    ax.set_xlabel("Pearson r")
    ax.set_title(f"C. SA independent of input (ΔR²={delta:+.3f})")
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax.invert_yaxis()
    for i, v in enumerate(vals):
        ax.text(v + (0.01 if v >= 0 else -0.01), i, f"{v:+.3f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=9)

    plt.tight_layout()
    out = Path("results/obj025_sa_vs_recon.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nSaved: {out}")

    out_json = Path("results/obj025_sa_vs_recon.json")
    with out_json.open("w") as f:
        json.dump({
            "raw_correlations_with_dist": {n: float(r) for n, r in rows},
            "partial_r_SA_given_inputs": pr_sa,
            "partial_r_reconR2_given_SA": pr_r2,
            "R2_input_only": r2_input,
            "R2_input_plus_SA": r2_input_plus_sa,
            "deltaR2_from_SA": delta,
            "n_configs": len(data),
        }, f, indent=2)
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
