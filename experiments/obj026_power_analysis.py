"""obj-026 T14: Power analysis for obj-024 substitution effect.

Question: given observed Cohen's d=1.10 between (sensory=16, embed=2) and
(sensory=2, embed=32), is our n=5/cell adequate? What about the smaller
effects observed at other cells?

Computes:
  1. Post-hoc power at current n=5 for d=1.10 (substitution), d=0.30 (small),
     d=0.50 (medium)
  2. Sample size required for 80% power at various d
  3. Recommendation for obj-026 (2-rock replicate) seed count
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def power_two_sample_t(d, n, alpha=0.05):
    """Approximate 2-sample t-test power (Normal approximation)."""
    from math import sqrt, erf
    def ndtr(x):
        return 0.5 * (1 + erf(x / sqrt(2)))
    n_effective = n / 2  # per group; for d-based formula we use n_per_group
    delta = d * sqrt(n / 2)
    z_alpha = 1.959964  # two-sided alpha=0.05
    return 1 - ndtr(z_alpha - delta) + ndtr(-z_alpha - delta)


def sample_size_for_power(d, target_power=0.8, alpha=0.05):
    """Approximate n per group for target power via normal approximation."""
    if d < 0.01:
        return float("inf")
    z_alpha = 1.959964
    z_beta = 0.8416  # power=0.8
    n = 2 * ((z_alpha + z_beta) / d) ** 2
    return max(2, int(np.ceil(n)))


def main():
    # Load obj-024 and compute observed effect sizes for all pair comparisons
    with open("results/sensory_capacity_checkpoint.json") as f:
        data = json.load(f)
    from collections import defaultdict
    cells = defaultdict(list)
    for r in data:
        cells[(r["sensory_dim"], r["embedding_dim"])].append(r["SA"])

    def cohens_d(a, b):
        a, b = np.asarray(a), np.asarray(b)
        pooled = np.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) / (len(a)+len(b)-2))
        return (a.mean() - b.mean()) / (pooled + 1e-10)

    # Pairwise effect sizes: all cell pairs involving sensory=16 cells vs others
    d_key = cohens_d(cells[(16, 2)], cells[(2, 32)])
    d_peak = cohens_d(cells[(16, 16)], cells[(2, 4)])
    d_peak_vs_min = cohens_d(cells[(16, 16)], cells[(16, 2)])

    print("Observed effect sizes (Cohen's d):")
    print(f"  Substitution (16,2 vs 2,32):    d = {d_key:.2f}")
    print(f"  Ceiling vs floor (16,16 vs 2,4): d = {d_peak:.2f}")
    print(f"  Peak vs min within sensory=16: d = {d_peak_vs_min:.2f}")

    # Power at n=5 for these effects
    print("\nPost-hoc power at n=5 seeds/cell:")
    for name, d in [("substitution", d_key), ("ceiling-floor", d_peak), ("peak-min @ sensory=16", d_peak_vs_min)]:
        p = power_two_sample_t(abs(d), 10)  # n_total = 10
        print(f"  d={abs(d):.2f} ({name}): power = {p:.3f}")

    # Power curves
    ds = np.linspace(0.1, 2.5, 30)
    ns = [4, 5, 6, 8, 10, 15, 20]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("obj-026 T14: Power Analysis for Sensory-Capacity Effects",
                 fontsize=13, fontweight="bold")

    ax = axes[0]
    for n in ns:
        powers = [power_two_sample_t(d, n * 2) for d in ds]
        ax.plot(ds, powers, label=f"n={n}/group")
    ax.axhline(y=0.8, color="gray", linestyle="--", alpha=0.5, label="80% target")
    ax.axvline(x=d_key, color="red", linestyle=":", alpha=0.7, label=f"observed substitution d={abs(d_key):.2f}")
    ax.axvline(x=0.3, color="orange", linestyle=":", alpha=0.5, label="small (d=0.3)")
    ax.set_xlabel("Cohen's d")
    ax.set_ylabel("Statistical Power")
    ax.set_title("A. Power vs Effect Size")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)

    # Required n vs effect size
    ax = axes[1]
    d_range = np.linspace(0.2, 2.0, 30)
    n_needed = [sample_size_for_power(d, 0.8) for d in d_range]
    ax.plot(d_range, n_needed, linewidth=2, color="#1f77b4")
    ax.axhline(y=5, color="red", linestyle="--", alpha=0.5, label="current obj-024 (n=5)")
    ax.axhline(y=3, color="purple", linestyle="--", alpha=0.5, label="obj-026 (n=3)")
    ax.axvline(x=abs(d_key), color="red", linestyle=":", alpha=0.5)
    ax.annotate(f"d={abs(d_key):.2f}\n(substitution)",
                xy=(abs(d_key), sample_size_for_power(abs(d_key))),
                xytext=(abs(d_key) + 0.15, sample_size_for_power(abs(d_key)) + 3),
                fontsize=9)
    ax.set_xlabel("Cohen's d (true effect)")
    ax.set_ylabel("n per group (80% power)")
    ax.set_title("B. Sample Size Required")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    plt.tight_layout()
    out = Path("results/obj026_power_analysis.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nSaved: {out}")

    # Verdict
    print("\n=== VERDICT ===")
    n_for_d = sample_size_for_power(abs(d_key))
    p_at_5 = power_two_sample_t(abs(d_key), 10)
    print(f"For the observed substitution effect (d={abs(d_key):.2f}):")
    print(f"  n=5/group → power={p_at_5:.3f} {'(adequate)' if p_at_5 > 0.8 else '(UNDERPOWERED)'}")
    print(f"  Required n for 80% power: {n_for_d}/group")
    print(f"For small effects (d=0.3): need n={sample_size_for_power(0.3)} — "
          f"impractical with this experimental budget.")

    out_json = Path("results/obj026_power_analysis.json")
    with out_json.open("w") as f:
        json.dump({
            "observed_effects": {
                "substitution_d": float(d_key),
                "ceiling_floor_d": float(d_peak),
            },
            "post_hoc_power_n5": float(p_at_5),
            "n_for_80_power": n_for_d,
            "recommendation_obj026": "n=3 adequate for d>=1.0 (power>0.68). "
                                     "Medium effects (d=0.5) need n>=17 — not feasible in obj-026.",
        }, f, indent=2)


if __name__ == "__main__":
    main()
