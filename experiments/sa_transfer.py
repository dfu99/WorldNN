"""obj-019 (design): SA transfer / generalization across rock variants.

Question: does a trained organism generalize to a slightly different rock?

The organism trains on Rock A (standard physics). We then evaluate SA
and performance on Rock B (modified physics) WITHOUT retraining. If SA
degrades gracefully (proportional to physics change), the alignment
operator R captures structural understanding, not memorized actions.

Transfer matrix:
  Train on: standard rock (push_radius=0.2, push_strength=0.12, move_speed=0.15)
  Evaluate on 6 variants:
    1. push_radius ±20% (0.16, 0.24) — "smaller/larger rock"
    2. push_strength ±30% (0.084, 0.156) — "heavier/lighter rock"
    3. move_speed ±20% (0.12, 0.18) — "slower/faster organism" (control)
  + 2 combined variants:
    4. smaller + heavier (harder)
    5. larger + lighter (easier)

For each variant, measure:
  - SA (on the variant's physics, using variant's optimal action)
  - Task performance (distance)
  - "SA retention" = SA_variant / SA_original

Prediction: SA should degrade GRACEFULLY — proportional to the physics
change magnitude. A 20% change in push_radius should cause ~10-20% SA
drop, not catastrophic failure. This would show the organism learned
the STRUCTURE of rock-pushing, not just the specific parameters.

Grid: 3 embed_dims × 3 seeds × 7 eval conditions = 63 evaluations
(but only 9 training runs — we reuse trained organisms)
"""

import sys
import json
import time
import math
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from worldnn.matter import RockPushMatter
from worldnn.organism import Organism
from coordination_quality import compute_optimal_action, measure_coordination_quality
from perception_ladder import train_organism_with_perception


# Rock variants: name → physics overrides
VARIANTS = {
    "standard":       {"push_radius": 0.20, "push_strength": 0.120, "move_speed": 0.15},
    "smaller_rock":   {"push_radius": 0.16, "push_strength": 0.120, "move_speed": 0.15},
    "larger_rock":    {"push_radius": 0.24, "push_strength": 0.120, "move_speed": 0.15},
    "heavier_rock":   {"push_radius": 0.20, "push_strength": 0.084, "move_speed": 0.15},
    "lighter_rock":   {"push_radius": 0.20, "push_strength": 0.156, "move_speed": 0.15},
    "slower_organism":{"push_radius": 0.20, "push_strength": 0.120, "move_speed": 0.12},
    "faster_organism":{"push_radius": 0.20, "push_strength": 0.120, "move_speed": 0.18},
    "hard_combo":     {"push_radius": 0.16, "push_strength": 0.084, "move_speed": 0.15},
    "easy_combo":     {"push_radius": 0.24, "push_strength": 0.156, "move_speed": 0.15},
}


def make_matter(variant_name, device="cpu"):
    """Create a RockPushMatter with the given variant's physics."""
    params = VARIANTS[variant_name]
    return RockPushMatter(
        emission_dim=8, action_dim=2, seed_dim=4,
        push_radius=params["push_radius"],
        push_strength=params["push_strength"],
        move_speed=params["move_speed"],
    ).to(device)


def eval_on_variant(organism, variant_matter, device="cpu"):
    """Evaluate a trained organism on a different rock variant.

    Returns SA and task performance WITHOUT retraining.
    """
    def oracle_fn(state, emission):
        return state

    # SA
    sa = measure_coordination_quality(
        organism, oracle_fn, variant_matter, n_samples=2000, device=device,
    )

    # Rollout performance
    target = torch.tensor([0.8, 0.8], device=device)
    organism.eval()
    dists = []
    with torch.no_grad():
        for _ in range(20):
            state = variant_matter.reset_state(256, device)
            for t in range(20):
                seed = torch.randn(256, variant_matter.seed_dim, device=device)
                action, _, _ = organism(state)
                state, _, _ = variant_matter(state, seed, action)
            rock_pos = state[:, :2]
            d = torch.norm(rock_pos - target.unsqueeze(0), dim=-1).mean().item()
            dists.append(d)

    return {
        "SA": sa["C_i"],
        "SA_std": sa["C_i_std"],
        "dist": sum(dists) / len(dists),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    embed_dims = [4, 16, 64]
    seeds = [42, 123, 456]

    all_results = []

    for emb in embed_dims:
        for s in seeds:
            print(f"\n=== Training: emb={emb}, seed={s} ===")
            torch.manual_seed(s)

            # Train on standard rock
            train_matter = make_matter("standard", device)
            organism = Organism(
                sensory_dim=4, embedding_dim=emb, action_dim=2,
            ).to(device)

            def oracle_fn(state, emission):
                return state

            metrics = train_organism_with_perception(
                train_matter, organism, oracle_fn,
                target_x=0.8, target_y=0.8,
                n_episodes=500, device=device,
            )

            # Evaluate on standard (baseline SA)
            baseline = eval_on_variant(organism, train_matter, device)
            print(f"  standard: SA={baseline['SA']:.3f}, dist={baseline['dist']:.3f}")

            # Evaluate on each variant
            for variant_name, params in VARIANTS.items():
                variant_matter = make_matter(variant_name, device)
                result = eval_on_variant(organism, variant_matter, device)

                sa_retention = result["SA"] / max(baseline["SA"], 1e-6)
                dist_change = result["dist"] - baseline["dist"]

                entry = {
                    "embedding_dim": emb,
                    "seed": s,
                    "variant": variant_name,
                    "SA": result["SA"],
                    "SA_std": result["SA_std"],
                    "dist": result["dist"],
                    "baseline_SA": baseline["SA"],
                    "baseline_dist": baseline["dist"],
                    "SA_retention": sa_retention,
                    "dist_change": dist_change,
                }
                all_results.append(entry)

                if variant_name != "standard":
                    print(f"  {variant_name:18s}: SA={result['SA']:.3f} "
                          f"(retention={sa_retention:.1%}), "
                          f"dist={result['dist']:.3f} (Δ={dist_change:+.3f})")

    # Save results
    with open(results_dir / "sa_transfer.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print("TRANSFER SUMMARY")
    print(f"{'='*60}")

    by_variant = defaultdict(list)
    for r in all_results:
        by_variant[r["variant"]].append(r)

    print(f"\n{'Variant':20s} {'Mean SA':>8s} {'Retention':>10s} {'Δdist':>8s}")
    print("-" * 50)
    for variant in VARIANTS:
        rs = by_variant[variant]
        mean_sa = sum(r["SA"] for r in rs) / len(rs)
        mean_ret = sum(r["SA_retention"] for r in rs) / len(rs)
        mean_dd = sum(r["dist_change"] for r in rs) / len(rs)
        print(f"{variant:20s} {mean_sa:>8.3f} {mean_ret:>9.1%} {mean_dd:>+8.3f}")

    # Generate plot
    try:
        generate_plot(all_results, results_dir)
    except Exception as e:
        print(f"Plot failed: {e}")


def generate_plot(results, results_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    by_variant = defaultdict(list)
    for r in results:
        by_variant[r["variant"]].append(r)

    variants_ordered = [
        "hard_combo", "smaller_rock", "heavier_rock", "slower_organism",
        "standard",
        "faster_organism", "lighter_rock", "larger_rock", "easy_combo",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: SA retention by variant
    ax = axes[0]
    means, stds, labels = [], [], []
    for v in variants_ordered:
        rs = by_variant[v]
        rets = [r["SA_retention"] for r in rs]
        means.append(np.mean(rets))
        stds.append(np.std(rets))
        labels.append(v.replace("_", "\n"))

    colors = ["#d62728" if m < 0.8 else "#ff7f0e" if m < 0.95 else "#2ca02c" for m in means]
    bars = ax.barh(range(len(labels)), means, xerr=stds, color=colors, alpha=0.85, capsize=3)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(1.0, color="gray", linestyle="--", alpha=0.5, label="No transfer loss")
    ax.axvline(0.8, color="red", linestyle=":", alpha=0.4, label="20% SA loss")
    ax.set_xlabel("SA Retention (variant SA / training SA)")
    ax.set_title("Does SA Generalize Across Rock Variants?")
    ax.legend(fontsize=8)

    # Panel 2: SA retention vs physics distance
    ax = axes[1]
    std_params = VARIANTS["standard"]
    for v, rs in by_variant.items():
        params = VARIANTS[v]
        # Physics "distance" = sum of relative changes
        phys_dist = sum(
            abs(params[k] - std_params[k]) / std_params[k]
            for k in ["push_radius", "push_strength", "move_speed"]
        )
        for r in rs:
            color = {"4": "#1f77b4", "16": "#ff7f0e", "64": "#2ca02c"}.get(str(r["embedding_dim"]), "gray")
            ax.scatter(phys_dist, r["SA_retention"], color=color, s=50, alpha=0.6,
                      edgecolors="black", linewidth=0.3)

    # Fit line
    all_phys = []
    all_ret = []
    for v, rs in by_variant.items():
        params = VARIANTS[v]
        phys_dist = sum(abs(params[k] - std_params[k]) / std_params[k] for k in ["push_radius", "push_strength", "move_speed"])
        for r in rs:
            all_phys.append(phys_dist)
            all_ret.append(r["SA_retention"])
    if len(all_phys) >= 5:
        corr = np.corrcoef(all_phys, all_ret)[0, 1]
        z = np.polyfit(all_phys, all_ret, 1)
        x_range = np.linspace(0, max(all_phys), 100)
        ax.plot(x_range, np.polyval(z, x_range), "k--", alpha=0.4)
        ax.text(0.05, 0.05, f"r = {corr:.3f}", transform=ax.transAxes,
                fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    for emb, color in [(4, "#1f77b4"), (16, "#ff7f0e"), (64, "#2ca02c")]:
        ax.scatter([], [], color=color, label=f"emb={emb}", s=40)
    ax.set_xlabel("Physics Distance (sum of relative parameter changes)")
    ax.set_ylabel("SA Retention")
    ax.set_title("SA Degrades Gracefully with Physics Change")
    ax.legend(fontsize=8)

    fig.suptitle("obj-019: SA Transfer Across Rock Variants", fontsize=14)
    plt.tight_layout()
    out = results_dir / "obj019_sa_transfer.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
