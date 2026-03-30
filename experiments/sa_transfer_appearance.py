"""obj-020: SA transfer with appearance + physics variation.

obj-019 showed 93-106% SA retention across physics variants, but the
emission matrix was identical — all rocks "looked" the same. This
experiment perturbs the emission projection matrix (state_proj) so
that variant rocks both LOOK and RESPOND differently.

Two axes of variation:
  1. Physics: push_radius, push_strength (as in obj-019)
  2. Appearance: state_proj perturbed by epsilon * randn

Grid:
  appearance_epsilon: [0.0, 0.1, 0.3, 0.5]
  physics_variant: [standard, smaller_rock, heavier_rock, hard_combo]
  embed_dim: [4, 16, 64]
  seeds: [42, 123, 456, 789, 1337]
  = 4 appearance × 4 physics × 3 embed × 5 seeds = 240 evaluations
  but only 15 training runs (3 embed × 5 seeds, all trained on standard)
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


PHYSICS = {
    "standard":     {"push_radius": 0.20, "push_strength": 0.120},
    "smaller_rock": {"push_radius": 0.16, "push_strength": 0.120},
    "heavier_rock": {"push_radius": 0.20, "push_strength": 0.084},
    "hard_combo":   {"push_radius": 0.16, "push_strength": 0.084},
}

APPEARANCE_EPSILONS = [0.0, 0.1, 0.3, 0.5]


def make_variant_matter(physics_name, appearance_eps, base_matter, device):
    """Create a matter variant with perturbed physics AND appearance."""
    params = PHYSICS[physics_name]
    variant = RockPushMatter(
        emission_dim=8, action_dim=2, seed_dim=4,
        push_radius=params["push_radius"],
        push_strength=params["push_strength"],
    ).to(device)

    # Copy base emission matrix and perturb
    with torch.no_grad():
        variant.state_proj.copy_(base_matter.state_proj)
        variant.emission_bias.copy_(base_matter.emission_bias)
        variant.seed_proj.copy_(base_matter.seed_proj)

        if appearance_eps > 0:
            variant.state_proj.add_(torch.randn_like(variant.state_proj) * appearance_eps)
            variant.emission_bias.add_(torch.randn_like(variant.emission_bias) * appearance_eps)

    return variant


def eval_on_variant(organism, variant_matter, device):
    """Evaluate SA and distance without retraining."""
    def oracle_fn(state, emission):
        return state

    sa = measure_coordination_quality(
        organism, oracle_fn, variant_matter, n_samples=2000, device=device,
    )

    target = torch.tensor([0.8, 0.8], device=device)
    organism.eval()
    dists = []
    with torch.no_grad():
        for _ in range(10):
            state = variant_matter.reset_state(256, device)
            for t in range(20):
                seed = torch.randn(256, variant_matter.seed_dim, device=device)
                action, _, _ = organism(state)
                state, _, _ = variant_matter(state, seed, action)
            rock_pos = state[:, :2]
            dists.append(torch.norm(rock_pos - target.unsqueeze(0), dim=-1).mean().item())

    return {"SA": sa["C_i"], "dist": sum(dists) / len(dists)}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    embed_dims = [4, 16, 64]
    seeds = [42, 123, 456, 789, 1337]
    dev = torch.device(device)

    all_results = []
    t0 = time.time()

    for emb in embed_dims:
        for s in seeds:
            print(f"\n=== Training: emb={emb}, seed={s} ===")
            torch.manual_seed(s)

            # Create and train on standard rock
            base_matter = RockPushMatter(
                emission_dim=8, action_dim=2, seed_dim=4,
            ).to(dev)
            organism = Organism(
                sensory_dim=4, embedding_dim=emb, action_dim=2,
            ).to(dev)

            def oracle_fn(state, emission):
                return state

            metrics = train_organism_with_perception(
                base_matter, organism, oracle_fn,
                target_x=0.8, target_y=0.8,
                n_episodes=500, device=device,
            )

            # Baseline SA on training matter
            baseline = eval_on_variant(organism, base_matter, dev)
            print(f"  baseline: SA={baseline['SA']:.3f}, dist={baseline['dist']:.3f}")

            # Evaluate on each combination
            for phys_name in PHYSICS:
                for eps in APPEARANCE_EPSILONS:
                    torch.manual_seed(s * 1000 + int(eps * 100))  # deterministic perturbation per seed
                    variant = make_variant_matter(phys_name, eps, base_matter, dev)
                    result = eval_on_variant(organism, variant, dev)

                    sa_retention = result["SA"] / max(baseline["SA"], 1e-6)
                    entry = {
                        "embedding_dim": emb,
                        "seed": s,
                        "physics": phys_name,
                        "appearance_eps": eps,
                        "SA": result["SA"],
                        "dist": result["dist"],
                        "baseline_SA": baseline["SA"],
                        "baseline_dist": baseline["dist"],
                        "SA_retention": sa_retention,
                    }
                    all_results.append(entry)

                    if phys_name != "standard" or eps > 0:
                        print(f"  {phys_name:15s} eps={eps}: SA={result['SA']:.3f} "
                              f"(ret={sa_retention:.1%})")

    elapsed = time.time() - t0
    print(f"\nTotal: {len(all_results)} evaluations in {elapsed/60:.1f} min")

    with open(results_dir / "sa_transfer_appearance.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print("TRANSFER SUMMARY: SA Retention by Physics × Appearance")
    print(f"{'='*60}")

    import numpy as np
    by_group = defaultdict(list)
    for r in all_results:
        by_group[(r["physics"], r["appearance_eps"])].append(r["SA_retention"])

    print(f"\n{'':20s}", end="")
    for eps in APPEARANCE_EPSILONS:
        print(f"  eps={eps:<4}", end="")
    print()
    for phys in PHYSICS:
        print(f"{phys:20s}", end="")
        for eps in APPEARANCE_EPSILONS:
            rets = by_group[(phys, eps)]
            print(f"  {np.mean(rets):5.1%}", end="")
        print()

    # Physics distance vs appearance distance analysis
    print(f"\n=== Correlation: SA retention vs perturbation magnitude ===")
    all_eps = [r["appearance_eps"] for r in all_results]
    all_ret = [r["SA_retention"] for r in all_results]
    r_eps = np.corrcoef(all_eps, all_ret)[0, 1]
    print(f"  r(appearance_eps, SA_retention) = {r_eps:.3f}")

    # Generate plot
    try:
        generate_plot(all_results, results_dir)
    except Exception as e:
        print(f"Plot failed: {e}")
        import traceback; traceback.print_exc()


def generate_plot(results, results_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: SA retention heatmap (physics × appearance)
    ax = axes[0]
    by_group = defaultdict(list)
    for r in results:
        by_group[(r["physics"], r["appearance_eps"])].append(r["SA_retention"])

    phys_names = list(PHYSICS.keys())
    data = np.zeros((len(phys_names), len(APPEARANCE_EPSILONS)))
    for i, phys in enumerate(phys_names):
        for j, eps in enumerate(APPEARANCE_EPSILONS):
            vals = by_group[(phys, eps)]
            data[i, j] = np.mean(vals) if vals else 0

    im = ax.imshow(data, cmap="RdYlGn", vmin=0.5, vmax=1.2, aspect="auto")
    ax.set_xticks(range(len(APPEARANCE_EPSILONS)))
    ax.set_xticklabels([f"ε={e}" for e in APPEARANCE_EPSILONS])
    ax.set_yticks(range(len(phys_names)))
    ax.set_yticklabels([p.replace("_", "\n") for p in phys_names], fontsize=9)
    ax.set_xlabel("Appearance perturbation (ε)")
    ax.set_title("(a) SA Retention: Physics × Appearance")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i,j]:.0%}", ha="center", va="center", fontsize=9,
                    color="black" if data[i, j] > 0.7 else "white")
    plt.colorbar(im, ax=ax, label="SA retention")

    # Panel 2: SA retention vs appearance epsilon (line per physics)
    ax = axes[1]
    colors = {"standard": "#1f77b4", "smaller_rock": "#ff7f0e",
              "heavier_rock": "#2ca02c", "hard_combo": "#d62728"}
    for phys in phys_names:
        means = [np.mean(by_group[(phys, eps)]) for eps in APPEARANCE_EPSILONS]
        stds = [np.std(by_group[(phys, eps)]) for eps in APPEARANCE_EPSILONS]
        ax.errorbar(APPEARANCE_EPSILONS, means, yerr=stds, marker="o",
                    label=phys, color=colors[phys], capsize=3, linewidth=2)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.4)
    ax.axhline(0.8, color="red", linestyle=":", alpha=0.4, label="80% retention")
    ax.set_xlabel("Appearance perturbation (ε)")
    ax.set_ylabel("SA Retention")
    ax.set_title("(b) SA Retention vs Appearance Change")
    ax.legend(fontsize=8)

    # Panel 3: SA retention by embed_dim (faceted by epsilon)
    ax = axes[2]
    by_emb_eps = defaultdict(list)
    for r in results:
        by_emb_eps[(r["embedding_dim"], r["appearance_eps"])].append(r["SA_retention"])
    for eps in APPEARANCE_EPSILONS:
        embs = [4, 16, 64]
        means = [np.mean(by_emb_eps[(e, eps)]) for e in embs]
        ax.plot(embs, means, marker="s", label=f"ε={eps}", linewidth=2)
    ax.set_xlabel("Embedding Dimension")
    ax.set_ylabel("SA Retention")
    ax.set_title("(c) Capacity Effect on Transfer")
    ax.legend(fontsize=8)
    ax.set_xscale("log", base=2)
    ax.set_xticks([4, 16, 64])
    ax.set_xticklabels([4, 16, 64])

    fig.suptitle("obj-020: SA Transfer with Appearance + Physics Variation", fontsize=14)
    plt.tight_layout()
    out = results_dir / "obj020_sa_transfer_appearance.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
