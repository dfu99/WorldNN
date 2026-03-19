"""obj-014: Expanded C_i sweep — all perception conditions.

obj-013 showed r=-0.867 with oracle + VAE(mu) lat=16. Now test whether
the correlation and threshold hold across ALL perception conditions:

  1. Oracle (direct 4D state)
  2. Oracle + noise (σ=0.1, 0.5)
  3. Raw emission (8D, no VAE)
  4. VAE mu lat=8
  5. VAE mu lat=16
  6. VAE mu lat=32

Each × embed_dim=[2,4,8,16,32] × 3 seeds = 6 × 5 × 3 = 90 configs

Reuses perception_ladder's train_organism_with_perception for custom
perception functions, and coordination_quality's measure_coordination_quality.
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
from worldnn.world import RockPushWorld
from worldnn.matter import RockPushMatter
from worldnn.organism import Organism
from worldnn.train import train_environment_rockpush
from perception_ladder import train_organism_with_perception, probe_vae_latent
from coordination_quality import compute_optimal_action, measure_coordination_quality


def run_config(level_name, matter, sensory_dim, embed_dim, seed,
               perception_fn, device="cuda", n_episodes=500):
    """Run one perception level + measure C_i."""
    torch.manual_seed(seed)
    dev = torch.device(device)

    organism = Organism(
        sensory_dim=sensory_dim,
        embedding_dim=embed_dim,
        action_dim=2,
    ).to(dev)

    t0 = time.time()
    metrics = train_organism_with_perception(
        matter, organism, perception_fn,
        target_x=0.8, target_y=0.8,
        n_episodes=n_episodes, device=device,
    )
    elapsed = time.time() - t0

    # Measure C_i
    ci_metrics = measure_coordination_quality(
        organism, perception_fn, matter, device=device,
    )

    n_tail = min(100, len(metrics["rock_distance"]))
    avg_dist = sum(metrics["rock_distance"][-n_tail:]) / n_tail
    avg_contact = sum(metrics["contact_rate"][-n_tail:]) / n_tail

    return {
        "level": level_name,
        "sensory_dim": sensory_dim,
        "embedding_dim": embed_dim,
        "seed": seed,
        "avg_dist_last100": avg_dist,
        "avg_contact_last100": avg_contact,
        "elapsed_s": elapsed,
        **ci_metrics,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    checkpoint_path = results_dir / "expanded_ci_checkpoint.json"
    results_path = results_dir / "expanded_ci_sweep.json"

    embed_dims = [2, 4, 8, 16, 32]
    seeds = [42, 123, 456]
    dev = torch.device(device)

    # Shared matter
    torch.manual_seed(0)
    matter = RockPushMatter(
        emission_dim=8, action_dim=2, seed_dim=4,
    ).to(dev)

    # Resume
    completed = []
    completed_keys = set()
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            completed = json.load(f)
        for r in completed:
            completed_keys.add((r["level"], r["embedding_dim"], r["seed"]))
        print(f"Resuming: {len(completed)} done")

    # ── Build perception levels ──
    levels = []

    # 1. Oracle
    def oracle_fn(state, emission):
        return state
    for emb in embed_dims:
        for s in seeds:
            levels.append(("oracle", 4, emb, s, oracle_fn))

    # 2. Oracle + noise
    for noise_std in [0.1, 0.5]:
        def make_noisy_fn(std):
            def fn(state, emission):
                return state + torch.randn_like(state) * std
            return fn
        for emb in embed_dims:
            for s in seeds:
                levels.append((
                    f"oracle_noise{noise_std}",
                    4, emb, s, make_noisy_fn(noise_std),
                ))

    # 3. Raw emission (8D)
    def emission_fn(state, emission):
        return emission
    for emb in embed_dims:
        for s in seeds:
            levels.append(("raw_emission", 8, emb, s, emission_fn))

    # 4-6. VAE mu at different latent dims
    vae_probe_results = {}
    for lat_dim in [8, 16, 32]:
        torch.manual_seed(0)
        world = RockPushWorld(
            emission_dim=8, channel_dim=8, env_latent_dim=lat_dim,
            embedding_dim=8, action_dim=2, seed_dim=4,
            channel_noise=0.01, target_x=0.8, target_y=0.8,
        ).to(dev)
        world.matter = matter  # Share matter

        print(f"Pre-training VAE (lat={lat_dim})...", end=" ", flush=True)
        vae_losses = train_environment_rockpush(world, n_steps=1500, device=dev)
        print(f"loss={vae_losses[-1]:.4f}")

        probe = probe_vae_latent(world, device=device)
        vae_probe_results[lat_dim] = probe
        print(f"  Probe R²: mean={probe['r2_mean']:.3f}")

        env = world.environment
        ch = world.channel
        def make_vae_fn(env_module, ch_module):
            def fn(state, emission):
                with torch.no_grad():
                    ch_out = ch_module(emission)
                    mu, _ = env_module.encode(ch_out)
                return mu
            return fn

        vae_fn = make_vae_fn(env, ch)
        for emb in embed_dims:
            for s in seeds:
                levels.append((f"vae_mu_lat{lat_dim}", lat_dim, emb, s, vae_fn))

    total = len(levels)
    print(f"\nobj-014 expanded sweep: {total} configs")
    t0 = time.time()

    for level_name, sensory_dim, emb, s, perc_fn in levels:
        key = (level_name, emb, s)
        if key in completed_keys:
            continue

        idx = len(completed) + 1
        print(f"[{idx}/{total}] {level_name}, emb={emb}, seed={s}",
              end=" ... ", flush=True)

        try:
            result = run_config(
                level_name, matter, sensory_dim, emb, s, perc_fn,
                device=device,
            )
            completed.append(result)
            completed_keys.add(key)
            print(f"dist={result['avg_dist_last100']:.3f}, "
                  f"C_i={result['C_i']:.3f} "
                  f"({result['elapsed_s']:.0f}s)")
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback; traceback.print_exc()
            completed.append({
                "level": level_name, "sensory_dim": sensory_dim,
                "embedding_dim": emb, "seed": s, "error": str(e),
            })
            completed_keys.add(key)

        if len(completed) % 6 == 0:
            with open(checkpoint_path, "w") as f:
                json.dump(completed, f, indent=2)
            elapsed = time.time() - t0
            done_this_run = len(completed) - (total - len(levels) + len(completed_keys))
            if done_this_run > 0:
                rate = done_this_run / elapsed
                left = total - len(completed_keys)
                if rate > 0:
                    print(f"  [checkpoint] ETA: {left/rate/60:.0f} min")

    elapsed = time.time() - t0

    output = {
        "results": completed,
        "vae_probe_results": {str(k): v for k, v in vae_probe_results.items()},
        "total_configs": total,
        "elapsed_seconds": elapsed,
    }
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    with open(checkpoint_path, "w") as f:
        json.dump(completed, f, indent=2)

    n_ok = len([r for r in completed if "error" not in r])
    print(f"\nDone: {n_ok}/{total} in {elapsed/60:.1f} min")

    print_summary(completed)

    try:
        generate_plot(completed, vae_probe_results, results_dir)
    except Exception as e:
        print(f"Plot failed: {e}")
        import traceback; traceback.print_exc()


def print_summary(completed):
    import numpy as np

    valid = [r for r in completed if "error" not in r]
    print("\n=== Summary by level × embed_dim ===")
    by_group = defaultdict(list)
    for r in valid:
        by_group[(r["level"], r["embedding_dim"])].append(r)

    levels_seen = sorted(set(r["level"] for r in valid))
    for level in levels_seen:
        print(f"\n  {level}:")
        for emb in [2, 4, 8, 16, 32]:
            results = by_group.get((level, emb), [])
            if results:
                dists = [r["avg_dist_last100"] for r in results]
                cis = [r["C_i"] for r in results]
                print(f"    emb={emb:>2}: dist={np.mean(dists):.3f}, "
                      f"C_i={np.mean(cis):.3f}")

    # Overall correlation
    if len(valid) >= 10:
        dists = [r["avg_dist_last100"] for r in valid]
        cis = [r["C_i"] for r in valid]
        corr = torch.corrcoef(torch.stack([
            torch.tensor(dists), torch.tensor(cis)
        ]))[0, 1].item()
        print(f"\n  Overall C_i vs distance correlation: r={corr:.3f}")

    # Threshold analysis
    print("\n=== Threshold analysis ===")
    for thresh in [0.5, 0.6, 0.7]:
        above = [r for r in valid if r["C_i"] >= thresh]
        below = [r for r in valid if r["C_i"] < thresh]
        a_learn = len([r for r in above if r["avg_dist_last100"] < 0.48])
        b_learn = len([r for r in below if r["avg_dist_last100"] < 0.48])
        print(f"  C_i >= {thresh}: {len(above)} configs, "
              f"{a_learn} learn ({a_learn/max(len(above),1)*100:.0f}%)")
        print(f"  C_i <  {thresh}: {len(below)} configs, "
              f"{b_learn} learn ({b_learn/max(len(below),1)*100:.0f}%)")


def generate_plot(completed, probe_results, results_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    valid = [r for r in completed if "error" not in r]
    if not valid:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    embed_dims = [2, 4, 8, 16, 32]

    # Color map by perception level
    level_colors = {
        "oracle": "#1f77b4",
        "oracle_noise0.1": "#6baed6",
        "oracle_noise0.5": "#bdd7e7",
        "raw_emission": "#2ca02c",
        "vae_mu_lat8": "#ff7f0e",
        "vae_mu_lat16": "#d62728",
        "vae_mu_lat32": "#9467bd",
    }
    level_labels = {
        "oracle": "Oracle",
        "oracle_noise0.1": "Oracle+N(0.1)",
        "oracle_noise0.5": "Oracle+N(0.5)",
        "raw_emission": "Raw emission (8D)",
        "vae_mu_lat8": "VAE μ lat=8",
        "vae_mu_lat16": "VAE μ lat=16",
        "vae_mu_lat32": "VAE μ lat=32",
    }

    by_group = defaultdict(lambda: defaultdict(list))
    for r in valid:
        by_group[r["level"]][r["embedding_dim"]].append(r)

    # Panel 1: Distance vs embed_dim by level
    ax = axes[0, 0]
    for level in level_colors:
        if level not in by_group:
            continue
        means = [np.mean([r["avg_dist_last100"] for r in by_group[level].get(e, [{"avg_dist_last100": np.nan}])]) for e in embed_dims]
        ax.plot(embed_dims, means, marker="o", label=level_labels.get(level, level),
                color=level_colors[level], linewidth=2)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Embedding Dimension")
    ax.set_ylabel("Rock-Target Distance (lower = better)")
    ax.set_title("Task Performance Across All Conditions")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_xscale("log", base=2)
    ax.set_xticks(embed_dims)
    ax.set_xticklabels(embed_dims)

    # Panel 2: C_i vs embed_dim by level
    ax = axes[0, 1]
    for level in level_colors:
        if level not in by_group:
            continue
        means = [np.mean([r["C_i"] for r in by_group[level].get(e, [{"C_i": 0}])]) for e in embed_dims]
        ax.plot(embed_dims, means, marker="s", label=level_labels.get(level, level),
                color=level_colors[level], linewidth=2)
    ax.axhline(0.6, color="red", linestyle=":", alpha=0.5, label="Threshold (0.6)")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel("Embedding Dimension")
    ax.set_ylabel("C_i (coordination quality)")
    ax.set_title("Coordination Quality by Condition")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_xscale("log", base=2)
    ax.set_xticks(embed_dims)
    ax.set_xticklabels(embed_dims)

    # Panel 3: THE KEY PLOT — C_i vs Distance (all conditions)
    ax = axes[1, 0]
    for r in valid:
        color = level_colors.get(r["level"], "gray")
        ax.scatter(r["C_i"], r["avg_dist_last100"],
                  color=color, s=r["embedding_dim"] * 3, alpha=0.6,
                  edgecolors="black", linewidth=0.3)

    # Legend for levels
    for level in level_colors:
        if level in by_group:
            ax.scatter([], [], color=level_colors[level],
                      label=level_labels.get(level, level), s=40)

    # Fit line
    all_ci = [r["C_i"] for r in valid]
    all_dist = [r["avg_dist_last100"] for r in valid]
    if len(all_ci) >= 10:
        z_fit = np.polyfit(all_ci, all_dist, 1)
        x_range = np.linspace(min(all_ci), max(all_ci), 100)
        ax.plot(x_range, np.polyval(z_fit, x_range), "k--", alpha=0.4)
        corr = np.corrcoef(all_ci, all_dist)[0, 1]
        ax.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax.transAxes,
                fontsize=11, fontweight="bold", verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.axvline(0.6, color="red", linestyle=":", alpha=0.4)
    ax.set_xlabel("Coordination Quality C_i")
    ax.set_ylabel("Rock-Target Distance")
    ax.set_title("C_i Predicts Performance (All Conditions)")
    ax.legend(fontsize=6, loc="upper right", ncol=2)

    # Panel 4: VAE probe R² if available
    ax = axes[1, 1]
    if probe_results:
        lats = sorted(int(k) for k in probe_results.keys())
        r2_means = [probe_results[str(l)]["r2_mean"] for l in lats]
        ax.bar([f"lat={l}" for l in lats], r2_means,
               color=["#ff7f0e", "#d62728", "#9467bd"][:len(lats)], alpha=0.85)
        ax.set_ylabel("R² (linear probe: z → state)")
        ax.set_title("VAE Latent Quality")
        ax.set_ylim(0, 1.05)
    else:
        # Show threshold analysis instead
        for thresh in [0.4, 0.5, 0.6, 0.7, 0.8]:
            above = [r for r in valid if r["C_i"] >= thresh]
            if above:
                learn_rate = len([r for r in above if r["avg_dist_last100"] < 0.48]) / len(above)
                ax.bar(f"≥{thresh}", learn_rate, color="steelblue", alpha=0.8)
        ax.set_ylabel("Learning Success Rate")
        ax.set_title("Success Rate by C_i Threshold")
        ax.set_ylim(0, 1.1)

    fig.suptitle("obj-014: Expanded C_i Sweep — Does r=-0.87 Hold Universally?", fontsize=14)
    plt.tight_layout()
    out = results_dir / "obj014_expanded_ci.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
