#!/usr/bin/env python3
"""Obj-008: Continuous state spaces — position targeting task.

Extends beyond binary state flipping to continuous position control.
The organism must push matter to a target position (0.8) through the
same lossy perception-action loop.

Key questions:
1. Does the embedding dim matter more for continuous control?
   (Binary task showed dim is almost irrelevant with PPO)
2. How does noise affect proportional control vs binary switching?
3. Is the env_lat=1 bottleneck still present for continuous tasks?

Sweep: noise × env_lat × embed_dim, PPO only, 3 seeds each.
"""

import sys
import os
import json
import itertools
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from worldnn.world import ContinuousWorld
from worldnn.train import train_environment_continuous, train_organism_ppo_continuous


def run_config(
    channel_noise: float,
    env_latent_dim: int,
    embedding_dim: int,
    seed: int = 42,
    device: str = "cpu",
    n_episodes: int = 500,
) -> dict:
    """Run one continuous-state configuration."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)

    world = ContinuousWorld(
        emission_dim=4,
        channel_dim=4,
        env_latent_dim=env_latent_dim,
        embedding_dim=embedding_dim,
        action_dim=2,
        seed_dim=4,
        channel_noise=channel_noise,
        force_scale=0.1,
        target_position=0.8,
        env_hidden=32,
        organism_hidden=32,
    ).to(dev)

    train_environment_continuous(
        world, n_steps=500, batch_size=256, lr=1e-3, beta=0.1, device=dev
    )

    metrics = train_organism_ppo_continuous(
        world, n_episodes=n_episodes, steps_per_episode=10,
        batch_size=512, lr=3e-4, clip_eps=0.2, ppo_epochs=4, device=dev,
    )

    final_dist = float(np.mean(metrics["mean_distance"][-20:]))
    early_dist = float(np.mean(metrics["mean_distance"][40:60])) if len(metrics["mean_distance"]) > 60 else 0.5

    return {
        "channel_noise": channel_noise,
        "env_latent_dim": env_latent_dim,
        "embedding_dim": embedding_dim,
        "seed": seed,
        "final_distance": final_dist,
        "early_distance": early_dist,
        "final_reward": float(np.mean(metrics["rewards"][-20:])),
        "distance_curve": metrics["mean_distance"],
    }


def run_study(results_dir: str = "results"):
    """Run the continuous state sweep."""
    os.makedirs(results_dir, exist_ok=True)

    noise_levels = [0.01, 0.1, 0.5, 1.0, 2.0]
    env_lats = [1, 2, 4]
    embed_dims = [1, 2, 4, 8]
    seeds = [42, 123, 456]

    configs = list(itertools.product(noise_levels, env_lats, embed_dims, seeds))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Total configs: {len(configs)}")

    checkpoint_path = f"{results_dir}/continuous_checkpoint.json"
    all_results = []
    start_idx = 0
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            all_results = json.load(f)["results"]
        start_idx = len(all_results)
        print(f"Resuming from {start_idx}/{len(configs)}")

    for i, (noise, lat, emb, seed) in enumerate(configs):
        if i < start_idx:
            continue
        print(
            f"  [{i+1}/{len(configs)}] noise={noise}, lat={lat}, "
            f"embed={emb}, seed={seed}",
            end="", flush=True,
        )
        result = run_config(noise, lat, emb, seed=seed, device=device)
        save_result = {k: v for k, v in result.items() if k != "distance_curve"}
        all_results.append(save_result)
        print(f"  -> dist={result['final_distance']:.3f}")

        with open(checkpoint_path, "w") as f:
            json.dump({"results": all_results}, f, indent=2)

    with open(f"{results_dir}/continuous_results.json", "w") as f:
        json.dump({"results": all_results}, f, indent=2)

    plot_results(all_results, results_dir)
    return all_results


def plot_results(results: list[dict], results_dir: str):
    """Generate comparison plots for continuous state task."""
    from collections import defaultdict

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Row 1: Distance-to-target heatmaps (noise × embed) for each env_lat
    for ax, lat in zip(axes[0], [1, 2, 4]):
        noise_levels = sorted(set(r["channel_noise"] for r in results if r["env_latent_dim"] == lat))
        embed_dims = sorted(set(r["embedding_dim"] for r in results if r["env_latent_dim"] == lat))

        matrix = np.full((len(embed_dims), len(noise_levels)), np.nan)
        for i, emb in enumerate(embed_dims):
            for j, noise in enumerate(noise_levels):
                vals = [r["final_distance"] for r in results
                        if r["channel_noise"] == noise and r["env_latent_dim"] == lat
                        and r["embedding_dim"] == emb]
                if vals:
                    matrix[i, j] = np.mean(vals)

        im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=0.5, origin="lower")
        ax.set_xticks(range(len(noise_levels)))
        ax.set_xticklabels([f"{n}" for n in noise_levels], rotation=45)
        ax.set_yticks(range(len(embed_dims)))
        ax.set_yticklabels([str(e) for e in embed_dims])
        ax.set_xlabel("Channel Noise σ")
        ax.set_ylabel("Embedding Dim")
        ax.set_title(f"env_lat={lat}", fontweight="bold")

        for i in range(len(embed_dims)):
            for j in range(len(noise_levels)):
                if not np.isnan(matrix[i, j]):
                    color = "white" if matrix[i, j] > 0.3 else "black"
                    ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                            fontsize=8, fontweight="bold", color=color)

        plt.colorbar(im, ax=ax, shrink=0.8, label="Mean Distance to Target")

    # Row 2: Curves
    # Left: embed_dim effect (averaged over seeds)
    ax = axes[1][0]
    for lat, color in [(1, "#E65100"), (2, "#1565C0"), (4, "#388E3C")]:
        embed_dims = sorted(set(r["embedding_dim"] for r in results if r["env_latent_dim"] == lat))
        # Use noise=0.1 as representative
        means = []
        for emb in embed_dims:
            vals = [r["final_distance"] for r in results
                    if r["env_latent_dim"] == lat and r["embedding_dim"] == emb
                    and r["channel_noise"] == 0.1]
            means.append(np.mean(vals) if vals else np.nan)
        ax.plot(embed_dims, means, "-o", color=color, linewidth=2, label=f"lat={lat}")

    ax.set_xlabel("Embedding Dimension")
    ax.set_ylabel("Mean Distance to Target")
    ax.set_title("Embed Dim Effect (noise=0.1)", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Middle: noise effect (averaged over embed dims)
    ax = axes[1][1]
    for lat, color in [(1, "#E65100"), (2, "#1565C0"), (4, "#388E3C")]:
        noise_levels = sorted(set(r["channel_noise"] for r in results if r["env_latent_dim"] == lat))
        means = []
        for noise in noise_levels:
            vals = [r["final_distance"] for r in results
                    if r["env_latent_dim"] == lat and r["channel_noise"] == noise]
            means.append(np.mean(vals) if vals else np.nan)
        ax.plot(noise_levels, means, "-o", color=color, linewidth=2, label=f"lat={lat}")

    ax.set_xlabel("Channel Noise σ")
    ax.set_ylabel("Mean Distance to Target")
    ax.set_title("Noise Effect (all embeds)", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")

    # Right: Binary vs Continuous comparison text
    ax = axes[1][2]
    ax.axis("off")
    ax.text(0.1, 0.5,
            "Continuous State Task\n\n"
            "• Target: position = 0.8\n"
            "• Reward: 1 - |pos - target|\n"
            "• Force: tanh(action · proj) × 0.1\n"
            "• 10 steps per episode\n\n"
            "Key comparison vs Binary:\n"
            "• Binary: dim irrelevant with PPO\n"
            "• Continuous: does dim matter\n"
            "  for proportional control?",
            transform=ax.transAxes, fontsize=11,
            verticalalignment="center",
            bbox=dict(boxstyle="round", facecolor="#E3F2FD", alpha=0.8))

    fig.suptitle(
        "Continuous State Control: Position Targeting Through Lossy Channels\n"
        "Mean distance to target (lower is better), 3 seeds",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    out = f"{results_dir}/continuous_state.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    run_study(args.results_dir)
