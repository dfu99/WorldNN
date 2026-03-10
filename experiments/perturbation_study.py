#!/usr/bin/env python3
"""Perturbation study: minimum organism capacity vs. information loss.

Sweeps over:
  - Channel noise levels (information loss in transmission)
  - Environment latent dimensions (compression in medium)
  - Organism embedding dimensions (brain capacity)

Measures:
  - Success rate of 1-bit state flip
  - Mutual information at each stage of the perception chain
  - Training convergence speed

This is the core experiment of WorldNN: empirically finding the
relationship between cumulative information loss and required
organism capacity.
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
from matplotlib.colors import Normalize
from matplotlib import cm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from worldnn.world import World
from worldnn.train import train_environment, train_organism
from worldnn.utils import compute_chain_mi


def run_single_config(
    channel_noise: float,
    env_latent_dim: int,
    embedding_dim: int,
    seed: int = 42,
    device: str = "cpu",
) -> dict:
    """Run one configuration and return metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    dev = torch.device(device)

    world = World(
        emission_dim=4,
        channel_dim=4,
        env_latent_dim=env_latent_dim,
        embedding_dim=embedding_dim,
        action_dim=2,
        seed_dim=4,
        channel_noise=channel_noise,
        channel_bandwidth=1.0,
        flip_difficulty=1.0,
        matter_hidden=32,
        env_hidden=32,
        organism_hidden=32,
    ).to(dev)

    # Phase 1: pre-train environment VAE
    env_losses = train_environment(
        world, n_steps=500, batch_size=256, lr=1e-3, beta=0.1, device=dev
    )

    # Phase 2: train organism
    metrics = train_organism(
        world,
        n_episodes=400,
        steps_per_episode=10,
        batch_size=512,
        lr=1e-3,
        device=dev,
    )

    # Evaluate: run episodes and compute MI
    world.eval()
    with torch.no_grad():
        traj = world.run_episode(1024, 10, target_state=1.0, device=dev)

        # Final success rate (average over last 20 episodes)
        final_success = np.mean(metrics["success_rates"][-20:])

        # Compute MI at each stage
        all_states = torch.cat(traj["states"])
        all_emissions = torch.cat(traj["emissions"])
        all_channel = torch.cat(traj["channel_out"])
        all_z = torch.cat(traj["z_latents"])
        all_embed = torch.cat(traj["embeddings"])

        mi = compute_chain_mi(
            all_states, all_emissions, all_channel, all_z, all_embed, n_samples=2000
        )

    channel_capacity = world.channel.theoretical_capacity()

    return {
        "channel_noise": channel_noise,
        "env_latent_dim": env_latent_dim,
        "embedding_dim": embedding_dim,
        "final_success": final_success,
        "channel_capacity": channel_capacity,
        "mi": mi,
        "reward_curve": metrics["rewards"],
        "success_curve": metrics["success_rates"],
        "env_final_loss": env_losses[-1],
    }


def run_perturbation_study(results_dir: str = "results"):
    """Run the full perturbation study sweep."""
    os.makedirs(results_dir, exist_ok=True)

    # Sweep parameters
    channel_noises = [0.01, 0.1, 0.5, 1.0, 2.0]
    env_latent_dims = [1, 2, 4]
    embedding_dims = [1, 2, 4, 8, 16]

    configs = list(
        itertools.product(channel_noises, env_latent_dims, embedding_dims)
    )

    print(f"Running {len(configs)} configurations...")
    all_results = []

    for i, (noise, lat, emb) in enumerate(configs):
        print(
            f"  [{i+1}/{len(configs)}] noise={noise}, env_lat={lat}, embed={emb}",
            end="",
            flush=True,
        )
        result = run_single_config(noise, lat, emb, seed=42)
        all_results.append(result)
        print(f"  → success={result['final_success']:.3f}")

    # Save raw results
    # Convert numpy floats for JSON serialization
    serializable = []
    for r in all_results:
        sr = {k: v for k, v in r.items() if k not in ("reward_curve", "success_curve")}
        sr["mi"] = {k: float(v) for k, v in r["mi"].items()}
        sr["final_success"] = float(r["final_success"])
        sr["channel_capacity"] = float(r["channel_capacity"])
        sr["env_final_loss"] = float(r["env_final_loss"])
        serializable.append(sr)

    with open(f"{results_dir}/perturbation_results.json", "w") as f:
        json.dump(serializable, f, indent=2)

    # Generate visualizations
    plot_results(all_results, results_dir)

    return all_results


def plot_results(results: list[dict], results_dir: str):
    """Generate all visualization plots from perturbation study results."""

    # ── Plot 1: Heatmap of success rate vs (channel_noise, embedding_dim) ──
    # For each env_latent_dim
    env_lats = sorted(set(r["env_latent_dim"] for r in results))
    noises = sorted(set(r["channel_noise"] for r in results))
    emb_dims = sorted(set(r["embedding_dim"] for r in results))

    fig, axes = plt.subplots(1, len(env_lats), figsize=(5 * len(env_lats), 4), sharey=True)
    if len(env_lats) == 1:
        axes = [axes]

    for ax, lat in zip(axes, env_lats):
        grid = np.zeros((len(noises), len(emb_dims)))
        for r in results:
            if r["env_latent_dim"] == lat:
                i = noises.index(r["channel_noise"])
                j = emb_dims.index(r["embedding_dim"])
                grid[i, j] = r["final_success"]

        im = ax.imshow(grid, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1, origin="lower")
        ax.set_xticks(range(len(emb_dims)))
        ax.set_xticklabels(emb_dims)
        ax.set_yticks(range(len(noises)))
        ax.set_yticklabels([f"{n:.2f}" for n in noises])
        ax.set_xlabel("Organism Embedding Dim")
        ax.set_ylabel("Channel Noise σ")
        ax.set_title(f"Env Latent Dim = {lat}")

        # Annotate cells
        for i in range(len(noises)):
            for j in range(len(emb_dims)):
                ax.text(j, i, f"{grid[i,j]:.2f}", ha="center", va="center", fontsize=8)

    plt.colorbar(im, ax=axes, label="Success Rate", shrink=0.8)
    fig.suptitle("1-Bit State Flip Success Rate\n(Organism Capacity vs. Information Loss)", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/success_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Plot 2: MI chain for different noise levels ──
    fig, ax = plt.subplots(figsize=(8, 5))
    stages = ["I(S;X)", "I(S;Y)", "I(S;Z)", "I(S;E)"]
    stage_labels = ["Matter→\nEmission", "Channel→\nOutput", "Env→\nLatent", "Organism→\nEmbedding"]

    # Pick middle env_latent and embedding for this plot
    mid_lat = env_lats[len(env_lats) // 2]
    mid_emb = emb_dims[len(emb_dims) // 2]

    colors = cm.viridis(np.linspace(0.2, 0.9, len(noises)))
    for noise, color in zip(noises, colors):
        matching = [
            r for r in results
            if r["channel_noise"] == noise
            and r["env_latent_dim"] == mid_lat
            and r["embedding_dim"] == mid_emb
        ]
        if matching:
            r = matching[0]
            mi_vals = [r["mi"][s] for s in stages]
            ax.plot(range(4), mi_vals, "o-", color=color, label=f"σ={noise:.2f}", linewidth=2, markersize=8)

    ax.set_xticks(range(4))
    ax.set_xticklabels(stage_labels)
    ax.set_ylabel("Mutual Information I(S; ·) [nats]")
    ax.set_title(f"Information Decay Through Perception Chain\n(env_lat={mid_lat}, embed={mid_emb})")
    ax.legend(title="Channel Noise")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/mi_chain.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Plot 3: Minimum embedding dim vs channel noise for threshold success ──
    fig, ax = plt.subplots(figsize=(7, 5))
    threshold = 0.6  # success rate threshold

    for lat, marker in zip(env_lats, ["o", "s", "D"]):
        min_embs = []
        for noise in noises:
            matching = sorted(
                [r for r in results if r["channel_noise"] == noise and r["env_latent_dim"] == lat],
                key=lambda r: r["embedding_dim"],
            )
            min_emb = None
            for r in matching:
                if r["final_success"] >= threshold:
                    min_emb = r["embedding_dim"]
                    break
            min_embs.append(min_emb)

        # Plot (skip None values)
        valid = [(n, e) for n, e in zip(noises, min_embs) if e is not None]
        if valid:
            ns, es = zip(*valid)
            ax.plot(ns, es, f"-{marker}", label=f"env_lat={lat}", linewidth=2, markersize=8)

    ax.set_xlabel("Channel Noise σ")
    ax.set_ylabel(f"Min Embedding Dim for >{threshold:.0%} Success")
    ax.set_title("Required Organism Capacity vs. Environmental Noise")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    plt.tight_layout()
    plt.savefig(f"{results_dir}/min_capacity_vs_noise.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Plot 4: Training curves for select configs ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Pick interesting configs to compare
    showcase = []
    for noise in [0.01, 0.5, 2.0]:
        for emb in [2, 8]:
            matching = [
                r for r in results
                if r["channel_noise"] == noise
                and r["embedding_dim"] == emb
                and r["env_latent_dim"] == mid_lat
            ]
            if matching:
                showcase.append(matching[0])

    for r in showcase:
        label = f"σ={r['channel_noise']}, emb={r['embedding_dim']}"
        axes[0].plot(r["reward_curve"], alpha=0.8, label=label)
        axes[1].plot(r["success_curve"], alpha=0.8, label=label)

    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Average Reward")
    axes[0].set_title("Training Reward Curves")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Success Rate")
    axes[1].set_title("Training Success Curves")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{results_dir}/training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved plots to {results_dir}/")


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    run_perturbation_study("results")
