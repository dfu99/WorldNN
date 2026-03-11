#!/usr/bin/env python3
"""Perturbation study with PPO: true capacity curves.

Re-runs the original 75-config sweep (obj-002) using PPO instead of
REINFORCE. obj-004 showed REINFORCE fails on env_lat=1 due to gradient
variance, not information loss. This sweep gives the true relationship
between information loss and required organism capacity.

Sweeps:
  - Channel noise: [0.01, 0.1, 0.5, 1.0, 2.0]
  - Environment latent dim: [1, 2, 4]
  - Organism embedding dim: [1, 2, 4, 8, 16]

Total: 75 configurations, each with PPO training.
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
from worldnn.train import train_environment, train_organism_ppo
from worldnn.utils import compute_chain_mi


def run_single_config(
    channel_noise: float,
    env_latent_dim: int,
    embedding_dim: int,
    seed: int = 42,
    device: str = "cpu",
) -> dict:
    """Run one configuration with PPO and return metrics."""
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

    # Phase 2: train organism with PPO
    metrics = train_organism_ppo(
        world,
        n_episodes=500,
        steps_per_episode=10,
        batch_size=512,
        lr=3e-4,
        clip_eps=0.2,
        ppo_epochs=4,
        device=dev,
    )

    # Evaluate: run episodes and compute MI
    world.eval()
    with torch.no_grad():
        traj = world.run_episode(1024, 10, target_state=1.0, device=dev)

        final_success = np.mean(metrics["success_rates"][-20:])

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


def run_perturbation_study_ppo(results_dir: str = "results"):
    """Run the full perturbation study sweep with PPO."""
    os.makedirs(results_dir, exist_ok=True)

    channel_noises = [0.01, 0.1, 0.5, 1.0, 2.0]
    env_latent_dims = [1, 2, 4]
    embedding_dims = [1, 2, 4, 8, 16]

    configs = list(
        itertools.product(channel_noises, env_latent_dims, embedding_dims)
    )

    print(f"Running {len(configs)} configurations with PPO...")
    all_results = []

    for i, (noise, lat, emb) in enumerate(configs):
        print(
            f"  [{i+1}/{len(configs)}] noise={noise}, env_lat={lat}, embed={emb}",
            end="",
            flush=True,
        )
        result = run_single_config(noise, lat, emb, seed=42)
        all_results.append(result)
        print(f"  -> success={result['final_success']:.3f}")

    # Save raw results
    serializable = []
    for r in all_results:
        sr = {k: v for k, v in r.items() if k not in ("reward_curve", "success_curve")}
        sr["mi"] = {k: float(v) for k, v in r["mi"].items()}
        sr["final_success"] = float(r["final_success"])
        sr["channel_capacity"] = float(r["channel_capacity"])
        sr["env_final_loss"] = float(r["env_final_loss"])
        serializable.append(sr)

    with open(f"{results_dir}/perturbation_ppo_results.json", "w") as f:
        json.dump(serializable, f, indent=2)

    # Generate visualizations
    plot_results(all_results, results_dir)
    plot_comparison(all_results, results_dir)

    return all_results


def plot_results(results: list[dict], results_dir: str):
    """Generate heatmaps, MI chain, and capacity curves for PPO sweep."""

    env_lats = sorted(set(r["env_latent_dim"] for r in results))
    noises = sorted(set(r["channel_noise"] for r in results))
    emb_dims = sorted(set(r["embedding_dim"] for r in results))

    # ── Plot 1: Success heatmaps ──
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

        for i in range(len(noises)):
            for j in range(len(emb_dims)):
                ax.text(j, i, f"{grid[i,j]:.2f}", ha="center", va="center", fontsize=8)

    plt.colorbar(im, ax=axes, label="Success Rate", shrink=0.8)
    fig.suptitle("1-Bit State Flip Success Rate (PPO)\nOrganism Capacity vs. Information Loss", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/success_heatmap_ppo.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Plot 2: MI chain ──
    fig, ax = plt.subplots(figsize=(8, 5))
    stages = ["I(S;X)", "I(S;Y)", "I(S;Z)", "I(S;E)"]
    stage_labels = ["Matter→\nEmission", "Channel→\nOutput", "Env→\nLatent", "Organism→\nEmbedding"]

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
    ax.set_title(f"Information Decay Through Perception Chain (PPO)\n(env_lat={mid_lat}, embed={mid_emb})")
    ax.legend(title="Channel Noise")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/mi_chain_ppo.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Plot 3: Min embedding dim vs noise ──
    fig, ax = plt.subplots(figsize=(7, 5))
    threshold = 0.6

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

        valid = [(n, e) for n, e in zip(noises, min_embs) if e is not None]
        if valid:
            ns, es = zip(*valid)
            ax.plot(ns, es, f"-{marker}", label=f"env_lat={lat}", linewidth=2, markersize=8)

    ax.set_xlabel("Channel Noise σ")
    ax.set_ylabel(f"Min Embedding Dim for >{threshold:.0%} Success")
    ax.set_title("Required Organism Capacity vs. Noise (PPO)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    plt.tight_layout()
    plt.savefig(f"{results_dir}/min_capacity_vs_noise_ppo.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved PPO plots to {results_dir}/")


def plot_comparison(ppo_results: list[dict], results_dir: str):
    """Compare PPO vs REINFORCE results (loads old REINFORCE data)."""
    reinforce_path = f"{results_dir}/perturbation_results.json"
    if not os.path.exists(reinforce_path):
        print("No REINFORCE results found for comparison, skipping.")
        return

    with open(reinforce_path) as f:
        reinforce_data = json.load(f)

    env_lats = sorted(set(r["env_latent_dim"] for r in ppo_results))
    noises = sorted(set(r["channel_noise"] for r in ppo_results))
    emb_dims = sorted(set(r["embedding_dim"] for r in ppo_results))

    # ── Comparison: PPO vs REINFORCE difference heatmaps ──
    fig, axes = plt.subplots(1, len(env_lats), figsize=(5 * len(env_lats), 4), sharey=True)
    if len(env_lats) == 1:
        axes = [axes]

    # Build lookup for REINFORCE results
    rf_lookup = {}
    for r in reinforce_data:
        key = (r["channel_noise"], r["env_latent_dim"], r["embedding_dim"])
        rf_lookup[key] = r["final_success"]

    for ax, lat in zip(axes, env_lats):
        grid = np.zeros((len(noises), len(emb_dims)))
        for r in ppo_results:
            if r["env_latent_dim"] == lat:
                key = (r["channel_noise"], lat, r["embedding_dim"])
                rf_success = rf_lookup.get(key, 0)
                i = noises.index(r["channel_noise"])
                j = emb_dims.index(r["embedding_dim"])
                grid[i, j] = r["final_success"] - rf_success

        im = ax.imshow(grid, aspect="auto", cmap="RdBu", vmin=-0.5, vmax=0.5, origin="lower")
        ax.set_xticks(range(len(emb_dims)))
        ax.set_xticklabels(emb_dims)
        ax.set_yticks(range(len(noises)))
        ax.set_yticklabels([f"{n:.2f}" for n in noises])
        ax.set_xlabel("Organism Embedding Dim")
        ax.set_ylabel("Channel Noise σ")
        ax.set_title(f"Env Latent Dim = {lat}")

        for i in range(len(noises)):
            for j in range(len(emb_dims)):
                val = grid[i, j]
                color = "white" if abs(val) > 0.25 else "black"
                ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=7, color=color)

    plt.colorbar(im, ax=axes, label="PPO − REINFORCE (Δ success)", shrink=0.8)
    fig.suptitle("PPO vs REINFORCE: Success Rate Improvement", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/ppo_vs_reinforce.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Summary stats ──
    improvements = []
    for r in ppo_results:
        key = (r["channel_noise"], r["env_latent_dim"], r["embedding_dim"])
        if key in rf_lookup:
            improvements.append(r["final_success"] - rf_lookup[key])

    if improvements:
        print(f"\nPPO vs REINFORCE comparison:")
        print(f"  Mean improvement: {np.mean(improvements):+.3f}")
        print(f"  Max improvement:  {np.max(improvements):+.3f}")
        print(f"  Configs improved: {sum(1 for x in improvements if x > 0.05)}/{len(improvements)}")

    print(f"Saved comparison plot to {results_dir}/ppo_vs_reinforce.png")


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    run_perturbation_study_ppo("results")
