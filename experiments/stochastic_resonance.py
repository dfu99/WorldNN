#!/usr/bin/env python3
"""Obj-006: Investigate stochastic resonance at env_lat=1.

At env_lat=1, the PPO sweep (obj-005) found that noise=0.5 outperforms
noise=0.01 (86% vs 72%). This is unexpected — more noise should hurt.

This experiment:
1. Finer noise grid around the peak (13 levels from 0.01 to 2.0)
2. Multiple seeds (5) for confidence intervals
3. REINFORCE control to check if resonance is PPO-specific
4. Also env_lat=2 as control (no resonance expected)
5. Rerun timed-out configs: noise=2.0, env_lat=4, embed=[1,2,4,8,16]

Total configs:
  - Resonance study: 13 noise × 2 env_lat × 2 algo × 5 seeds = 260
    (using embed=4 as representative)
  - Timeout reruns: 5 configs × 1 seed (PPO only)
  Total: 265 configs
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

from worldnn.world import World
from worldnn.train import train_environment, train_organism_ppo, train_organism
from worldnn.utils import compute_chain_mi


def run_config(
    channel_noise: float,
    env_latent_dim: int,
    embedding_dim: int,
    algorithm: str,  # "ppo" or "reinforce"
    seed: int = 42,
    device: str = "cpu",
    n_episodes: int = 500,
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
    if algorithm == "ppo":
        metrics = train_organism_ppo(
            world, n_episodes=n_episodes, steps_per_episode=10,
            batch_size=512, lr=3e-4, clip_eps=0.2, ppo_epochs=4, device=dev,
        )
    else:
        metrics = train_organism(
            world, n_episodes=n_episodes, steps_per_episode=10,
            batch_size=512, lr=3e-4, device=dev,
        )

    final_success = float(np.mean(metrics["success_rates"][-20:]))

    return {
        "channel_noise": channel_noise,
        "env_latent_dim": env_latent_dim,
        "embedding_dim": embedding_dim,
        "algorithm": algorithm,
        "seed": seed,
        "final_success": final_success,
        "env_final_loss": float(env_losses[-1]),
        "success_curve": metrics["success_rates"],
    }


def run_stochastic_resonance_study(results_dir: str = "results"):
    """Run the full stochastic resonance investigation."""
    os.makedirs(results_dir, exist_ok=True)

    # Finer noise grid centered around the 0.5 peak
    noise_grid = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.5, 2.0]
    env_lats = [1, 2]
    algorithms = ["ppo", "reinforce"]
    seeds = [42, 123, 456, 789, 1337]
    embed_dim = 4  # representative

    # Part 1: Resonance study
    resonance_configs = list(itertools.product(noise_grid, env_lats, algorithms, seeds))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Part 1: {len(resonance_configs)} resonance configs")

    # Resume from checkpoint
    checkpoint_path = f"{results_dir}/stochastic_resonance_checkpoint.json"
    all_results = []
    start_idx = 0
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            data = json.load(f)
        all_results = data["resonance_results"]
        start_idx = len(all_results)
        print(f"  Resuming from checkpoint: {start_idx}/{len(resonance_configs)} done")

    for i, (noise, lat, algo, seed) in enumerate(resonance_configs):
        if i < start_idx:
            continue
        print(
            f"  [{i+1}/{len(resonance_configs)}] noise={noise}, lat={lat}, "
            f"algo={algo}, seed={seed}",
            end="", flush=True,
        )
        result = run_config(noise, lat, embed_dim, algo, seed=seed, device=device)
        # Don't save success_curve in checkpoint (too large)
        save_result = {k: v for k, v in result.items() if k != "success_curve"}
        all_results.append(save_result)
        print(f"  -> {result['final_success']:.3f}")

        # Checkpoint every config
        with open(checkpoint_path, "w") as f:
            json.dump({"resonance_results": all_results, "timeout_results": []}, f, indent=2)

    # Part 2: Timeout reruns (noise=2.0, env_lat=4, all embed dims)
    print("\nPart 2: Timeout reruns (noise=2.0, env_lat=4)")
    timeout_configs = [(2.0, 4, emb) for emb in [1, 2, 4, 8, 16]]
    timeout_results = []

    for noise, lat, emb in timeout_configs:
        print(f"  noise={noise}, lat={lat}, embed={emb}", end="", flush=True)
        result = run_config(noise, lat, emb, "ppo", seed=42, device=device, n_episodes=800)
        save_result = {k: v for k, v in result.items() if k != "success_curve"}
        timeout_results.append(save_result)
        print(f"  -> {result['final_success']:.3f}")

    # Save final results
    final_data = {
        "resonance_results": all_results,
        "timeout_results": timeout_results,
    }
    with open(f"{results_dir}/stochastic_resonance_results.json", "w") as f:
        json.dump(final_data, f, indent=2)

    # Generate plots
    plot_resonance(all_results, results_dir)
    plot_algorithm_comparison(all_results, results_dir)

    return final_data


def plot_resonance(results: list[dict], results_dir: str):
    """Plot the stochastic resonance curve with confidence intervals."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, lat in zip(axes, [1, 2]):
        for algo, color, marker in [("ppo", "#2196F3", "o"), ("reinforce", "#FF5722", "s")]:
            subset = [r for r in results if r["env_latent_dim"] == lat and r["algorithm"] == algo]
            if not subset:
                continue

            # Group by noise level
            noise_levels = sorted(set(r["channel_noise"] for r in subset))
            means, stds, noises = [], [], []

            for noise in noise_levels:
                vals = [r["final_success"] for r in subset if r["channel_noise"] == noise]
                if vals:
                    means.append(np.mean(vals))
                    stds.append(np.std(vals))
                    noises.append(noise)

            means = np.array(means)
            stds = np.array(stds)
            noises = np.array(noises)

            ax.plot(noises, means, f"-{marker}", color=color, label=algo.upper(),
                    linewidth=2, markersize=7)
            ax.fill_between(noises, means - stds, means + stds, color=color, alpha=0.15)

        ax.set_xlabel("Channel Noise σ", fontsize=12)
        ax.set_ylabel("Success Rate", fontsize=12)
        ax.set_title(f"env_latent_dim = {lat}", fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xscale("log")

        # Mark the resonance peak if at lat=1
        if lat == 1:
            ppo_subset = [r for r in results if r["env_latent_dim"] == 1 and r["algorithm"] == "ppo"]
            if ppo_subset:
                noise_means = {}
                for r in ppo_subset:
                    noise_means.setdefault(r["channel_noise"], []).append(r["final_success"])
                noise_means = {k: np.mean(v) for k, v in noise_means.items()}
                if noise_means:
                    peak_noise = max(noise_means, key=noise_means.get)
                    peak_val = noise_means[peak_noise]
                    ax.annotate(
                        f"Peak: σ={peak_noise}\n{peak_val:.1%}",
                        xy=(peak_noise, peak_val),
                        xytext=(peak_noise * 3, peak_val - 0.15),
                        arrowprops=dict(arrowstyle="->", color="black"),
                        fontsize=10, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                    )

    fig.suptitle(
        "Stochastic Resonance in Perception-Action Loops\n"
        "Success rate vs. channel noise (embed_dim=4, 5 seeds)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(f"{results_dir}/stochastic_resonance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {results_dir}/stochastic_resonance.png")


def plot_algorithm_comparison(results: list[dict], results_dir: str):
    """Plot PPO vs REINFORCE resonance difference."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for lat, color, ls in [(1, "#9C27B0", "-"), (2, "#4CAF50", "--")]:
        noise_levels = sorted(set(
            r["channel_noise"] for r in results if r["env_latent_dim"] == lat
        ))

        diffs, diff_stds, noises = [], [], []
        for noise in noise_levels:
            ppo_vals = [r["final_success"] for r in results
                        if r["channel_noise"] == noise and r["env_latent_dim"] == lat
                        and r["algorithm"] == "ppo"]
            rf_vals = [r["final_success"] for r in results
                       if r["channel_noise"] == noise and r["env_latent_dim"] == lat
                       and r["algorithm"] == "reinforce"]
            if ppo_vals and rf_vals:
                diff = np.mean(ppo_vals) - np.mean(rf_vals)
                # Propagate uncertainty
                std = np.sqrt(np.std(ppo_vals)**2 + np.std(rf_vals)**2)
                diffs.append(diff)
                diff_stds.append(std)
                noises.append(noise)

        if diffs:
            diffs = np.array(diffs)
            diff_stds = np.array(diff_stds)
            ax.plot(noises, diffs, f"-o", color=color, linestyle=ls,
                    label=f"env_lat={lat}", linewidth=2, markersize=7)
            ax.fill_between(noises, diffs - diff_stds, diffs + diff_stds,
                            color=color, alpha=0.15)

    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Channel Noise σ", fontsize=12)
    ax.set_ylabel("PPO − REINFORCE (Δ success)", fontsize=12)
    ax.set_title(
        "PPO Advantage Over REINFORCE\n"
        "Does stochastic resonance depend on the optimizer?",
        fontsize=13,
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    plt.tight_layout()
    plt.savefig(f"{results_dir}/resonance_algo_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {results_dir}/resonance_algo_comparison.png")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    run_stochastic_resonance_study(args.results_dir)
