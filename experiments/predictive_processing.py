#!/usr/bin/env python3
"""Obj-007: Does predictive processing improve learning in lossy environments?

Compare PPO vs PPO+Prediction across conditions where learning is hardest:
- env_lat=1 (low-dimensional bottleneck)
- Various noise levels (0.01, 0.1, 0.5, 1.0)
- Various prediction loss coefficients (0.0, 0.05, 0.1, 0.5)

The hypothesis: organisms that predict their next observation build better
internal models and learn faster/more reliably, especially when perception
is severely bottlenecked.
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
from worldnn.organism import PredictiveOrganism
from worldnn.train import train_environment, train_organism_ppo, train_organism_ppo_predictive


def make_world(env_latent_dim, channel_noise, embedding_dim, predictive=False, device="cpu"):
    """Create a World, optionally with a PredictiveOrganism."""
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
    )

    if predictive:
        # Replace organism with PredictiveOrganism
        world.organism = PredictiveOrganism(
            sensory_dim=env_latent_dim,
            embedding_dim=embedding_dim,
            action_dim=2,
            hidden_size=32,
        )

    return world.to(torch.device(device))


def run_config(
    channel_noise: float,
    env_latent_dim: int,
    embedding_dim: int,
    pred_coef: float,
    seed: int = 42,
    device: str = "cpu",
    n_episodes: int = 500,
) -> dict:
    """Run one configuration."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)

    predictive = pred_coef > 0
    world = make_world(env_latent_dim, channel_noise, embedding_dim,
                       predictive=predictive, device=device)

    # Phase 1: pre-train environment VAE
    train_environment(world, n_steps=500, batch_size=256, lr=1e-3, beta=0.1, device=dev)

    # Phase 2: train organism
    if predictive:
        metrics = train_organism_ppo_predictive(
            world, n_episodes=n_episodes, steps_per_episode=10,
            batch_size=512, lr=3e-4, clip_eps=0.2, ppo_epochs=4,
            pred_coef=pred_coef, device=dev,
        )
    else:
        metrics = train_organism_ppo(
            world, n_episodes=n_episodes, steps_per_episode=10,
            batch_size=512, lr=3e-4, clip_eps=0.2, ppo_epochs=4, device=dev,
        )

    final_success = float(np.mean(metrics["success_rates"][-20:]))
    early_success = float(np.mean(metrics["success_rates"][40:60])) if len(metrics["success_rates"]) > 60 else 0.0

    result = {
        "channel_noise": channel_noise,
        "env_latent_dim": env_latent_dim,
        "embedding_dim": embedding_dim,
        "pred_coef": pred_coef,
        "seed": seed,
        "final_success": final_success,
        "early_success": early_success,
        "success_curve": metrics["success_rates"],
    }
    if "pred_losses" in metrics:
        result["final_pred_loss"] = float(np.mean(metrics["pred_losses"][-20:]))

    return result


def run_study(results_dir: str = "results"):
    """Run the predictive processing comparison."""
    os.makedirs(results_dir, exist_ok=True)

    noise_levels = [0.01, 0.1, 0.5, 1.0]
    env_lats = [1, 2]
    pred_coefs = [0.0, 0.05, 0.1, 0.5]  # 0.0 = baseline PPO
    seeds = [42, 123, 456]
    embed_dim = 4

    configs = list(itertools.product(noise_levels, env_lats, pred_coefs, seeds))

    device = os.environ.get("WORLDNN_DEVICE", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("WORLDNN_DEVICE=cuda set but CUDA unavailable")
    print(f"Device: {device}")
    print(f"Total configs: {len(configs)}")

    # Resume from checkpoint
    checkpoint_path = f"{results_dir}/predictive_checkpoint.json"
    all_results = []
    start_idx = 0
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            all_results = json.load(f)["results"]
        start_idx = len(all_results)
        print(f"Resuming from {start_idx}/{len(configs)}")

    for i, (noise, lat, pred_c, seed) in enumerate(configs):
        if i < start_idx:
            continue
        label = f"pred={pred_c}" if pred_c > 0 else "baseline"
        print(
            f"  [{i+1}/{len(configs)}] noise={noise}, lat={lat}, "
            f"{label}, seed={seed}",
            end="", flush=True,
        )
        result = run_config(noise, lat, embed_dim, pred_c, seed=seed, device=device)
        # Don't save curves in checkpoint
        save_result = {k: v for k, v in result.items() if k != "success_curve"}
        all_results.append(save_result)
        print(f"  -> {result['final_success']:.3f}")

        with open(checkpoint_path, "w") as f:
            json.dump({"results": all_results}, f, indent=2)

    # Save final
    with open(f"{results_dir}/predictive_results.json", "w") as f:
        json.dump({"results": all_results}, f, indent=2)

    plot_results(all_results, results_dir)
    return all_results


def plot_results(results: list[dict], results_dir: str):
    """Generate comparison plots."""
    from collections import defaultdict

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax_row, lat in zip(axes, [1, 2]):
        noise_levels = sorted(set(r["channel_noise"] for r in results if r["env_latent_dim"] == lat))
        pred_coefs = sorted(set(r["pred_coef"] for r in results))
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(pred_coefs)))

        # Left: final success
        ax = ax_row[0]
        for pred_c, color in zip(pred_coefs, colors):
            means, stds, noises = [], [], []
            for noise in noise_levels:
                vals = [r["final_success"] for r in results
                        if r["channel_noise"] == noise and r["env_latent_dim"] == lat
                        and r["pred_coef"] == pred_c]
                if vals:
                    means.append(np.mean(vals))
                    stds.append(np.std(vals))
                    noises.append(noise)
            label = f"pred={pred_c}" if pred_c > 0 else "PPO baseline"
            ax.plot(noises, means, "-o", color=color, label=label, linewidth=2, markersize=6)
            if stds:
                means, stds = np.array(means), np.array(stds)
                ax.fill_between(noises, means - stds, means + stds, color=color, alpha=0.12)

        ax.set_xlabel("Channel Noise σ")
        ax.set_ylabel("Final Success Rate")
        ax.set_title(f"env_lat={lat}: Final Performance", fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")
        ax.set_ylim(-0.05, 1.05)

        # Right: early success (learning speed)
        ax = ax_row[1]
        for pred_c, color in zip(pred_coefs, colors):
            means, noises = [], []
            for noise in noise_levels:
                vals = [r["early_success"] for r in results
                        if r["channel_noise"] == noise and r["env_latent_dim"] == lat
                        and r["pred_coef"] == pred_c]
                if vals:
                    means.append(np.mean(vals))
                    noises.append(noise)
            label = f"pred={pred_c}" if pred_c > 0 else "PPO baseline"
            ax.plot(noises, means, "-o", color=color, label=label, linewidth=2, markersize=6)

        ax.set_xlabel("Channel Noise σ")
        ax.set_ylabel("Early Success Rate (ep 40-60)")
        ax.set_title(f"env_lat={lat}: Learning Speed", fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")
        ax.set_ylim(-0.05, 1.05)

    fig.suptitle(
        "Predictive Processing: Does Predicting Next Observation Improve Learning?\n"
        "PPO vs PPO+Prediction (embed=4, 3 seeds)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    out = f"{results_dir}/predictive_processing.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")

    # Improvement heatmap
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, lat in zip(axes, [1, 2]):
        noise_levels = sorted(set(r["channel_noise"] for r in results if r["env_latent_dim"] == lat))
        pred_coefs_nonzero = sorted(set(r["pred_coef"] for r in results if r["pred_coef"] > 0))

        matrix = np.full((len(pred_coefs_nonzero), len(noise_levels)), np.nan)
        for i, pred_c in enumerate(pred_coefs_nonzero):
            for j, noise in enumerate(noise_levels):
                pred_vals = [r["final_success"] for r in results
                             if r["channel_noise"] == noise and r["env_latent_dim"] == lat
                             and r["pred_coef"] == pred_c]
                base_vals = [r["final_success"] for r in results
                             if r["channel_noise"] == noise and r["env_latent_dim"] == lat
                             and r["pred_coef"] == 0.0]
                if pred_vals and base_vals:
                    matrix[i, j] = np.mean(pred_vals) - np.mean(base_vals)

        im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-0.2, vmax=0.2)
        ax.set_xticks(range(len(noise_levels)))
        ax.set_xticklabels([f"{n}" for n in noise_levels])
        ax.set_yticks(range(len(pred_coefs_nonzero)))
        ax.set_yticklabels([f"{p}" for p in pred_coefs_nonzero])
        ax.set_xlabel("Channel Noise σ")
        ax.set_ylabel("Prediction Coef")
        ax.set_title(f"env_lat={lat}: Δ Success vs Baseline", fontweight="bold")

        for i in range(len(pred_coefs_nonzero)):
            for j in range(len(noise_levels)):
                if not np.isnan(matrix[i, j]):
                    color = "white" if abs(matrix[i, j]) > 0.1 else "black"
                    sign = "+" if matrix[i, j] > 0 else ""
                    ax.text(j, i, f"{sign}{matrix[i, j]:.2f}", ha="center", va="center",
                            fontsize=9, fontweight="bold", color=color)

        plt.colorbar(im, ax=ax, label="Δ Success Rate", shrink=0.8)

    fig.suptitle(
        "Predictive Processing Improvement Over PPO Baseline",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    out = f"{results_dir}/predictive_improvement.png"
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
