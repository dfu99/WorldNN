#!/usr/bin/env python3
"""Deep analysis of env_latent_dim=1 failure mode.

Key puzzle: some env_lat=1 configs show I(S;E) ~ 0.62 but only ~5% success,
while env_lat=2 configs with similar I(S;E) ~ 0.63 achieve ~77% success.

Hypotheses tested:
  H1: MI overestimated — KSG estimator is noisy for 1D binary→1D continuous
  H2: Information present but not separable — distributions overlap
  H3: VAE training insufficient — 1D latent can't reconstruct well enough

Approach: re-train key configs, extract z|state=0 and z|state=1 distributions,
compute separability metrics, and visualize.
"""

import sys
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from worldnn.world import World
from worldnn.train import train_environment, train_organism
from worldnn.utils import compute_chain_mi


def train_and_collect(env_latent_dim, channel_noise=0.1, embedding_dim=4, seed=42):
    """Train a world and collect latent distributions by state."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device("cpu")

    world = World(
        emission_dim=4, channel_dim=4,
        env_latent_dim=env_latent_dim, embedding_dim=embedding_dim,
        action_dim=2, seed_dim=4,
        channel_noise=channel_noise, channel_bandwidth=1.0,
        flip_difficulty=1.0,
        matter_hidden=32, env_hidden=32, organism_hidden=32,
    ).to(dev)

    env_losses = train_environment(world, n_steps=500, batch_size=256, lr=1e-3, beta=0.1, device=dev)
    metrics = train_organism(world, n_episodes=400, steps_per_episode=10, batch_size=512, lr=1e-3, device=dev)

    # Collect latent distributions
    world.eval()
    with torch.no_grad():
        traj = world.run_episode(2048, 10, target_state=1.0, device=dev)
        states = torch.cat(traj["states"]).numpy()
        z_latents = torch.cat(traj["z_latents"]).numpy()
        embeddings = torch.cat(traj["embeddings"]).numpy()
        emissions = torch.cat(traj["emissions"]).numpy()
        channel_out = torch.cat(traj["channel_out"]).numpy()
        mu = torch.cat(traj["mu"]).numpy()
        logvar = torch.cat(traj["logvar"]).numpy()

    final_success = np.mean(metrics["success_rates"][-20:])

    return {
        "states": states,
        "z": z_latents,
        "mu": mu,
        "logvar": logvar,
        "embeddings": embeddings,
        "emissions": emissions,
        "channel_out": channel_out,
        "env_losses": env_losses,
        "success_curve": metrics["success_rates"],
        "final_success": final_success,
    }


def compute_separability(z, states):
    """Compute separability metrics for latent space."""
    z0 = z[states == 0]
    z1 = z[states == 1]

    # 1. Linear separability: optimal threshold accuracy (per-dim, take best)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
        z0 = z0.reshape(-1, 1)
        z1 = z1.reshape(-1, 1)

    best_acc = 0
    best_dim = 0
    for d in range(z.shape[1]):
        thresholds = np.linspace(z[:, d].min(), z[:, d].max(), 200)
        for t in thresholds:
            pred = (z[:, d] > t).astype(float)
            acc = max(
                np.mean(pred == states),
                np.mean((1 - pred) == states)
            )
            if acc > best_acc:
                best_acc = acc
                best_dim = d

    # 2. Mean separation / pooled std (Cohen's d, per best dim)
    mean0 = z0[:, best_dim].mean()
    mean1 = z1[:, best_dim].mean()
    pooled_std = np.sqrt((z0[:, best_dim].var() + z1[:, best_dim].var()) / 2)
    cohens_d = abs(mean1 - mean0) / (pooled_std + 1e-10)

    # 3. Overlap coefficient (using KDE)
    try:
        kde0 = gaussian_kde(z0[:, best_dim])
        kde1 = gaussian_kde(z1[:, best_dim])
        x_grid = np.linspace(z[:, best_dim].min(), z[:, best_dim].max(), 500)
        overlap = np.trapz(np.minimum(kde0(x_grid), kde1(x_grid)), x_grid)
    except Exception:
        overlap = float("nan")

    return {
        "threshold_accuracy": best_acc,
        "best_dim": best_dim,
        "cohens_d": cohens_d,
        "overlap_coeff": overlap,
        "mean_s0": float(mean0),
        "mean_s1": float(mean1),
        "std_s0": float(z0[:, best_dim].std()),
        "std_s1": float(z1[:, best_dim].std()),
    }


def main():
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # Run key configs
    configs = [
        {"env_latent_dim": 1, "channel_noise": 0.1, "label": "env_lat=1, noise=0.1"},
        {"env_latent_dim": 2, "channel_noise": 0.1, "label": "env_lat=2, noise=0.1"},
        {"env_latent_dim": 4, "channel_noise": 0.1, "label": "env_lat=4, noise=0.1"},
        {"env_latent_dim": 1, "channel_noise": 0.01, "label": "env_lat=1, noise=0.01"},
        {"env_latent_dim": 2, "channel_noise": 0.01, "label": "env_lat=2, noise=0.01"},
    ]

    collected = []
    for cfg in configs:
        print(f"Training: {cfg['label']}...", flush=True)
        data = train_and_collect(cfg["env_latent_dim"], cfg["channel_noise"])
        sep = compute_separability(data["z"], data["states"])
        data["sep"] = sep
        data["config"] = cfg
        collected.append(data)
        print(f"  success={data['final_success']:.3f}, "
              f"thresh_acc={sep['threshold_accuracy']:.3f}, "
              f"cohens_d={sep['cohens_d']:.3f}, "
              f"overlap={sep['overlap_coeff']:.3f}")

    # ── Figure 1: Latent distributions z|state for env_lat=1 vs 2 vs 4 ──
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Top row: z distributions for env_lat=1, 2, 4 (all noise=0.1)
    for col, data in enumerate(collected[:3]):
        ax = fig.add_subplot(gs[0, col])
        z = data["z"]
        states = data["states"]
        cfg = data["config"]
        sep = data["sep"]

        dim = sep["best_dim"]
        z0 = z[states == 0][:, dim] if z.ndim > 1 else z[states == 0]
        z1 = z[states == 1][:, dim] if z.ndim > 1 else z[states == 1]

        ax.hist(z0, bins=60, alpha=0.6, density=True, label="state=0", color="tab:blue")
        ax.hist(z1, bins=60, alpha=0.6, density=True, label="state=1", color="tab:red")
        ax.set_title(f"{cfg['label']}\nsuccess={data['final_success']:.1%}")
        ax.set_xlabel(f"z[{dim}] (best separating dim)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

        # Add separability stats
        txt = (f"d={sep['cohens_d']:.2f}\n"
               f"acc={sep['threshold_accuracy']:.1%}\n"
               f"ovlp={sep['overlap_coeff']:.2f}")
        ax.text(0.98, 0.98, txt, transform=ax.transAxes, va="top", ha="right",
                fontsize=8, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Bottom-left: VAE reconstruction loss comparison
    ax = fig.add_subplot(gs[1, 0])
    for data in collected[:3]:
        ax.plot(data["env_losses"], alpha=0.8, label=data["config"]["label"])
    ax.set_xlabel("VAE Training Step")
    ax.set_ylabel("VAE Loss")
    ax.set_title("Environment VAE Training")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Bottom-middle: Separability summary bar chart
    ax = fig.add_subplot(gs[1, 1])
    labels = [d["config"]["label"] for d in collected]
    thresh_accs = [d["sep"]["threshold_accuracy"] for d in collected]
    successes = [d["final_success"] for d in collected]
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w/2, thresh_accs, w, label="Threshold Accuracy", color="steelblue")
    ax.bar(x + w/2, successes, w, label="Task Success Rate", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace(", ", "\n") for l in labels], fontsize=7)
    ax.set_ylabel("Rate")
    ax.set_title("Separability vs. Task Success")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")

    # Bottom-right: Cohen's d vs success
    ax = fig.add_subplot(gs[1, 2])
    ds = [d["sep"]["cohens_d"] for d in collected]
    ax.scatter(ds, successes, s=100, c=["tab:red", "tab:green", "tab:blue", "tab:orange", "tab:purple"],
               edgecolors="black", zorder=5)
    for i, data in enumerate(collected):
        ax.annotate(data["config"]["label"], (ds[i], successes[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=7)
    ax.set_xlabel("Cohen's d (latent separation)")
    ax.set_ylabel("Task Success Rate")
    ax.set_title("Effect Size vs. Task Success")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Diagnosing env_latent_dim=1 Failure Mode", fontsize=14, y=0.98)
    plt.savefig(f"{results_dir}/latent_failure_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {results_dir}/latent_failure_analysis.png")

    # ── Figure 2: VAE latent space 2D scatter (for env_lat=2) ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, data in zip(axes, [collected[1], collected[2]]):  # env_lat=2, env_lat=4
        z = data["z"]
        states = data["states"]
        cfg = data["config"]

        # Use first 2 latent dims
        mask0 = states == 0
        mask1 = states == 1
        n_plot = min(2000, mask0.sum(), mask1.sum())
        idx0 = np.random.choice(np.where(mask0)[0], n_plot, replace=False)
        idx1 = np.random.choice(np.where(mask1)[0], n_plot, replace=False)

        ax.scatter(z[idx0, 0], z[idx0, 1], alpha=0.3, s=5, label="state=0", c="tab:blue")
        ax.scatter(z[idx1, 0], z[idx1, 1], alpha=0.3, s=5, label="state=1", c="tab:red")
        ax.set_xlabel("z[0]")
        ax.set_ylabel("z[1]")
        ax.set_title(f"{cfg['label']}\nsuccess={data['final_success']:.1%}")
        ax.legend(markerscale=5)

    fig.suptitle("VAE Latent Space: State Clustering", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/latent_scatter_2d.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {results_dir}/latent_scatter_2d.png")

    # ── Figure 3: Information bottleneck diagram ──
    # Show MI at each stage + separability + success as a pipeline diagram
    fig, ax = plt.subplots(figsize=(14, 6))

    # For noise=0.1, compare env_lat=1 vs 2
    d1 = collected[0]  # env_lat=1
    d2 = collected[1]  # env_lat=2

    from worldnn.utils import estimate_mi_ksg

    stages = ["I(S;X)", "I(S;Y)", "I(S;Z)", "I(S;E)"]
    stage_names = ["Emission", "Channel\nOutput", "Env\nLatent z", "Organism\nEmbedding"]

    for data, color, label in [(d1, "tab:red", "env_lat=1"), (d2, "tab:green", "env_lat=2")]:
        s = data["states"].reshape(-1, 1)
        idx = np.random.choice(len(s), min(2000, len(s)), replace=False)
        mis = [
            estimate_mi_ksg(s[idx], data["emissions"][idx]),
            estimate_mi_ksg(s[idx], data["channel_out"][idx]),
            estimate_mi_ksg(s[idx], data["z"][idx]),
            estimate_mi_ksg(s[idx], data["embeddings"][idx]),
        ]
        ax.plot(range(4), mis, "o-", color=color, linewidth=2.5, markersize=10, label=label)

        # Add threshold accuracy at z stage
        z_acc = data["sep"]["threshold_accuracy"]
        ax.annotate(f"z-sep: {z_acc:.1%}", (2, mis[2]),
                    textcoords="offset points", xytext=(10, -15 if label=="env_lat=1" else 15),
                    fontsize=9, color=color, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.5))

    ax.set_xticks(range(4))
    ax.set_xticklabels(stage_names)
    ax.set_ylabel("Mutual Information I(S; .) [nats]")
    ax.set_title("MI Chain with Separability Annotations (noise=0.1, embed=4)")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add success rates as text
    ax.text(3.3, 0.05, f"env_lat=1 success: {d1['final_success']:.1%}", color="tab:red",
            fontsize=10, fontweight="bold")
    ax.text(3.3, 0.12, f"env_lat=2 success: {d2['final_success']:.1%}", color="tab:green",
            fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(f"{results_dir}/mi_chain_annotated.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {results_dir}/mi_chain_annotated.png")

    # Print summary
    print("\n" + "="*70)
    print("ANALYSIS SUMMARY")
    print("="*70)
    for data in collected:
        cfg = data["config"]
        sep = data["sep"]
        print(f"\n{cfg['label']}:")
        print(f"  Task success:       {data['final_success']:.1%}")
        print(f"  Threshold accuracy: {sep['threshold_accuracy']:.1%}")
        print(f"  Cohen's d:          {sep['cohens_d']:.3f}")
        print(f"  Distribution overlap: {sep['overlap_coeff']:.3f}")
        print(f"  z|s=0: mean={sep['mean_s0']:.3f}, std={sep['std_s0']:.3f}")
        print(f"  z|s=1: mean={sep['mean_s1']:.3f}, std={sep['std_s1']:.3f}")

    print("\n" + "="*70)
    print("CONCLUSIONS")
    print("="*70)
    d1_sep = collected[0]["sep"]
    d2_sep = collected[1]["sep"]
    print(f"\nenv_lat=1 threshold accuracy: {d1_sep['threshold_accuracy']:.1%}")
    print(f"env_lat=2 threshold accuracy: {d2_sep['threshold_accuracy']:.1%}")
    if d1_sep["threshold_accuracy"] < 0.65:
        print("\n-> H1 SUPPORTED: env_lat=1 z is genuinely non-separable.")
        print("   The MI estimate was misleadingly high (KSG noise with 1D latent).")
    elif d1_sep["threshold_accuracy"] > 0.80:
        print("\n-> H3 SUPPORTED: Information IS present in z but organism can't learn it.")
        print("   The RL training is the bottleneck, not the representation.")
    else:
        print("\n-> H2 SUPPORTED: Information partially present but not cleanly separable.")
        print("   The 1D bottleneck mixes state with irrelevant channel features.")
    print(f"\nCohen's d: env_lat=1={d1_sep['cohens_d']:.2f}, env_lat=2={d2_sep['cohens_d']:.2f}")
    print(f"Overlap: env_lat=1={d1_sep['overlap_coeff']:.2f}, env_lat=2={d2_sep['overlap_coeff']:.2f}")


if __name__ == "__main__":
    main()
