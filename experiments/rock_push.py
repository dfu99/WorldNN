"""Rock-push experiment (obj-009): multi-object 2D task with multi-channel perception.

Tests whether higher-dimensional state (4D vs 1-bit/1D) creates genuine
organism capacity requirements. Sweeps embedding_dim to find where it
starts to matter.

Grid:
  channel_noise: [0.01, 0.1, 0.5, 1.0]
  env_latent_dim: [4, 8]
  embedding_dim: [2, 4, 8, 16, 32]
  seeds: 3
  = 4 × 2 × 5 × 3 = 120 configs
  1000 episodes each (up from 500) for sufficient learning
"""

import sys
import json
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from worldnn.world import RockPushWorld
from worldnn.train import train_environment_rockpush, train_organism_ppo_rockpush


def run_config(
    channel_noise: float,
    env_latent_dim: int,
    embedding_dim: int,
    seed: int,
    n_vae_steps: int = 1500,
    n_episodes: int = 1000,
    steps_per_episode: int = 20,
    batch_size: int = 512,
    device: str = "cpu",
) -> dict:
    """Run one configuration and return metrics."""
    torch.manual_seed(seed)

    dev = torch.device(device)
    world = RockPushWorld(
        emission_dim=8,
        channel_dim=8,
        env_latent_dim=env_latent_dim,
        embedding_dim=embedding_dim,
        action_dim=2,
        seed_dim=4,
        channel_noise=channel_noise,
        target_x=0.8,
        target_y=0.8,
    ).to(dev)

    # Phase 1: Pre-train VAE
    vae_losses = train_environment_rockpush(
        world, n_steps=n_vae_steps, batch_size=256, device=dev
    )

    # Phase 2: Train organism with PPO
    metrics = train_organism_ppo_rockpush(
        world,
        n_episodes=n_episodes,
        steps_per_episode=steps_per_episode,
        batch_size=batch_size,
        device=dev,
    )

    return {
        "channel_noise": channel_noise,
        "env_latent_dim": env_latent_dim,
        "embedding_dim": embedding_dim,
        "seed": seed,
        "final_rock_distance": metrics["rock_distance"][-1],
        "final_reward": metrics["rewards"][-1],
        "final_contact_rate": metrics["contact_rate"][-1],
        "best_rock_distance": min(metrics["rock_distance"]),
        "vae_final_loss": vae_losses[-1],
        "converged_distance": sum(metrics["rock_distance"][-20:]) / 20,
    }


def main():
    parser = argparse.ArgumentParser(description="Rock-push experiment")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-episodes", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True)
    checkpoint_path = results_dir / "rockpush_checkpoint.json"
    results_path = results_dir / "rockpush_results.json"

    # Define sweep grid
    noises = [0.01, 0.1, 0.5, 1.0]
    latent_dims = [4, 8]
    embed_dims = [2, 4, 8, 16, 32]
    seeds = [42, 123, 456]

    # Build config list
    configs = []
    for noise in noises:
        for lat in latent_dims:
            for emb in embed_dims:
                for s in seeds:
                    configs.append((noise, lat, emb, s))

    # Resume from checkpoint if exists
    completed = []
    completed_keys = set()
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            completed = json.load(f)
        for r in completed:
            key = (r["channel_noise"], r["env_latent_dim"], r["embedding_dim"], r["seed"])
            completed_keys.add(key)
        print(f"Resuming from checkpoint: {len(completed)}/{len(configs)} done")

    total = len(configs)
    print(f"Rock-push experiment: {total} configs")
    print(f"Device: {args.device}")
    t0 = time.time()

    for i, (noise, lat, emb, s) in enumerate(configs):
        key = (noise, lat, emb, s)
        if key in completed_keys:
            continue

        print(f"[{len(completed)+1}/{total}] noise={noise}, lat={lat}, "
              f"emb={emb}, seed={s}", end=" ... ", flush=True)

        try:
            result = run_config(
                channel_noise=noise,
                env_latent_dim=lat,
                embedding_dim=emb,
                seed=s,
                n_episodes=args.n_episodes,
                device=args.device,
            )
            completed.append(result)
            print(f"dist={result['final_rock_distance']:.3f}, "
                  f"contact={result['final_contact_rate']:.3f}")
        except Exception as e:
            print(f"FAILED: {e}")
            completed.append({
                "channel_noise": noise, "env_latent_dim": lat,
                "embedding_dim": emb, "seed": s,
                "error": str(e),
            })

        # Checkpoint
        if len(completed) % args.checkpoint_every == 0:
            with open(checkpoint_path, "w") as f:
                json.dump(completed, f)

    # Save final results
    elapsed = time.time() - t0
    output = {
        "rockpush_results": completed,
        "total_configs": total,
        "completed": len([r for r in completed if "error" not in r]),
        "elapsed_seconds": elapsed,
    }
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    with open(checkpoint_path, "w") as f:
        json.dump(completed, f)

    print(f"\nDone: {len(completed)}/{total} in {elapsed/60:.1f} min")
    print(f"Results: {results_path}")

    # Generate summary plot
    try:
        generate_plot(completed, results_dir)
    except Exception as e:
        print(f"Plot generation failed: {e}")


def generate_plot(results, results_dir):
    """Generate summary visualization."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from collections import defaultdict

    # Filter out errors
    valid = [r for r in results if "error" not in r]
    if not valid:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Rock distance by embedding dim (aggregated over noise/lat)
    ax = axes[0, 0]
    by_emb = defaultdict(list)
    for r in valid:
        by_emb[r["embedding_dim"]].append(r["converged_distance"])
    embs = sorted(by_emb.keys())
    means = [np.mean(by_emb[e]) for e in embs]
    stds = [np.std(by_emb[e]) for e in embs]
    ax.bar(range(len(embs)), means, yerr=stds, tick_label=[str(e) for e in embs],
           color="steelblue", alpha=0.8, capsize=4)
    ax.set_xlabel("Embedding Dimension")
    ax.set_ylabel("Mean Rock-Target Distance")
    ax.set_title("Does Embedding Dim Matter?")

    # 2. Heatmap: noise × embedding dim (fixed lat=4)
    ax = axes[0, 1]
    noises = sorted(set(r["channel_noise"] for r in valid))
    embed_dims = sorted(set(r["embedding_dim"] for r in valid))
    grid = np.full((len(noises), len(embed_dims)), np.nan)
    for r in valid:
        if r["env_latent_dim"] == 4:
            ni = noises.index(r["channel_noise"])
            ei = embed_dims.index(r["embedding_dim"])
            if np.isnan(grid[ni, ei]):
                grid[ni, ei] = r["converged_distance"]
            else:
                grid[ni, ei] = (grid[ni, ei] + r["converged_distance"]) / 2
    im = ax.imshow(grid, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(embed_dims)))
    ax.set_xticklabels([str(e) for e in embed_dims])
    ax.set_yticks(range(len(noises)))
    ax.set_yticklabels([str(n) for n in noises])
    ax.set_xlabel("Embedding Dim")
    ax.set_ylabel("Channel Noise")
    ax.set_title("Rock Distance (env_lat=4)")
    plt.colorbar(im, ax=ax, label="Distance")

    # 3. Effect of env_latent_dim
    ax = axes[1, 0]
    by_lat = defaultdict(list)
    for r in valid:
        by_lat[r["env_latent_dim"]].append(r["converged_distance"])
    lats = sorted(by_lat.keys())
    means = [np.mean(by_lat[l]) for l in lats]
    stds = [np.std(by_lat[l]) for l in lats]
    ax.bar(range(len(lats)), means, yerr=stds, tick_label=[str(l) for l in lats],
           color="coral", alpha=0.8, capsize=4)
    ax.set_xlabel("Environment Latent Dim")
    ax.set_ylabel("Mean Rock-Target Distance")
    ax.set_title("Environment Compression Effect")

    # 4. Contact rate by embedding dim
    ax = axes[1, 1]
    by_emb_contact = defaultdict(list)
    for r in valid:
        by_emb_contact[r["embedding_dim"]].append(r["final_contact_rate"])
    embs = sorted(by_emb_contact.keys())
    means = [np.mean(by_emb_contact[e]) for e in embs]
    stds = [np.std(by_emb_contact[e]) for e in embs]
    ax.bar(range(len(embs)), means, yerr=stds, tick_label=[str(e) for e in embs],
           color="mediumpurple", alpha=0.8, capsize=4)
    ax.set_xlabel("Embedding Dimension")
    ax.set_ylabel("Contact Rate")
    ax.set_title("Does Larger Brain → More Contact?")

    fig.suptitle("Rock-Push Experiment (obj-009): Multi-Object 2D Task", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "rockpush_results.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {results_dir / 'rockpush_results.png'}")


if __name__ == "__main__":
    main()
