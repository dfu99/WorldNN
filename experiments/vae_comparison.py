"""VAE vs Oracle comparison (obj-009-vae): Run VAE pipeline on same grid as oracle.

Focused grid to quantify the perception bottleneck:
  channel_noise: [0.01, 0.1]  (near-perfect and moderate)
  env_latent_dim: 4
  embedding_dim: [2, 4, 8, 16, 32]
  seeds: 5
  = 2 × 5 × 5 = 50 configs, 500 episodes each

Compares directly against oracle_expanded.json + oracle_gpu.json.
"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from worldnn.world import RockPushWorld
from worldnn.train import train_environment_rockpush, train_organism_ppo_rockpush


def run_vae_config(
    channel_noise: float,
    env_latent_dim: int,
    embedding_dim: int,
    seed: int,
    n_vae_steps: int = 1500,
    n_episodes: int = 500,
    steps_per_episode: int = 20,
    batch_size: int = 256,
    device: str = "cuda",
) -> dict:
    """Run one VAE configuration."""
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
    t0 = time.time()
    vae_losses = train_environment_rockpush(
        world, n_steps=n_vae_steps, batch_size=256, device=dev
    )
    vae_time = time.time() - t0

    # Phase 2: Train organism with PPO
    t1 = time.time()
    metrics = train_organism_ppo_rockpush(
        world,
        n_episodes=n_episodes,
        steps_per_episode=steps_per_episode,
        batch_size=batch_size,
        device=dev,
    )
    ppo_time = time.time() - t1

    # Compute converged metrics (last 100 episodes)
    n_tail = min(100, len(metrics["rock_distance"]))
    avg_dist = sum(metrics["rock_distance"][-n_tail:]) / n_tail
    avg_contact = sum(metrics["contact_rate"][-n_tail:]) / n_tail

    return {
        "channel_noise": channel_noise,
        "env_latent_dim": env_latent_dim,
        "embedding_dim": embedding_dim,
        "seed": seed,
        "final_dist": metrics["rock_distance"][-1],
        "avg_dist_last100": avg_dist,
        "final_contact": metrics["contact_rate"][-1],
        "avg_contact_last100": avg_contact,
        "vae_final_loss": vae_losses[-1],
        "vae_time_s": vae_time,
        "ppo_time_s": ppo_time,
        "elapsed_s": vae_time + ppo_time,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    checkpoint_path = results_dir / "vae_comparison_checkpoint.json"
    results_path = results_dir / "vae_comparison.json"

    # Grid: focused comparison
    noises = [0.01, 0.1]
    env_latent_dim = 4
    embed_dims = [2, 4, 8, 16, 32]
    seeds = [42, 123, 456, 789, 1337]

    configs = []
    for noise in noises:
        for emb in embed_dims:
            for s in seeds:
                configs.append((noise, emb, s))

    # Resume from checkpoint
    completed = []
    completed_keys = set()
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            completed = json.load(f)
        for r in completed:
            key = (r["channel_noise"], r["embedding_dim"], r["seed"])
            completed_keys.add(key)
        print(f"Resuming: {len(completed)}/{len(configs)} done")

    total = len(configs)
    remaining = total - len(completed)
    print(f"VAE comparison: {total} configs ({remaining} remaining)")
    t0 = time.time()

    for i, (noise, emb, s) in enumerate(configs):
        key = (noise, emb, s)
        if key in completed_keys:
            continue

        idx = len(completed) + 1
        print(f"[{idx}/{total}] noise={noise}, emb={emb}, seed={s}", end=" ... ", flush=True)

        try:
            result = run_vae_config(
                channel_noise=noise,
                env_latent_dim=env_latent_dim,
                embedding_dim=emb,
                seed=s,
                device=device,
            )
            completed.append(result)
            print(f"dist={result['avg_dist_last100']:.3f}, "
                  f"contact={result['avg_contact_last100']:.3f}, "
                  f"vae_loss={result['vae_final_loss']:.4f} "
                  f"({result['elapsed_s']:.0f}s)")
        except Exception as e:
            print(f"FAILED: {e}")
            completed.append({
                "channel_noise": noise, "env_latent_dim": env_latent_dim,
                "embedding_dim": emb, "seed": s, "error": str(e),
            })

        # Checkpoint every 5 configs
        if len(completed) % 5 == 0:
            with open(checkpoint_path, "w") as f:
                json.dump(completed, f, indent=2)
            elapsed = time.time() - t0
            rate = (len(completed) - (total - remaining)) / max(elapsed, 1)
            if rate > 0:
                eta = (remaining - (len(completed) - (total - remaining))) / rate
                print(f"  [checkpoint] ETA: {eta/60:.0f} min remaining")

    # Save final
    elapsed = time.time() - t0
    with open(results_path, "w") as f:
        json.dump(completed, f, indent=2)
    with open(checkpoint_path, "w") as f:
        json.dump(completed, f, indent=2)

    n_ok = len([r for r in completed if "error" not in r])
    print(f"\nDone: {n_ok}/{total} successful in {elapsed/60:.1f} min")
    print(f"Results: {results_path}")


if __name__ == "__main__":
    main()
