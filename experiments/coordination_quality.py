"""obj-013: Coordination quality C_i — mu fix + alignment measurement.

Two goals:
1. Fix the VAE pipeline by using deterministic mu (not stochastic z)
2. Measure coordination quality C_i: how well the organism's learned
   projection aligns sensory input with optimal actions

C_i = E[cos(organism_action, optimal_action)] across states

If C_i predicts task performance across all conditions (oracle, VAE,
different embed dims), that's the novel empirical contribution.

Grid:
  perception: [oracle, vae_mu_lat16]
  channel_noise: [0.01]  (just one noise level to keep it focused)
  embedding_dim: [2, 4, 8, 16, 32]
  seeds: [42, 123, 456, 789, 1337]
  = 50 configs, 500 episodes each
"""

import sys
import json
import time
import math
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from worldnn.world import RockPushWorld
from worldnn.matter import RockPushMatter
from worldnn.organism import Organism
from worldnn.train import train_environment_rockpush, train_organism_ppo_rockpush


def compute_optimal_action(state, target_x=0.8, target_y=0.8):
    """Compute the optimal action for rock-push.

    Optimal strategy: move toward the rock, then push it toward the target.
    Returns a normalized direction vector.
    """
    rock_pos = state[:, :2]
    org_pos = state[:, 2:4]
    target = torch.tensor([target_x, target_y], device=state.device)

    # Vector from organism to rock
    to_rock = rock_pos - org_pos
    rock_dist = torch.norm(to_rock, dim=-1, keepdim=True).clamp(min=1e-6)

    # Vector from rock to target
    to_target = target.unsqueeze(0) - rock_pos
    target_dist = torch.norm(to_target, dim=-1, keepdim=True).clamp(min=1e-6)

    # If far from rock, move toward rock. If near rock, push toward target.
    contact_threshold = 0.15
    near_rock = (rock_dist < contact_threshold).float()

    # Blend: far → go to rock, near → push toward target
    optimal = (1 - near_rock) * (to_rock / rock_dist) + near_rock * (to_target / target_dist)

    # Normalize
    opt_norm = torch.norm(optimal, dim=-1, keepdim=True).clamp(min=1e-6)
    return optimal / opt_norm


def measure_coordination_quality(organism, perception_fn, matter,
                                  n_samples=5000, device="cuda"):
    """Measure C_i: cosine similarity between organism's action and optimal action.

    Also returns:
    - embedding_utilization: mean norm of embedding (how much of the bottleneck is used)
    - action_magnitude: mean norm of action output
    """
    dev = torch.device(device)
    organism.eval()

    all_cos_sim = []
    all_emb_norm = []
    all_act_norm = []

    with torch.no_grad():
        for _ in range(n_samples // 256 + 1):
            state = matter.reset_state(256, dev)
            seed = torch.randn(256, matter.seed_dim, device=dev)
            action = torch.randn(256, 2, device=dev) * 0.1
            next_state, emission, _ = matter(state, seed, action)

            obs = perception_fn(next_state, emission)
            action_mean, embedding, _ = organism(obs)

            optimal = compute_optimal_action(next_state)

            # Cosine similarity
            cos_sim = F.cosine_similarity(action_mean, optimal, dim=-1)
            all_cos_sim.append(cos_sim)
            all_emb_norm.append(torch.norm(embedding, dim=-1))
            all_act_norm.append(torch.norm(action_mean, dim=-1))

    cos_sims = torch.cat(all_cos_sim)[:n_samples]
    emb_norms = torch.cat(all_emb_norm)[:n_samples]
    act_norms = torch.cat(all_act_norm)[:n_samples]

    return {
        "C_i": cos_sims.mean().item(),
        "C_i_std": cos_sims.std().item(),
        "C_i_positive_frac": (cos_sims > 0).float().mean().item(),
        "embedding_utilization": emb_norms.mean().item(),
        "action_magnitude": act_norms.mean().item(),
    }


def run_vae_mu_config(channel_noise, embedding_dim, seed,
                       n_vae_steps=1500, n_episodes=500, device="cuda"):
    """Run VAE pipeline with mu (deterministic) instead of z."""
    torch.manual_seed(seed)
    dev = torch.device(device)

    world = RockPushWorld(
        emission_dim=8, channel_dim=8, env_latent_dim=16,
        embedding_dim=embedding_dim, action_dim=2, seed_dim=4,
        channel_noise=channel_noise, target_x=0.8, target_y=0.8,
    ).to(dev)
    world.use_mu = True  # THE FIX

    # Phase 1: Pre-train VAE
    t0 = time.time()
    vae_losses = train_environment_rockpush(
        world, n_steps=n_vae_steps, batch_size=256, device=dev
    )
    vae_time = time.time() - t0

    # Phase 2: Train organism with PPO (now using mu)
    t1 = time.time()
    metrics = train_organism_ppo_rockpush(
        world, n_episodes=n_episodes, steps_per_episode=20,
        batch_size=256, device=dev,
    )
    ppo_time = time.time() - t1

    # Phase 3: Measure coordination quality
    env = world.environment
    ch = world.channel
    def vae_mu_fn(state, emission):
        with torch.no_grad():
            ch_out = ch(emission)
            mu, _ = env.encode(ch_out)
        return mu

    ci_metrics = measure_coordination_quality(
        world.organism, vae_mu_fn, world.matter, device=device,
    )

    n_tail = min(100, len(metrics["rock_distance"]))
    avg_dist = sum(metrics["rock_distance"][-n_tail:]) / n_tail
    avg_contact = sum(metrics["contact_rate"][-n_tail:]) / n_tail

    return {
        "perception": "vae_mu_lat16",
        "channel_noise": channel_noise,
        "embedding_dim": embedding_dim,
        "seed": seed,
        "avg_dist_last100": avg_dist,
        "avg_contact_last100": avg_contact,
        "vae_final_loss": vae_losses[-1],
        "elapsed_s": vae_time + ppo_time,
        **ci_metrics,
    }


def run_oracle_config(embedding_dim, seed, n_episodes=500, device="cuda"):
    """Run oracle + measure C_i."""
    torch.manual_seed(seed)
    dev = torch.device(device)

    matter = RockPushMatter(
        emission_dim=8, action_dim=2, seed_dim=4,
    ).to(dev)

    organism = Organism(
        sensory_dim=4, embedding_dim=embedding_dim, action_dim=2,
    ).to(dev)

    # Train with oracle PPO — inline to avoid cross-experiment imports
    sys.path.insert(0, str(Path(__file__).parent))
    from vae_lat16_sweep import train_oracle_ppo
    t0 = time.time()
    metrics = train_oracle_ppo(
        matter, organism, embedding_dim, seed,
        n_episodes=n_episodes, device=device,
    )
    elapsed = time.time() - t0

    # Measure C_i
    def oracle_fn(state, emission):
        return state
    ci_metrics = measure_coordination_quality(
        organism, oracle_fn, matter, device=device,
    )

    n_tail = min(100, len(metrics["rock_distance"]))
    avg_dist = sum(metrics["rock_distance"][-n_tail:]) / n_tail
    avg_contact = sum(metrics["contact_rate"][-n_tail:]) / n_tail

    return {
        "perception": "oracle",
        "channel_noise": 0.0,
        "embedding_dim": embedding_dim,
        "seed": seed,
        "avg_dist_last100": avg_dist,
        "avg_contact_last100": avg_contact,
        "vae_final_loss": None,
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
    checkpoint_path = results_dir / "coordination_quality_checkpoint.json"
    results_path = results_dir / "coordination_quality.json"

    embed_dims = [2, 4, 8, 16, 32]
    seeds = [42, 123, 456, 789, 1337]

    # Build configs: oracle + vae_mu
    configs = []
    for emb in embed_dims:
        for s in seeds:
            configs.append(("oracle", 0.0, emb, s))
    for emb in embed_dims:
        for s in seeds:
            configs.append(("vae_mu_lat16", 0.01, emb, s))

    # Resume
    completed = []
    completed_keys = set()
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            completed = json.load(f)
        for r in completed:
            completed_keys.add((r["perception"], r["embedding_dim"], r["seed"]))
        print(f"Resuming: {len(completed)}/{len(configs)} done")

    total = len(configs)
    remaining = total - len(completed_keys)
    print(f"obj-013 sweep: {total} configs ({remaining} remaining)")
    t0 = time.time()

    for perc, noise, emb, s in configs:
        key = (perc, emb, s)
        if key in completed_keys:
            continue

        idx = len(completed) + 1
        print(f"[{idx}/{total}] {perc}, emb={emb}, seed={s}",
              end=" ... ", flush=True)

        try:
            if perc == "oracle":
                result = run_oracle_config(emb, s, device=device)
            else:
                result = run_vae_mu_config(noise, emb, s, device=device)
            completed.append(result)
            completed_keys.add(key)
            print(f"dist={result['avg_dist_last100']:.3f}, "
                  f"C_i={result['C_i']:.3f}, "
                  f"emb_util={result['embedding_utilization']:.3f} "
                  f"({result['elapsed_s']:.0f}s)")
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback; traceback.print_exc()
            completed.append({
                "perception": perc, "channel_noise": noise,
                "embedding_dim": emb, "seed": s, "error": str(e),
            })
            completed_keys.add(key)

        if len(completed) % 5 == 0:
            with open(checkpoint_path, "w") as f:
                json.dump(completed, f, indent=2)
            elapsed = time.time() - t0
            done_this_run = len(completed) - (total - remaining)
            if done_this_run > 0:
                rate = done_this_run / elapsed
                left = remaining - done_this_run
                if rate > 0:
                    print(f"  [checkpoint] ETA: {left/rate/60:.0f} min")

    elapsed = time.time() - t0
    with open(results_path, "w") as f:
        json.dump(completed, f, indent=2)
    with open(checkpoint_path, "w") as f:
        json.dump(completed, f, indent=2)

    n_ok = len([r for r in completed if "error" not in r])
    print(f"\nDone: {n_ok}/{total} in {elapsed/60:.1f} min")

    print_summary(completed)

    try:
        generate_plot(completed, results_dir)
    except Exception as e:
        print(f"Plot failed: {e}")
        import traceback; traceback.print_exc()


def print_summary(completed):
    import numpy as np

    valid = [r for r in completed if "error" not in r]

    print("\n=== Summary by perception × embed_dim ===")
    by_group = defaultdict(list)
    for r in valid:
        by_group[(r["perception"], r["embedding_dim"])].append(r)

    for perc in ["oracle", "vae_mu_lat16"]:
        print(f"\n  {perc}:")
        for emb in [2, 4, 8, 16, 32]:
            results = by_group.get((perc, emb), [])
            if results:
                dists = [r["avg_dist_last100"] for r in results]
                cis = [r["C_i"] for r in results]
                print(f"    emb={emb:>2}: dist={np.mean(dists):.3f}±{np.std(dists):.3f}, "
                      f"C_i={np.mean(cis):.3f}±{np.std(cis):.3f}")

    # Correlation between C_i and distance
    if len(valid) >= 10:
        dists = [r["avg_dist_last100"] for r in valid]
        cis = [r["C_i"] for r in valid]
        d_arr = torch.tensor(dists)
        c_arr = torch.tensor(cis)
        corr = torch.corrcoef(torch.stack([d_arr, c_arr]))[0, 1].item()
        print(f"\n  C_i vs distance correlation: r={corr:.3f}")


def generate_plot(completed, results_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    valid = [r for r in completed if "error" not in r]
    if not valid:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    embed_dims = [2, 4, 8, 16, 32]

    by_group = defaultdict(lambda: defaultdict(list))
    for r in valid:
        by_group[r["perception"]][r["embedding_dim"]].append(r)

    colors = {"oracle": "steelblue", "vae_mu_lat16": "coral"}
    labels = {"oracle": "Oracle (direct state)", "vae_mu_lat16": "VAE lat=16 (mu fix)"}

    # Panel 1: Distance vs embed_dim
    ax = axes[0, 0]
    for perc in ["oracle", "vae_mu_lat16"]:
        if perc not in by_group:
            continue
        means = [np.mean([r["avg_dist_last100"] for r in by_group[perc].get(e, [{"avg_dist_last100": np.nan}])]) for e in embed_dims]
        stds = [np.std([r["avg_dist_last100"] for r in by_group[perc].get(e, [{"avg_dist_last100": 0}])]) for e in embed_dims]
        ax.errorbar(embed_dims, means, yerr=stds, marker="o", label=labels[perc],
                    color=colors[perc], capsize=3, linewidth=2)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Random")
    ax.set_xlabel("Embedding Dimension")
    ax.set_ylabel("Rock-Target Distance (lower = better)")
    ax.set_title("Task Performance: Oracle vs VAE(mu)")
    ax.legend(fontsize=8)
    ax.set_xscale("log", base=2)
    ax.set_xticks(embed_dims)
    ax.set_xticklabels(embed_dims)

    # Panel 2: C_i vs embed_dim
    ax = axes[0, 1]
    for perc in ["oracle", "vae_mu_lat16"]:
        if perc not in by_group:
            continue
        means = [np.mean([r["C_i"] for r in by_group[perc].get(e, [{"C_i": 0}])]) for e in embed_dims]
        stds = [np.std([r["C_i"] for r in by_group[perc].get(e, [{"C_i": 0}])]) for e in embed_dims]
        ax.errorbar(embed_dims, means, yerr=stds, marker="s", label=labels[perc],
                    color=colors[perc], capsize=3, linewidth=2)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5, label="No alignment")
    ax.set_xlabel("Embedding Dimension")
    ax.set_ylabel("C_i (cosine similarity with optimal)")
    ax.set_title("Coordination Quality by Capacity")
    ax.legend(fontsize=8)
    ax.set_xscale("log", base=2)
    ax.set_xticks(embed_dims)
    ax.set_xticklabels(embed_dims)

    # Panel 3: C_i vs Distance (THE KEY PLOT)
    ax = axes[1, 0]
    for perc in ["oracle", "vae_mu_lat16"]:
        if perc not in by_group:
            continue
        for emb in embed_dims:
            results = by_group[perc].get(emb, [])
            for r in results:
                ax.scatter(r["C_i"], r["avg_dist_last100"],
                          color=colors[perc], s=emb * 3, alpha=0.6,
                          edgecolors="black", linewidth=0.5)
    # Legend
    for perc in ["oracle", "vae_mu_lat16"]:
        ax.scatter([], [], color=colors[perc], label=labels[perc], s=50)
    for emb in [2, 32]:
        ax.scatter([], [], color="gray", s=emb * 3, label=f"emb={emb}", edgecolors="black", linewidth=0.5)

    # Fit line
    all_ci = [r["C_i"] for r in valid]
    all_dist = [r["avg_dist_last100"] for r in valid]
    if len(all_ci) >= 5:
        z_fit = np.polyfit(all_ci, all_dist, 1)
        x_range = np.linspace(min(all_ci), max(all_ci), 100)
        ax.plot(x_range, np.polyval(z_fit, x_range), "k--", alpha=0.4, label=f"linear fit")
        corr = np.corrcoef(all_ci, all_dist)[0, 1]
        ax.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax.transAxes,
                fontsize=10, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("Coordination Quality C_i")
    ax.set_ylabel("Rock-Target Distance")
    ax.set_title("C_i Predicts Task Performance")
    ax.legend(fontsize=7, loc="upper right")

    # Panel 4: Embedding utilization vs embed_dim
    ax = axes[1, 1]
    for perc in ["oracle", "vae_mu_lat16"]:
        if perc not in by_group:
            continue
        means = [np.mean([r["embedding_utilization"] for r in by_group[perc].get(e, [{"embedding_utilization": 0}])]) for e in embed_dims]
        ax.plot(embed_dims, means, marker="^", label=labels[perc],
                color=colors[perc], linewidth=2)
    ax.set_xlabel("Embedding Dimension")
    ax.set_ylabel("Embedding Norm (utilization)")
    ax.set_title("How Much of the Bottleneck Is Used?")
    ax.legend(fontsize=8)
    ax.set_xscale("log", base=2)
    ax.set_xticks(embed_dims)
    ax.set_xticklabels(embed_dims)

    fig.suptitle("obj-013: Coordination Quality — Does C_i Predict Performance?", fontsize=14)
    plt.tight_layout()
    out = results_dir / "obj013_coordination_quality.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
