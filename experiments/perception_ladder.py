"""Perception Ladder (obj-011): Where between oracle and VAE does learning break?

Diagnoses VAE failure and maps the transition from working (oracle) to
broken (VAE) perception with intermediate conditions:

  Level 0: Oracle — direct 4D state [rock_x, rock_y, org_x, org_y]
  Level 1: Oracle + noise — state + Gaussian noise (σ=0.1, 0.5, 1.0)
  Level 2: Linear projection — state projected through random matrix (like emissions)
  Level 3: Linear projection + noise — emission-like with channel noise
  Level 4: VAE large latent — env_latent_dim=8,16,32 (vs standard 4)
  Level 5: VAE standard — env_latent_dim=4 (the broken case)

Also runs a VAE latent probe: after VAE training, fit a linear regressor
from z → state to measure how much state info the VAE preserves.

Grid per level: embed_dim=[8,32] × 3 seeds = 6 configs each
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
from worldnn.train import train_environment_rockpush


def train_organism_with_perception(
    matter, organism, perception_fn, target_x, target_y,
    n_episodes=500, steps_per_episode=20, batch_size=256,
    lr=3e-4, gamma=0.99, entropy_coef=0.01,
    action_std_init=0.8, action_std_final=0.2, clip_eps=0.2,
    ppo_epochs=4, device="cuda",
):
    """Train organism with a custom perception function.

    perception_fn: state → observation (what the organism sees)
    """
    dev = torch.device(device)
    target = torch.tensor([target_x, target_y], device=dev)
    seed_dim = matter.seed_dim

    log_std = nn.Parameter(
        torch.full((2,), math.log(action_std_init), device=dev)
    )
    optimizer = torch.optim.Adam(
        list(organism.parameters()) + [log_std], lr=lr
    )

    metrics = {"rewards": [], "rock_distance": [], "contact_rate": []}

    for ep in range(n_episodes):
        organism.train()
        state = matter.reset_state(batch_size, dev)
        action = None

        all_obs, all_actions, all_log_probs = [], [], []
        all_rewards, all_values = [], []
        contact_sum = 0.0

        for t in range(steps_per_episode):
            seed = torch.randn(batch_size, seed_dim, device=dev)
            if action is None:
                action = torch.zeros(batch_size, 2, device=dev)

            next_state, emission, contact = matter(state, seed, action)

            # Custom perception: state → observation
            with torch.no_grad():
                obs = perception_fn(next_state, emission)

            action_mean, embedding, value = organism(obs)
            std = log_std.exp().unsqueeze(0).expand_as(action_mean)
            dist = torch.distributions.Normal(action_mean, std)
            action_sample = dist.sample()
            lp = dist.log_prob(action_sample).sum(dim=-1)

            # Reward
            rock_pos = next_state[:, :2]
            org_pos = next_state[:, 2:4]
            rock_dist = torch.norm(rock_pos - target, dim=-1)
            org_rock_dist = torch.norm(rock_pos - org_pos, dim=-1)
            reward = (1.0 - rock_dist) + 0.2 * (1.0 - org_rock_dist) + 0.1 * contact

            all_obs.append(obs.detach())
            all_actions.append(action_sample.detach())
            all_log_probs.append(lp.detach())
            all_rewards.append(reward.detach())
            all_values.append(value.detach())
            contact_sum += contact.mean().item()

            state = next_state.detach()
            action = action_sample.detach()

        # Compute returns & advantages
        T = len(all_rewards)
        returns = []
        G = torch.zeros(batch_size, device=dev)
        for t_idx in reversed(range(T)):
            G = all_rewards[t_idx] + gamma * G
            returns.insert(0, G)
        returns = torch.stack(returns)
        values = torch.stack(all_values)
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        obs_batch = torch.stack(all_obs)
        act_batch = torch.stack(all_actions)
        old_lp = torch.stack(all_log_probs)

        for _ in range(ppo_epochs):
            for t_idx in range(T):
                action_mean, _, value = organism(obs_batch[t_idx])
                std = log_std.exp().unsqueeze(0).expand_as(action_mean)
                d = torch.distributions.Normal(action_mean, std)
                new_lp = d.log_prob(act_batch[t_idx]).sum(dim=-1)

                ratio = (new_lp - old_lp[t_idx]).exp()
                clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
                policy_loss = -torch.min(
                    ratio * advantages[t_idx], clipped * advantages[t_idx]
                ).mean()
                value_loss = F.mse_loss(value, returns[t_idx])
                entropy = d.entropy().sum(dim=-1).mean()
                loss = policy_loss + 0.5 * value_loss - entropy_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(organism.parameters()) + [log_std], 1.0
                )
                optimizer.step()

        with torch.no_grad():
            rock_pos = state[:, :2]
            final_dist = torch.norm(rock_pos - target, dim=-1).mean().item()
        metrics["rewards"].append(
            sum(r.mean().item() for r in all_rewards) / T
        )
        metrics["rock_distance"].append(final_dist)
        metrics["contact_rate"].append(contact_sum / T)

    return metrics


def probe_vae_latent(world, n_samples=10000, device="cuda"):
    """Fit a linear probe: z → state. Returns R² for each state variable."""
    dev = torch.device(device)
    world.eval()

    states, latents = [], []
    with torch.no_grad():
        for _ in range(n_samples // 256 + 1):
            state = world.matter.reset_state(256, dev)
            seed = torch.randn(256, world.seed_dim, device=dev)
            action = torch.randn(256, world.action_dim, device=dev) * 0.1
            _, emission, _ = world.matter(state, seed, action)
            channel_out = world.channel(emission)
            z, _, mu, _ = world.environment(channel_out)
            states.append(state.cpu())
            latents.append(mu.cpu())  # Use mu (deterministic) not z (stochastic)

    states = torch.cat(states)[:n_samples]
    latents = torch.cat(latents)[:n_samples]

    # Linear regression: z → state (least squares)
    # state = latents @ W + b
    X = torch.cat([latents, torch.ones(len(latents), 1)], dim=1)
    W, _, _, _ = torch.linalg.lstsq(X, states)
    pred = X @ W

    # R² per state variable
    ss_res = ((states - pred) ** 2).mean(dim=0)
    ss_tot = ((states - states.mean(dim=0)) ** 2).mean(dim=0)
    r2 = 1 - ss_res / (ss_tot + 1e-8)

    return {
        "r2_rock_x": r2[0].item(),
        "r2_rock_y": r2[1].item(),
        "r2_org_x": r2[2].item(),
        "r2_org_y": r2[3].item(),
        "r2_mean": r2.mean().item(),
        "residual_mean": ss_res.mean().item(),
    }


def run_perception_level(level_name, matter, sensory_dim, embed_dim, seed,
                         perception_fn, device="cuda", n_episodes=500):
    """Run one perception level config."""
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

    n_tail = min(100, len(metrics["rock_distance"]))
    avg_dist = sum(metrics["rock_distance"][-n_tail:]) / n_tail
    avg_contact = sum(metrics["contact_rate"][-n_tail:]) / n_tail

    return {
        "level": level_name,
        "embedding_dim": embed_dim,
        "seed": seed,
        "avg_dist_last100": avg_dist,
        "avg_contact_last100": avg_contact,
        "final_dist": metrics["rock_distance"][-1],
        "elapsed_s": elapsed,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    checkpoint_path = results_dir / "perception_ladder_checkpoint.json"
    results_path = results_dir / "perception_ladder.json"

    embed_dims = [8, 32]
    seeds = [42, 123, 456]
    dev = torch.device(device)

    # Create shared matter (same random projection for all levels)
    torch.manual_seed(0)
    matter = RockPushMatter(
        emission_dim=8, action_dim=2, seed_dim=4,
    ).to(dev)

    # Resume from checkpoint
    completed = []
    completed_keys = set()
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            completed = json.load(f)
        for r in completed:
            completed_keys.add((r["level"], r["embedding_dim"], r["seed"]))
        print(f"Resuming: {len(completed)} done")

    # ── Define perception levels ──
    levels = []

    # Level 0: Oracle (direct state)
    def oracle_fn(state, emission):
        return state
    for emb in embed_dims:
        for s in seeds:
            levels.append(("L0_oracle", 4, emb, s, oracle_fn))

    # Level 1: Oracle + noise
    for noise_std in [0.1, 0.5, 1.0]:
        def make_noisy_fn(std):
            def fn(state, emission):
                return state + torch.randn_like(state) * std
            return fn
        for emb in embed_dims:
            for s in seeds:
                levels.append((
                    f"L1_oracle_noise{noise_std}",
                    4, emb, s, make_noisy_fn(noise_std),
                ))

    # Level 2: Linear projection (like emission but without seed noise)
    # Project 4D state through a random 4→4 matrix (invertible but scrambled)
    torch.manual_seed(99)
    proj_matrix = torch.randn(4, 4, device=dev) * 0.5
    proj_bias = torch.randn(4, device=dev) * 0.2
    def linear_proj_fn(state, emission):
        return state @ proj_matrix + proj_bias
    for emb in embed_dims:
        for s in seeds:
            levels.append(("L2_linear_proj", 4, emb, s, linear_proj_fn))

    # Level 3: Emission (8D) directly — bypass VAE, organism sees raw channel output
    # This tests if the emission encoding itself is the problem
    for emb in embed_dims:
        for s in seeds:
            def emission_fn(state, emission):
                return emission
            levels.append(("L3_raw_emission", 8, emb, s, emission_fn))

    # Level 4: Emission + channel noise
    from worldnn.channels import Channel
    torch.manual_seed(0)
    channel_low = Channel(input_dim=8, output_dim=8, noise_std=0.01).to(dev)
    def emission_channel_fn(state, emission):
        return channel_low(emission)
    for emb in embed_dims:
        for s in seeds:
            levels.append(("L4_emission_channel", 8, emb, s, emission_channel_fn))

    # Level 5-7: VAE with different latent dims
    vae_probe_results = {}
    for lat_dim in [8, 16, 32]:
        torch.manual_seed(0)
        world = RockPushWorld(
            emission_dim=8, channel_dim=8, env_latent_dim=lat_dim,
            embedding_dim=8, action_dim=2, seed_dim=4,
            channel_noise=0.01, target_x=0.8, target_y=0.8,
        ).to(dev)
        # Share the same matter
        world.matter = matter

        print(f"Pre-training VAE (lat={lat_dim})...", end=" ", flush=True)
        vae_losses = train_environment_rockpush(world, n_steps=1500, device=dev)
        print(f"loss={vae_losses[-1]:.4f}")

        # Probe VAE latent quality
        probe = probe_vae_latent(world, device=device)
        vae_probe_results[lat_dim] = probe
        print(f"  Probe R²: rock_x={probe['r2_rock_x']:.3f}, "
              f"rock_y={probe['r2_rock_y']:.3f}, "
              f"org_x={probe['r2_org_x']:.3f}, "
              f"org_y={probe['r2_org_y']:.3f}, "
              f"mean={probe['r2_mean']:.3f}")

        # Create perception function using this trained VAE
        env = world.environment
        ch = world.channel
        def make_vae_fn(env_module, ch_module):
            def fn(state, emission):
                with torch.no_grad():
                    ch_out = ch_module(emission)
                    mu, _ = env_module.encode(ch_out)
                return mu  # Use deterministic mu
            return fn

        vae_fn = make_vae_fn(env, ch)
        for emb in embed_dims:
            for s in seeds:
                levels.append((f"L5_vae_lat{lat_dim}", lat_dim, emb, s, vae_fn))

    # Also test standard VAE lat=4
    torch.manual_seed(0)
    world4 = RockPushWorld(
        emission_dim=8, channel_dim=8, env_latent_dim=4,
        embedding_dim=8, action_dim=2, seed_dim=4,
        channel_noise=0.01, target_x=0.8, target_y=0.8,
    ).to(dev)
    world4.matter = matter
    print(f"Pre-training VAE (lat=4)...", end=" ", flush=True)
    vae_losses4 = train_environment_rockpush(world4, n_steps=1500, device=dev)
    print(f"loss={vae_losses4[-1]:.4f}")
    probe4 = probe_vae_latent(world4, device=device)
    vae_probe_results[4] = probe4
    print(f"  Probe R²: rock_x={probe4['r2_rock_x']:.3f}, "
          f"rock_y={probe4['r2_rock_y']:.3f}, "
          f"org_x={probe4['r2_org_x']:.3f}, "
          f"org_y={probe4['r2_org_y']:.3f}, "
          f"mean={probe4['r2_mean']:.3f}")
    env4 = world4.environment
    ch4 = world4.channel
    vae_fn4 = make_vae_fn(env4, ch4)
    for emb in embed_dims:
        for s in seeds:
            levels.append(("L5_vae_lat4", 4, emb, s, vae_fn4))

    total = len(levels)
    print(f"\nPerception ladder: {total} configs")
    t0 = time.time()

    for i, (level_name, sensory_dim, emb, s, perc_fn) in enumerate(levels):
        key = (level_name, emb, s)
        if key in completed_keys:
            continue

        idx = len(completed) + 1
        print(f"[{idx}/{total}] {level_name}, emb={emb}, seed={s}",
              end=" ... ", flush=True)

        try:
            result = run_perception_level(
                level_name, matter, sensory_dim, emb, s, perc_fn,
                device=device, n_episodes=500,
            )
            completed.append(result)
            print(f"dist={result['avg_dist_last100']:.3f}, "
                  f"contact={result['avg_contact_last100']:.3f} "
                  f"({result['elapsed_s']:.0f}s)")
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback; traceback.print_exc()
            completed.append({
                "level": level_name, "embedding_dim": emb,
                "seed": s, "error": str(e),
            })

        # Checkpoint every 6 configs
        if len(completed) % 6 == 0:
            with open(checkpoint_path, "w") as f:
                json.dump(completed, f, indent=2)

    elapsed = time.time() - t0

    # Save final results
    output = {
        "ladder_results": completed,
        "vae_probe_results": vae_probe_results,
        "total_configs": total,
        "elapsed_seconds": elapsed,
    }
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    with open(checkpoint_path, "w") as f:
        json.dump(completed, f, indent=2)

    print(f"\nDone: {len(completed)}/{total} in {elapsed/60:.1f} min")

    # Print summary
    print("\n=== Perception Ladder Summary ===")
    by_level = defaultdict(list)
    for r in completed:
        if "error" not in r:
            by_level[r["level"]].append(r["avg_dist_last100"])
    for level in sorted(by_level):
        vals = by_level[level]
        import numpy as np
        print(f"  {level:25s}: {np.mean(vals):.3f} ± {np.std(vals):.3f} (n={len(vals)})")

    print("\n=== VAE Probe Results ===")
    for lat, probe in sorted(vae_probe_results.items()):
        print(f"  lat={lat}: R²={probe['r2_mean']:.3f} "
              f"(rx={probe['r2_rock_x']:.3f}, ry={probe['r2_rock_y']:.3f}, "
              f"ox={probe['r2_org_x']:.3f}, oy={probe['r2_org_y']:.3f})")

    # Generate plot
    try:
        generate_plot(completed, vae_probe_results, results_dir)
    except Exception as e:
        print(f"Plot failed: {e}")
        import traceback; traceback.print_exc()


def generate_plot(results, probe_results, results_dir):
    """Generate perception ladder visualization."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    valid = [r for r in results if "error" not in r]
    if not valid:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Distance by perception level (embed=32 only)
    ax = axes[0]
    by_level = defaultdict(list)
    for r in valid:
        if r["embedding_dim"] == 32:
            by_level[r["level"]].append(r["avg_dist_last100"])

    # Order levels logically
    level_order = [
        "L0_oracle",
        "L1_oracle_noise0.1", "L1_oracle_noise0.5", "L1_oracle_noise1.0",
        "L2_linear_proj",
        "L3_raw_emission", "L4_emission_channel",
        "L5_vae_lat32", "L5_vae_lat16", "L5_vae_lat8", "L5_vae_lat4",
    ]
    level_labels = [
        "Oracle", "Oracle+N(0.1)", "Oracle+N(0.5)", "Oracle+N(1.0)",
        "Linear proj", "Raw emission", "Emission+ch",
        "VAE lat=32", "VAE lat=16", "VAE lat=8", "VAE lat=4",
    ]

    present = [(l, lb) for l, lb in zip(level_order, level_labels) if l in by_level]
    x_vals = range(len(present))
    means = [np.mean(by_level[l]) for l, _ in present]
    stds = [np.std(by_level[l]) for l, _ in present]
    colors = []
    for l, _ in present:
        if "oracle" in l.lower() and "noise" not in l:
            colors.append("steelblue")
        elif "noise" in l:
            colors.append("skyblue")
        elif "linear" in l or "emission" in l:
            colors.append("goldenrod")
        else:
            colors.append("coral")

    ax.bar(x_vals, means, yerr=stds, color=colors, alpha=0.85, capsize=3)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Random")
    ax.set_xticks(x_vals)
    ax.set_xticklabels([lb for _, lb in present], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Rock-Target Distance (lower = better)")
    ax.set_title("Perception Ladder (embed=32)")
    ax.set_ylim(0, 0.6)

    # Panel 2: embed=8 vs embed=32 comparison
    ax = axes[1]
    by_level_8 = defaultdict(list)
    by_level_32 = defaultdict(list)
    for r in valid:
        if r["embedding_dim"] == 8:
            by_level_8[r["level"]].append(r["avg_dist_last100"])
        elif r["embedding_dim"] == 32:
            by_level_32[r["level"]].append(r["avg_dist_last100"])

    present2 = [(l, lb) for l, lb in zip(level_order, level_labels)
                if l in by_level_8 or l in by_level_32]
    x2 = np.arange(len(present2))
    w = 0.35
    m8 = [np.mean(by_level_8.get(l, [np.nan])) for l, _ in present2]
    m32 = [np.mean(by_level_32.get(l, [np.nan])) for l, _ in present2]
    ax.bar(x2 - w/2, m8, w, label="embed=8", color="mediumpurple", alpha=0.8)
    ax.bar(x2 + w/2, m32, w, label="embed=32", color="steelblue", alpha=0.8)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xticks(x2)
    ax.set_xticklabels([lb for _, lb in present2], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Rock-Target Distance")
    ax.set_title("Capacity Effect by Perception Level")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 0.6)

    # Panel 3: VAE probe R²
    ax = axes[2]
    if probe_results:
        lats = sorted(probe_results.keys())
        vars_names = ["rock_x", "rock_y", "org_x", "org_y"]
        x3 = np.arange(len(lats))
        w3 = 0.18
        for i, var in enumerate(vars_names):
            vals = [probe_results[l][f"r2_{var}"] for l in lats]
            ax.bar(x3 + i * w3, vals, w3, label=var, alpha=0.8)
        ax.set_xticks(x3 + 1.5 * w3)
        ax.set_xticklabels([f"lat={l}" for l in lats])
        ax.set_ylabel("R² (linear probe)")
        ax.set_title("VAE Latent → State Recovery")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.05)
    else:
        ax.text(0.5, 0.5, "No probe data", ha="center", va="center",
                transform=ax.transAxes)

    fig.suptitle("Perception Ladder (obj-011): Where Does Learning Break?", fontsize=14)
    plt.tight_layout()
    out = results_dir / "perception_ladder.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
