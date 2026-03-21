"""obj-015: C_i dynamics during training — does alignment trajectory predict success?

Measure C_i every 50 episodes during training to see:
1. Do successful configs show rapid C_i rise early?
2. Do failed configs plateau below threshold?
3. Can we predict final performance from early C_i trajectory?

Focused grid (fewer configs, more measurement):
  perception: [oracle, raw_emission, vae_mu_lat16]
  embedding_dim: [2, 8, 32]
  seeds: [42, 123, 456]
  = 3 × 3 × 3 = 27 configs, measured every 50 episodes over 500
"""

import sys
import json
import time
import math
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from worldnn.world import RockPushWorld
from worldnn.matter import RockPushMatter
from worldnn.organism import Organism
from worldnn.train import train_environment_rockpush
from coordination_quality import compute_optimal_action, measure_coordination_quality


def train_with_ci_tracking(
    matter, organism, perception_fn, target_x, target_y,
    n_episodes=500, steps_per_episode=20, batch_size=256,
    lr=3e-4, gamma=0.99, entropy_coef=0.01,
    action_std_init=0.8, action_std_final=0.2, clip_eps=0.2,
    ppo_epochs=4, ci_interval=50, device="cuda",
):
    """Train organism with periodic C_i measurement.

    Returns metrics dict with ci_trajectory: list of (episode, C_i) tuples.
    """
    dev = torch.device(device)
    target = torch.tensor([target_x, target_y], device=dev)

    log_std = nn.Parameter(
        torch.full((2,), math.log(action_std_init), device=dev)
    )
    optimizer = torch.optim.Adam(
        list(organism.parameters()) + [log_std], lr=lr
    )

    metrics = {
        "rewards": [], "rock_distance": [], "contact_rate": [],
        "ci_trajectory": [],
    }

    # Measure C_i before training (episode 0)
    ci_pre = measure_coordination_quality(
        organism, perception_fn, matter, n_samples=2000, device=device,
    )
    metrics["ci_trajectory"].append({"episode": 0, **ci_pre})

    for ep in range(n_episodes):
        organism.train()
        state = matter.reset_state(batch_size, dev)
        action = None

        all_obs, all_actions, all_log_probs = [], [], []
        all_rewards, all_values = [], []
        contact_sum = 0.0

        for t in range(steps_per_episode):
            seed = torch.randn(batch_size, matter.seed_dim, device=dev)
            if action is None:
                action = torch.zeros(batch_size, 2, device=dev)

            next_state, emission, contact = matter(state, seed, action)

            with torch.no_grad():
                obs = perception_fn(next_state, emission)

            action_mean, embedding, value = organism(obs)
            std = log_std.exp().unsqueeze(0).expand_as(action_mean)
            dist = torch.distributions.Normal(action_mean, std)
            action_sample = dist.sample()
            lp = dist.log_prob(action_sample).sum(dim=-1)

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

        # Returns & advantages
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

        # Periodic C_i measurement
        if (ep + 1) % ci_interval == 0:
            ci_snap = measure_coordination_quality(
                organism, perception_fn, matter, n_samples=2000, device=device,
            )
            metrics["ci_trajectory"].append({"episode": ep + 1, **ci_snap})

    return metrics


def run_config(level_name, matter, sensory_dim, embed_dim, seed,
               perception_fn, device="cuda"):
    """Run one config with C_i tracking."""
    torch.manual_seed(seed)
    dev = torch.device(device)

    organism = Organism(
        sensory_dim=sensory_dim,
        embedding_dim=embed_dim,
        action_dim=2,
    ).to(dev)

    t0 = time.time()
    metrics = train_with_ci_tracking(
        matter, organism, perception_fn,
        target_x=0.8, target_y=0.8,
        n_episodes=500, ci_interval=50, device=device,
    )
    elapsed = time.time() - t0

    n_tail = min(100, len(metrics["rock_distance"]))
    avg_dist = sum(metrics["rock_distance"][-n_tail:]) / n_tail

    return {
        "level": level_name,
        "embedding_dim": embed_dim,
        "seed": seed,
        "avg_dist_last100": avg_dist,
        "ci_trajectory": metrics["ci_trajectory"],
        "elapsed_s": elapsed,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    checkpoint_path = results_dir / "ci_dynamics_checkpoint.json"
    results_path = results_dir / "ci_dynamics.json"

    embed_dims = [2, 8, 32]
    seeds = [42, 123, 456]
    dev = torch.device(device)

    # Shared matter
    torch.manual_seed(0)
    matter = RockPushMatter(
        emission_dim=8, action_dim=2, seed_dim=4,
    ).to(dev)

    # Resume
    completed = []
    completed_keys = set()
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            completed = json.load(f)
        for r in completed:
            completed_keys.add((r["level"], r["embedding_dim"], r["seed"]))
        print(f"Resuming: {len(completed)} done")

    # Build levels
    levels = []

    # Oracle
    def oracle_fn(state, emission):
        return state
    for emb in embed_dims:
        for s in seeds:
            levels.append(("oracle", 4, emb, s, oracle_fn))

    # Raw emission
    def emission_fn(state, emission):
        return emission
    for emb in embed_dims:
        for s in seeds:
            levels.append(("raw_emission", 8, emb, s, emission_fn))

    # VAE mu lat=16
    torch.manual_seed(0)
    world = RockPushWorld(
        emission_dim=8, channel_dim=8, env_latent_dim=16,
        embedding_dim=8, action_dim=2, seed_dim=4,
        channel_noise=0.01, target_x=0.8, target_y=0.8,
    ).to(dev)
    world.matter = matter

    print("Pre-training VAE (lat=16)...", end=" ", flush=True)
    vae_losses = train_environment_rockpush(world, n_steps=1500, device=dev)
    print(f"loss={vae_losses[-1]:.4f}")

    env = world.environment
    ch = world.channel
    def make_vae_fn(env_module, ch_module):
        def fn(state, emission):
            with torch.no_grad():
                ch_out = ch_module(emission)
                mu, _ = env_module.encode(ch_out)
            return mu
        return fn
    vae_fn = make_vae_fn(env, ch)
    for emb in embed_dims:
        for s in seeds:
            levels.append(("vae_mu_lat16", 16, emb, s, vae_fn))

    total = len(levels)
    print(f"\nobj-015 C_i dynamics: {total} configs")
    t0 = time.time()

    for level_name, sensory_dim, emb, s, perc_fn in levels:
        key = (level_name, emb, s)
        if key in completed_keys:
            continue

        idx = len(completed) + 1
        print(f"[{idx}/{total}] {level_name}, emb={emb}, seed={s}",
              end=" ... ", flush=True)

        try:
            result = run_config(
                level_name, matter, sensory_dim, emb, s, perc_fn,
                device=device,
            )
            completed.append(result)
            completed_keys.add(key)
            final_ci = result["ci_trajectory"][-1]["C_i"]
            print(f"dist={result['avg_dist_last100']:.3f}, "
                  f"final_C_i={final_ci:.3f} "
                  f"({result['elapsed_s']:.0f}s)")
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback; traceback.print_exc()
            completed.append({
                "level": level_name, "embedding_dim": emb,
                "seed": s, "error": str(e),
            })
            completed_keys.add(key)

        if len(completed) % 3 == 0:
            with open(checkpoint_path, "w") as f:
                json.dump(completed, f, indent=2)

    elapsed = time.time() - t0
    with open(results_path, "w") as f:
        json.dump({"results": completed, "elapsed_seconds": elapsed}, f, indent=2)
    with open(checkpoint_path, "w") as f:
        json.dump(completed, f, indent=2)

    n_ok = len([r for r in completed if "error" not in r])
    print(f"\nDone: {n_ok}/{total} in {elapsed/60:.1f} min")

    try:
        generate_plot(completed, results_dir)
    except Exception as e:
        print(f"Plot failed: {e}")
        import traceback; traceback.print_exc()


def generate_plot(completed, results_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    valid = [r for r in completed if "error" not in r]
    if not valid:
        return

    level_colors = {
        "oracle": "#1f77b4",
        "raw_emission": "#2ca02c",
        "vae_mu_lat16": "#d62728",
    }
    emb_styles = {2: ":", 8: "--", 32: "-"}
    emb_markers = {2: "o", 8: "s", 32: "D"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: C_i trajectory over training (all configs)
    ax = axes[0, 0]
    for r in valid:
        traj = r["ci_trajectory"]
        eps = [t["episode"] for t in traj]
        cis = [t["C_i"] for t in traj]
        color = level_colors.get(r["level"], "gray")
        style = emb_styles.get(r["embedding_dim"], "-")
        ax.plot(eps, cis, color=color, linestyle=style, alpha=0.5, linewidth=1.5)
    ax.axhline(0.6, color="red", linestyle=":", alpha=0.4, label="Threshold (0.6)")
    ax.set_xlabel("Training Episode")
    ax.set_ylabel("C_i (coordination quality)")
    ax.set_title("C_i Trajectory During Training")
    # Legend
    for level, color in level_colors.items():
        ax.plot([], [], color=color, linewidth=2, label=level)
    for emb, style in emb_styles.items():
        ax.plot([], [], color="gray", linestyle=style, linewidth=1.5, label=f"emb={emb}")
    ax.legend(fontsize=7, loc="upper left", ncol=2)

    # Panel 2: Mean C_i trajectory by condition (averaged over seeds)
    ax = axes[0, 1]
    by_group = defaultdict(list)
    for r in valid:
        by_group[(r["level"], r["embedding_dim"])].append(r)

    for (level, emb), runs in sorted(by_group.items()):
        # Align trajectories
        all_eps = sorted(set(t["episode"] for r in runs for t in r["ci_trajectory"]))
        mean_cis = []
        for ep in all_eps:
            vals = []
            for r in runs:
                for t in r["ci_trajectory"]:
                    if t["episode"] == ep:
                        vals.append(t["C_i"])
            mean_cis.append(np.mean(vals) if vals else np.nan)

        color = level_colors.get(level, "gray")
        style = emb_styles.get(emb, "-")
        marker = emb_markers.get(emb, "o")
        ax.plot(all_eps, mean_cis, color=color, linestyle=style,
                marker=marker, markersize=4, linewidth=2,
                label=f"{level} e={emb}")

    ax.axhline(0.6, color="red", linestyle=":", alpha=0.4)
    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Mean C_i")
    ax.set_title("Mean C_i Trajectory (averaged over seeds)")
    ax.legend(fontsize=6, loc="upper left", ncol=2)

    # Panel 3: Early C_i (ep=100) vs final distance
    ax = axes[1, 0]
    for r in valid:
        # Find C_i at episode 100
        ci_100 = None
        for t in r["ci_trajectory"]:
            if t["episode"] == 100:
                ci_100 = t["C_i"]
                break
        if ci_100 is not None:
            color = level_colors.get(r["level"], "gray")
            ax.scatter(ci_100, r["avg_dist_last100"], color=color,
                      s=r["embedding_dim"] * 3, alpha=0.7,
                      edgecolors="black", linewidth=0.3)

    # Fit line
    early_cis, final_dists = [], []
    for r in valid:
        for t in r["ci_trajectory"]:
            if t["episode"] == 100:
                early_cis.append(t["C_i"])
                final_dists.append(r["avg_dist_last100"])
    if len(early_cis) >= 5:
        corr = np.corrcoef(early_cis, final_dists)[0, 1]
        z_fit = np.polyfit(early_cis, final_dists, 1)
        x_range = np.linspace(min(early_cis), max(early_cis), 100)
        ax.plot(x_range, np.polyval(z_fit, x_range), "k--", alpha=0.4)
        ax.text(0.05, 0.95, f"r = {corr:.3f}\n(ep=100 C_i)", transform=ax.transAxes,
                fontsize=10, fontweight="bold", verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    for level, color in level_colors.items():
        ax.scatter([], [], color=color, label=level, s=40)
    ax.set_xlabel("C_i at Episode 100")
    ax.set_ylabel("Final Rock-Target Distance")
    ax.set_title("Early C_i Predicts Final Performance")
    ax.legend(fontsize=8)

    # Panel 4: C_i slope (first 200 episodes) vs final distance
    ax = axes[1, 1]
    slopes, dists_for_slope = [], []
    colors_for_slope = []
    for r in valid:
        traj = r["ci_trajectory"]
        early = [(t["episode"], t["C_i"]) for t in traj if t["episode"] <= 200]
        if len(early) >= 3:
            eps_arr = np.array([e for e, c in early])
            ci_arr = np.array([c for e, c in early])
            slope = np.polyfit(eps_arr, ci_arr, 1)[0] * 100  # per 100 episodes
            slopes.append(slope)
            dists_for_slope.append(r["avg_dist_last100"])
            colors_for_slope.append(level_colors.get(r["level"], "gray"))

    if slopes:
        for s, d, c in zip(slopes, dists_for_slope, colors_for_slope):
            ax.scatter(s, d, color=c, s=50, alpha=0.7, edgecolors="black", linewidth=0.3)
        if len(slopes) >= 5:
            corr_slope = np.corrcoef(slopes, dists_for_slope)[0, 1]
            ax.text(0.05, 0.95, f"r = {corr_slope:.3f}\n(C_i slope)", transform=ax.transAxes,
                    fontsize=10, fontweight="bold", verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("C_i Slope (per 100 episodes, first 200)")
    ax.set_ylabel("Final Rock-Target Distance")
    ax.set_title("C_i Learning Rate Predicts Success")
    for level, color in level_colors.items():
        ax.scatter([], [], color=color, label=level, s=40)
    ax.legend(fontsize=8)

    fig.suptitle("obj-015: C_i Dynamics During Training", fontsize=14)
    plt.tight_layout()
    out = results_dir / "obj015_ci_dynamics.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
