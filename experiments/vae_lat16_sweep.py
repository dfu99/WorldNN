"""Full pipeline sweep (obj-012): VAE lat=16 + oracle baseline comparison.

obj-011 showed VAE lat=16 preserves 82% of state (R²=0.817) and enables
learning (0.460 at emb=32). Now run the full embed_dim sweep to see if
the monotonic capacity curve from the oracle replicates through the VAE.

Grid:
  perception: [oracle, vae_lat16]
  channel_noise: [0.01, 0.1]  (noise only applies to VAE conditions)
  embedding_dim: [2, 4, 8, 16, 32]
  seeds: [42, 123, 456, 789, 1337]
  = oracle: 1 × 5 × 5 = 25 configs
  + VAE:    2 × 5 × 5 = 50 configs
  = 75 total, 500 episodes each
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


def run_vae_config(channel_noise, env_latent_dim, embedding_dim, seed,
                   n_vae_steps=1500, n_episodes=500, device="cuda"):
    """Run a single VAE pipeline config."""
    torch.manual_seed(seed)
    dev = torch.device(device)

    world = RockPushWorld(
        emission_dim=8, channel_dim=8,
        env_latent_dim=env_latent_dim,
        embedding_dim=embedding_dim,
        action_dim=2, seed_dim=4,
        channel_noise=channel_noise,
        target_x=0.8, target_y=0.8,
    ).to(dev)

    t0 = time.time()
    vae_losses = train_environment_rockpush(
        world, n_steps=n_vae_steps, batch_size=256, device=dev
    )
    vae_time = time.time() - t0

    t1 = time.time()
    metrics = train_organism_ppo_rockpush(
        world, n_episodes=n_episodes, steps_per_episode=20,
        batch_size=256, device=dev,
    )
    ppo_time = time.time() - t1

    n_tail = min(100, len(metrics["rock_distance"]))
    avg_dist = sum(metrics["rock_distance"][-n_tail:]) / n_tail
    avg_contact = sum(metrics["contact_rate"][-n_tail:]) / n_tail

    return {
        "perception": "vae_lat16",
        "channel_noise": channel_noise,
        "env_latent_dim": env_latent_dim,
        "embedding_dim": embedding_dim,
        "seed": seed,
        "final_dist": metrics["rock_distance"][-1],
        "avg_dist_last100": avg_dist,
        "final_contact": metrics["contact_rate"][-1],
        "avg_contact_last100": avg_contact,
        "vae_final_loss": vae_losses[-1],
        "elapsed_s": vae_time + ppo_time,
    }


def train_oracle_ppo(matter, organism, embedding_dim, seed,
                     target_x=0.8, target_y=0.8,
                     n_episodes=500, steps_per_episode=20, batch_size=256,
                     lr=3e-4, gamma=0.99, entropy_coef=0.01,
                     action_std_init=0.8, action_std_final=0.2,
                     clip_eps=0.2, ppo_epochs=4, device="cuda"):
    """Train organism with oracle perception (direct state observation)."""
    dev = torch.device(device)
    target = torch.tensor([target_x, target_y], device=dev)

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
            seed_t = torch.randn(batch_size, matter.seed_dim, device=dev)
            if action is None:
                action = torch.zeros(batch_size, 2, device=dev)

            next_state, emission, contact = matter(state, seed_t, action)

            # Oracle: organism sees raw 4D state
            obs = next_state.detach()

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

            all_obs.append(obs)
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


def run_oracle_config(embedding_dim, seed, n_episodes=500, device="cuda"):
    """Run a single oracle config."""
    torch.manual_seed(seed)
    dev = torch.device(device)

    matter = RockPushMatter(
        emission_dim=8, action_dim=2, seed_dim=4,
    ).to(dev)

    organism = Organism(
        sensory_dim=4,  # direct 4D state observation
        embedding_dim=embedding_dim,
        action_dim=2,
    ).to(dev)

    t0 = time.time()
    metrics = train_oracle_ppo(
        matter, organism, embedding_dim, seed,
        n_episodes=n_episodes, device=device,
    )
    elapsed = time.time() - t0

    n_tail = min(100, len(metrics["rock_distance"]))
    avg_dist = sum(metrics["rock_distance"][-n_tail:]) / n_tail
    avg_contact = sum(metrics["contact_rate"][-n_tail:]) / n_tail

    return {
        "perception": "oracle",
        "channel_noise": 0.0,
        "env_latent_dim": 0,
        "embedding_dim": embedding_dim,
        "seed": seed,
        "final_dist": metrics["rock_distance"][-1],
        "avg_dist_last100": avg_dist,
        "final_contact": metrics["contact_rate"][-1],
        "avg_contact_last100": avg_contact,
        "vae_final_loss": None,
        "elapsed_s": elapsed,
    }


def make_config_key(r):
    return (r["perception"], r.get("channel_noise", 0), r["embedding_dim"], r["seed"])


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    checkpoint_path = results_dir / "vae_lat16_checkpoint.json"
    results_path = results_dir / "vae_lat16_sweep.json"

    noises = [0.01, 0.1]
    env_latent_dim = 16
    embed_dims = [2, 4, 8, 16, 32]
    seeds = [42, 123, 456, 789, 1337]

    # Build config list: oracle first, then VAE
    configs = []
    for emb in embed_dims:
        for s in seeds:
            configs.append(("oracle", 0.0, emb, s))
    for noise in noises:
        for emb in embed_dims:
            for s in seeds:
                configs.append(("vae_lat16", noise, emb, s))

    # Resume from checkpoint
    completed = []
    completed_keys = set()
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            completed = json.load(f)
        for r in completed:
            completed_keys.add(make_config_key(r))
        print(f"Resuming: {len(completed)}/{len(configs)} done")

    total = len(configs)
    remaining = total - len(completed_keys)
    print(f"obj-012 sweep: {total} configs ({remaining} remaining)")
    print(f"  Oracle: {len(embed_dims) * len(seeds)} configs")
    print(f"  VAE lat=16: {len(noises) * len(embed_dims) * len(seeds)} configs")
    t0 = time.time()

    for perc, noise, emb, s in configs:
        key = (perc, noise, emb, s)
        if key in completed_keys:
            continue

        idx = len(completed) + 1
        label = f"{perc}" if perc == "oracle" else f"vae_lat16 noise={noise}"
        print(f"[{idx}/{total}] {label}, emb={emb}, seed={s}",
              end=" ... ", flush=True)

        try:
            if perc == "oracle":
                result = run_oracle_config(emb, s, device=device)
            else:
                result = run_vae_config(noise, env_latent_dim, emb, s, device=device)
            completed.append(result)
            completed_keys.add(key)
            print(f"dist={result['avg_dist_last100']:.3f}, "
                  f"contact={result['avg_contact_last100']:.3f} "
                  f"({result['elapsed_s']:.0f}s)")
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback; traceback.print_exc()
            completed.append({
                "perception": perc, "channel_noise": noise,
                "env_latent_dim": env_latent_dim if perc != "oracle" else 0,
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
        label = r["perception"]
        if r["perception"] == "vae_lat16":
            label = f"vae_lat16_n{r['channel_noise']}"
        by_group[(label, r["embedding_dim"])].append(r["avg_dist_last100"])

    for label in ["oracle", "vae_lat16_n0.01", "vae_lat16_n0.1"]:
        print(f"\n  {label}:")
        for emb in [2, 4, 8, 16, 32]:
            vals = by_group.get((label, emb), [])
            if vals:
                print(f"    emb={emb:>2}: {np.mean(vals):.3f} ± {np.std(vals):.3f} (n={len(vals)})")

    # Capacity gap: emb=2 vs emb=32
    print("\n=== Capacity gap (emb=2 minus emb=32, higher = more capacity effect) ===")
    for label in ["oracle", "vae_lat16_n0.01", "vae_lat16_n0.1"]:
        v2 = by_group.get((label, 2), [])
        v32 = by_group.get((label, 32), [])
        if v2 and v32:
            gap = np.mean(v2) - np.mean(v32)
            print(f"  {label}: {gap:+.3f} (emb=2: {np.mean(v2):.3f}, emb=32: {np.mean(v32):.3f})")


def generate_plot(completed, results_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    valid = [r for r in completed if "error" not in r]
    if not valid:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    embed_dims = [2, 4, 8, 16, 32]

    # Collect by group
    by_group = defaultdict(lambda: defaultdict(list))
    for r in valid:
        label = r["perception"]
        if r["perception"] == "vae_lat16":
            label = f"VAE lat=16 (σ={r['channel_noise']})"
        else:
            label = "Oracle (direct state)"
        by_group[label][r["embedding_dim"]].append(r["avg_dist_last100"])

    colors = {
        "Oracle (direct state)": "steelblue",
        "VAE lat=16 (σ=0.01)": "coral",
        "VAE lat=16 (σ=0.1)": "firebrick",
    }

    # Panel 1: Distance vs embed_dim by perception
    ax = axes[0]
    for label in ["Oracle (direct state)", "VAE lat=16 (σ=0.01)", "VAE lat=16 (σ=0.1)"]:
        if label not in by_group:
            continue
        means = [np.mean(by_group[label].get(e, [np.nan])) for e in embed_dims]
        stds = [np.std(by_group[label].get(e, [0])) for e in embed_dims]
        ax.errorbar(embed_dims, means, yerr=stds, marker="o", label=label,
                    color=colors.get(label), capsize=3, linewidth=2)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Random baseline")
    ax.set_xlabel("Embedding Dimension")
    ax.set_ylabel("Rock-Target Distance (lower = better)")
    ax.set_title("Capacity Curve: Oracle vs VAE lat=16")
    ax.legend(fontsize=8)
    ax.set_xscale("log", base=2)
    ax.set_xticks(embed_dims)
    ax.set_xticklabels(embed_dims)
    ax.set_ylim(0.2, 0.55)

    # Panel 2: Capacity gap by perception level
    ax = axes[1]
    groups_for_gap = {}
    for r in valid:
        key = r["perception"]
        if key == "vae_lat16":
            key = f"vae_n{r['channel_noise']}"
        if key not in groups_for_gap:
            groups_for_gap[key] = defaultdict(list)
        groups_for_gap[key][r["embedding_dim"]].append(r["avg_dist_last100"])

    gap_labels, gap_vals = [], []
    for key, nice in [("oracle", "Oracle"), ("vae_n0.01", "VAE σ=0.01"), ("vae_n0.1", "VAE σ=0.1")]:
        if key in groups_for_gap:
            v2 = groups_for_gap[key].get(2, [])
            v32 = groups_for_gap[key].get(32, [])
            if v2 and v32:
                gap = np.mean(v2) - np.mean(v32)
                gap_labels.append(nice)
                gap_vals.append(gap)
    bar_colors = ["steelblue", "coral", "firebrick"][:len(gap_labels)]
    ax.bar(gap_labels, gap_vals, color=bar_colors, alpha=0.85)
    ax.set_ylabel("Capacity Gap (emb=2 − emb=32)")
    ax.set_title("How Much Does Capacity Matter?")
    ax.axhline(0, color="gray", linestyle="-", alpha=0.3)

    # Panel 3: Contact rate vs embed_dim
    ax = axes[2]
    by_contact = defaultdict(lambda: defaultdict(list))
    for r in valid:
        label = r["perception"]
        if r["perception"] == "vae_lat16":
            label = f"VAE lat=16 (σ={r['channel_noise']})"
        else:
            label = "Oracle (direct state)"
        by_contact[label][r["embedding_dim"]].append(r["avg_contact_last100"])

    for label in ["Oracle (direct state)", "VAE lat=16 (σ=0.01)", "VAE lat=16 (σ=0.1)"]:
        if label not in by_contact:
            continue
        means = [np.mean(by_contact[label].get(e, [np.nan])) for e in embed_dims]
        ax.plot(embed_dims, means, marker="s", label=label,
                color=colors.get(label), linewidth=2)
    ax.set_xlabel("Embedding Dimension")
    ax.set_ylabel("Contact Rate")
    ax.set_title("Contact Rate by Capacity")
    ax.legend(fontsize=8)
    ax.set_xscale("log", base=2)
    ax.set_xticks(embed_dims)
    ax.set_xticklabels(embed_dims)

    fig.suptitle("obj-012: Full Pipeline — Oracle vs VAE lat=16 Capacity Sweep", fontsize=14)
    plt.tight_layout()
    out = results_dir / "obj012_oracle_vs_vae_lat16.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
