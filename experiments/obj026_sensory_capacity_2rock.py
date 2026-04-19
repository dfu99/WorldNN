"""obj-026: 2-rock sensory-capacity replicate (Reviewer E mitigation).

Replicates obj-024's sensory × capacity grid on the 2-rock (6D state) task to
test whether the substitution pattern holds across task dimensionality. If
it does, the task-similarity concern is substantially reduced.

Design:
  - Task: 2-rock (MultiRockMatter with n_rocks=2), 6D state
  - emission_dim=16 (same as obj-024)
  - sensory_dim ∈ {2, 4, 8, 16}
  - embedding_dim ∈ {2, 4, 8, 16, 32}
  - 3 seeds (fewer than obj-024's 5 — this is a replicate, not a primary)
  - Total: 4 × 5 × 3 = 60 configs, 800 episodes each

If the pattern replicates:
  - Reviewer E risk drops from High to Medium
  - The rate-distortion result from obj-025 T3 becomes generalizable
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
import numpy as np
from worldnn.matter import MultiRockMatter
from worldnn.organism import Organism


def compute_optimal_action_2rock(state, targets):
    """Optimal: approach the rock farthest from its target and push toward it."""
    rock_positions = state[:, :4].reshape(-1, 2, 2)
    org_pos = state[:, 4:6]
    rock_dists = torch.stack([
        torch.norm(rock_positions[:, i] - targets[i].unsqueeze(0), dim=-1)
        for i in range(2)
    ], dim=-1)
    worst_idx = rock_dists.argmax(dim=-1)
    worst_rock = torch.zeros_like(org_pos)
    worst_target = torch.zeros_like(org_pos)
    for i in range(2):
        mask = (worst_idx == i).unsqueeze(-1).float()
        worst_rock += mask * rock_positions[:, i]
        worst_target += mask * targets[i].unsqueeze(0)
    to_rock = worst_rock - org_pos
    rock_dist = torch.norm(to_rock, dim=-1, keepdim=True).clamp(min=1e-6)
    to_target = worst_target - worst_rock
    target_dist = torch.norm(to_target, dim=-1, keepdim=True).clamp(min=1e-6)
    near = (rock_dist < 0.15).float()
    optimal = (1 - near) * (to_rock / rock_dist) + near * (to_target / target_dist)
    return optimal / torch.norm(optimal, dim=-1, keepdim=True).clamp(min=1e-6)


def train_organism_2rock(
    matter, organism, sensory_dim, targets,
    n_episodes=800, steps_per_episode=20, batch_size=256,
    lr=3e-4, gamma=0.99, entropy_coef=0.01,
    action_std_init=0.8, action_std_final=0.2, clip_eps=0.2,
    ppo_epochs=4, device="cuda",
):
    dev = torch.device(device)
    seed_dim = matter.seed_dim

    log_std = nn.Parameter(torch.full((2,), math.log(action_std_init), device=dev))
    optimizer = torch.optim.Adam(list(organism.parameters()) + [log_std], lr=lr)

    metrics = {"rewards": [], "mean_dist": [], "contact_rate": []}

    for ep in range(n_episodes):
        organism.train()
        state = matter.reset_state(batch_size, dev)
        action = None
        all_obs, all_actions, all_log_probs = [], [], []
        all_rewards, all_values = [], []
        contact_sum = 0.0

        frac = min(ep / (n_episodes * 0.7), 1.0)
        current_std = action_std_init + (action_std_final - action_std_init) * frac

        for t in range(steps_per_episode):
            seed = torch.randn(batch_size, seed_dim, device=dev)
            if action is None:
                action = torch.zeros(batch_size, 2, device=dev)

            next_state, emission, contact = matter(state, seed, action)
            obs = emission[:, :sensory_dim]

            action_mean, embedding, value = organism(obs)
            std = torch.full_like(action_mean, current_std)
            dist = torch.distributions.Normal(action_mean, std)
            action_sample = dist.sample()
            lp = dist.log_prob(action_sample).sum(dim=-1)

            # Reward: mean distance from both rocks to their targets
            rock_positions = next_state[:, :4].reshape(-1, 2, 2)
            per_rock_dist = torch.stack([
                torch.norm(rock_positions[:, i] - targets[i].unsqueeze(0), dim=-1)
                for i in range(2)
            ], dim=-1)
            mean_rock_dist = per_rock_dist.mean(dim=-1)
            reward = (1.0 - mean_rock_dist) + 0.1 * contact

            all_obs.append(obs.detach())
            all_actions.append(action_sample.detach())
            all_log_probs.append(lp.detach())
            all_rewards.append(reward.detach())
            all_values.append(value.detach())
            contact_sum += contact.mean().item()

            state = next_state.detach()
            action = action_sample.detach()

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
                std = torch.full_like(action_mean, current_std)
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
                nn.utils.clip_grad_norm_(list(organism.parameters()) + [log_std], 1.0)
                optimizer.step()

        with torch.no_grad():
            rock_positions = state[:, :4].reshape(-1, 2, 2)
            per_rock_dist = torch.stack([
                torch.norm(rock_positions[:, i] - targets[i].unsqueeze(0), dim=-1)
                for i in range(2)
            ], dim=-1)
            final_dist = per_rock_dist.mean().item()
        metrics["rewards"].append(sum(r.mean().item() for r in all_rewards) / T)
        metrics["mean_dist"].append(final_dist)
        metrics["contact_rate"].append(contact_sum / T)

    return metrics


def compute_sa_2rock(matter, organism, sensory_dim, targets, n_samples=2000, device="cuda"):
    dev = torch.device(device)
    organism.eval()
    cos_sims = []
    with torch.no_grad():
        batch = 256
        for _ in range(n_samples // batch + 1):
            state = matter.reset_state(batch, dev)
            seed = torch.randn(batch, matter.seed_dim, device=dev)
            action = torch.randn(batch, 2, device=dev) * 0.1
            next_state, emission, _ = matter(state, seed, action)
            a_optimal = compute_optimal_action_2rock(next_state, targets)
            obs = emission[:, :sensory_dim]
            a_learned, _, _ = organism(obs)
            a_learned = a_learned / (a_learned.norm(dim=-1, keepdim=True) + 1e-8)
            cos_sim = (a_learned * a_optimal).sum(dim=-1)
            cos_sims.append(cos_sim.cpu())
    return torch.cat(cos_sims)[:n_samples].mean().item()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    checkpoint_path = results_dir / "obj026_sensory_capacity_2rock_checkpoint.json"
    results_path = results_dir / "obj026_sensory_capacity_2rock.json"

    sensory_dims = [2, 4, 8, 16]
    embed_dims = [2, 4, 8, 16, 32]
    seeds = [42, 123, 456]
    dev = torch.device(device)

    torch.manual_seed(0)
    matter = MultiRockMatter(
        emission_dim=16, action_dim=2, seed_dim=4, n_rocks=2,
    ).to(dev)
    targets = torch.tensor([[0.2, 0.8], [0.8, 0.2]], device=dev)

    completed = []
    completed_keys = set()
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            completed = json.load(f)
        for r in completed:
            completed_keys.add((r["sensory_dim"], r["embedding_dim"], r["seed"]))
        print(f"Resuming: {len(completed)} done")

    configs = [(sd, ed, s) for sd in sensory_dims for ed in embed_dims for s in seeds]
    total = len(configs)
    print(f"\n2-rock sensory-capacity replicate: {total} configs")
    t0 = time.time()

    for (sd, ed, s) in configs:
        if (sd, ed, s) in completed_keys:
            continue

        idx = len(completed) + 1
        print(f"[{idx}/{total}] sensory={sd}, emb={ed}, seed={s}", end=" ... ", flush=True)

        try:
            torch.manual_seed(s)
            organism = Organism(sensory_dim=sd, embedding_dim=ed, action_dim=2).to(dev)
            metrics = train_organism_2rock(
                matter, organism, sd, targets, n_episodes=800, device=device,
            )
            sa = compute_sa_2rock(matter, organism, sd, targets, device=device)
            n_tail = min(100, len(metrics["mean_dist"]))
            avg_dist = sum(metrics["mean_dist"][-n_tail:]) / n_tail
            avg_contact = sum(metrics["contact_rate"][-n_tail:]) / n_tail
            result = {
                "sensory_dim": sd, "embedding_dim": ed, "seed": s,
                "avg_dist": avg_dist, "avg_contact": avg_contact, "SA": sa,
                "final_dist": metrics["mean_dist"][-1],
                "n_params": sum(p.numel() for p in organism.parameters()),
            }
            completed.append(result)
            print(f"SA={sa:.3f}, dist={avg_dist:.3f}")
        except Exception as e:
            print(f"FAILED: {e}")
            completed.append({
                "sensory_dim": sd, "embedding_dim": ed, "seed": s, "error": str(e),
            })

        if len(completed) % 5 == 0:
            with open(checkpoint_path, "w") as f:
                json.dump(completed, f, indent=2)

    elapsed = time.time() - t0
    with open(results_path, "w") as f:
        json.dump({"results": completed, "elapsed_s": elapsed}, f, indent=2)
    with open(checkpoint_path, "w") as f:
        json.dump(completed, f, indent=2)

    print(f"\nDone: {len(completed)}/{total} in {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
