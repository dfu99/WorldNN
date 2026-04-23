"""obj-027: Asymmetry-scaling curve at larger embedding capacity.

Extends obj-024 (which capped at embed_dim=32) to test whether the SA
substitution pattern persists, saturates, or reverses at larger capacity.
Exploits A40 48GB VRAM with a batch=1024 regime we could not run on the
A4500.

Design:
  - Task: 1-rock rock-push (same as obj-024)
  - sensory_dim ∈ {2, 8, 16} (extremes + middle)
  - embed_dim ∈ {16, 32, 64, 128, 256, 512}
  - n_seeds = 5
  - 800 episodes, batch=1024 (vs obj-024's 256)
  - Total: 3 × 6 × 5 = 90 configs

Hypothesis:
  - Sensory=16 peak SA continues rising past embed=32 until saturation;
    identify the saturation point. If it keeps rising, the
    "capacity ceiling" claim in §5.7 needs revision.
  - Sensory=2 peak SA stays ≈0 regardless of embed (rate-distortion bound).
  - Sensory=8 traces an intermediate curve.
"""

import os
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
from worldnn.matter import RockPushMatter
from worldnn.organism import Organism


def train_organism_sensory(
    matter, organism, sensory_dim, target_x, target_y,
    n_episodes=800, steps_per_episode=20, batch_size=1024,
    lr=3e-4, gamma=0.99, entropy_coef=0.01,
    action_std_init=0.8, action_std_final=0.2, clip_eps=0.2,
    ppo_epochs=4, device="cpu",
):
    dev = torch.device(device)
    target = torch.tensor([target_x, target_y], device=dev)
    seed_dim = matter.seed_dim
    log_std = nn.Parameter(torch.full((2,), math.log(action_std_init), device=dev))
    optimizer = torch.optim.Adam(list(organism.parameters()) + [log_std], lr=lr)

    metrics = {"rewards": [], "rock_distance": [], "contact_rate": []}

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

            action_mean, _, value = organism(obs)
            std = torch.full_like(action_mean, current_std)
            dist = torch.distributions.Normal(action_mean, std)
            action_sample = dist.sample()
            lp = dist.log_prob(action_sample).sum(dim=-1)

            rock_pos = next_state[:, :2]
            org_pos = next_state[:, 2:4]
            rock_dist = torch.norm(rock_pos - target, dim=-1)
            org_rock_dist = torch.norm(rock_pos - org_pos, dim=-1)
            reward = (1.0 - rock_dist) + 0.2 * (1.0 - org_rock_dist) + 0.1 * contact

            all_obs.append(obs.detach()); all_actions.append(action_sample.detach())
            all_log_probs.append(lp.detach()); all_rewards.append(reward.detach())
            all_values.append(value.detach()); contact_sum += contact.mean().item()
            state = next_state.detach(); action = action_sample.detach()

        T = len(all_rewards)
        returns = []
        G = torch.zeros(batch_size, device=dev)
        for t_idx in reversed(range(T)):
            G = all_rewards[t_idx] + gamma * G
            returns.insert(0, G)
        returns = torch.stack(returns); values = torch.stack(all_values)
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
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(list(organism.parameters()) + [log_std], 1.0)
                optimizer.step()

        with torch.no_grad():
            rock_pos = state[:, :2]
            final_dist = torch.norm(rock_pos - target, dim=-1).mean().item()
        metrics["rewards"].append(sum(r.mean().item() for r in all_rewards) / T)
        metrics["rock_distance"].append(final_dist)
        metrics["contact_rate"].append(contact_sum / T)

    return metrics


def compute_sa_sensory(matter, organism, sensory_dim, n_samples=2000, device="cpu"):
    dev = torch.device(device)
    organism.eval()
    cos_sims = []
    with torch.no_grad():
        batch = 512
        for _ in range(n_samples // batch + 1):
            state = matter.reset_state(batch, dev)
            seed = torch.randn(batch, matter.seed_dim, device=dev)
            action = torch.randn(batch, 2, device=dev) * 0.1
            next_state, emission, _ = matter(state, seed, action)
            rock_pos = next_state[:, :2]; org_pos = next_state[:, 2:4]
            a_optimal = rock_pos - org_pos
            a_optimal = a_optimal / (a_optimal.norm(dim=-1, keepdim=True) + 1e-8)
            obs = emission[:, :sensory_dim]
            a_learned, _, _ = organism(obs)
            a_learned = a_learned / (a_learned.norm(dim=-1, keepdim=True) + 1e-8)
            cos_sims.append((a_learned * a_optimal).sum(dim=-1).cpu())
    return torch.cat(cos_sims)[:n_samples].mean().item()


def main():
    device = os.environ.get("WORLDNN_DEVICE", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("WORLDNN_DEVICE=cuda set but CUDA unavailable")
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    ck = results_dir / "obj027_asymmetry_scaling_checkpoint.json"
    out = results_dir / "obj027_asymmetry_scaling.json"

    sensory_dims = [2, 8, 16]
    embed_dims = [16, 32, 64, 128, 256, 512]
    seeds = [42, 123, 456, 789, 1337]
    dev = torch.device(device)

    torch.manual_seed(0)
    matter = RockPushMatter(emission_dim=16, action_dim=2, seed_dim=4).to(dev)

    completed = []
    completed_keys = set()
    if ck.exists():
        completed = json.load(ck.open())
        for r in completed:
            completed_keys.add((r["sensory_dim"], r["embedding_dim"], r["seed"]))
        print(f"Resuming: {len(completed)} done")

    configs = [(sd, ed, s) for sd in sensory_dims for ed in embed_dims for s in seeds]
    total = len(configs)
    print(f"\nAsymmetry scaling: {total} configs, batch=1024")
    t0 = time.time()

    for (sd, ed, s) in configs:
        if (sd, ed, s) in completed_keys:
            continue
        idx = len(completed) + 1
        print(f"[{idx}/{total}] sensory={sd}, emb={ed}, seed={s}", end=" ... ", flush=True)
        torch.manual_seed(s)
        organism = Organism(sensory_dim=sd, embedding_dim=ed, action_dim=2).to(dev)
        metrics = train_organism_sensory(
            matter, organism, sd, target_x=0.8, target_y=0.8,
            n_episodes=800, batch_size=1024, device=device,
        )
        sa = compute_sa_sensory(matter, organism, sd, device=device)
        n_tail = min(100, len(metrics["rock_distance"]))
        avg_dist = sum(metrics["rock_distance"][-n_tail:]) / n_tail
        avg_contact = sum(metrics["contact_rate"][-n_tail:]) / n_tail
        result = {
            "sensory_dim": sd, "embedding_dim": ed, "seed": s,
            "avg_dist": avg_dist, "avg_contact": avg_contact, "SA": sa,
            "final_dist": metrics["rock_distance"][-1],
            "n_params": sum(p.numel() for p in organism.parameters()),
        }
        completed.append(result)
        print(f"SA={sa:.3f}, dist={avg_dist:.3f}, params={result['n_params']}")
        if len(completed) % 5 == 0:
            json.dump(completed, ck.open("w"), indent=2)

    elapsed = time.time() - t0
    json.dump({"results": completed, "elapsed_s": elapsed}, out.open("w"), indent=2)
    json.dump(completed, ck.open("w"), indent=2)
    print(f"\nDone: {len(completed)}/{total} in {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
