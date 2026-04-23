"""obj-028: 1D continuous positioning sensory-capacity sweep.

Tests whether the sensory-capacity substitution pattern from obj-024 holds
on a qualitatively different task family (1D continuous positioning, not
manipulation). Addresses Reviewer E task-similarity risk without relying
on multi-rock scaling that caused obj-026's floor effect.

Task: ContinuousMatter — organism observes a 1D position signal through
a 4D emission, learns to push its position toward a target. Simpler than
rock-push; no contact mechanics.

Design:
  - sensory_dim ∈ {1, 2, 4} (emission_dim=4)
  - embedding_dim ∈ {2, 4, 8, 16, 32}
  - n_seeds = 5
  - 500 episodes, batch=1024
  - Total: 3 × 5 × 5 = 75 configs
"""

import os
import sys
import json
import time
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from worldnn.matter import ContinuousMatter
from worldnn.organism import Organism


def train_1d(
    matter, organism, sensory_dim, target,
    n_episodes=500, steps_per_episode=20, batch_size=1024,
    lr=3e-4, gamma=0.99, entropy_coef=0.01,
    action_std_init=0.5, action_std_final=0.1, clip_eps=0.2,
    ppo_epochs=4, device="cpu",
):
    dev = torch.device(device)
    seed_dim = matter.seed_dim
    log_std = nn.Parameter(torch.full((1,), math.log(action_std_init), device=dev))
    optimizer = torch.optim.Adam(list(organism.parameters()) + [log_std], lr=lr)
    final_dist_trace = []

    for ep in range(n_episodes):
        organism.train()
        state = matter.reset_state(batch_size, dev)
        action = None
        all_obs, all_actions, all_log_probs = [], [], []
        all_rewards, all_values = [], []

        frac = min(ep / (n_episodes * 0.7), 1.0)
        current_std = action_std_init + (action_std_final - action_std_init) * frac

        for t in range(steps_per_episode):
            seed = torch.randn(batch_size, seed_dim, device=dev)
            if action is None:
                action = torch.zeros(batch_size, 1, device=dev)
            next_state, emission, _ = matter(state, seed, action)
            obs = emission[:, :sensory_dim]

            action_mean, _, value = organism(obs)
            # Organism outputs action_dim=2 by default; we'll use the first dim for 1D
            action_mean_1d = action_mean[:, :1]
            std = torch.full_like(action_mean_1d, current_std)
            dist = torch.distributions.Normal(action_mean_1d, std)
            action_sample = dist.sample()
            lp = dist.log_prob(action_sample).sum(dim=-1)

            pos_dist = torch.abs(next_state - target)
            reward = 1.0 - pos_dist

            all_obs.append(obs.detach()); all_actions.append(action_sample.detach())
            all_log_probs.append(lp.detach()); all_rewards.append(reward.detach())
            all_values.append(value.detach())
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
                action_mean_1d = action_mean[:, :1]
                std = torch.full_like(action_mean_1d, current_std)
                d = torch.distributions.Normal(action_mean_1d, std)
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
            final_dist_trace.append(torch.abs(state - target).mean().item())

    return final_dist_trace


def compute_sa_1d(matter, organism, sensory_dim, target, n_samples=2000, device="cpu"):
    dev = torch.device(device)
    organism.eval()
    cos_sims = []
    with torch.no_grad():
        batch = 512
        for _ in range(n_samples // batch + 1):
            state = matter.reset_state(batch, dev)
            seed = torch.randn(batch, matter.seed_dim, device=dev)
            action = torch.randn(batch, 1, device=dev) * 0.1
            next_state, emission, _ = matter(state, seed, action)
            a_optimal = torch.sign(target - next_state)  # scalar direction
            obs = emission[:, :sensory_dim]
            a_learned_full, _, _ = organism(obs)
            a_learned = a_learned_full[:, :1]
            a_learned_norm = a_learned / (a_learned.abs() + 1e-8)
            cos = (a_learned_norm * a_optimal.unsqueeze(-1)).sum(dim=-1)
            cos_sims.append(cos.cpu())
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
    ck = results_dir / "obj028_1d_sensory_capacity_checkpoint.json"
    out = results_dir / "obj028_1d_sensory_capacity.json"

    sensory_dims = [1, 2, 4]
    embed_dims = [2, 4, 8, 16, 32]
    seeds = [42, 123, 456, 789, 1337]
    dev = torch.device(device)

    torch.manual_seed(0)
    matter = ContinuousMatter(emission_dim=4, action_dim=1, seed_dim=4).to(dev)
    target = torch.tensor(0.8, device=dev)

    completed = []
    completed_keys = set()
    if ck.exists():
        completed = json.load(ck.open())
        for r in completed:
            completed_keys.add((r["sensory_dim"], r["embedding_dim"], r["seed"]))
        print(f"Resuming: {len(completed)} done")

    configs = [(sd, ed, s) for sd in sensory_dims for ed in embed_dims for s in seeds]
    total = len(configs)
    print(f"\n1D positioning sensory-capacity: {total} configs, batch=1024")
    t0 = time.time()

    for (sd, ed, s) in configs:
        if (sd, ed, s) in completed_keys:
            continue
        idx = len(completed) + 1
        print(f"[{idx}/{total}] sensory={sd}, emb={ed}, seed={s}", end=" ... ", flush=True)
        torch.manual_seed(s)
        organism = Organism(sensory_dim=sd, embedding_dim=ed, action_dim=2).to(dev)
        trace = train_1d(matter, organism, sd, target, n_episodes=500, batch_size=1024, device=device)
        sa = compute_sa_1d(matter, organism, sd, target, device=device)
        avg_dist = sum(trace[-50:]) / min(50, len(trace))
        result = {
            "sensory_dim": sd, "embedding_dim": ed, "seed": s,
            "avg_dist": avg_dist, "SA": sa,
            "final_dist": trace[-1],
            "n_params": sum(p.numel() for p in organism.parameters()),
        }
        completed.append(result)
        print(f"SA={sa:.3f}, dist={avg_dist:.3f}")
        if len(completed) % 5 == 0:
            json.dump(completed, ck.open("w"), indent=2)

    elapsed = time.time() - t0
    json.dump({"results": completed, "elapsed_s": elapsed}, out.open("w"), indent=2)
    json.dump(completed, ck.open("w"), indent=2)
    print(f"\nDone: {len(completed)}/{total} in {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
