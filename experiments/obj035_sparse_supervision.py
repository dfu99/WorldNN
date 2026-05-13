"""obj-035: Sparse oracle-supervision ablation.

Question: does the SA ceiling persist under reduced oracle supervision?

PPO with an auxiliary loss term: at every gradient step, add
  w * ||action_mean(obs) - oracle_action(state)||²
where w ∈ {0.0, 0.01, 0.1, 1.0}. w=0 is the pure-PPO baseline (matches
obj-024); higher w pulls the policy toward oracle imitation.

Hypothesis: if SA is a side-effect of oracle supervision, mean SA
should rise monotonically with w. If SA captures structure that PPO
discovers on its own, mean SA should be roughly constant in w.

Setup: sensory=16, embed=16, 200 ep, batch=128, 3 seeds, 4 supervision
weights = 12 configs. CPU.
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
import numpy as np
from worldnn.matter import RockPushMatter
from worldnn.organism import Organism


def train_with_supervision(
    matter, organism, sensory_dim, target,
    supervision_weight=0.0,
    n_episodes=200, steps_per_episode=20, batch_size=128,
    lr=3e-4, gamma=0.99, entropy_coef=0.01,
    action_std_init=0.8, action_std_final=0.2, clip_eps=0.2,
    ppo_epochs=4, device="cpu",
):
    dev = torch.device(device)
    seed_dim = matter.seed_dim
    log_std = nn.Parameter(torch.full((2,), math.log(action_std_init), device=dev))
    optimizer = torch.optim.Adam(list(organism.parameters()) + [log_std], lr=lr)

    final_dist = None

    for ep in range(n_episodes):
        organism.train()
        state = matter.reset_state(batch_size, dev)
        action = None
        all_obs, all_actions, all_log_probs = [], [], []
        all_rewards, all_values, all_states = [], [], []

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
            rock_pos = next_state[:, :2]; org_pos = next_state[:, 2:4]
            rock_dist = torch.norm(rock_pos - target, dim=-1)
            org_rock_dist = torch.norm(rock_pos - org_pos, dim=-1)
            reward = (1.0 - rock_dist) + 0.2 * (1.0 - org_rock_dist) + 0.1 * contact
            all_obs.append(obs.detach()); all_actions.append(action_sample.detach())
            all_log_probs.append(lp.detach()); all_rewards.append(reward.detach())
            all_values.append(value.detach()); all_states.append(next_state.detach())
            state = next_state.detach(); action = action_sample.detach()

        T = len(all_rewards)
        returns_list = []
        G = torch.zeros(batch_size, device=dev)
        for t_idx in reversed(range(T)):
            G = all_rewards[t_idx] + gamma * G
            returns_list.insert(0, G)
        returns = torch.stack(returns_list); values = torch.stack(all_values)
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_batch = torch.stack(all_obs)
        act_batch = torch.stack(all_actions)
        old_lp = torch.stack(all_log_probs)
        states_batch = torch.stack(all_states)

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
                # === Sparse-supervision auxiliary loss ===
                if supervision_weight > 0:
                    s = states_batch[t_idx]
                    rock_t = s[:, :2]; org_t = s[:, 2:4]
                    a_opt = rock_t - org_t
                    a_opt = a_opt / (a_opt.norm(dim=-1, keepdim=True) + 1e-8)
                    aux = F.mse_loss(action_mean, a_opt)
                    loss = loss + supervision_weight * aux
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(list(organism.parameters()) + [log_std], 1.0)
                optimizer.step()

        final_dist = torch.norm(state[:, :2] - target, dim=-1).mean().item()
    return final_dist


def compute_sa(matter, organism, sensory_dim, n=2000, device="cpu"):
    dev = torch.device(device)
    organism.eval()
    cos_sims = []
    with torch.no_grad():
        batch = 256
        for _ in range(n // batch + 1):
            state = matter.reset_state(batch, dev)
            seed = torch.randn(batch, matter.seed_dim, device=dev)
            action = torch.randn(batch, 2, device=dev) * 0.1
            ns, em, _ = matter(state, seed, action)
            rock = ns[:, :2]; org = ns[:, 2:4]
            a_opt = (rock - org); a_opt = a_opt / (a_opt.norm(dim=-1, keepdim=True) + 1e-8)
            obs = em[:, :sensory_dim]
            a_l, _, _ = organism(obs); a_l = a_l / (a_l.norm(dim=-1, keepdim=True) + 1e-8)
            cos_sims.append((a_l * a_opt).sum(dim=-1).cpu())
    return torch.cat(cos_sims)[:n].mean().item()


def main():
    device = os.environ.get("WORLDNN_DEVICE", "cpu")
    print(f"Device: {device}")
    sensory_dim = 16; embed_dim = 16
    seeds = [42, 123, 456]
    weights = [0.0, 0.01, 0.1, 1.0]

    dev = torch.device(device)
    torch.manual_seed(0)
    matter = RockPushMatter(emission_dim=16, action_dim=2, seed_dim=4).to(dev)
    target = torch.tensor([0.8, 0.8], device=dev)
    results = []
    t0 = time.time()
    for w in weights:
        for s in seeds:
            torch.manual_seed(s)
            organism = Organism(sensory_dim=sensory_dim, embedding_dim=embed_dim,
                                  action_dim=2).to(dev)
            print(f"  w={w:>4} seed={s} ...", end=" ", flush=True)
            final_dist = train_with_supervision(
                matter, organism, sensory_dim, target,
                supervision_weight=w, n_episodes=200, batch_size=128, device=device)
            sa = compute_sa(matter, organism, sensory_dim, device=device)
            results.append({
                "supervision_weight": w, "seed": s,
                "sensory_dim": sensory_dim, "embedding_dim": embed_dim,
                "SA": sa, "final_dist": final_dist,
            })
            print(f"SA={sa:+.3f} dist={final_dist:.3f}")
    elapsed = time.time() - t0
    out = Path("results/obj035_sparse_supervision.json")
    out.write_text(json.dumps({"results": results, "elapsed_s": elapsed}, indent=2))
    print(f"\nDone in {elapsed/60:.1f} min. Saved {out}")

    # Summary
    from collections import defaultdict
    by_w = defaultdict(list)
    for r in results:
        by_w[r["supervision_weight"]].append(r["SA"])
    print(f"\n{'weight':>8} {'mean_SA':>10} {'std_SA':>10}")
    for w in weights:
        arr = np.array(by_w[w])
        print(f"{w:>8} {arr.mean():>+10.3f} {arr.std():>10.3f}")


if __name__ == "__main__":
    main()
