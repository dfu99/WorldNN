"""obj-036: 1D positioning sensory-capacity sweep (CPU-fast variant).

Addresses Reviewer E's single-environment disposition without retraining
the obj-024 grid. Uses ContinuousMatter (1D positioning, not contact-
based manipulation) so the task family is qualitatively different from
rock-push. If the SA pattern holds, the information-theoretic-ceiling
claim generalizes; if it floors, we have a clean scope statement.

Slim grid for CPU/cron-tick scale: 3 sensory × 5 embed × 3 seeds × 150
episodes batch=128 = 45 configs, ~10-15 min.
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
from worldnn.matter import ContinuousMatter
from worldnn.organism import Organism


def train_1d(
    matter, organism, sensory_dim, target_pos,
    n_episodes=150, steps_per_episode=20, batch_size=128,
    lr=3e-4, gamma=0.99, entropy_coef=0.01,
    action_std_init=0.5, action_std_final=0.1, clip_eps=0.2,
    ppo_epochs=4, device="cpu",
):
    dev = torch.device(device)
    seed_dim = matter.seed_dim
    log_std = nn.Parameter(torch.full((1,), math.log(action_std_init), device=dev))
    optimizer = torch.optim.Adam(list(organism.parameters()) + [log_std], lr=lr)
    final_dist = None

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

            action_full, _, value = organism(obs)
            action_mean = action_full[:, :1]
            std = torch.full_like(action_mean, current_std)
            dist = torch.distributions.Normal(action_mean, std)
            action_sample = dist.sample()
            lp = dist.log_prob(action_sample).sum(dim=-1)
            pos_dist = torch.abs(next_state - target_pos)
            reward = 1.0 - pos_dist

            all_obs.append(obs.detach())
            all_actions.append(action_sample.detach())
            all_log_probs.append(lp.detach())
            all_rewards.append(reward.detach())
            all_values.append(value.detach())
            state = next_state.detach()
            action = action_sample.detach()

        T = len(all_rewards)
        returns_list = []
        G = torch.zeros(batch_size, device=dev)
        for t_idx in reversed(range(T)):
            G = all_rewards[t_idx] + gamma * G
            returns_list.insert(0, G)
        returns = torch.stack(returns_list)
        values = torch.stack(all_values)
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        obs_batch = torch.stack(all_obs)
        act_batch = torch.stack(all_actions)
        old_lp = torch.stack(all_log_probs)

        for _ in range(ppo_epochs):
            for t_idx in range(T):
                action_full, _, value = organism(obs_batch[t_idx])
                action_mean = action_full[:, :1]
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
        final_dist = torch.abs(state - target_pos).mean().item()
    return final_dist


def compute_sa_1d(matter, organism, sensory_dim, target_pos, n=2000, device="cpu"):
    """SA = E[cos(action_1d, sign(target - state))]."""
    dev = torch.device(device)
    organism.eval()
    cos_sims = []
    with torch.no_grad():
        batch = 256
        for _ in range(n // batch + 1):
            state = matter.reset_state(batch, dev)
            seed = torch.randn(batch, matter.seed_dim, device=dev)
            action = torch.randn(batch, 1, device=dev) * 0.1
            ns, em, _ = matter(state, seed, action)
            # Optimal: move toward target. action is scalar; sign(target-ns)
            a_opt_sign = torch.sign(target_pos - ns)
            obs = em[:, :sensory_dim]
            a_l_full, _, _ = organism(obs)
            a_l = a_l_full[:, 0]  # 1D
            a_l_sign = torch.sign(a_l)
            # SA in 1D = cos = sign agreement (1 or -1)
            cos = a_l_sign * a_opt_sign  # ∈ {-1, 0, 1}
            cos_sims.append(cos.cpu())
    return torch.cat(cos_sims)[:n].mean().item()


def main():
    device = os.environ.get("WORLDNN_DEVICE", "cpu")
    print(f"Device: {device}")
    sensory_dims = [1, 2, 4]
    embed_dims = [2, 4, 8, 16, 32]
    seeds = [42, 123, 456]
    dev = torch.device(device)
    torch.manual_seed(0)
    matter = ContinuousMatter(emission_dim=4, action_dim=1, seed_dim=4).to(dev)
    target_pos = torch.tensor(0.8, device=dev)

    results = []
    t0 = time.time()
    total = len(sensory_dims) * len(embed_dims) * len(seeds)
    i = 0
    for sd in sensory_dims:
        for ed in embed_dims:
            for s in seeds:
                i += 1
                torch.manual_seed(s)
                organism = Organism(sensory_dim=sd, embedding_dim=ed,
                                      action_dim=2).to(dev)
                print(f"  [{i:>2}/{total}] sensory={sd} embed={ed:>2} seed={s} ...",
                      end=" ", flush=True)
                final_dist = train_1d(matter, organism, sd, target_pos,
                                       n_episodes=150, batch_size=128, device=device)
                sa = compute_sa_1d(matter, organism, sd, target_pos, device=device)
                results.append({
                    "sensory_dim": sd, "embedding_dim": ed, "seed": s,
                    "SA": sa, "final_dist": final_dist,
                })
                print(f"SA={sa:+.3f} dist={final_dist:.3f}")
    elapsed = time.time() - t0
    out = Path("results/obj036_1d_sensory_capacity.json")
    out.write_text(json.dumps({"results": results, "elapsed_s": elapsed}, indent=2))
    print(f"\nDone in {elapsed/60:.1f} min. Saved {out}")

    # Summary
    from collections import defaultdict
    by_cell = defaultdict(list)
    for r in results:
        by_cell[(r["sensory_dim"], r["embedding_dim"])].append(r["SA"])
    print(f"\nMean SA by (sensory, embed):")
    print(f"{'sensory':>8}" + "".join(f"  emb={e:<4}" for e in embed_dims))
    for sd in sensory_dims:
        row = f"{sd:>8}"
        for ed in embed_dims:
            arr = np.array(by_cell.get((sd, ed), []))
            row += f"  {arr.mean():>+6.3f}"
        print(row)


if __name__ == "__main__":
    main()
