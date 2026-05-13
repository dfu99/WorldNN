"""obj-034: Outcome Alignment (OA) instrumentation.

Answers the PI's earlier "intent vs real" question with a measured
quantity rather than just a conceptual response.

OA = E[ cos(action_xy, Δrock_xy) | contact ]

Computed on a slim training run (sensory ∈ {2, 4, 8, 16} × embed=16 ×
3 seeds, 200 ep each — small enough to run CPU-only in minutes) so we
can compare OA with SA per-config and see whether OA is redundant
with SA or carries complementary signal.
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


def train_and_collect(
    matter, organism, sensory_dim, target,
    n_episodes=200, steps_per_episode=20, batch_size=128,
    lr=3e-4, gamma=0.99, entropy_coef=0.01,
    action_std_init=0.8, action_std_final=0.2, clip_eps=0.2,
    ppo_epochs=4, contact_threshold=0.3, device="cpu",
):
    dev = torch.device(device)
    seed_dim = matter.seed_dim
    log_std = nn.Parameter(torch.full((2,), math.log(action_std_init), device=dev))
    optimizer = torch.optim.Adam(list(organism.parameters()) + [log_std], lr=lr)

    contact_action = []
    contact_delta = []
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
                action = torch.zeros(batch_size, 2, device=dev)
            next_state, emission, contact = matter(state, seed, action)
            obs = emission[:, :sensory_dim]

            action_mean, _, value = organism(obs)
            std = torch.full_like(action_mean, current_std)
            dist = torch.distributions.Normal(action_mean, std)
            action_sample = dist.sample()
            lp = dist.log_prob(action_sample).sum(dim=-1)

            # === OA instrumentation (last 50 episodes only, to capture
            #    post-training behavior, and only on contact steps) ===
            if ep >= n_episodes - 50:
                contact_mask = contact > contact_threshold
                if contact_mask.any():
                    rock_t = state[:, :2]
                    rock_t1 = next_state[:, :2]
                    delta_rock = rock_t1 - rock_t
                    contact_action.append(action_sample[contact_mask].detach().cpu())
                    contact_delta.append(delta_rock[contact_mask].detach().cpu())

            rock_pos = next_state[:, :2]
            org_pos = next_state[:, 2:4]
            rock_dist = torch.norm(rock_pos - target, dim=-1)
            org_rock_dist = torch.norm(rock_pos - org_pos, dim=-1)
            reward = (1.0 - rock_dist) + 0.2 * (1.0 - org_rock_dist) + 0.1 * contact

            all_obs.append(obs.detach()); all_actions.append(action_sample.detach())
            all_log_probs.append(lp.detach()); all_rewards.append(reward.detach())
            all_values.append(value.detach())
            state = next_state.detach(); action = action_sample.detach()

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

        final_dist = torch.norm(state[:, :2] - target, dim=-1).mean().item()

    if contact_action:
        ca = torch.cat(contact_action, dim=0)
        cd = torch.cat(contact_delta, dim=0)
    else:
        ca = torch.zeros(0, 2)
        cd = torch.zeros(0, 2)
    return ca.numpy(), cd.numpy(), final_dist


def compute_sa(matter, organism, sensory_dim, target, n=2000, device="cpu"):
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
            a_opt = (rock - org)
            a_opt = a_opt / (a_opt.norm(dim=-1, keepdim=True) + 1e-8)
            obs = em[:, :sensory_dim]
            a_l, _, _ = organism(obs)
            a_l = a_l / (a_l.norm(dim=-1, keepdim=True) + 1e-8)
            cos_sims.append((a_l * a_opt).sum(dim=-1).cpu())
    return torch.cat(cos_sims)[:n].mean().item()


def compute_oa(action, delta_rock):
    """OA = mean cos(action_xy, Δrock_xy) on contact steps."""
    if len(action) == 0:
        return float("nan"), 0
    a = action / (np.linalg.norm(action, axis=-1, keepdims=True) + 1e-8)
    d = delta_rock / (np.linalg.norm(delta_rock, axis=-1, keepdims=True) + 1e-8)
    cos = (a * d).sum(axis=-1)
    return float(cos.mean()), int(len(cos))


def main():
    device = os.environ.get("WORLDNN_DEVICE", "cpu")
    print(f"Device: {device}")

    sensory_dims = [2, 4, 8, 16]
    embed_dim = 16
    seeds = [42, 123, 456]

    dev = torch.device(device)
    torch.manual_seed(0)
    matter = RockPushMatter(emission_dim=16, action_dim=2, seed_dim=4).to(dev)
    target = torch.tensor([0.8, 0.8], device=dev)

    results = []
    t0 = time.time()
    for sd in sensory_dims:
        for s in seeds:
            torch.manual_seed(s)
            organism = Organism(sensory_dim=sd, embedding_dim=embed_dim,
                                  action_dim=2).to(dev)
            print(f"  sensory={sd} seed={s} ...", end=" ", flush=True)
            ca, cd, final_dist = train_and_collect(
                matter, organism, sd, target,
                n_episodes=200, batch_size=128, device=device)
            oa, n_contact = compute_oa(ca, cd)
            sa = compute_sa(matter, organism, sd, target, device=device)
            results.append({
                "sensory_dim": sd, "embedding_dim": embed_dim, "seed": s,
                "SA": sa, "OA": oa, "n_contact_samples": n_contact,
                "final_dist": final_dist,
            })
            print(f"SA={sa:+.3f} OA={oa:+.3f} dist={final_dist:.3f} n_contact={n_contact}")

    elapsed = time.time() - t0
    out = Path("results/obj034_outcome_alignment.json")
    out.write_text(json.dumps({
        "results": results,
        "elapsed_s": elapsed,
    }, indent=2))
    print(f"\nDone in {elapsed/60:.1f} min. Saved {out}")

    # Quick analysis
    import collections
    by_sensory = collections.defaultdict(list)
    for r in results:
        by_sensory[r["sensory_dim"]].append(r)
    print("\nBy sensory_dim:")
    print(f"{'sensory':>8} {'mean_SA':>10} {'mean_OA':>10} {'mean_dist':>10}")
    for sd in sensory_dims:
        rs = by_sensory[sd]
        sa_m = np.mean([r["SA"] for r in rs])
        oa_vals = [r["OA"] for r in rs if not np.isnan(r["OA"])]
        oa_m = np.mean(oa_vals) if oa_vals else float("nan")
        d_m = np.mean([r["final_dist"] for r in rs])
        print(f"{sd:>8} {sa_m:>+10.3f} {oa_m:>+10.3f} {d_m:>10.3f}")

    # Correlation SA vs OA across all configs
    all_sa = np.array([r["SA"] for r in results])
    all_oa = np.array([r["OA"] for r in results])
    all_d = np.array([r["final_dist"] for r in results])
    valid = ~np.isnan(all_oa)
    if valid.sum() >= 4:
        r_sa_oa = np.corrcoef(all_sa[valid], all_oa[valid])[0, 1]
        r_oa_d = np.corrcoef(all_oa[valid], all_d[valid])[0, 1]
        r_sa_d = np.corrcoef(all_sa[valid], all_d[valid])[0, 1]
        print(f"\nCorrelations across {valid.sum()} valid configs:")
        print(f"  r(SA, OA)   = {r_sa_oa:+.3f}")
        print(f"  r(SA, dist) = {r_sa_d:+.3f}")
        print(f"  r(OA, dist) = {r_oa_d:+.3f}")


if __name__ == "__main__":
    main()
