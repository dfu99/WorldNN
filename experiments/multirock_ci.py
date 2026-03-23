"""obj-017: Multi-rock C_i sweep — second task for ICLR validation.

3-rock push (8D state, 2D action, 16D emission). Tests whether C_i
threshold and perception×capacity interaction generalize to higher-
dimensional tasks.

Grid:
  perception: [oracle, oracle_noise0.1, raw_emission, vae_mu_lat16, vae_mu_lat32]
  embedding_dim: [4, 16, 64]
  seeds: [42, 123, 456, 789, 1337, 2024, 3141]
  = 5 × 3 × 7 = 105 configs + 21 random baselines = 126 total
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
from worldnn.matter import MultiRockMatter
from worldnn.world import RockPushWorld
from worldnn.organism import Organism
from worldnn.channels import Channel
from worldnn.environment import EnvironmentVAE
from worldnn.train import train_environment_rockpush


def compute_optimal_action_multirock(state, targets, n_rocks=3):
    """Optimal action: move toward the rock that is farthest from its target,
    then push it toward target. Greedy sequential strategy."""
    rock_positions = state[:, :n_rocks*2].reshape(-1, n_rocks, 2)  # [batch, n, 2]
    org_pos = state[:, n_rocks*2:n_rocks*2+2]  # [batch, 2]

    # Find which rock is farthest from its target
    rock_dists = torch.zeros(state.shape[0], n_rocks, device=state.device)
    for i in range(n_rocks):
        rock_dists[:, i] = torch.norm(rock_positions[:, i] - targets[i].unsqueeze(0), dim=-1)

    # Target the farthest rock
    worst_rock_idx = rock_dists.argmax(dim=-1)  # [batch]

    # Get position of the worst rock for each batch element
    worst_rock_pos = torch.zeros_like(org_pos)
    worst_target = torch.zeros_like(org_pos)
    for i in range(n_rocks):
        mask = (worst_rock_idx == i).unsqueeze(-1).float()
        worst_rock_pos += mask * rock_positions[:, i]
        worst_target += mask * targets[i].unsqueeze(0)

    # Move toward the worst rock
    to_rock = worst_rock_pos - org_pos
    rock_dist = torch.norm(to_rock, dim=-1, keepdim=True).clamp(min=1e-6)

    to_target = worst_target - worst_rock_pos
    target_dist = torch.norm(to_target, dim=-1, keepdim=True).clamp(min=1e-6)

    contact_threshold = 0.15
    near_rock = (rock_dist < contact_threshold).float()
    optimal = (1 - near_rock) * (to_rock / rock_dist) + near_rock * (to_target / target_dist)
    opt_norm = torch.norm(optimal, dim=-1, keepdim=True).clamp(min=1e-6)
    return optimal / opt_norm


def measure_ci_multirock(organism, perception_fn, matter, targets,
                          n_samples=5000, device="cuda"):
    """Measure C_i for multi-rock task."""
    dev = torch.device(device)
    organism.eval()
    all_cos = []
    with torch.no_grad():
        for _ in range(n_samples // 256 + 1):
            state = matter.reset_state(256, dev)
            seed = torch.randn(256, matter.seed_dim, device=dev)
            action = torch.randn(256, 2, device=dev) * 0.1
            next_state, emission, _ = matter(state, seed, action)
            obs = perception_fn(next_state, emission)
            action_mean, _, _ = organism(obs)
            optimal = compute_optimal_action_multirock(next_state, targets)
            cos_sim = F.cosine_similarity(action_mean, optimal, dim=-1)
            all_cos.append(cos_sim)
    cos_all = torch.cat(all_cos)[:n_samples]
    return {
        "C_i": cos_all.mean().item(),
        "C_i_std": cos_all.std().item(),
        "C_i_positive_frac": (cos_all > 0).float().mean().item(),
    }


def train_multirock_ppo(
    matter, organism, perception_fn, targets,
    n_episodes=500, steps_per_episode=20, batch_size=256,
    lr=3e-4, gamma=0.99, entropy_coef=0.01,
    action_std_init=0.8, clip_eps=0.2, ppo_epochs=4, device="cuda",
):
    """Train organism for multi-rock task with custom perception."""
    dev = torch.device(device)
    n_rocks = matter.n_rocks

    log_std = nn.Parameter(torch.full((2,), math.log(action_std_init), device=dev))
    optimizer = torch.optim.Adam(list(organism.parameters()) + [log_std], lr=lr)

    metrics = {"rewards": [], "mean_rock_dist": [], "contact_rate": []}

    for ep in range(n_episodes):
        organism.train()
        state = matter.reset_state(batch_size, dev)
        action = None

        all_obs, all_actions, all_lp, all_rewards, all_values = [], [], [], [], []
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

            # Reward: sum of (1 - dist_to_target) for each rock + contact bonus
            rock_positions = next_state[:, :n_rocks*2].reshape(-1, n_rocks, 2)
            reward = torch.zeros(batch_size, device=dev)
            for i in range(n_rocks):
                rd = torch.norm(rock_positions[:, i] - targets[i].unsqueeze(0), dim=-1)
                reward += (1.0 - rd) / n_rocks
            org_pos = next_state[:, n_rocks*2:n_rocks*2+2]
            # Approach bonus: get close to any rock
            min_org_rock = torch.stack([
                torch.norm(rock_positions[:, i] - org_pos, dim=-1)
                for i in range(n_rocks)
            ], dim=-1).min(dim=-1).values
            reward += 0.2 * (1.0 - min_org_rock)
            reward += 0.1 * contact

            all_obs.append(obs.detach())
            all_actions.append(action_sample.detach())
            all_lp.append(lp.detach())
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

        obs_batch = torch.stack(all_obs)
        act_batch = torch.stack(all_actions)
        old_lp = torch.stack(all_lp)

        for _ in range(ppo_epochs):
            for t_idx in range(T):
                am, _, v = organism(obs_batch[t_idx])
                std = log_std.exp().unsqueeze(0).expand_as(am)
                d = torch.distributions.Normal(am, std)
                new_lp = d.log_prob(act_batch[t_idx]).sum(dim=-1)
                ratio = (new_lp - old_lp[t_idx]).exp()
                clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
                ploss = -torch.min(ratio * advantages[t_idx], clipped * advantages[t_idx]).mean()
                vloss = F.mse_loss(v, returns[t_idx])
                ent = d.entropy().sum(dim=-1).mean()
                loss = ploss + 0.5 * vloss - entropy_coef * ent
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(organism.parameters()) + [log_std], 1.0)
                optimizer.step()

        with torch.no_grad():
            rp = state[:, :n_rocks*2].reshape(-1, n_rocks, 2)
            mean_rd = sum(
                torch.norm(rp[:, i] - targets[i].unsqueeze(0), dim=-1).mean().item()
                for i in range(n_rocks)
            ) / n_rocks
        metrics["rewards"].append(sum(r.mean().item() for r in all_rewards) / T)
        metrics["mean_rock_dist"].append(mean_rd)
        metrics["contact_rate"].append(contact_sum / T)

    return metrics


def run_config(level_name, matter, sensory_dim, embed_dim, seed,
               perception_fn, targets, device="cuda"):
    torch.manual_seed(seed)
    dev = torch.device(device)
    organism = Organism(sensory_dim=sensory_dim, embedding_dim=embed_dim, action_dim=2).to(dev)

    t0 = time.time()
    metrics = train_multirock_ppo(
        matter, organism, perception_fn, targets,
        n_episodes=500, device=device,
    )
    elapsed = time.time() - t0

    ci = measure_ci_multirock(organism, perception_fn, matter, targets, device=device)

    n_tail = min(100, len(metrics["mean_rock_dist"]))
    avg_dist = sum(metrics["mean_rock_dist"][-n_tail:]) / n_tail

    return {
        "level": level_name,
        "embedding_dim": embed_dim,
        "seed": seed,
        "avg_dist_last100": avg_dist,
        "elapsed_s": elapsed,
        **ci,
    }


def run_baseline(matter, sensory_dim, embed_dim, seed, perception_fn, targets, device="cuda"):
    torch.manual_seed(seed)
    dev = torch.device(device)
    organism = Organism(sensory_dim=sensory_dim, embedding_dim=embed_dim, action_dim=2).to(dev)
    n_rocks = matter.n_rocks

    ci = measure_ci_multirock(organism, perception_fn, matter, targets, device=device)

    # Random policy distance
    dists = []
    with torch.no_grad():
        for _ in range(20):
            state = matter.reset_state(256, dev)
            for t in range(20):
                sd = torch.randn(256, matter.seed_dim, device=dev)
                a = torch.randn(256, 2, device=dev) * 0.3
                state, _, _ = matter(state, sd, a)
            rp = state[:, :n_rocks*2].reshape(-1, n_rocks, 2)
            md = sum(
                torch.norm(rp[:, i] - targets[i].unsqueeze(0), dim=-1).mean().item()
                for i in range(n_rocks)
            ) / n_rocks
            dists.append(md)

    return {
        "level": "random_baseline",
        "embedding_dim": embed_dim,
        "seed": seed,
        "avg_dist_last100": sum(dists) / len(dists),
        "elapsed_s": 0,
        **ci,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    checkpoint_path = results_dir / "multirock_ci_checkpoint.json"
    results_path = results_dir / "multirock_ci.json"

    embed_dims = [4, 16, 64]
    seeds = [42, 123, 456, 789, 1337, 2024, 3141]
    dev = torch.device(device)

    # Targets for the 3 rocks
    targets = [
        torch.tensor([0.2, 0.8], device=dev),
        torch.tensor([0.8, 0.8], device=dev),
        torch.tensor([0.5, 0.2], device=dev),
    ]

    torch.manual_seed(0)
    matter = MultiRockMatter(
        emission_dim=16, action_dim=2, seed_dim=4,
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

    levels = []

    # Random baseline
    def oracle_fn(state, emission):
        return state
    for emb in embed_dims:
        for s in seeds:
            levels.append(("random_baseline", 8, emb, s, oracle_fn, True))

    # Oracle
    for emb in embed_dims:
        for s in seeds:
            levels.append(("oracle", 8, emb, s, oracle_fn, False))

    # Oracle + noise
    def make_noisy(std):
        def fn(state, emission):
            return state + torch.randn_like(state) * std
        return fn
    for emb in embed_dims:
        for s in seeds:
            levels.append(("oracle_noise0.1", 8, emb, s, make_noisy(0.1), False))

    # Raw emission (16D)
    def emission_fn(state, emission):
        return emission
    for emb in embed_dims:
        for s in seeds:
            levels.append(("raw_emission", 16, emb, s, emission_fn, False))

    # VAE mu lat=16 and lat=32
    vae_probes = {}
    for lat_dim in [16, 32]:
        torch.manual_seed(0)
        # Build a temporary world just for the VAE
        channel = Channel(input_dim=16, output_dim=16, noise_std=0.01).to(dev)
        env_vae = EnvironmentVAE(channel_dim=16, latent_dim=lat_dim, hidden_size=64, action_dim=2).to(dev)

        print(f"Pre-training VAE (lat={lat_dim}, 16D input)...", end=" ", flush=True)
        optimizer = torch.optim.Adam(env_vae.parameters(), lr=1e-3)
        for step in range(2000):
            state = matter.reset_state(256, dev)
            seed_t = torch.randn(256, matter.seed_dim, device=dev)
            action_t = torch.randn(256, 2, device=dev) * 0.1
            with torch.no_grad():
                _, emission, _ = matter(state, seed_t, action_t)
                ch_out = channel(emission)
            z, y_hat, mu, logvar = env_vae(ch_out)
            loss = env_vae.vae_loss(ch_out, y_hat, mu, logvar, beta=0.1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"loss={loss.item():.4f}")

        ch_frozen = channel
        env_frozen = env_vae
        def make_vae_fn(env_m, ch_m):
            def fn(state, emission):
                with torch.no_grad():
                    co = ch_m(emission)
                    mu, _ = env_m.encode(co)
                return mu
            return fn
        vae_fn = make_vae_fn(env_frozen, ch_frozen)
        for emb in embed_dims:
            for s in seeds:
                levels.append((f"vae_mu_lat{lat_dim}", lat_dim, emb, s, vae_fn, False))

    total = len(levels)
    print(f"\nobj-017 multi-rock sweep: {total} configs")
    t0 = time.time()

    for level_name, sensory_dim, emb, s, perc_fn, is_baseline in levels:
        key = (level_name, emb, s)
        if key in completed_keys:
            continue

        idx = len(completed) + 1
        print(f"[{idx}/{total}] {level_name}, emb={emb}, seed={s}", end=" ... ", flush=True)

        try:
            if is_baseline:
                result = run_baseline(matter, sensory_dim, emb, s, perc_fn, targets, device=device)
            else:
                result = run_config(level_name, matter, sensory_dim, emb, s, perc_fn, targets, device=device)
            completed.append(result)
            completed_keys.add(key)
            print(f"dist={result['avg_dist_last100']:.3f}, C_i={result['C_i']:.3f} ({result['elapsed_s']:.0f}s)")
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback; traceback.print_exc()
            completed.append({"level": level_name, "embedding_dim": emb, "seed": s, "error": str(e)})
            completed_keys.add(key)

        if len(completed) % 7 == 0:
            with open(checkpoint_path, "w") as f:
                json.dump(completed, f, indent=2)
            elapsed = time.time() - t0
            remaining = total - len(completed_keys)
            done = len(completed_keys)
            if done > 0 and elapsed > 0:
                print(f"  [checkpoint] {done}/{total}, ETA: {remaining/(done/elapsed)/60:.0f} min")

    elapsed = time.time() - t0
    with open(results_path, "w") as f:
        json.dump({"results": completed, "elapsed_seconds": elapsed}, f, indent=2)
    with open(checkpoint_path, "w") as f:
        json.dump(completed, f, indent=2)

    valid = [r for r in completed if "error" not in r]
    trained = [r for r in valid if r["level"] != "random_baseline"]
    baselines = [r for r in valid if r["level"] == "random_baseline"]
    print(f"\nDone: {len(valid)}/{total} in {elapsed/60:.1f} min")

    # Summary
    import numpy as np
    if baselines:
        bl_d = np.mean([r["avg_dist_last100"] for r in baselines])
        print(f"Random baseline: dist={bl_d:.3f}")
    if trained:
        dists = [r["avg_dist_last100"] for r in trained]
        cis = [r["C_i"] for r in trained]
        r_val = np.corrcoef(dists, cis)[0, 1]
        print(f"Overall r = {r_val:.3f} (n={len(trained)})")

        by_level = defaultdict(list)
        for r in trained:
            by_level[r["level"]].append(r)
        for lev in sorted(by_level):
            rs = by_level[lev]
            print(f"  {lev:20s}: dist={np.mean([r['avg_dist_last100'] for r in rs]):.3f}, "
                  f"C_i={np.mean([r['C_i'] for r in rs]):.3f}")

    # Interaction test
    if len(trained) >= 20:
        from math import erfc, sqrt
        perc_q = {"oracle": 4, "oracle_noise0.1": 3, "raw_emission": 2, "vae_mu_lat32": 1, "vae_mu_lat16": 0}
        Y = np.array([r["avg_dist_last100"] for r in trained])
        Xp = np.array([perc_q.get(r["level"], 0) for r in trained], dtype=float)
        Xc = np.array([np.log2(r["embedding_dim"]) for r in trained])
        Xi = Xp * Xc
        n = len(Y)
        X_full = np.column_stack([np.ones(n), Xp, Xc, Xi])
        X_red = np.column_stack([np.ones(n), Xp, Xc])
        beta_f = np.linalg.lstsq(X_full, Y, rcond=None)[0]
        ss_f = np.sum((Y - X_full @ beta_f)**2)
        beta_r = np.linalg.lstsq(X_red, Y, rcond=None)[0]
        ss_r = np.sum((Y - X_red @ beta_r)**2)
        df = n - 4
        F_stat = ((ss_r - ss_f) / 1) / (ss_f / df)
        p_val = erfc(sqrt(F_stat) / sqrt(2))
        print(f"\nInteraction test: F(1,{df})={F_stat:.1f}, p={p_val:.2e}")
        print(f"β_interaction = {beta_f[3]:.5f}")


if __name__ == "__main__":
    main()
