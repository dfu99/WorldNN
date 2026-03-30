"""obj-021: 2-rock fallback (6D state) — cleaner second task for ICLR.

3-rock (8D) showed r=-0.300 (marginal). 2-rock (6D) is simpler and
may produce a cleaner SA pattern while still being a meaningful step
up from 4D rock-push.

State: [r1x, r1y, r2x, r2y, ox, oy] = 6D
Emission: 12D (fixed projection)
Action: 2D

Grid: oracle + raw_emission + vae_mu_lat16 × embed=[4,16,64] × 5 seeds
    = 3 × 3 × 5 = 45 configs + 15 baselines = 60 total
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
from worldnn.organism import Organism
from worldnn.channels import Channel
from worldnn.environment import EnvironmentVAE
from worldnn.train import train_environment_rockpush
from coordination_quality import measure_coordination_quality


def compute_optimal_action_2rock(state, targets):
    """Optimal: approach farthest-from-target rock, push it."""
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


def measure_ci_2rock(organism, perception_fn, matter, targets, n_samples=2000, device="cuda"):
    dev = torch.device(device)
    organism.eval()
    all_cos = []
    with torch.no_grad():
        for _ in range(n_samples // 256 + 1):
            state = matter.reset_state(256, dev)
            seed = torch.randn(256, matter.seed_dim, device=dev)
            action = torch.randn(256, 2, device=dev) * 0.1
            ns, emission, _ = matter(state, seed, action)
            obs = perception_fn(ns, emission)
            am, _, _ = organism(obs)
            opt = compute_optimal_action_2rock(ns, targets)
            all_cos.append(F.cosine_similarity(am, opt, dim=-1))
    return {"C_i": torch.cat(all_cos)[:n_samples].mean().item()}


def train_2rock_ppo(matter, organism, perception_fn, targets,
                     n_episodes=1000, steps_per_episode=20, batch_size=256,
                     lr=3e-4, gamma=0.99, entropy_coef=0.01,
                     action_std_init=0.8, clip_eps=0.2, ppo_epochs=4, device="cuda"):
    dev = torch.device(device)
    log_std = nn.Parameter(torch.full((2,), math.log(action_std_init), device=dev))
    opt = torch.optim.Adam(list(organism.parameters()) + [log_std], lr=lr)
    metrics = {"rewards": [], "mean_rock_dist": []}

    for ep in range(n_episodes):
        organism.train()
        state = matter.reset_state(batch_size, dev)
        action = torch.zeros(batch_size, 2, device=dev)
        all_o, all_a, all_lp, all_r, all_v = [], [], [], [], []

        for t in range(steps_per_episode):
            seed = torch.randn(batch_size, matter.seed_dim, device=dev)
            ns, emission, contact = matter(state, seed, action)
            with torch.no_grad():
                obs = perception_fn(ns, emission)
            am, emb, val = organism(obs)
            std = log_std.exp().unsqueeze(0).expand_as(am)
            dist = torch.distributions.Normal(am, std)
            a_s = dist.sample()
            lp = dist.log_prob(a_s).sum(dim=-1)

            rp = ns[:, :4].reshape(-1, 2, 2)
            rock_dists = torch.stack([
                torch.norm(rp[:, i] - targets[i].unsqueeze(0), dim=-1) for i in range(2)
            ], dim=-1)
            rew = (1.0 - rock_dists.mean(dim=-1))
            rew += 0.3 * (1.0 - rock_dists.max(dim=-1).values)
            org_pos = ns[:, 4:6]
            worst_idx = rock_dists.argmax(dim=-1)
            worst_rock = torch.zeros_like(org_pos)
            for i in range(2):
                mask = (worst_idx == i).unsqueeze(-1).float()
                worst_rock += mask * rp[:, i]
            rew += 0.4 * (1.0 - torch.norm(worst_rock - org_pos, dim=-1))
            rew += 0.15 * contact

            all_o.append(obs.detach()); all_a.append(a_s.detach())
            all_lp.append(lp.detach()); all_r.append(rew.detach()); all_v.append(val.detach())
            state = ns.detach(); action = a_s.detach()

        T = len(all_r); rets = []; G = torch.zeros(batch_size, device=dev)
        for ti in reversed(range(T)):
            G = all_r[ti] + gamma * G; rets.insert(0, G)
        rets = torch.stack(rets); vals = torch.stack(all_v)
        adv = rets - vals; adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ob = torch.stack(all_o); ab = torch.stack(all_a); olp = torch.stack(all_lp)

        for _ in range(ppo_epochs):
            for ti in range(T):
                am2, _, v2 = organism(ob[ti])
                std2 = log_std.exp().unsqueeze(0).expand_as(am2)
                d2 = torch.distributions.Normal(am2, std2)
                nlp = d2.log_prob(ab[ti]).sum(dim=-1)
                r = (nlp - olp[ti]).exp(); cl = torch.clamp(r, 1-clip_eps, 1+clip_eps)
                pl = -torch.min(r * adv[ti], cl * adv[ti]).mean()
                vl = F.mse_loss(v2, rets[ti]); ent = d2.entropy().sum(dim=-1).mean()
                loss = pl + 0.5 * vl - entropy_coef * ent
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(list(organism.parameters()) + [log_std], 1.0)
                opt.step()

        with torch.no_grad():
            rp = state[:, :4].reshape(-1, 2, 2)
            md = sum(torch.norm(rp[:, i] - targets[i].unsqueeze(0), dim=-1).mean().item() for i in range(2)) / 2
        metrics["mean_rock_dist"].append(md)
        metrics["rewards"].append(sum(r.mean().item() for r in all_r) / T)

    return metrics


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    results_path = results_dir / "tworock_ci.json"
    checkpoint_path = results_dir / "tworock_ci_checkpoint.json"

    embed_dims = [4, 16, 64]
    seeds = [42, 123, 456, 789, 1337]
    dev = torch.device(device)
    targets = [torch.tensor([0.2, 0.8], device=dev), torch.tensor([0.8, 0.2], device=dev)]

    torch.manual_seed(0)
    matter = MultiRockMatter(emission_dim=12, action_dim=2, seed_dim=4, n_rocks=2,
                              move_speed=0.20, push_strength=0.15).to(dev)

    completed = []
    completed_keys = set()
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            completed = json.load(f)
        for r in completed:
            completed_keys.add((r["level"], r["embedding_dim"], r["seed"]))
        print(f"Resuming: {len(completed)} done")

    levels = []

    # Baselines
    def oracle_fn(state, emission): return state
    for emb in embed_dims:
        for s in seeds:
            levels.append(("baseline", 6, emb, s, oracle_fn, True))

    # Oracle
    for emb in embed_dims:
        for s in seeds:
            levels.append(("oracle", 6, emb, s, oracle_fn, False))

    # Raw emission (12D)
    def emission_fn(state, emission): return emission
    for emb in embed_dims:
        for s in seeds:
            levels.append(("raw_emission", 12, emb, s, emission_fn, False))

    # VAE mu lat=16
    torch.manual_seed(0)
    channel = Channel(input_dim=12, output_dim=12, noise_std=0.01).to(dev)
    env_vae = EnvironmentVAE(channel_dim=12, latent_dim=16, hidden_size=64, action_dim=2).to(dev)
    vae_opt = torch.optim.Adam(env_vae.parameters(), lr=1e-3)
    print("Pre-training VAE (lat=16, 12D)...", end=" ", flush=True)
    for step in range(2000):
        state = matter.reset_state(256, dev)
        seed_t = torch.randn(256, matter.seed_dim, device=dev)
        action_t = torch.randn(256, 2, device=dev) * 0.1
        with torch.no_grad():
            _, emission, _ = matter(state, seed_t, action_t)
            ch_out = channel(emission)
        z, y_hat, mu, logvar = env_vae(ch_out)
        loss = env_vae.vae_loss(ch_out, y_hat, mu, logvar, beta=0.1)
        vae_opt.zero_grad(); loss.backward(); vae_opt.step()
    print(f"loss={loss.item():.4f}")

    def make_vae_fn(env_m, ch_m):
        def fn(state, emission):
            with torch.no_grad():
                co = ch_m(emission); mu, _ = env_m.encode(co)
            return mu
        return fn
    vae_fn = make_vae_fn(env_vae, channel)
    for emb in embed_dims:
        for s in seeds:
            levels.append(("vae_mu_lat16", 16, emb, s, vae_fn, False))

    total = len(levels)
    print(f"\nobj-021 2-rock sweep: {total} configs")
    t0 = time.time()

    for level_name, sensory_dim, emb, s, perc_fn, is_baseline in levels:
        key = (level_name, emb, s)
        if key in completed_keys:
            continue

        idx = len(completed) + 1
        print(f"[{idx}/{total}] {level_name}, emb={emb}, seed={s}", end=" ... ", flush=True)

        try:
            torch.manual_seed(s)
            organism = Organism(sensory_dim=sensory_dim, embedding_dim=emb,
                                action_dim=2, hidden_size=64).to(dev)

            if is_baseline:
                ci = measure_ci_2rock(organism, perc_fn, matter, targets, device=device)
                with torch.no_grad():
                    state = matter.reset_state(512, dev)
                    rp = state[:, :4].reshape(-1, 2, 2)
                    bl_dist = sum(torch.norm(rp[:, i] - targets[i].unsqueeze(0), dim=-1).mean().item() for i in range(2)) / 2
                result = {"level": level_name, "embedding_dim": emb, "seed": s,
                          "avg_dist_last100": bl_dist, **ci, "elapsed_s": 0}
            else:
                t1 = time.time()
                metrics = train_2rock_ppo(matter, organism, perc_fn, targets,
                                           n_episodes=1000, device=device)
                elapsed = time.time() - t1
                ci = measure_ci_2rock(organism, perc_fn, matter, targets, device=device)
                n_tail = min(100, len(metrics["mean_rock_dist"]))
                avg_dist = sum(metrics["mean_rock_dist"][-n_tail:]) / n_tail
                result = {"level": level_name, "embedding_dim": emb, "seed": s,
                          "avg_dist_last100": avg_dist, **ci, "elapsed_s": elapsed}

            completed.append(result)
            completed_keys.add(key)
            print(f"dist={result['avg_dist_last100']:.3f}, SA={result['C_i']:.3f} ({result['elapsed_s']:.0f}s)")
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback; traceback.print_exc()
            completed.append({"level": level_name, "embedding_dim": emb, "seed": s, "error": str(e)})
            completed_keys.add(key)

        if len(completed) % 5 == 0:
            with open(checkpoint_path, "w") as f:
                json.dump(completed, f, indent=2)

    elapsed = time.time() - t0
    with open(results_path, "w") as f:
        json.dump({"results": completed, "elapsed_seconds": elapsed}, f, indent=2)

    import numpy as np
    valid = [r for r in completed if "error" not in r]
    baselines = [r for r in valid if r["level"] == "baseline"]
    trained = [r for r in valid if r["level"] != "baseline"]
    bl_dist = np.mean([r["avg_dist_last100"] for r in baselines])
    print(f"\nDone: {len(valid)}/{total} in {elapsed/60:.1f} min")
    print(f"Baseline: dist={bl_dist:.4f}")

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
            print(f"  {lev:20s}: dist={np.mean([r['avg_dist_last100'] for r in rs]):.4f}, "
                  f"SA={np.mean([r['C_i'] for r in rs]):.3f}")

        # Interaction test
        from math import erfc, sqrt
        perc_q = {"oracle": 2, "raw_emission": 1, "vae_mu_lat16": 0}
        Y = np.array(dists); Xp = np.array([perc_q.get(r["level"], 0) for r in trained], dtype=float)
        Xc = np.array([np.log2(r["embedding_dim"]) for r in trained])
        Xi = Xp * Xc; n = len(Y)
        X_full = np.column_stack([np.ones(n), Xp, Xc, Xi])
        X_red = np.column_stack([np.ones(n), Xp, Xc])
        bf = np.linalg.lstsq(X_full, Y, rcond=None)[0]
        sf = np.sum((Y - X_full @ bf)**2)
        br = np.linalg.lstsq(X_red, Y, rcond=None)[0]
        sr = np.sum((Y - X_red @ br)**2)
        df = n - 4; F = ((sr - sf) / 1) / (sf / df)
        p = erfc(sqrt(max(F, 0)) / sqrt(2))
        print(f"\nInteraction: F(1,{df})={F:.1f}, p={p:.2e}, β_int={bf[3]:.5f}")


if __name__ == "__main__":
    main()
