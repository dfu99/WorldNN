"""obj-018 (design only): Self-supervised SA proxy candidates.

SA requires oracle access (optimal action a*). For practical relevance,
we need proxies that estimate SA without knowing a*. Three candidates:

Proxy A — Prediction Consistency:
  Train a simple forward model: given (obs, action) → predicted next obs.
  Measure how consistent the organism's actions are with its own world
  model. If SA is high, the organism acts purposefully → predictions are
  accurate. If SA is low, actions are random → predictions are poor.
  Proxy = -mean_prediction_error (lower error = higher proxy SA).

Proxy B — Action Stability Under Observation Noise:
  Perturb the observation with small noise and measure how much the
  action changes. A well-aligned organism has learned a stable mapping
  from perception to action — small input changes cause small output
  changes. A poorly aligned organism has a noisy, unstable mapping.
  Proxy = -mean(||a(obs) - a(obs + ε)||) (lower change = higher proxy SA).

Proxy C — Value-Action Alignment:
  The value function estimates future reward. If the organism is well-
  aligned, high-value states should correlate with actions that improve
  the situation. Measure the correlation between value gradient direction
  and action direction in observation space.
  Proxy = mean(cos(∇_obs V(obs), a(obs))) approximately.
  Practical version: cos(V(obs+δa) - V(obs-δa), a(obs)) using finite
  differences along the action direction.

Each proxy is computed alongside the true SA for validation.

DO NOT SUBMIT TO PACE — design only. Awaiting PI approval.
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
from worldnn.matter import RockPushMatter
from worldnn.organism import Organism
from coordination_quality import compute_optimal_action, measure_coordination_quality


class SimpleForwardModel(nn.Module):
    """Predicts next observation given current obs + action."""
    def __init__(self, obs_dim, action_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, obs_dim),
        )

    def forward(self, obs, action):
        return self.net(torch.cat([obs, action], dim=-1))


def proxy_a_prediction_consistency(organism, matter, perception_fn,
                                    n_samples=2000, device="cuda"):
    """Proxy A: Train forward model, measure prediction error.

    Lower prediction error → organism acts more purposefully → higher SA.
    """
    dev = torch.device(device)
    organism.eval()

    # Collect (obs, action, next_obs) transitions
    obs_list, act_list, next_obs_list = [], [], []
    with torch.no_grad():
        for _ in range(n_samples // 256 + 1):
            state = matter.reset_state(256, dev)
            seed = torch.randn(256, matter.seed_dim, device=dev)
            action_rand = torch.zeros(256, 2, device=dev)
            next_state, emission, _ = matter(state, seed, action_rand)
            obs = perception_fn(next_state, emission)

            action_mean, _, _ = organism(obs)

            # Take the organism's action and see what happens
            seed2 = torch.randn(256, matter.seed_dim, device=dev)
            next_state2, emission2, _ = matter(next_state, seed2, action_mean)
            next_obs = perception_fn(next_state2, emission2)

            obs_list.append(obs)
            act_list.append(action_mean)
            next_obs_list.append(next_obs)

    obs_all = torch.cat(obs_list)[:n_samples]
    act_all = torch.cat(act_list)[:n_samples]
    next_obs_all = torch.cat(next_obs_list)[:n_samples]

    # Train forward model
    obs_dim = obs_all.shape[-1]
    fwd = SimpleForwardModel(obs_dim, 2).to(dev)
    opt = torch.optim.Adam(fwd.parameters(), lr=1e-3)

    n_train = int(0.8 * len(obs_all))
    for epoch in range(200):
        idx = torch.randperm(n_train)[:256]
        pred = fwd(obs_all[idx], act_all[idx])
        loss = F.mse_loss(pred, next_obs_all[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()

    # Evaluate prediction error on held-out data
    fwd.eval()
    with torch.no_grad():
        pred_test = fwd(obs_all[n_train:], act_all[n_train:])
        test_error = F.mse_loss(pred_test, next_obs_all[n_train:]).item()

    return {"proxy_a_pred_error": test_error, "proxy_a_value": -test_error}


def proxy_b_action_stability(organism, matter, perception_fn,
                              noise_std=0.05, n_samples=2000, device="cuda"):
    """Proxy B: Action stability under observation noise.

    Perturb obs → measure action change. Stable = well-aligned.
    """
    dev = torch.device(device)
    organism.eval()

    all_diffs = []
    with torch.no_grad():
        for _ in range(n_samples // 256 + 1):
            state = matter.reset_state(256, dev)
            seed = torch.randn(256, matter.seed_dim, device=dev)
            action = torch.zeros(256, 2, device=dev)
            next_state, emission, _ = matter(state, seed, action)
            obs = perception_fn(next_state, emission)

            # Clean action
            action_clean, _, _ = organism(obs)

            # Perturbed action
            obs_noisy = obs + torch.randn_like(obs) * noise_std
            action_noisy, _, _ = organism(obs_noisy)

            diff = torch.norm(action_clean - action_noisy, dim=-1)
            all_diffs.append(diff)

    diffs = torch.cat(all_diffs)[:n_samples]
    mean_diff = diffs.mean().item()

    return {"proxy_b_action_diff": mean_diff, "proxy_b_value": -mean_diff}


def proxy_c_value_action_alignment(organism, matter, perception_fn,
                                    delta=0.01, n_samples=2000, device="cuda"):
    """Proxy C: Value-action alignment via finite differences.

    If actions improve the value function, they're purposeful.
    cos(V(obs + δ*a_dir) - V(obs - δ*a_dir), a) measures this.
    """
    dev = torch.device(device)
    organism.eval()

    all_cos = []
    with torch.no_grad():
        for _ in range(n_samples // 256 + 1):
            state = matter.reset_state(256, dev)
            seed = torch.randn(256, matter.seed_dim, device=dev)
            action = torch.zeros(256, 2, device=dev)
            next_state, emission, _ = matter(state, seed, action)
            obs = perception_fn(next_state, emission)

            action_mean, _, value_center = organism(obs)

            # Normalize action direction
            a_norm = action_mean / (torch.norm(action_mean, dim=-1, keepdim=True) + 1e-8)

            # Finite difference: perturb obs along action direction
            # (project action into obs space via first 2 dims as proxy)
            obs_dim = obs.shape[-1]
            perturbation = torch.zeros_like(obs)
            perturbation[:, :2] = a_norm * delta  # perturb spatial dims

            _, _, value_plus = organism(obs + perturbation)
            _, _, value_minus = organism(obs - perturbation)

            value_gradient = value_plus - value_minus  # scalar per sample

            # If value increases along action direction, alignment is positive
            alignment = value_gradient  # positive = action improves value
            all_cos.append(alignment)

    alignments = torch.cat(all_cos)[:n_samples]
    mean_alignment = alignments.mean().item()

    return {"proxy_c_value_align": mean_alignment, "proxy_c_value": mean_alignment}


def proxy_d_action_entropy(organism, matter, perception_fn,
                            n_samples=2000, device="cuda"):
    """Proxy D: Action dispersion (entropy-like).

    Measure the variance of actions across different states. A well-aligned
    organism produces structured, state-dependent actions → high variance
    (different actions for different states). A poorly aligned organism
    outputs similar random actions regardless of state → low variance.
    """
    dev = torch.device(device)
    organism.eval()

    all_actions = []
    with torch.no_grad():
        for _ in range(n_samples // 256 + 1):
            state = matter.reset_state(256, dev)
            seed = torch.randn(256, matter.seed_dim, device=dev)
            action = torch.zeros(256, 2, device=dev)
            next_state, emission, _ = matter(state, seed, action)
            obs = perception_fn(next_state, emission)
            action_mean, _, _ = organism(obs)
            all_actions.append(action_mean)

    actions = torch.cat(all_actions)[:n_samples]

    # Measure: variance of actions across states (structured behavior)
    action_var = actions.var(dim=0).sum().item()
    # Also measure mean magnitude (purposeful vs near-zero)
    action_mag = torch.norm(actions, dim=-1).mean().item()

    return {"proxy_d_action_var": action_var, "proxy_d_mag": action_mag,
            "proxy_d_value": action_var * action_mag}


def proxy_e_policy_consistency(organism, matter, perception_fn,
                                n_samples=2000, device="cuda"):
    """Proxy E: Policy consistency (self-predictability).

    Train a simple model to predict the organism's own actions from
    observations. If the organism has a consistent policy, this model has
    low error. If the policy is noisy/random, the model has high error.

    Proxy = 1 - normalized_prediction_error (higher = more consistent).
    """
    dev = torch.device(device)
    organism.eval()

    obs_list, act_list = [], []
    with torch.no_grad():
        for _ in range(n_samples // 256 + 1):
            state = matter.reset_state(256, dev)
            seed = torch.randn(256, matter.seed_dim, device=dev)
            action = torch.zeros(256, 2, device=dev)
            next_state, emission, _ = matter(state, seed, action)
            obs = perception_fn(next_state, emission)
            action_mean, _, _ = organism(obs)
            obs_list.append(obs)
            act_list.append(action_mean)

    obs_all = torch.cat(obs_list)[:n_samples]
    act_all = torch.cat(act_list)[:n_samples]

    # Train simple predictor: obs → action
    obs_dim = obs_all.shape[-1]
    predictor = nn.Sequential(
        nn.Linear(obs_dim, 32),
        nn.ReLU(),
        nn.Linear(32, 2),
    ).to(dev)
    opt = torch.optim.Adam(predictor.parameters(), lr=1e-3)

    n_train = int(0.8 * len(obs_all))
    for epoch in range(200):
        idx = torch.randperm(n_train, device=dev)[:256]
        pred = predictor(obs_all[idx])
        loss = F.mse_loss(pred, act_all[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()

    # Test error
    predictor.eval()
    with torch.no_grad():
        pred_test = predictor(obs_all[n_train:])
        test_error = F.mse_loss(pred_test, act_all[n_train:]).item()
        # Normalize by action variance
        act_var = act_all[n_train:].var().item() + 1e-8
        normalized_error = test_error / act_var

    # Higher consistency (lower normalized error) = higher proxy value
    consistency = 1.0 - min(normalized_error, 1.0)

    return {"proxy_e_error": test_error, "proxy_e_norm_error": normalized_error,
            "proxy_e_value": consistency}


def validate_proxies(embed_dims=[2, 8, 32], seeds=[42, 123, 456],
                     device="cpu"):
    """Run all proxies alongside true SA on a small grid to validate correlation."""
    torch.manual_seed(0)
    matter = RockPushMatter(emission_dim=8, action_dim=2, seed_dim=4).to(device)

    def oracle_fn(state, emission):
        return state

    results = []
    total = len(embed_dims) * len(seeds)
    idx = 0

    for emb in embed_dims:
        for s in seeds:
            idx += 1
            print(f"[{idx}/{total}] emb={emb}, seed={s}", end=" ... ", flush=True)
            torch.manual_seed(s)
            organism = Organism(sensory_dim=4, embedding_dim=emb, action_dim=2).to(device)

            # Quick train (100 episodes, small batch for CPU speed)
            from perception_ladder import train_organism_with_perception
            metrics = train_organism_with_perception(
                matter, organism, oracle_fn,
                target_x=0.8, target_y=0.8,
                n_episodes=100, batch_size=64, device=device,
            )

            # True SA
            sa = measure_coordination_quality(
                organism, oracle_fn, matter, n_samples=1000, device=device,
            )

            # Proxies
            pa = proxy_a_prediction_consistency(
                organism, matter, oracle_fn, n_samples=1000, device=device,
            )
            pb = proxy_b_action_stability(
                organism, matter, oracle_fn, n_samples=1000, device=device,
            )
            pc = proxy_c_value_action_alignment(
                organism, matter, oracle_fn, n_samples=1000, device=device,
            )

            n_tail = min(50, len(metrics["rock_distance"]))
            avg_dist = sum(metrics["rock_distance"][-n_tail:]) / n_tail

            result = {
                "embedding_dim": emb, "seed": s,
                "avg_dist": avg_dist,
                "true_SA": sa["C_i"],
                **pa, **pb, **pc,
            }
            results.append(result)
            print(f"SA={sa['C_i']:.3f}, "
                  f"pA={pa['proxy_a_value']:.4f}, "
                  f"pB={pb['proxy_b_value']:.4f}, "
                  f"pC={pc['proxy_c_value']:.4f}")

    return results


def analyze_proxies(results):
    """Compute correlation of each proxy with true SA."""
    import numpy as np

    true_sa = np.array([r["true_SA"] for r in results])
    dists = np.array([r["avg_dist"] for r in results])

    print(f"\n=== Proxy Validation (n={len(results)}) ===")
    print(f"True SA vs distance: r = {np.corrcoef(true_sa, dists)[0,1]:.3f}")

    for proxy_name, key in [
        ("A: Prediction consistency", "proxy_a_value"),
        ("B: Action stability", "proxy_b_value"),
        ("C: Value-action alignment", "proxy_c_value"),
    ]:
        vals = np.array([r[key] for r in results])
        r_sa = np.corrcoef(vals, true_sa)[0, 1]
        r_dist = np.corrcoef(vals, dists)[0, 1]
        print(f"  {proxy_name}:")
        print(f"    r(proxy, true SA) = {r_sa:+.3f}")
        print(f"    r(proxy, dist)    = {r_dist:+.3f}")


if __name__ == "__main__":
    print("=== SA Proxy Design Validation (CPU, small grid) ===")
    print("This is a design validation, not a production run.\n")
    results = validate_proxies(device="cpu")
    analyze_proxies(results)
