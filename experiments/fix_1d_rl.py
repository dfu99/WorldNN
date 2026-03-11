#!/usr/bin/env python3
"""Fix RL learning from 1D latent space (obj-003 follow-up).

The env_lat=1 latent is well-separated (97% threshold accuracy) but
REINFORCE fails to learn from it. This experiment tests targeted fixes:

  Fix A: Supervised sensory pre-training — train organism's encoder to
         predict state from z before RL starts (warm-start the perception).
  Fix B: PPO instead of REINFORCE — lower variance gradient estimator.
  Fix C: Remove sensory filter gate for 1D — the sigmoid gate on scalar
         input may kill signal. Feed z directly to encoder.
  Fix D: Combined — sensory pre-training + PPO + no gate.

All runs use env_lat=1, noise=0.1, embed=4 (the failing config).
Baseline: original REINFORCE (should match ~5% success).
"""

import sys
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from worldnn.world import World
from worldnn.organism import Organism
from worldnn.train import train_environment, gaussian_log_prob


# ── Fix C: Organism with no sensory gate ──────────────────────────────

class OrganismNoGate(Organism):
    """Organism that bypasses sensory filter — feeds z directly to encoder."""

    def forward(self, z):
        # Skip gate, feed z directly to encoder
        embedding = self.encoder(z)
        action = self.policy(embedding)
        value = self.value_head(embedding).squeeze(-1)
        return action, embedding, value


# ── Fix A: Supervised sensory pre-training ────────────────────────────

def pretrain_sensory(world, n_steps=300, batch_size=512, lr=1e-3, device=None):
    """Pre-train organism encoder to predict state from z."""
    if device is None:
        device = next(world.parameters()).device

    # Freeze everything except organism encoder
    classifier = nn.Linear(world.organism.embedding_dim, 1).to(device)
    params = list(world.organism.encoder.parameters()) + list(classifier.parameters())

    # Also train sensory filter if it exists
    if hasattr(world.organism, 'sensory_filter'):
        params += list(world.organism.sensory_filter.parameters())

    optimizer = torch.optim.Adam(params, lr=lr)
    losses = []

    for step in range(n_steps):
        state = world.matter.reset_state(batch_size, device)
        seed = torch.randn(batch_size, world.seed_dim, device=device)
        action = torch.zeros(batch_size, world.action_dim, device=device)

        with torch.no_grad():
            _, emission, _ = world.matter(state, seed, action)
            channel_out = world.channel(emission)
            z, _, _, _ = world.environment(channel_out)

        # Forward through organism sensory path
        _, embedding, _ = world.organism(z)
        pred = classifier(embedding).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(pred, state)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses


# ── Fix B: PPO training ───────────────────────────────────────────────

def train_organism_ppo(
    world, n_episodes=500, steps_per_episode=10, batch_size=512,
    lr=3e-4, gamma=0.99, entropy_coef=0.01,
    action_std_init=0.8, action_std_final=0.2,
    clip_eps=0.2, ppo_epochs=4, device=None,
):
    """Train organism with PPO instead of REINFORCE."""
    if device is None:
        device = next(world.parameters()).device

    log_std = nn.Parameter(
        torch.full((world.action_dim,), math.log(action_std_init), device=device)
    )
    optimizer = torch.optim.Adam(
        list(world.organism.parameters()) + [log_std], lr=lr
    )

    metrics = {"rewards": [], "success_rates": [], "policy_losses": []}

    for ep in range(n_episodes):
        world.organism.train()

        # Anneal exploration
        frac = ep / max(n_episodes - 1, 1)
        target_std = action_std_init + frac * (action_std_final - action_std_init)

        # ── Collect rollout ──
        state = world.matter.reset_state(batch_size, device)
        action = None

        all_z, all_actions, all_log_probs = [], [], []
        all_rewards, all_values = [], []

        for t in range(steps_per_episode):
            result = world.step(state, action)
            z = result["z"].detach()

            action_mean, embedding, value = world.organism(z)
            std = log_std.exp().unsqueeze(0).expand_as(action_mean)
            dist = torch.distributions.Normal(action_mean, std)
            action_sample = dist.sample()
            lp = dist.log_prob(action_sample).sum(dim=-1)

            propagated = world.environment.propagate_action(action_sample)
            next_state, _, flip_prob = world.matter(state, result["seed"], propagated)

            reward = (next_state == 1.0).float()
            shaped = reward + 0.1 * flip_prob * (1.0 - state)

            all_z.append(z)
            all_actions.append(action_sample.detach())
            all_log_probs.append(lp.detach())
            all_rewards.append(shaped.detach())
            all_values.append(value.detach())

            state = next_state.detach()
            action = propagated.detach()

        # ── Compute returns & advantages ──
        T = len(all_rewards)
        returns = []
        G = torch.zeros(batch_size, device=device)
        for t in reversed(range(T)):
            G = all_rewards[t] + gamma * G
            returns.insert(0, G)
        returns = torch.stack(returns)
        values = torch.stack(all_values)
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ── PPO update ──
        z_batch = torch.stack(all_z)       # [T, B, sensory_dim]
        act_batch = torch.stack(all_actions)
        old_lp = torch.stack(all_log_probs)

        total_loss_val = 0.0
        for _ in range(ppo_epochs):
            for t in range(T):
                action_mean, _, value = world.organism(z_batch[t])
                std = log_std.exp().unsqueeze(0).expand_as(action_mean)
                dist = torch.distributions.Normal(action_mean, std)
                new_lp = dist.log_prob(act_batch[t]).sum(dim=-1)

                ratio = (new_lp - old_lp[t]).exp()
                clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
                policy_loss = -torch.min(ratio * advantages[t], clipped * advantages[t]).mean()
                value_loss = F.mse_loss(value, returns[t])
                entropy = dist.entropy().sum(dim=-1).mean()

                loss = policy_loss + 0.5 * value_loss - entropy_coef * entropy
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(world.organism.parameters()) + [log_std], 1.0)
                optimizer.step()
                total_loss_val += loss.item()

        with torch.no_grad():
            success = (state == 1.0).float().mean().item()
        avg_reward = sum(r.mean().item() for r in all_rewards) / T
        metrics["rewards"].append(avg_reward)
        metrics["success_rates"].append(success)
        metrics["policy_losses"].append(total_loss_val / (T * ppo_epochs))

    return metrics


# ── Main experiment ───────────────────────────────────────────────────

def run_config(label, use_gate=True, pretrain=False, use_ppo=False, seed=42):
    """Run one config and return metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device("cpu")

    world = World(
        emission_dim=4, channel_dim=4,
        env_latent_dim=1, embedding_dim=4,
        action_dim=2, seed_dim=4,
        channel_noise=0.1, channel_bandwidth=1.0,
        flip_difficulty=1.0,
        matter_hidden=32, env_hidden=32, organism_hidden=32,
    ).to(dev)

    # Optionally replace organism with no-gate version
    if not use_gate:
        world.organism = OrganismNoGate(
            sensory_dim=1, embedding_dim=4, action_dim=2, hidden_size=32,
        ).to(dev)

    # Phase 1: Train environment VAE
    env_losses = train_environment(world, n_steps=500, batch_size=256, lr=1e-3, beta=0.1, device=dev)

    # Optional: supervised sensory pre-training
    pretrain_losses = None
    if pretrain:
        pretrain_losses = pretrain_sensory(world, n_steps=300, batch_size=512, lr=1e-3, device=dev)

    # Phase 2: Train organism
    if use_ppo:
        metrics = train_organism_ppo(
            world, n_episodes=500, steps_per_episode=10, batch_size=512,
            lr=3e-4, device=dev,
        )
    else:
        from worldnn.train import train_organism
        metrics = train_organism(
            world, n_episodes=500, steps_per_episode=10, batch_size=512,
            lr=3e-4, device=dev,
        )

    final_success = np.mean(metrics["success_rates"][-30:])
    print(f"  {label}: final_success={final_success:.3f}")

    return {
        "label": label,
        "success_rates": metrics["success_rates"],
        "final_success": final_success,
        "env_losses": env_losses,
        "pretrain_losses": pretrain_losses,
    }


def main():
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    configs = [
        {"label": "Baseline (REINFORCE)", "use_gate": True, "pretrain": False, "use_ppo": False},
        {"label": "A: Sensory pretrain", "use_gate": True, "pretrain": True, "use_ppo": False},
        {"label": "B: PPO", "use_gate": True, "pretrain": False, "use_ppo": True},
        {"label": "C: No gate", "use_gate": False, "pretrain": False, "use_ppo": False},
        {"label": "D: Pretrain+PPO+NoGate", "use_gate": False, "pretrain": True, "use_ppo": True},
        {"label": "E: Pretrain+PPO", "use_gate": True, "pretrain": True, "use_ppo": True},
    ]

    all_results = []
    for cfg in configs:
        print(f"Running: {cfg['label']}...")
        result = run_config(**cfg)
        all_results.append(result)

    # ── Figure: Learning curves comparison ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: success rate curves
    ax = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
    for res, color in zip(all_results, colors):
        # Smooth with running mean
        sr = np.array(res["success_rates"])
        window = 20
        smoothed = np.convolve(sr, np.ones(window)/window, mode="valid")
        ax.plot(smoothed, label=f"{res['label']} ({res['final_success']:.1%})",
                color=color, linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success Rate")
    ax.set_title("env_lat=1, noise=0.1, embed=4: Fix Comparison")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="random baseline")

    # Right: final success bar chart
    ax = axes[1]
    labels = [r["label"] for r in all_results]
    finals = [r["final_success"] for r in all_results]
    bars = ax.barh(range(len(labels)), finals, color=colors, edgecolor="black")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Final Success Rate (last 30 episodes)")
    ax.set_title("Which Fix Enables 1D Latent Learning?")
    ax.set_xlim(0, 1.05)
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3, axis="x")
    for bar, val in zip(bars, finals):
        ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                f"{val:.1%}", va="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(f"{results_dir}/fix_1d_rl.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {results_dir}/fix_1d_rl.png")

    # Print summary
    print("\n" + "=" * 60)
    print("FIX COMPARISON SUMMARY (env_lat=1, noise=0.1, embed=4)")
    print("=" * 60)
    for res in sorted(all_results, key=lambda x: -x["final_success"]):
        marker = "***" if res["final_success"] > 0.6 else "   "
        print(f"  {marker} {res['label']:30s} {res['final_success']:.1%}")
    print("=" * 60)

    best = max(all_results, key=lambda x: x["final_success"])
    baseline = all_results[0]
    print(f"\nBaseline: {baseline['final_success']:.1%}")
    print(f"Best fix: {best['label']} at {best['final_success']:.1%}")
    if best["final_success"] > 0.5:
        print("SUCCESS: At least one fix enables learning from 1D latent!")
    else:
        print("PARTIAL: Improvements found but no fix reaches >50% success.")
        print("Consider: more training episodes, different architectures, or")
        print("the 1D RL problem may require fundamentally different approach.")


if __name__ == "__main__":
    main()
