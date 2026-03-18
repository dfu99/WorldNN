"""Training loop for the World simulation.

Two-phase training:
1. Pre-train the environment VAE on channel signals (unsupervised)
2. Train the organism via policy gradient (REINFORCE or PPO)

The matter's transition network is fixed (not learned) — it represents
the physical laws governing the object. The channel is also fixed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from worldnn.world import World, ContinuousWorld, RockPushWorld
from worldnn.organism import PredictiveOrganism


def train_environment(
    world: World,
    n_steps: int = 1000,
    batch_size: int = 256,
    lr: float = 1e-3,
    beta: float = 0.1,
    device: torch.device | None = None,
) -> list[float]:
    """Pre-train the environment VAE on channel signals."""
    if device is None:
        device = next(world.parameters()).device

    optimizer = torch.optim.Adam(world.environment.parameters(), lr=lr)
    losses = []

    for step in range(n_steps):
        state = world.matter.reset_state(batch_size, device)
        seed = torch.randn(batch_size, world.seed_dim, device=device)
        action = torch.randn(batch_size, world.action_dim, device=device) * 0.1

        with torch.no_grad():
            _, emission, _ = world.matter(state, seed, action)
            channel_out = world.channel(emission)

        z, y_hat, mu, logvar = world.environment(channel_out)
        loss = world.environment.vae_loss(channel_out, y_hat, mu, logvar, beta=beta)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses


def gaussian_log_prob(action: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    """Compute log probability under diagonal Gaussian."""
    std = log_std.exp()
    var = std ** 2
    log_prob = -0.5 * (((action - mean) ** 2) / var + 2 * log_std + math.log(2 * math.pi))
    return log_prob.sum(dim=-1)  # sum over action dims


def train_organism(
    world: World,
    n_episodes: int = 500,
    steps_per_episode: int = 10,
    batch_size: int = 256,
    lr: float = 3e-4,
    gamma: float = 0.99,
    entropy_coef: float = 0.01,
    action_std_init: float = 0.8,
    action_std_final: float = 0.2,
    device: torch.device | None = None,
) -> dict[str, list[float]]:
    """Train the organism to flip matter state using REINFORCE with Gaussian policy."""
    if device is None:
        device = next(world.parameters()).device

    # Learnable log-std for exploration
    log_std = torch.nn.Parameter(
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
        target_log_std = math.log(action_std_init + frac * (action_std_final - action_std_init))

        # Run episode manually to collect log_probs
        state = world.matter.reset_state(batch_size, device)
        action = None

        log_probs = []
        rewards_list = []
        values_list = []
        states_list = []

        for t in range(steps_per_episode):
            result = world.step(state, action)

            # Organism forward pass gives action mean
            z = result["z"]
            action_mean, embedding, value = world.organism(z)

            # Sample action from Gaussian policy
            std = log_std.exp().unsqueeze(0).expand_as(action_mean)
            action_dist = torch.distributions.Normal(action_mean, std)
            action_sample = action_dist.sample()
            lp = action_dist.log_prob(action_sample).sum(dim=-1)

            # Propagate action through environment
            propagated = world.environment.propagate_action(action_sample)

            # Re-run matter transition with the sampled action
            next_state, _, flip_prob = world.matter(state, result["seed"], propagated)

            # Reward: +1 if in target state, 0 otherwise (shaped)
            reward = (next_state == 1.0).float()
            # Also give partial credit for flip probability when state is 0
            shaped_reward = reward + 0.1 * flip_prob * (1.0 - state)

            log_probs.append(lp)
            rewards_list.append(shaped_reward.detach())
            values_list.append(value)
            states_list.append(state.clone())

            state = next_state.detach()
            action = propagated.detach()

        # Compute returns
        T = len(rewards_list)
        returns = []
        G = torch.zeros(batch_size, device=device)
        for t in reversed(range(T)):
            G = rewards_list[t] + gamma * G
            returns.insert(0, G)

        # Normalize returns for stability
        all_returns = torch.stack(returns)
        ret_mean = all_returns.mean()
        ret_std = all_returns.std() + 1e-8
        all_returns = (all_returns - ret_mean) / ret_std

        # Policy gradient loss
        policy_loss = torch.tensor(0.0, device=device)
        value_loss = torch.tensor(0.0, device=device)
        entropy_bonus = torch.tensor(0.0, device=device)

        for t in range(T):
            advantage = (all_returns[t] - values_list[t]).detach()
            policy_loss = policy_loss - (log_probs[t] * advantage).mean()
            value_loss = value_loss + F.mse_loss(values_list[t], returns[t].detach())
            entropy_bonus = entropy_bonus + (log_std + 0.5 * math.log(2 * math.pi * math.e)).sum()

        total_loss = policy_loss / T + 0.5 * value_loss / T - entropy_coef * entropy_bonus / T

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(world.organism.parameters()) + [log_std], 1.0)
        optimizer.step()

        # Track metrics
        with torch.no_grad():
            success = (state == 1.0).float().mean().item()

        avg_reward = sum(r.mean().item() for r in rewards_list) / T
        metrics["rewards"].append(avg_reward)
        metrics["success_rates"].append(success)
        metrics["policy_losses"].append(total_loss.item())

    return metrics


def train_organism_ppo(
    world: World,
    n_episodes: int = 500,
    steps_per_episode: int = 10,
    batch_size: int = 512,
    lr: float = 3e-4,
    gamma: float = 0.99,
    entropy_coef: float = 0.01,
    action_std_init: float = 0.8,
    action_std_final: float = 0.2,
    clip_eps: float = 0.2,
    ppo_epochs: int = 4,
    device: torch.device | None = None,
) -> dict[str, list[float]]:
    """Train the organism to flip matter state using PPO.

    PPO's clipped objective provides lower-variance gradient estimates
    than REINFORCE, which is critical for low-dimensional latent inputs
    (env_lat=1) where REINFORCE fails (~5% vs PPO's ~87%).
    """
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
        z_batch = torch.stack(all_z)
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


def train_organism_ppo_predictive(
    world: World,
    n_episodes: int = 500,
    steps_per_episode: int = 10,
    batch_size: int = 512,
    lr: float = 3e-4,
    gamma: float = 0.99,
    entropy_coef: float = 0.01,
    pred_coef: float = 0.1,
    action_std_init: float = 0.8,
    action_std_final: float = 0.2,
    clip_eps: float = 0.2,
    ppo_epochs: int = 4,
    device: torch.device | None = None,
) -> dict[str, list[float]]:
    """Train organism with PPO + predictive processing auxiliary loss.

    The organism predicts the next observation z and receives an auxiliary
    loss proportional to the prediction error. This encourages the organism
    to build a world model alongside its policy.

    Requires world.organism to be a PredictiveOrganism.
    """
    if device is None:
        device = next(world.parameters()).device

    assert isinstance(world.organism, PredictiveOrganism), \
        "world.organism must be a PredictiveOrganism for predictive training"

    log_std = nn.Parameter(
        torch.full((world.action_dim,), math.log(action_std_init), device=device)
    )
    optimizer = torch.optim.Adam(
        list(world.organism.parameters()) + [log_std], lr=lr
    )

    metrics = {"rewards": [], "success_rates": [], "policy_losses": [], "pred_losses": []}

    for ep in range(n_episodes):
        world.organism.train()

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

        # ── PPO + Prediction update ──
        z_batch = torch.stack(all_z)
        act_batch = torch.stack(all_actions)
        old_lp = torch.stack(all_log_probs)

        total_loss_val = 0.0
        total_pred_loss = 0.0
        for _ in range(ppo_epochs):
            for t in range(T):
                action_mean, embedding, value, pred_z = \
                    world.organism.forward_with_prediction(z_batch[t])
                std = log_std.exp().unsqueeze(0).expand_as(action_mean)
                dist = torch.distributions.Normal(action_mean, std)
                new_lp = dist.log_prob(act_batch[t]).sum(dim=-1)

                ratio = (new_lp - old_lp[t]).exp()
                clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
                policy_loss = -torch.min(ratio * advantages[t], clipped * advantages[t]).mean()
                value_loss = F.mse_loss(value, returns[t])
                entropy = dist.entropy().sum(dim=-1).mean()

                # Prediction loss: predict next z from current embedding + action
                if t < T - 1:
                    pred_loss = F.mse_loss(pred_z, z_batch[t + 1])
                else:
                    pred_loss = torch.tensor(0.0, device=device)

                loss = policy_loss + 0.5 * value_loss - entropy_coef * entropy \
                    + pred_coef * pred_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(world.organism.parameters()) + [log_std], 1.0)
                optimizer.step()
                total_loss_val += loss.item()
                total_pred_loss += pred_loss.item()

        with torch.no_grad():
            success = (state == 1.0).float().mean().item()
        avg_reward = sum(r.mean().item() for r in all_rewards) / T
        metrics["rewards"].append(avg_reward)
        metrics["success_rates"].append(success)
        metrics["policy_losses"].append(total_loss_val / (T * ppo_epochs))
        metrics["pred_losses"].append(total_pred_loss / (T * ppo_epochs))

    return metrics


def train_environment_continuous(
    world: ContinuousWorld,
    n_steps: int = 1000,
    batch_size: int = 256,
    lr: float = 1e-3,
    beta: float = 0.1,
    device: torch.device | None = None,
) -> list[float]:
    """Pre-train VAE for continuous matter (same as binary, but samples continuous states)."""
    if device is None:
        device = next(world.parameters()).device

    optimizer = torch.optim.Adam(world.environment.parameters(), lr=lr)
    losses = []

    for step in range(n_steps):
        state = world.matter.reset_state(batch_size, device)
        seed = torch.randn(batch_size, world.seed_dim, device=device)
        action = torch.randn(batch_size, world.action_dim, device=device) * 0.1

        with torch.no_grad():
            _, emission, _ = world.matter(state, seed, action)
            channel_out = world.channel(emission)

        z, y_hat, mu, logvar = world.environment(channel_out)
        loss = world.environment.vae_loss(channel_out, y_hat, mu, logvar, beta=beta)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses


def train_organism_ppo_continuous(
    world: ContinuousWorld,
    n_episodes: int = 500,
    steps_per_episode: int = 10,
    batch_size: int = 512,
    lr: float = 3e-4,
    gamma: float = 0.99,
    entropy_coef: float = 0.01,
    action_std_init: float = 0.8,
    action_std_final: float = 0.2,
    clip_eps: float = 0.2,
    ppo_epochs: int = 4,
    device: torch.device | None = None,
) -> dict[str, list[float]]:
    """Train organism to push continuous matter toward target position using PPO."""
    if device is None:
        device = next(world.parameters()).device

    target = world.target_position

    log_std = nn.Parameter(
        torch.full((world.action_dim,), math.log(action_std_init), device=device)
    )
    optimizer = torch.optim.Adam(
        list(world.organism.parameters()) + [log_std], lr=lr
    )

    metrics = {"rewards": [], "mean_distance": [], "policy_losses": []}

    for ep in range(n_episodes):
        world.organism.train()

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
            next_state, _, force = world.matter(state, result["seed"], propagated)

            # Reward: negative distance to target (dense, shaped)
            distance = (next_state - target).abs()
            reward = 1.0 - distance  # 1.0 when at target, 0.0 when at opposite end

            all_z.append(z)
            all_actions.append(action_sample.detach())
            all_log_probs.append(lp.detach())
            all_rewards.append(reward.detach())
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
        z_batch = torch.stack(all_z)
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
            final_dist = (state - target).abs().mean().item()
        avg_reward = sum(r.mean().item() for r in all_rewards) / T
        metrics["rewards"].append(avg_reward)
        metrics["mean_distance"].append(final_dist)
        metrics["policy_losses"].append(total_loss_val / (T * ppo_epochs))

    return metrics


def train_environment_rockpush(
    world: RockPushWorld,
    n_steps: int = 1000,
    batch_size: int = 256,
    lr: float = 1e-3,
    beta: float = 0.1,
    device: torch.device | None = None,
) -> list[float]:
    """Pre-train VAE for rock-push multi-channel emissions."""
    if device is None:
        device = next(world.parameters()).device

    optimizer = torch.optim.Adam(world.environment.parameters(), lr=lr)
    losses = []

    for step in range(n_steps):
        state = world.matter.reset_state(batch_size, device)
        seed = torch.randn(batch_size, world.seed_dim, device=device)
        action = torch.randn(batch_size, world.action_dim, device=device) * 0.1

        with torch.no_grad():
            _, emission, _ = world.matter(state, seed, action)
            channel_out = world.channel(emission)

        z, y_hat, mu, logvar = world.environment(channel_out)
        loss = world.environment.vae_loss(channel_out, y_hat, mu, logvar, beta=beta)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses


def train_organism_ppo_rockpush(
    world: RockPushWorld,
    n_episodes: int = 500,
    steps_per_episode: int = 20,
    batch_size: int = 512,
    lr: float = 3e-4,
    gamma: float = 0.99,
    entropy_coef: float = 0.01,
    action_std_init: float = 0.8,
    action_std_final: float = 0.2,
    clip_eps: float = 0.2,
    ppo_epochs: int = 4,
    device: torch.device | None = None,
) -> dict[str, list[float]]:
    """Train organism to push rock to target using PPO.

    Reward structure:
      - Primary: negative rock-to-target distance (dense)
      - Approach bonus: reward for getting closer to rock
      - Contact bonus: reward for touching rock
    """
    if device is None:
        device = next(world.parameters()).device

    target = torch.tensor([world.target_x, world.target_y], device=device)

    log_std = nn.Parameter(
        torch.full((world.action_dim,), math.log(action_std_init), device=device)
    )
    optimizer = torch.optim.Adam(
        list(world.organism.parameters()) + [log_std], lr=lr
    )

    metrics = {
        "rewards": [], "rock_distance": [], "policy_losses": [],
        "contact_rate": [],
    }

    for ep in range(n_episodes):
        world.organism.train()

        state = world.matter.reset_state(batch_size, device)
        action = None

        all_z, all_actions, all_log_probs = [], [], []
        all_rewards, all_values = [], []
        contact_sum = 0.0

        for t in range(steps_per_episode):
            result = world.step(state, action)
            # Use mu (deterministic) when world.use_mu is set — stochastic z
            # destroys spatial signal for directional control (obj-012)
            z = result["mu"].detach() if getattr(world, "use_mu", False) else result["z"].detach()

            action_mean, embedding, value = world.organism(z)
            std = log_std.exp().unsqueeze(0).expand_as(action_mean)
            dist = torch.distributions.Normal(action_mean, std)
            action_sample = dist.sample()
            lp = dist.log_prob(action_sample).sum(dim=-1)

            # For rock-push, pass action directly to matter (no environment
            # transform) — directional control requires action semantics
            next_state, _, contact = world.matter(state, result["seed"], action_sample)

            # Multi-component reward
            rock_pos = next_state[:, :2]
            org_pos = next_state[:, 2:4]
            rock_dist = torch.norm(rock_pos - target, dim=-1)
            org_rock_dist = torch.norm(rock_pos - org_pos, dim=-1)

            # Dense reward: proximity of rock to target
            rock_reward = 1.0 - rock_dist
            # Approach bonus: organism closer to rock
            approach = 0.2 * (1.0 - org_rock_dist)
            # Contact bonus
            contact_bonus = 0.1 * contact
            reward = rock_reward + approach + contact_bonus

            all_z.append(z)
            all_actions.append(action_sample.detach())
            all_log_probs.append(lp.detach())
            all_rewards.append(reward.detach())
            all_values.append(value.detach())
            contact_sum += contact.mean().item()

            state = next_state.detach()
            action = action_sample.detach()

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
        z_batch = torch.stack(all_z)
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
            rock_pos = state[:, :2]
            final_dist = torch.norm(rock_pos - target, dim=-1).mean().item()
        avg_reward = sum(r.mean().item() for r in all_rewards) / T
        metrics["rewards"].append(avg_reward)
        metrics["rock_distance"].append(final_dist)
        metrics["policy_losses"].append(total_loss_val / (T * ppo_epochs))
        metrics["contact_rate"].append(contact_sum / T)

    return metrics


def train_oracle_ppo_rockpush(
    matter,
    embedding_dim: int = 8,
    n_episodes: int = 1000,
    steps_per_episode: int = 20,
    batch_size: int = 512,
    lr: float = 3e-4,
    gamma: float = 0.99,
    entropy_coef: float = 0.01,
    action_std_init: float = 0.8,
    action_std_final: float = 0.2,
    clip_eps: float = 0.2,
    ppo_epochs: int = 4,
    target_x: float = 0.8,
    target_y: float = 0.8,
    device: torch.device | None = None,
) -> dict[str, list[float]]:
    """Oracle baseline: organism directly observes 4D state (no VAE/channel).

    This isolates whether the rock-push task is learnable at all, independent
    of the perception pipeline. If the oracle succeeds but the full pipeline
    fails, the bottleneck is perception. If both fail, the task/reward design
    needs work.

    Uses a standalone Organism with sensory_dim=4 (raw state input).
    """
    from worldnn.organism import Organism
    from worldnn.matter import RockPushMatter

    if device is None:
        device = torch.device("cpu")

    org = Organism(
        sensory_dim=4,  # directly observes [rock_x, rock_y, org_x, org_y]
        embedding_dim=embedding_dim,
        action_dim=matter.action_dim,
        hidden_size=64,
    ).to(device)

    target = torch.tensor([target_x, target_y], device=device)

    log_std = nn.Parameter(
        torch.full((matter.action_dim,), math.log(action_std_init), device=device)
    )
    optimizer = torch.optim.Adam(
        list(org.parameters()) + [log_std], lr=lr
    )

    metrics = {
        "rewards": [], "rock_distance": [], "policy_losses": [],
        "contact_rate": [],
    }

    for ep in range(n_episodes):
        org.train()

        state = matter.reset_state(batch_size, device)
        action = torch.zeros(batch_size, matter.action_dim, device=device)

        all_obs, all_actions, all_log_probs = [], [], []
        all_rewards, all_values = [], []
        contact_sum = 0.0

        for t in range(steps_per_episode):
            seed = torch.randn(batch_size, matter.seed_dim, device=device)

            # Oracle: organism sees raw state
            obs = state.detach()
            action_mean, embedding, value = org(obs)
            std = log_std.exp().unsqueeze(0).expand_as(action_mean)
            dist = torch.distributions.Normal(action_mean, std)
            action_sample = dist.sample()
            lp = dist.log_prob(action_sample).sum(dim=-1)

            next_state, _, contact = matter(state, seed, action_sample)

            rock_pos = next_state[:, :2]
            org_pos = next_state[:, 2:4]
            rock_dist = torch.norm(rock_pos - target, dim=-1)
            org_rock_dist = torch.norm(rock_pos - org_pos, dim=-1)

            rock_reward = 1.0 - rock_dist
            approach = 0.2 * (1.0 - org_rock_dist)
            contact_bonus = 0.1 * contact
            reward = rock_reward + approach + contact_bonus

            all_obs.append(obs)
            all_actions.append(action_sample.detach())
            all_log_probs.append(lp.detach())
            all_rewards.append(reward.detach())
            all_values.append(value.detach())
            contact_sum += contact.mean().item()

            state = next_state.detach()

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
        obs_batch = torch.stack(all_obs)
        act_batch = torch.stack(all_actions)
        old_lp = torch.stack(all_log_probs)

        total_loss_val = 0.0
        for _ in range(ppo_epochs):
            for t in range(T):
                action_mean, _, value = org(obs_batch[t])
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
                nn.utils.clip_grad_norm_(list(org.parameters()) + [log_std], 1.0)
                optimizer.step()
                total_loss_val += loss.item()

        with torch.no_grad():
            rock_pos = state[:, :2]
            final_dist = torch.norm(rock_pos - target, dim=-1).mean().item()
        avg_reward = sum(r.mean().item() for r in all_rewards) / T
        metrics["rewards"].append(avg_reward)
        metrics["rock_distance"].append(final_dist)
        metrics["policy_losses"].append(total_loss_val / (T * ppo_epochs))
        metrics["contact_rate"].append(contact_sum / T)

    return metrics
