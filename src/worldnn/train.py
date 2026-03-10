"""Training loop for the World simulation.

Two-phase training:
1. Pre-train the environment VAE on channel signals (unsupervised)
2. Train the organism via policy gradient (REINFORCE with Gaussian policy)

The matter's transition network is fixed (not learned) — it represents
the physical laws governing the object. The channel is also fixed.
"""

import torch
import torch.nn.functional as F
import math
from worldnn.world import World


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
