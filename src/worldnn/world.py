"""World: the complete simulation loop tying all components together.

The world orchestrates one full perception-action cycle:

    seed ──► Matter ──► Channel ──► Environment ──► Organism ──► Action
                ▲                                                  │
                └──────────── (action propagated back) ◄───────────┘

Each step:
1. Matter produces emission based on state + seed + action
2. Emission passes through channel (noise + bandwidth)
3. Environment VAE encodes to latent z
4. Organism observes z, produces action
5. Action propagates back through environment to matter
6. Matter transitions to next state
"""

import torch
import torch.nn as nn

from worldnn.matter import Matter, ContinuousMatter, RockPushMatter
from worldnn.channels import Channel
from worldnn.environment import EnvironmentVAE
from worldnn.organism import Organism


class World(nn.Module):
    """Complete perception-action loop simulation."""

    def __init__(
        self,
        emission_dim: int = 4,
        channel_dim: int = 4,
        env_latent_dim: int = 2,
        embedding_dim: int = 4,
        action_dim: int = 2,
        seed_dim: int = 4,
        channel_noise: float = 0.1,
        channel_bandwidth: float = 1.0,
        flip_difficulty: float = 1.0,
        matter_hidden: int = 32,
        env_hidden: int = 32,
        organism_hidden: int = 32,
    ):
        super().__init__()

        self.seed_dim = seed_dim
        self.action_dim = action_dim

        self.matter = Matter(
            emission_dim=emission_dim,
            action_dim=action_dim,
            seed_dim=seed_dim,
            hidden_size=matter_hidden,
            flip_difficulty=flip_difficulty,
        )

        self.channel = Channel(
            input_dim=emission_dim,
            output_dim=channel_dim,
            noise_std=channel_noise,
            bandwidth=channel_bandwidth,
        )

        self.environment = EnvironmentVAE(
            channel_dim=channel_dim,
            latent_dim=env_latent_dim,
            hidden_size=env_hidden,
            action_dim=action_dim,
        )

        self.organism = Organism(
            sensory_dim=env_latent_dim,
            embedding_dim=embedding_dim,
            action_dim=action_dim,
            hidden_size=organism_hidden,
        )

    def step(
        self,
        state: torch.Tensor,
        prev_action: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Execute one full perception-action cycle.

        Args:
            state: [batch] current matter state
            prev_action: [batch, action_dim] previous action (None for first step)

        Returns:
            dict with all intermediate values for analysis
        """
        batch_size = state.shape[0]
        device = state.device

        # Generate random seed
        seed = torch.randn(batch_size, self.seed_dim, device=device)

        # Default action if none provided
        if prev_action is None:
            prev_action = torch.zeros(batch_size, self.action_dim, device=device)

        # 1. Matter: state transition + emission
        next_state, emission, flip_prob = self.matter(state, seed, prev_action)

        # 2. Channel: add noise and bandwidth limitation
        channel_out = self.channel(emission)

        # 3. Environment: VAE encode to latent
        z, y_hat, mu, logvar = self.environment(channel_out)

        # 4. Organism: observe z, produce action
        action, embedding, value = self.organism(z)

        # 5. Action propagation through environment
        propagated_action = self.environment.propagate_action(action)

        return {
            "state": state,
            "next_state": next_state,
            "seed": seed,
            "emission": emission,
            "channel_out": channel_out,
            "z": z,
            "y_hat": y_hat,
            "mu": mu,
            "logvar": logvar,
            "embedding": embedding,
            "action": propagated_action,
            "raw_action": action,
            "value": value,
            "flip_prob": flip_prob,
        }

    def run_episode(
        self,
        batch_size: int,
        n_steps: int,
        target_state: float = 1.0,
        device: torch.device | None = None,
    ) -> dict[str, list[torch.Tensor]]:
        """Run a full episode of perception-action steps.

        The organism's goal is to flip matter to target_state.

        Returns:
            dict mapping field names to lists of tensors (one per step)
        """
        if device is None:
            device = next(self.parameters()).device

        state = self.matter.reset_state(batch_size, device)
        action = None

        trajectory = {
            "states": [],
            "actions": [],
            "emissions": [],
            "z_latents": [],
            "embeddings": [],
            "flip_probs": [],
            "rewards": [],
            "values": [],
            "mu": [],
            "logvar": [],
            "y_hat": [],
            "channel_out": [],
        }

        for t in range(n_steps):
            result = self.step(state, action)

            # Reward: +1 if matter is in target state, -1 otherwise
            reward = (result["next_state"] == target_state).float() * 2 - 1

            trajectory["states"].append(state)
            trajectory["actions"].append(result["action"])
            trajectory["emissions"].append(result["emission"])
            trajectory["z_latents"].append(result["z"])
            trajectory["embeddings"].append(result["embedding"])
            trajectory["flip_probs"].append(result["flip_prob"])
            trajectory["rewards"].append(reward)
            trajectory["values"].append(result["value"])
            trajectory["mu"].append(result["mu"])
            trajectory["logvar"].append(result["logvar"])
            trajectory["y_hat"].append(result["y_hat"])
            trajectory["channel_out"].append(result["channel_out"])

            # Advance
            state = result["next_state"].detach()
            action = result["action"]

        return trajectory


class ContinuousWorld(nn.Module):
    """World with continuous 1D matter state (position targeting).

    The organism must push matter to a target position (default: 0.8).
    Reward is proportional to proximity to target. This requires
    proportional control, unlike the binary flip task.
    """

    def __init__(
        self,
        emission_dim: int = 4,
        channel_dim: int = 4,
        env_latent_dim: int = 2,
        embedding_dim: int = 4,
        action_dim: int = 2,
        seed_dim: int = 4,
        channel_noise: float = 0.1,
        channel_bandwidth: float = 1.0,
        force_scale: float = 0.1,
        target_position: float = 0.8,
        env_hidden: int = 32,
        organism_hidden: int = 32,
    ):
        super().__init__()

        self.seed_dim = seed_dim
        self.action_dim = action_dim
        self.target_position = target_position

        self.matter = ContinuousMatter(
            emission_dim=emission_dim,
            action_dim=action_dim,
            seed_dim=seed_dim,
            force_scale=force_scale,
        )

        self.channel = Channel(
            input_dim=emission_dim,
            output_dim=channel_dim,
            noise_std=channel_noise,
            bandwidth=channel_bandwidth,
        )

        self.environment = EnvironmentVAE(
            channel_dim=channel_dim,
            latent_dim=env_latent_dim,
            hidden_size=env_hidden,
            action_dim=action_dim,
        )

        self.organism = Organism(
            sensory_dim=env_latent_dim,
            embedding_dim=embedding_dim,
            action_dim=action_dim,
            hidden_size=organism_hidden,
        )

    def step(
        self,
        state: torch.Tensor,
        prev_action: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """One perception-action cycle for continuous matter."""
        batch_size = state.shape[0]
        device = state.device

        seed = torch.randn(batch_size, self.seed_dim, device=device)

        if prev_action is None:
            prev_action = torch.zeros(batch_size, self.action_dim, device=device)

        next_state, emission, force = self.matter(state, seed, prev_action)
        channel_out = self.channel(emission)
        z, y_hat, mu, logvar = self.environment(channel_out)
        action, embedding, value = self.organism(z)
        propagated_action = self.environment.propagate_action(action)

        return {
            "state": state,
            "next_state": next_state,
            "seed": seed,
            "emission": emission,
            "channel_out": channel_out,
            "z": z,
            "y_hat": y_hat,
            "mu": mu,
            "logvar": logvar,
            "embedding": embedding,
            "action": propagated_action,
            "raw_action": action,
            "value": value,
            "force": force,
        }


class RockPushWorld(nn.Module):
    """World with multi-object 2D physics: organism pushes rock to target.

    State is 4D [rock_x, rock_y, org_x, org_y]. Emissions have two channels:
    light (position-dependent) and vibration (contact-dependent). This requires
    richer internal models than 1-bit/1D tasks.

    Target: rock at (target_x, target_y). Reward is distance-based.
    """

    def __init__(
        self,
        emission_dim: int = 8,
        channel_dim: int = 8,
        env_latent_dim: int = 4,
        embedding_dim: int = 8,
        action_dim: int = 2,
        seed_dim: int = 4,
        channel_noise: float = 0.1,
        channel_bandwidth: float = 1.0,
        move_speed: float = 0.1,
        push_radius: float = 0.15,
        push_strength: float = 0.08,
        target_x: float = 0.8,
        target_y: float = 0.8,
        env_hidden: int = 32,
        organism_hidden: int = 32,
    ):
        super().__init__()

        self.seed_dim = seed_dim
        self.action_dim = action_dim
        self.target_x = target_x
        self.target_y = target_y

        self.matter = RockPushMatter(
            emission_dim=emission_dim,
            action_dim=action_dim,
            seed_dim=seed_dim,
            move_speed=move_speed,
            push_radius=push_radius,
            push_strength=push_strength,
        )

        self.channel = Channel(
            input_dim=emission_dim,
            output_dim=channel_dim,
            noise_std=channel_noise,
            bandwidth=channel_bandwidth,
        )

        self.environment = EnvironmentVAE(
            channel_dim=channel_dim,
            latent_dim=env_latent_dim,
            hidden_size=env_hidden,
            action_dim=action_dim,
        )

        self.organism = Organism(
            sensory_dim=env_latent_dim,
            embedding_dim=embedding_dim,
            action_dim=action_dim,
            hidden_size=organism_hidden,
        )

    def step(
        self,
        state: torch.Tensor,
        prev_action: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """One perception-action cycle for rock-push task."""
        batch_size = state.shape[0]
        device = state.device

        seed = torch.randn(batch_size, self.seed_dim, device=device)

        if prev_action is None:
            prev_action = torch.zeros(batch_size, self.action_dim, device=device)

        next_state, emission, contact = self.matter(state, seed, prev_action)
        channel_out = self.channel(emission)
        z, y_hat, mu, logvar = self.environment(channel_out)
        action, embedding, value = self.organism(z)
        propagated_action = self.environment.propagate_action(action)

        return {
            "state": state,
            "next_state": next_state,
            "seed": seed,
            "emission": emission,
            "channel_out": channel_out,
            "z": z,
            "y_hat": y_hat,
            "mu": mu,
            "logvar": logvar,
            "embedding": embedding,
            "action": propagated_action,
            "raw_action": action,
            "value": value,
            "contact": contact,
        }
