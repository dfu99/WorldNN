"""Matter: Mealy machines whose state transitions are driven by neural networks.

Matter has hidden state S and produces observable emissions X on each step.
The transition function depends on:
  - current state S_t
  - random seed (shared global source of stochasticity)
  - action A_t received from the environment

Architecture (simplified for binary-state study):
    ┌──────────────────────────────────────────┐
    │              Matter                       │
    │                                           │
    │  State S ∈ {0,1}                          │
    │                                           │
    │  Emission: X = state_signal + seed_noise  │
    │            (state info + stochasticity)    │
    │                                           │
    │  Transition: P(flip) = σ(match(A, target))│
    │              (action must match pattern)   │
    └──────────────────────────────────────────┘

For the binary-state case, we use explicit physics rather than a random NN,
so the relationship between actions and state flips is learnable.
A learned NN transition can be swapped in for more complex scenarios.
"""

import torch
import torch.nn as nn


class Matter(nn.Module):
    """A piece of matter with binary state and explicit physics.

    The organism must discover the hidden target action pattern that
    causes state flips. Emissions encode state information with noise.
    """

    def __init__(
        self,
        emission_dim: int = 4,
        action_dim: int = 2,
        seed_dim: int = 4,
        hidden_size: int = 32,
        flip_difficulty: float = 1.0,
    ):
        super().__init__()
        self.emission_dim = emission_dim
        self.action_dim = action_dim
        self.seed_dim = seed_dim
        self.flip_difficulty = flip_difficulty

        # Fixed emission patterns for each state (physics)
        # State 0 emits one pattern, state 1 emits another
        self.register_buffer(
            "emission_0", torch.randn(emission_dim) * 0.5
        )
        self.register_buffer(
            "emission_1", torch.randn(emission_dim) * 0.5 + 1.0
        )

        # Hidden target action that causes flips
        self.register_buffer(
            "flip_target", torch.randn(action_dim) * 0.5
        )

        # Seed-to-noise projection (fixed linear transform)
        self.register_buffer(
            "seed_proj", torch.randn(seed_dim, emission_dim) * 0.2
        )

    def forward(
        self, state: torch.Tensor, seed: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One step of the Mealy machine.

        Args:
            state: [batch] current binary state (0 or 1 as float)
            seed: [batch, seed_dim] random seed vector
            action: [batch, action_dim] action from organism

        Returns:
            next_state: [batch] new binary state
            emission: [batch, emission_dim] observable output
            flip_prob: [batch] probability that state flipped
        """
        batch = state.shape[0]

        # ── Emission: state-dependent signal + seed-derived noise ──
        # Mealy property: output depends on state AND input (seed)
        s = state.unsqueeze(-1)  # [batch, 1]
        base_emission = (1 - s) * self.emission_0 + s * self.emission_1
        noise = seed @ self.seed_proj  # [batch, emission_dim]
        emission = base_emission + noise

        # ── State transition: action must match target pattern ──
        distance = torch.sum((action - self.flip_target) ** 2, dim=-1)
        # Logit: positive when action is close to target
        flip_logit = (2.0 - distance * 2.0) / self.flip_difficulty
        flip_prob = torch.sigmoid(flip_logit)

        # Stochastic flip
        if self.training:
            flip = (torch.rand_like(flip_prob) < flip_prob).float()
            flip = flip + flip_prob - flip_prob.detach()  # STE
        else:
            flip = (flip_prob > 0.5).float()

        # XOR: flip toggles state
        next_state = state * (1 - flip) + (1 - state) * flip

        return next_state, emission, flip_prob

    def reset_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize random binary states."""
        return torch.randint(0, 2, (batch_size,), device=device).float()


class ContinuousMatter(nn.Module):
    """Matter with continuous 1D state (position on [0, 1]).

    The organism must push the state toward a target position.
    Emissions encode position with noise; actions apply force.

    Physics:
        position' = clamp(position + force * action_strength, 0, 1)
        emission = position_signal + seed_noise
        reward = -|position - target|  (closer is better)

    This requires finer control than binary flip — the organism must
    learn proportional control, not just binary switching.
    """

    def __init__(
        self,
        emission_dim: int = 4,
        action_dim: int = 2,
        seed_dim: int = 4,
        force_scale: float = 0.1,
    ):
        super().__init__()
        self.emission_dim = emission_dim
        self.action_dim = action_dim
        self.seed_dim = seed_dim
        self.force_scale = force_scale

        # Emission: position is encoded as a learned linear projection
        # (different from binary — here position maps smoothly to emission)
        self.register_buffer(
            "emission_weight", torch.randn(1, emission_dim) * 0.5
        )
        self.register_buffer(
            "emission_bias", torch.randn(emission_dim) * 0.3
        )

        # Seed-to-noise projection
        self.register_buffer(
            "seed_proj", torch.randn(seed_dim, emission_dim) * 0.2
        )

        # Action-to-force projection: maps action_dim → scalar force
        self.register_buffer(
            "force_proj", torch.randn(action_dim) * 0.5
        )

    def forward(
        self, state: torch.Tensor, seed: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One step of continuous matter.

        Args:
            state: [batch] current position in [0, 1]
            seed: [batch, seed_dim] random seed
            action: [batch, action_dim] action from organism

        Returns:
            next_state: [batch] new position in [0, 1]
            emission: [batch, emission_dim] observable output
            force_applied: [batch] scalar force that was applied
        """
        # ── Emission: smooth function of position + noise ──
        pos = state.unsqueeze(-1)  # [batch, 1]
        base_emission = pos * self.emission_weight + self.emission_bias
        noise = seed @ self.seed_proj
        emission = base_emission + noise

        # ── State transition: action → force → position change ──
        force = (action * self.force_proj).sum(dim=-1)  # [batch]
        force = torch.tanh(force) * self.force_scale  # bounded force

        next_state = torch.clamp(state + force, 0.0, 1.0)

        return next_state, emission, force

    def reset_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize random positions in [0, 1]."""
        return torch.rand(batch_size, device=device)


class RockPushMatter(nn.Module):
    """Multi-object matter: organism pushes a rock to a target in 2D.

    State: [rock_x, rock_y, org_x, org_y] — 4D continuous state.
    The organism moves in 2D; when close to the rock, its movement
    pushes the rock. Goal: get rock to target position.

    Multi-channel emissions:
      - Light channel (emission_dim // 2): encodes rock position relative
        to organism, intensity falls off with distance
      - Vibration channel (emission_dim // 2): contact signal, strong when
        organism touches rock, zero otherwise

    This creates genuine capacity requirements because:
      1. 4D state requires richer internal models than 1D
      2. Two information channels must be integrated
      3. Sequential strategy: approach → contact → push direction
    """

    def __init__(
        self,
        emission_dim: int = 8,
        action_dim: int = 2,
        seed_dim: int = 4,
        move_speed: float = 0.1,
        push_radius: float = 0.15,
        push_strength: float = 0.08,
    ):
        super().__init__()
        self.emission_dim = emission_dim
        self.action_dim = action_dim
        self.seed_dim = seed_dim
        self.move_speed = move_speed
        self.push_radius = push_radius
        self.push_strength = push_strength
        self.state_dim = 4  # [rock_x, rock_y, org_x, org_y]

        # Split emission into light and vibration channels
        self.light_dim = emission_dim // 2
        self.vib_dim = emission_dim - self.light_dim

        # Light projection: relative position → light emission
        self.register_buffer(
            "light_proj", torch.randn(2, self.light_dim) * 0.5
        )
        self.register_buffer(
            "light_bias", torch.randn(self.light_dim) * 0.2
        )

        # Vibration projection: contact strength → vibration emission
        self.register_buffer(
            "vib_proj", torch.randn(1, self.vib_dim) * 0.5
        )
        self.register_buffer(
            "vib_bias", torch.randn(self.vib_dim) * 0.1
        )

        # Seed-to-noise projection
        self.register_buffer(
            "seed_proj", torch.randn(seed_dim, emission_dim) * 0.15
        )

        # Action-to-movement: maps action_dim → 2D movement direction
        self.register_buffer(
            "move_proj", torch.randn(action_dim, 2) * 0.5
        )

    def forward(
        self, state: torch.Tensor, seed: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One step of rock-push physics.

        Args:
            state: [batch, 4] = [rock_x, rock_y, org_x, org_y]
            seed: [batch, seed_dim] random seed
            action: [batch, action_dim] organism's action

        Returns:
            next_state: [batch, 4] updated positions
            emission: [batch, emission_dim] multi-channel emission
            contact: [batch] contact strength (0-1)
        """
        rock_pos = state[:, :2]   # [batch, 2]
        org_pos = state[:, 2:4]   # [batch, 2]

        # ── Organism movement ──
        movement = torch.tanh(action @ self.move_proj) * self.move_speed
        new_org = torch.clamp(org_pos + movement, 0.0, 1.0)

        # ── Contact detection ──
        rel_pos = rock_pos - new_org  # [batch, 2]
        distance = torch.norm(rel_pos, dim=-1, keepdim=False)  # [batch]
        contact = torch.exp(-((distance / self.push_radius) ** 2))  # smooth contact

        # ── Rock physics: pushed when organism is close ──
        # Push direction = from organism toward rock
        push_dir = rel_pos / (distance.unsqueeze(-1) + 1e-6)  # [batch, 2]
        push_force = contact.unsqueeze(-1) * push_dir * self.push_strength
        new_rock = torch.clamp(rock_pos + push_force, 0.0, 1.0)

        # ── Multi-channel emission ──
        # Light: encodes rock-organism relative position, dimmer with distance
        light_intensity = 1.0 / (1.0 + distance.unsqueeze(-1))  # [batch, 1]
        light_signal = rel_pos @ self.light_proj * light_intensity + self.light_bias

        # Vibration: strong contact signal, zero when far
        vib_signal = contact.unsqueeze(-1) @ self.vib_proj + self.vib_bias

        # Combine channels + seed noise
        emission = torch.cat([light_signal, vib_signal], dim=-1)
        noise = seed @ self.seed_proj
        emission = emission + noise

        # ── Assemble next state ──
        next_state = torch.cat([new_rock, new_org], dim=-1)

        return next_state, emission, contact

    def reset_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize random positions: rock and organism in [0.1, 0.9]."""
        return torch.rand(batch_size, self.state_dim, device=device) * 0.8 + 0.1
