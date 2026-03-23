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
        move_speed: float = 0.15,
        push_radius: float = 0.2,
        push_strength: float = 0.12,
    ):
        super().__init__()
        self.emission_dim = emission_dim
        self.action_dim = action_dim
        self.seed_dim = seed_dim
        self.move_speed = move_speed
        self.push_radius = push_radius
        self.push_strength = push_strength
        self.state_dim = 4  # [rock_x, rock_y, org_x, org_y]

        # Emission encodes 4 state variables + contact through a fixed
        # projection. We use a full state→emission matrix to ensure all
        # state information is present in the signal.
        # state_vec = [rock_x, rock_y, org_x, org_y, contact]
        self.register_buffer(
            "state_proj", torch.randn(5, emission_dim) * 0.5
        )
        self.register_buffer(
            "emission_bias", torch.randn(emission_dim) * 0.2
        )

        # Seed-to-noise projection
        self.register_buffer(
            "seed_proj", torch.randn(seed_dim, emission_dim) * 0.15
        )

        # Action-to-movement: first 2 dims of action are (dx, dy) directly
        assert action_dim >= 2, "action_dim must be >= 2 for 2D movement"

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

        # ── Organism movement ── (first 2 action dims = dx, dy directly)
        movement = torch.tanh(action[:, :2]) * self.move_speed
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

        # ── Emission: all state information projected through fixed matrix ──
        state_vec = torch.stack([
            new_rock[:, 0], new_rock[:, 1],
            new_org[:, 0], new_org[:, 1],
            contact,
        ], dim=-1)  # [batch, 5]
        emission = state_vec @ self.state_proj + self.emission_bias
        noise = seed @ self.seed_proj
        emission = emission + noise

        # ── Assemble next state ──
        next_state = torch.cat([new_rock, new_org], dim=-1)

        return next_state, emission, contact

    def reset_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize random positions: rock and organism in [0.1, 0.9]."""
        return torch.rand(batch_size, self.state_dim, device=device) * 0.8 + 0.1


class MultiRockMatter(nn.Module):
    """Multi-object matter: organism pushes 3 rocks to 3 targets in 2D.

    State: [r1x, r1y, r2x, r2y, r3x, r3y, ox, oy] — 8D continuous.
    The organism moves in 2D and pushes any rock it contacts.
    Goal: get all 3 rocks to their respective targets.

    This is the 8D second task required for ICLR submission:
    - 8D state (vs 4D rock-push) stresses capacity requirements
    - 3 objects require the organism to attend and prioritize
    - Sequential strategy: approach nearest off-target rock → push → repeat
    """

    def __init__(
        self,
        emission_dim: int = 16,
        action_dim: int = 2,
        seed_dim: int = 4,
        move_speed: float = 0.15,
        push_radius: float = 0.2,
        push_strength: float = 0.12,
        n_rocks: int = 3,
    ):
        super().__init__()
        self.emission_dim = emission_dim
        self.action_dim = action_dim
        self.seed_dim = seed_dim
        self.move_speed = move_speed
        self.push_radius = push_radius
        self.push_strength = push_strength
        self.n_rocks = n_rocks
        self.state_dim = n_rocks * 2 + 2  # n_rocks×(x,y) + org(x,y) = 8D

        # Emission projection: state_vec → emission
        # state_vec = [r1x, r1y, r2x, r2y, r3x, r3y, ox, oy, c1, c2, c3]
        proj_dim = self.state_dim + n_rocks  # positions + contact per rock
        self.register_buffer(
            "state_proj", torch.randn(proj_dim, emission_dim) * 0.5
        )
        self.register_buffer(
            "emission_bias", torch.randn(emission_dim) * 0.2
        )
        self.register_buffer(
            "seed_proj", torch.randn(seed_dim, emission_dim) * 0.15
        )

    def forward(
        self, state: torch.Tensor, seed: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One step of multi-rock physics.

        Returns:
            next_state: [batch, 8]
            emission: [batch, emission_dim]
            contact: [batch] total contact (sum across rocks)
        """
        n = self.n_rocks
        rock_positions = state[:, :n*2].reshape(-1, n, 2)  # [batch, n, 2]
        org_pos = state[:, n*2:n*2+2]  # [batch, 2]

        # Organism movement
        movement = torch.tanh(action[:, :2]) * self.move_speed
        new_org = torch.clamp(org_pos + movement, 0.0, 1.0)

        # Per-rock contact and push
        contacts = []
        new_rocks = []
        for i in range(n):
            rock_i = rock_positions[:, i, :]  # [batch, 2]
            rel = rock_i - new_org
            dist = torch.norm(rel, dim=-1)
            contact_i = torch.exp(-((dist / self.push_radius) ** 2))
            contacts.append(contact_i)

            push_dir = rel / (dist.unsqueeze(-1) + 1e-6)
            push_force = contact_i.unsqueeze(-1) * push_dir * self.push_strength
            new_rock_i = torch.clamp(rock_i + push_force, 0.0, 1.0)
            new_rocks.append(new_rock_i)

        # Assemble state and emission
        new_rocks_flat = torch.cat(new_rocks, dim=-1)  # [batch, n*2]
        next_state = torch.cat([new_rocks_flat, new_org], dim=-1)  # [batch, 8]

        contact_stack = torch.stack(contacts, dim=-1)  # [batch, n]
        total_contact = contact_stack.sum(dim=-1)  # [batch]

        # Emission: full state + per-rock contacts
        state_vec = torch.cat([next_state, contact_stack], dim=-1)  # [batch, 8+3=11]
        emission = state_vec @ self.state_proj + self.emission_bias
        noise = seed @ self.seed_proj
        emission = emission + noise

        return next_state, emission, total_contact

    def reset_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Random positions for 3 rocks + organism in [0.1, 0.9]."""
        return torch.rand(batch_size, self.state_dim, device=device) * 0.8 + 0.1
