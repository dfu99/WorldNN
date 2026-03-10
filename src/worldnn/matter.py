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
