"""Organism: sensorimotor complex + embedding + action policy.

The organism is the agent that must:
1. Sample a subset of environmental signals (sensory filtering)
2. Compress them into an internal embedding (bounded capacity)
3. Produce actions based on the embedding (policy)

The embedding dimension is the key variable: how much internal model
capacity does the organism need to reliably flip 1 bit of matter state?

Architecture:
    ┌──────────────────────────────────────────────────────┐
    │                    Organism                           │
    │                                                       │
    │  z (from env) ──► Sensory ──► Embedding ──► Policy ──► A (action)
    │                   Filter      (bottleneck)              │
    │                   (subset)    dim = embed_dim            │
    │                                                       │
    │  Also receives: reward signal for RL training           │
    └──────────────────────────────────────────────────────┘

The sensory filter selects/weights which latent dimensions to attend to.
The embedding is a hard bottleneck — the organism's "brain capacity."
The policy maps embedding to action using a simple MLP.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Organism(nn.Module):
    """An organism with bounded internal capacity.

    The embedding_dim parameter controls the organism's "brain size" —
    the maximum amount of information it can maintain about the world.
    """

    def __init__(
        self,
        sensory_dim: int = 2,
        embedding_dim: int = 4,
        action_dim: int = 2,
        hidden_size: int = 32,
    ):
        super().__init__()
        self.sensory_dim = sensory_dim
        self.embedding_dim = embedding_dim
        self.action_dim = action_dim

        # Sensory filter: selects/weights environmental signals
        self.sensory_filter = nn.Sequential(
            nn.Linear(sensory_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, sensory_dim),
            nn.Sigmoid(),  # attention-like gating
        )

        # Embedding: compress sensory input to bounded representation
        # This is the key bottleneck — organism's internal model capacity
        self.encoder = nn.Sequential(
            nn.Linear(sensory_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embedding_dim),
            nn.Tanh(),  # bounded embedding
        )

        # Policy: embedding → action
        self.policy = nn.Sequential(
            nn.Linear(embedding_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )

        # Value head for advantage estimation
        self.value_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full perception-to-action pass.

        Args:
            z: [batch, sensory_dim] latent from environment

        Returns:
            action: [batch, action_dim] action to take
            embedding: [batch, embedding_dim] internal representation
            value: [batch] estimated value
        """
        # Sensory filtering — which signals to attend
        gate = self.sensory_filter(z)
        filtered = z * gate

        # Compress to embedding (the information bottleneck)
        embedding = self.encoder(filtered)

        # Produce action
        action = self.policy(embedding)

        # Value estimate
        value = self.value_head(embedding).squeeze(-1)

        return action, embedding, value

    def get_action_distribution(
        self, z: torch.Tensor, action_std: float = 0.5
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get action mean and sample with exploration noise."""
        action_mean, embedding, value = self.forward(z)

        if self.training:
            noise = torch.randn_like(action_mean) * action_std
            action = action_mean + noise
        else:
            action = action_mean

        return action, value


class PredictiveOrganism(Organism):
    """Organism with predictive processing: predicts next observation.

    Extends the base Organism with a prediction head that forecasts the
    next z given current embedding + action. The prediction error serves
    as an auxiliary learning signal (self-supervised), complementing the
    RL reward. This implements a simplified version of predictive coding /
    active inference.

    Architecture addition:
        embedding + action ──► Predictor MLP ──► predicted_next_z
        prediction_error = ||predicted_next_z - actual_next_z||²
    """

    def __init__(
        self,
        sensory_dim: int = 2,
        embedding_dim: int = 4,
        action_dim: int = 2,
        hidden_size: int = 32,
    ):
        super().__init__(
            sensory_dim=sensory_dim,
            embedding_dim=embedding_dim,
            action_dim=action_dim,
            hidden_size=hidden_size,
        )

        # Prediction head: given embedding + action, predict next z
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, sensory_dim),
        )

    def predict_next(
        self, embedding: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Predict next observation z given current embedding and action.

        Args:
            embedding: [batch, embedding_dim] current internal representation
            action: [batch, action_dim] action being taken

        Returns:
            predicted_z: [batch, sensory_dim] predicted next observation
        """
        combined = torch.cat([embedding, action], dim=-1)
        return self.predictor(combined)

    def forward_with_prediction(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass including next-observation prediction.

        Returns:
            action, embedding, value, predicted_next_z
        """
        action, embedding, value = self.forward(z)
        predicted_next_z = self.predict_next(embedding, action)
        return action, embedding, value, predicted_next_z
