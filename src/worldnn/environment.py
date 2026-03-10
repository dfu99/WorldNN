"""Environment: VAE that encodes/decodes channel signals.

The environment sits between raw channel outputs and the organism.
It represents the medium (atmosphere, water, distance) that further
compresses and distorts information. Modeled as a VAE to capture
learned lossy compression with a structured latent space.

Architecture:
    ┌───────────────────────────────────────────────┐
    │              Environment VAE                   │
    │                                                │
    │  Y (channel) ──► Encoder ──► μ, σ ──► z ──► Decoder ──► Y_hat
    │                              (latent)                     │
    │                                                           │
    │  Organism sees: z (latent) — NOT Y directly               │
    │  Action feedback: A ──► Decoder ──► propagated back       │
    └───────────────────────────────────────────────┘

The latent dimension controls how much information the environment
preserves. This is a key parameter in the perturbation study.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnvironmentVAE(nn.Module):
    """VAE environment that compresses channel signals.

    The organism observes the latent z, not the raw channel output.
    The latent dimension controls environmental information preservation.
    """

    def __init__(
        self,
        channel_dim: int = 4,
        latent_dim: int = 2,
        hidden_size: int = 32,
        action_dim: int = 2,
    ):
        super().__init__()
        self.channel_dim = channel_dim
        self.latent_dim = latent_dim

        # Encoder: channel signal → latent
        self.encoder = nn.Sequential(
            nn.Linear(channel_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_size, latent_dim)
        self.logvar_head = nn.Linear(hidden_size, latent_dim)

        # Decoder: latent → reconstructed channel signal
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, channel_dim),
        )

        # Action propagation: how actions pass back through environment
        self.action_transform = nn.Sequential(
            nn.Linear(action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )

    def encode(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode channel signal to latent distribution."""
        h = self.encoder(y)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        return mu, logvar

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """Sample from latent distribution using reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent back to channel space."""
        return self.decoder(z)

    def forward(
        self, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full encode-decode pass.

        Args:
            y: [batch, channel_dim] channel signal

        Returns:
            z: [batch, latent_dim] latent (what organism sees)
            y_hat: [batch, channel_dim] reconstruction
            mu: [batch, latent_dim] latent mean
            logvar: [batch, latent_dim] latent log-variance
        """
        mu, logvar = self.encode(y)
        z = self.reparameterize(mu, logvar)
        y_hat = self.decode(z)
        return z, y_hat, mu, logvar

    def propagate_action(self, action: torch.Tensor) -> torch.Tensor:
        """Transform action as it passes back through the environment."""
        return self.action_transform(action)

    def vae_loss(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float = 1.0,
    ) -> torch.Tensor:
        """VAE ELBO loss = reconstruction + beta * KL divergence."""
        recon = F.mse_loss(y_hat, y, reduction="mean")
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon + beta * kl
