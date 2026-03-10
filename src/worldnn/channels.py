"""Channels: the physical media that carry information from matter to observers.

Channels represent measurable quantities emitted by matter: light, sound,
heat, chemical gradients, etc. Each channel applies its own characteristic
noise and bandwidth limitation.

Architecture:
    ┌──────────────────────────────────┐
    │          Channel                 │
    │                                  │
    │  X (emission) ──► linear ──► noise ──► Y (channel output)
    │                   transform     addition
    └──────────────────────────────────┘

The channel is NOT learned — it represents physical constraints.
We parameterize it to sweep over different levels of information loss.
"""

import torch
import torch.nn as nn


class Channel(nn.Module):
    """A noisy communication channel between matter and environment.

    Models bandwidth limitation (linear projection) and additive noise.
    Not learned — represents fixed physical constraints of the medium.
    """

    def __init__(
        self,
        input_dim: int = 4,
        output_dim: int = 4,
        noise_std: float = 0.1,
        bandwidth: float = 1.0,
    ):
        super().__init__()
        self.noise_std = noise_std
        self.bandwidth = bandwidth
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Fixed linear transform (not learned) — represents physical medium
        # Initialize as scaled identity + small random for mixing
        weight = torch.eye(input_dim, output_dim) * bandwidth
        if input_dim != output_dim:
            weight = torch.randn(input_dim, output_dim) * 0.5 * bandwidth
        self.register_buffer("weight", weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transmit signal through noisy channel.

        Args:
            x: [batch, input_dim] raw emission from matter

        Returns:
            y: [batch, output_dim] noisy, bandwidth-limited signal
        """
        # Bandwidth limitation (linear transform)
        y = x @ self.weight

        # Additive Gaussian noise
        if self.noise_std > 0:
            noise = torch.randn_like(y) * self.noise_std
            y = y + noise

        return y

    def theoretical_capacity(self) -> float:
        """Shannon capacity of the channel (bits), assuming Gaussian."""
        # C = 0.5 * log2(1 + SNR) per dimension
        snr = (self.bandwidth**2) / (self.noise_std**2 + 1e-10)
        import math

        return 0.5 * self.output_dim * math.log2(1 + snr)
