"""Utility functions for mutual information estimation and analysis."""

import torch
import numpy as np
from scipy.special import digamma
from scipy.spatial import cKDTree


def estimate_mi_ksg(x: np.ndarray, y: np.ndarray, k: int = 3) -> float:
    """Estimate mutual information using KSG estimator (Kraskov et al. 2004).

    Args:
        x: [N, dx] samples
        y: [N, dy] samples
        k: number of nearest neighbors

    Returns:
        Estimated I(X; Y) in nats

    Warning:
        Audit 2026-05-05 verified this implementation under-reports MI on
        2-D Gaussian samples by 30-100%: at rho=0.3 (truth 0.05) and 0.6
        (truth 0.22) it returns the clamped-zero output; at rho=0.9
        (truth 0.83) it recovers ~66%. The ``max(0.0, mi)`` clamp masks
        negative raw outputs that would otherwise signal failed estimation.
        For paper-grade MI claims (especially with high-dimensional y),
        prefer the linear-probe Gaussian-MI lower bound used in
        ``experiments/obj025_mi_vs_sensory.py`` instead. Current behavior
        is pinned by ``TestKSGEstimator`` in ``tests/test_components.py``.
    """
    n = x.shape[0]
    if n < k + 1:
        return 0.0

    # Ensure 2D
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    xy = np.concatenate([x, y], axis=1)

    tree_xy = cKDTree(xy)
    tree_x = cKDTree(x)
    tree_y = cKDTree(y)

    # Find k-th nearest neighbor distance in joint space
    dists, _ = tree_xy.query(xy, k=k + 1)  # +1 because self is included
    eps = dists[:, -1]  # k-th neighbor distance

    # Count neighbors within eps in marginal spaces
    nx = np.array(
        [tree_x.query_ball_point(x[i], eps[i] - 1e-10, return_length=True) - 1 for i in range(n)]
    )
    ny = np.array(
        [tree_y.query_ball_point(y[i], eps[i] - 1e-10, return_length=True) - 1 for i in range(n)]
    )

    # KSG estimator
    mi = digamma(k) - np.mean(digamma(nx + 1) + digamma(ny + 1)) + digamma(n)
    return max(0.0, mi)


def compute_chain_mi(
    state: torch.Tensor,
    emission: torch.Tensor,
    channel_out: torch.Tensor,
    z: torch.Tensor,
    embedding: torch.Tensor,
    n_samples: int = 1000,
) -> dict[str, float]:
    """Compute mutual information at each stage of the perception chain.

    Returns I(S;X), I(S;Y), I(S;Z), I(S;E) where:
        S = hidden state, X = emission, Y = channel output,
        Z = env latent, E = organism embedding
    """
    # Convert to numpy, take subset for speed
    idx = np.random.choice(len(state), min(n_samples, len(state)), replace=False)
    s = state[idx].cpu().numpy().reshape(-1, 1)
    x = emission[idx].cpu().numpy()
    y = channel_out[idx].cpu().numpy()
    z_np = z[idx].cpu().numpy()
    e = embedding[idx].cpu().numpy()

    return {
        "I(S;X)": estimate_mi_ksg(s, x),
        "I(S;Y)": estimate_mi_ksg(s, y),
        "I(S;Z)": estimate_mi_ksg(s, z_np),
        "I(S;E)": estimate_mi_ksg(s, e),
    }
