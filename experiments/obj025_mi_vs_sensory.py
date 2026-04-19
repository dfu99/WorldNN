"""
obj-025 T3: Rate-distortion bound for sensory-capacity tradeoff.

Compute information-content measures of obs[:sensory_dim] about state S on
RockPushMatter for sensory_dim ∈ {2,4,8,16}. Overlay with peak SA from obj-024.

Uses TWO complementary estimators (KSG gets wobbly in high-dim):
  1. Normalized-KSG I(S; obs) in nats — nonparametric
  2. Linear-probe R²(state | obs) — robust, interpretable, maps to Gaussian MI

Addresses Reviewer C: "Define your doubt as I(X;W) - I(Z;W). Estimate it with
KSG. Show me the curve."
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from worldnn.matter import RockPushMatter
from worldnn.utils import estimate_mi_ksg


def zscore(x: np.ndarray) -> np.ndarray:
    std = x.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return (x - x.mean(axis=0, keepdims=True)) / std


def linear_probe_r2(x: np.ndarray, y: np.ndarray) -> float:
    """R² of predicting y from x via linear regression (averaged per y-dim)."""
    if x.ndim == 1: x = x.reshape(-1, 1)
    if y.ndim == 1: y = y.reshape(-1, 1)
    x = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    y_pred = x @ beta
    ss_res = ((y - y_pred) ** 2).sum(axis=0)
    ss_tot = ((y - y.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    return float(r2.mean())


def gaussian_mi_from_r2(r2: float, state_dim: int) -> float:
    """Lower-bound MI (nats) from avg linear-probe R² assuming Gaussian residuals."""
    r2 = min(max(r2, 0.0), 0.999999)
    return -0.5 * state_dim * np.log(1.0 - r2)


def sample_state_emission(matter: RockPushMatter, n_samples: int, device: str = "cpu"):
    dev = torch.device(device)
    matter = matter.to(dev)
    states = matter.reset_state(n_samples, dev)
    seeds = torch.randn(n_samples, matter.seed_dim, device=dev)
    actions = torch.randn(n_samples, matter.action_dim, device=dev) * 0.5
    with torch.no_grad():
        next_state, emission, _ = matter(states, seeds, actions)
    return next_state.cpu().numpy(), emission.cpu().numpy()


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    matter = RockPushMatter(
        emission_dim=16, action_dim=2, seed_dim=4,
        move_speed=0.15, push_radius=0.2, push_strength=0.12,
    )

    sensory_dims = [2, 4, 8, 16]
    n_samples = 3000

    print(f"Sampling {n_samples} (state, emission) pairs...")
    states, emissions = sample_state_emission(matter, n_samples, "cpu")
    print(f"  states shape: {states.shape}")
    print(f"  emissions shape: {emissions.shape}")

    # Normalize for KSG stability
    s_norm = zscore(states)
    e_norm = zscore(emissions)

    print("\nComputing I(S; obs) via normalized-KSG (k=5) + linear probe R²...")
    mi_ksg, r2_probe, mi_gauss = {}, {}, {}
    state_dim = states.shape[1]
    for sd in sensory_dims:
        obs = e_norm[:, :sd]
        mi_ksg[sd] = estimate_mi_ksg(s_norm, obs, k=5)
        r2_probe[sd] = linear_probe_r2(emissions[:, :sd], states)
        mi_gauss[sd] = gaussian_mi_from_r2(r2_probe[sd], state_dim)
        print(f"  sensory_dim={sd:2d}: "
              f"KSG={mi_ksg[sd]:.3f} nats | "
              f"R²(S|obs)={r2_probe[sd]:.3f} → Gauss-MI={mi_gauss[sd]:.3f} nats")

    # Load obj-024 peak SA per sensory_dim
    with open("results/sensory_capacity_checkpoint.json") as f:
        obj024 = json.load(f)
    from collections import defaultdict
    cells = defaultdict(list)
    for r in obj024:
        cells[(r["sensory_dim"], r["embedding_dim"])].append(r["SA"])
    peak_sa = {}
    for sd in sensory_dims:
        means = [np.mean(cells[(sd, ed)]) for ed in [2, 4, 8, 16, 32] if (sd, ed) in cells]
        peak_sa[sd] = max(means) if means else 0.0

    # === Figure ===
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "obj-025 T3: Information Content Bounds SA Ceiling\n"
        "Rate-distortion argument for sensory-capacity tradeoff",
        fontsize=13, fontweight="bold",
    )

    sd_arr = np.array(sensory_dims)
    mi_ksg_arr = np.array([mi_ksg[sd] for sd in sensory_dims])
    r2_arr = np.array([r2_probe[sd] for sd in sensory_dims])
    mi_gauss_arr = np.array([mi_gauss[sd] for sd in sensory_dims])
    sa_arr = np.array([peak_sa[sd] for sd in sensory_dims])

    ax = axes[0]
    ax.plot(sd_arr, mi_ksg_arr, "o-", label="KSG I(S; obs)", color="#1f77b4", linewidth=2, markersize=9)
    ax.plot(sd_arr, mi_gauss_arr, "s--", label="Gaussian-MI (linear probe)", color="#2ca02c", linewidth=2, markersize=9)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sensory_dims); ax.set_xticklabels(sensory_dims)
    ax.set_xlabel("Sensory Dim"); ax.set_ylabel("Mutual Information (nats)")
    ax.set_title("A. I(S; obs) vs Sensory Dim")
    ax.grid(alpha=0.3); ax.legend()

    ax = axes[1]
    ax.plot(sd_arr, r2_arr, "D-", color="#9467bd", linewidth=2, markersize=9)
    for x, y in zip(sd_arr, r2_arr):
        ax.annotate(f"R²={y:.3f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sensory_dims); ax.set_xticklabels(sensory_dims)
    ax.set_xlabel("Sensory Dim"); ax.set_ylabel("R² (linear state recovery)")
    ax.set_title("B. Linear Decodability of State")
    ax.set_ylim(0, 1.05); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(mi_gauss_arr, sa_arr, "s-", color="#d62728", linewidth=2, markersize=10)
    for sd, x, y in zip(sensory_dims, mi_gauss_arr, sa_arr):
        ax.annotate(f"s={sd}", (x, y), textcoords="offset points", xytext=(8, -4), fontsize=10)
    ax.set_xlabel("Gaussian-MI I(S; obs) (nats)")
    ax.set_ylabel("Peak SA achieved")
    ax.set_title("C. Peak SA vs Information Content")
    ax.grid(alpha=0.3); ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    out = Path("results/obj025_mi_vs_sensory.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nSaved: {out}")

    # Save numeric results
    r_ksg = float(np.corrcoef(mi_ksg_arr, sa_arr)[0, 1]) if mi_ksg_arr.std() > 1e-6 else 0.0
    r_gauss = float(np.corrcoef(mi_gauss_arr, sa_arr)[0, 1])
    r_r2 = float(np.corrcoef(r2_arr, sa_arr)[0, 1])

    out_json = Path("results/obj025_mi_vs_sensory.json")
    with out_json.open("w") as f:
        json.dump({
            "sensory_dims": sensory_dims,
            "KSG_MI_nats": {str(k): float(v) for k, v in mi_ksg.items()},
            "linear_probe_R2": {str(k): float(v) for k, v in r2_probe.items()},
            "Gaussian_MI_nats": {str(k): float(v) for k, v in mi_gauss.items()},
            "peak_SA_by_sensory_dim": {str(k): float(v) for k, v in peak_sa.items()},
            "correlations_with_peak_SA": {
                "KSG_MI": r_ksg, "Gaussian_MI": r_gauss, "R2_probe": r_r2,
            },
            "n_samples": n_samples,
        }, f, indent=2)
    print(f"Saved: {out_json}")

    print(f"\nCorrelation(Gaussian-MI, peak_SA) = {r_gauss:.4f}")
    print(f"Correlation(R²_probe, peak_SA) = {r_r2:.4f}")


if __name__ == "__main__":
    main()
