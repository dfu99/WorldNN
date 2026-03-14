#!/usr/bin/env python3
"""Theoretical bounds: minimum embedding dimension for reliable state flipping.

Derives analytical bounds on the minimum organism capacity needed to
flip a 1-bit matter state, as a function of channel noise and
environment compression.

Key results:
1. Channel capacity: C = 0.5 * log2(1 + SNR) per dimension
2. Environment bottleneck: I(S;Z) ≤ min(I(S;Y), env_latent_dim * max_rate)
3. Fano's inequality: P(error) ≥ 1 - [I(S;E) + 1] / log2(|S|)
4. Minimum embedding dim: d_min such that d_min dimensions can carry
   enough MI to distinguish S with P(error) < threshold

For our binary state (|S|=2, H(S)=1 bit):
  - Need I(S;E) > 1 - h(P_target) where h is binary entropy
  - For 90% success: I(S;E) > 1 - h(0.1) ≈ 0.531 bits
  - For 99% success: I(S;E) > 1 - h(0.01) ≈ 0.919 bits
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def binary_entropy(p: float) -> float:
    """Binary entropy H(p) in bits."""
    if p <= 0 or p >= 1:
        return 0.0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def channel_capacity_awgn(noise_std: float, signal_power: float = 1.0, n_dims: int = 4) -> float:
    """AWGN channel capacity in bits (total across dimensions).

    C = n/2 * log2(1 + P/σ²) for n-dimensional Gaussian channel.
    """
    if noise_std <= 0:
        return float('inf')
    snr = signal_power / (noise_std ** 2)
    return n_dims * 0.5 * np.log2(1 + snr)


def vae_rate(latent_dim: int, beta: float = 0.1) -> float:
    """Approximate maximum rate through VAE bottleneck.

    For a β-VAE with latent_dim dimensions, the rate is bounded by:
    R ≤ latent_dim * (rate_per_dim)

    With β=0.1, the KL term is weakly penalized, so each latent dim
    can carry close to its theoretical max. For Gaussian latents with
    unit prior, each dim carries at most ~1-2 bits for typical learned
    posteriors. We use an empirical upper bound of 1.5 bits/dim.
    """
    rate_per_dim = 1.5  # empirical bits per latent dim (approximate)
    return latent_dim * rate_per_dim


def min_mi_for_success(target_success: float) -> float:
    """Minimum mutual information I(S;E) needed for given success rate.

    From Fano's inequality for binary S:
    P(error) ≥ (H(S) - I(S;E) - 1) / log2(|S|)

    Rearranging for binary (H(S)=1, |S|=2):
    I(S;E) ≥ 1 - h(P_error) where P_error = 1 - target_success
    """
    p_error = 1 - target_success
    return max(0.0, 1 - binary_entropy(p_error))


def compute_mi_chain(noise_std: float, env_latent_dim: int,
                     emission_dim: int = 4, channel_dim: int = 4) -> dict:
    """Compute MI bounds at each stage of the perception chain.

    Returns theoretical upper bounds on MI at each stage.
    """
    # I(S; X): matter emission encodes binary state deterministically
    # With 4-dim emission, the two states produce distinct patterns
    # I(S;X) = H(S) = 1 bit (for binary state)
    i_sx = 1.0  # bits

    # I(S; Y): channel adds Gaussian noise
    # I(S;Y) ≤ min(I(S;X), C_channel)
    c_channel = channel_capacity_awgn(noise_std, n_dims=channel_dim)
    i_sy = min(i_sx, c_channel)

    # I(S; Z): environment VAE compresses to latent
    # I(S;Z) ≤ min(I(S;Y), R_vae)
    r_vae = vae_rate(env_latent_dim)
    i_sz = min(i_sy, r_vae)

    # I(S; E): organism embedding
    # I(S;E) ≤ I(S;Z) (data processing inequality)
    # Additionally limited by embedding capacity
    i_se = i_sz  # upper bound (organism is the learner, not a fixed channel)

    return {
        "I(S;X)": i_sx,
        "I(S;Y)": i_sy,
        "I(S;Z)": i_sz,
        "I(S;E)": i_se,
        "C_channel": c_channel,
        "R_vae": r_vae,
    }


def compute_min_embedding_dim(noise_std: float, env_latent_dim: int,
                               target_success: float = 0.9) -> dict:
    """Compute minimum embedding dimension needed for target success.

    The embedding must have enough capacity to carry the required MI.
    Each embedding dimension (with tanh activation, bounded [-1,1]) can
    carry at most ~1 bit of information about binary S.

    Returns the bound and the bottleneck location.
    """
    chain = compute_mi_chain(noise_std, env_latent_dim)
    mi_needed = min_mi_for_success(target_success)

    # Available MI at the organism's input
    available_mi = chain["I(S;Z)"]

    # Is there enough information available?
    if available_mi < mi_needed:
        return {
            "feasible": False,
            "bottleneck": "channel" if chain["C_channel"] < chain["R_vae"] else "environment",
            "available_mi": available_mi,
            "needed_mi": mi_needed,
            "min_embed_dim": float('inf'),
        }

    # Each embedding dim carries ~1 bit max for binary classification
    # But with tanh activation and continuous values, effective rate is lower
    effective_bits_per_dim = 0.8  # empirical: tanh-bounded dim ≈ 0.8 bits
    min_dim = max(1, int(np.ceil(mi_needed / effective_bits_per_dim)))

    return {
        "feasible": True,
        "bottleneck": "none" if available_mi > 2 * mi_needed else
                     ("channel" if chain["C_channel"] < chain["R_vae"] else "environment"),
        "available_mi": available_mi,
        "needed_mi": mi_needed,
        "min_embed_dim": min_dim,
    }


def plot_theoretical_bounds(results_dir: str = "results"):
    """Generate comprehensive theoretical bounds visualizations."""
    os.makedirs(results_dir, exist_ok=True)

    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    # ── Panel A: MI chain as function of noise ──
    ax_a = fig.add_subplot(gs[0, 0])
    noise_range = np.logspace(-2, 1, 100)

    for lat, ls in [(1, "-"), (2, "--"), (4, ":")]:
        mi_vals = [compute_mi_chain(n, lat) for n in noise_range]
        ax_a.plot(noise_range, [m["I(S;Y)"] for m in mi_vals],
                  ls=ls, color="#1565C0", linewidth=2,
                  label=f"I(S;Y) lat={lat}" if lat == 1 else None)
        ax_a.plot(noise_range, [m["I(S;Z)"] for m in mi_vals],
                  ls=ls, color="#E65100", linewidth=2,
                  label=f"I(S;Z) lat={lat}")

    ax_a.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5, label="H(S)=1 bit")
    ax_a.set_xlabel("Channel Noise σ")
    ax_a.set_ylabel("Mutual Information (bits)")
    ax_a.set_title("A. Information Through the Chain", fontweight="bold")
    ax_a.legend(fontsize=8)
    ax_a.grid(True, alpha=0.3)
    ax_a.set_xscale("log")
    ax_a.set_ylim(0, 1.2)

    # ── Panel B: Channel capacity ──
    ax_b = fig.add_subplot(gs[0, 1])
    caps = [channel_capacity_awgn(n) for n in noise_range]
    ax_b.plot(noise_range, caps, color="#1565C0", linewidth=2)
    ax_b.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="1 bit (needed)")

    # Mark where capacity crosses 1 bit
    cross_idx = np.argmin(np.abs(np.array(caps) - 1.0))
    ax_b.axvline(x=noise_range[cross_idx], color="red", linestyle=":", alpha=0.3)
    ax_b.annotate(f"σ≈{noise_range[cross_idx]:.2f}",
                  xy=(noise_range[cross_idx], 1.0), xytext=(noise_range[cross_idx]*3, 2),
                  arrowprops=dict(arrowstyle="->"), fontsize=10)

    ax_b.set_xlabel("Channel Noise σ")
    ax_b.set_ylabel("Channel Capacity (bits)")
    ax_b.set_title("B. AWGN Channel Capacity (4D)", fontweight="bold")
    ax_b.legend(fontsize=10)
    ax_b.grid(True, alpha=0.3)
    ax_b.set_xscale("log")
    ax_b.set_yscale("log")

    # ── Panel C: Min MI needed for various success rates ──
    ax_c = fig.add_subplot(gs[0, 2])
    success_range = np.linspace(0.51, 0.999, 200)
    mi_needed = [min_mi_for_success(s) for s in success_range]
    ax_c.plot(success_range, mi_needed, color="#7B1FA2", linewidth=2)
    for target, marker in [(0.75, "o"), (0.9, "s"), (0.95, "D"), (0.99, "^")]:
        mi = min_mi_for_success(target)
        ax_c.plot(target, mi, marker, color="#E65100", markersize=8, zorder=5)
        ax_c.annotate(f"{target:.0%}: {mi:.3f}b", xy=(target, mi),
                      xytext=(target - 0.15, mi + 0.05), fontsize=9)

    ax_c.set_xlabel("Target Success Rate")
    ax_c.set_ylabel("Min I(S;E) Needed (bits)")
    ax_c.set_title("C. Fano's Inequality Bound", fontweight="bold")
    ax_c.grid(True, alpha=0.3)

    # ── Panel D: Feasibility map (noise × env_lat) ──
    ax_d = fig.add_subplot(gs[1, 0])
    noise_grid = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    lat_grid = [1, 2, 3, 4, 6, 8]
    feas_matrix = np.zeros((len(lat_grid), len(noise_grid)))

    for i, lat in enumerate(lat_grid):
        for j, noise in enumerate(noise_grid):
            result = compute_min_embedding_dim(noise, lat, target_success=0.9)
            if result["feasible"]:
                feas_matrix[i, j] = result["available_mi"]
            else:
                feas_matrix[i, j] = result["available_mi"]

    im = ax_d.imshow(feas_matrix, aspect="auto", cmap="YlGnBu", origin="lower")
    ax_d.set_xticks(range(len(noise_grid)))
    ax_d.set_xticklabels([f"{n}" for n in noise_grid], rotation=45, fontsize=8)
    ax_d.set_yticks(range(len(lat_grid)))
    ax_d.set_yticklabels([str(l) for l in lat_grid])
    ax_d.set_xlabel("Channel Noise σ")
    ax_d.set_ylabel("Env Latent Dim")
    ax_d.set_title("D. Available MI (bits)", fontweight="bold")

    for i in range(len(lat_grid)):
        for j in range(len(noise_grid)):
            v = feas_matrix[i, j]
            color = "white" if v > 0.6 else "black"
            ax_d.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color=color)
    plt.colorbar(im, ax=ax_d, shrink=0.8)

    # ── Panel E: Predicted success vs. empirical (from obj-002/005) ──
    ax_e = fig.add_subplot(gs[1, 1])

    # Theoretical prediction: success ≈ sigmoid(k * (available_MI - needed_MI))
    # This is a rough model — actual success depends on optimizer too
    noise_pred = np.logspace(-2, 0.7, 50)
    for lat, color, ls in [(1, "#E65100", "-"), (2, "#1565C0", "--"), (4, "#388E3C", ":")]:
        predicted_success = []
        for n in noise_pred:
            chain = compute_mi_chain(n, lat)
            mi = chain["I(S;Z)"]
            mi_needed_90 = min_mi_for_success(0.9)
            # Sigmoid model: s ≈ 1/(1 + exp(-k*(MI - threshold)))
            s = 1.0 / (1.0 + np.exp(-8 * (mi - mi_needed_90)))
            predicted_success.append(s)
        ax_e.plot(noise_pred, predicted_success, ls=ls, color=color,
                  linewidth=2, label=f"lat={lat}")

    # Overlay empirical data points from obj-005 (approximate)
    empirical = {
        (1, 0.01): 0.72, (1, 0.1): 0.80, (1, 0.5): 0.86, (1, 1.0): 0.70, (1, 2.0): 0.58,
        (2, 0.01): 0.88, (2, 0.1): 0.87, (2, 0.5): 0.85, (2, 1.0): 0.75, (2, 2.0): 0.63,
        (4, 0.01): 0.88, (4, 0.1): 0.87, (4, 0.5): 0.84, (4, 1.0): 0.74,
    }
    for (lat, noise), success in empirical.items():
        color = {1: "#E65100", 2: "#1565C0", 4: "#388E3C"}[lat]
        ax_e.plot(noise, success, "o", color=color, markersize=6, alpha=0.7)

    ax_e.set_xlabel("Channel Noise σ")
    ax_e.set_ylabel("Predicted Success Rate")
    ax_e.set_title("E. Theory vs Empirical (PPO)", fontweight="bold")
    ax_e.legend(fontsize=10)
    ax_e.grid(True, alpha=0.3)
    ax_e.set_xscale("log")
    ax_e.set_ylim(-0.05, 1.05)

    # Note about the theory-empirical gap
    ax_e.text(0.02, 0.02,
              "Lines: theory\nDots: empirical (obj-005)\nGap at lat=1, low noise:\nstochastic resonance",
              transform=ax_e.transAxes, fontsize=8,
              bbox=dict(boxstyle="round", facecolor="#FFF9C4", alpha=0.8))

    # ── Panel F: Bottleneck identification ──
    ax_f = fig.add_subplot(gs[1, 2])

    # For each noise level, show which stage is the bottleneck
    noise_fine = np.logspace(-2, 1, 200)
    for lat, color in [(1, "#E65100"), (2, "#1565C0"), (4, "#388E3C")]:
        bottleneck_locs = []
        for n in noise_fine:
            chain = compute_mi_chain(n, lat)
            # Bottleneck = stage with smallest MI
            if chain["C_channel"] < chain["R_vae"]:
                # Channel is bottleneck: how much does it reduce?
                reduction = 1.0 - chain["I(S;Y)"]
            else:
                # VAE is bottleneck
                reduction = chain["I(S;Y)"] - chain["I(S;Z)"]
            bottleneck_locs.append(reduction)
        ax_f.plot(noise_fine, bottleneck_locs, color=color, linewidth=2, label=f"lat={lat}")

    ax_f.set_xlabel("Channel Noise σ")
    ax_f.set_ylabel("MI Lost at Bottleneck (bits)")
    ax_f.set_title("F. Where Information Is Lost", fontweight="bold")
    ax_f.legend(fontsize=10)
    ax_f.grid(True, alpha=0.3)
    ax_f.set_xscale("log")

    fig.suptitle(
        "Theoretical Bounds on Perception-Action Loops\n"
        "Information-Theoretic Limits for Binary State Flipping",
        fontsize=15, fontweight="bold", y=0.98,
    )

    out = f"{results_dir}/theoretical_bounds.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def print_bounds_table():
    """Print a table of theoretical bounds for common configurations."""
    print("\n" + "=" * 80)
    print("THEORETICAL BOUNDS: Min Embedding Dim for 90% Success")
    print("=" * 80)
    print(f"{'Noise':>8} {'Lat':>5} {'C_chan':>8} {'R_vae':>8} {'I(S;Z)':>8} "
          f"{'Feasible':>10} {'MinDim':>8} {'Bottleneck':>12}")
    print("-" * 80)

    for noise in [0.01, 0.1, 0.5, 1.0, 2.0]:
        for lat in [1, 2, 4]:
            chain = compute_mi_chain(noise, lat)
            result = compute_min_embedding_dim(noise, lat, target_success=0.9)
            dim_str = str(result["min_embed_dim"]) if result["feasible"] else "∞"
            print(f"{noise:>8.2f} {lat:>5} {chain['C_channel']:>8.3f} "
                  f"{chain['R_vae']:>8.3f} {chain['I(S;Z)']:>8.3f} "
                  f"{'YES' if result['feasible'] else 'NO':>10} "
                  f"{dim_str:>8} {result['bottleneck']:>12}")

    print("=" * 80)
    print("\nNote: Bounds assume optimal coding. Actual RL performance")
    print("depends on optimizer (PPO >> REINFORCE) and training dynamics.")
    print("The stochastic resonance at lat=1 is NOT predicted by these")
    print("bounds — it's an optimizer phenomenon, not information-theoretic.\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    print_bounds_table()
    plot_theoretical_bounds(args.results_dir)
