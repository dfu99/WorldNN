"""obj-029: Signal-dependent vs Gaussian noise sensitivity (Reviewer D q2).

Reviewer D asks: "Real organisms use signal-dependent noise. Your channel
adds Gaussian noise. Does the conclusion change?"

Without retraining (no GPU access right now), we can answer empirically by
recomputing the linear-probe recon ceiling R²(state | obs) under both noise
models on the same RockPushMatter. If the ceiling is robust, the
information-theoretic argument from §5.7 carries through. If it breaks,
we have a scope finding to disclose.

Two noise models on the same emission y = state·proj + bias:
  - Gaussian:        y' = y + σ·N(0, I)            (current paper)
  - Signal-dep:     y' = y + c·|y|·N(0, I)         (Weber-law style)

For each noise level, compute R²(state | y') via linear probe. Plot both
curves; report the ratio at matched effective noise.
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


def linear_probe_r2(x, y):
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    beta, *_ = np.linalg.lstsq(x_aug, y, rcond=None)
    pred = x_aug @ beta
    ss_res = ((y - pred) ** 2).sum(axis=0)
    ss_tot = ((y - y.mean(axis=0)) ** 2).sum(axis=0)
    return float((1 - ss_res / (ss_tot + 1e-12)).mean())


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    matter = RockPushMatter(emission_dim=8, action_dim=2, seed_dim=4)
    n = 3000
    dev = torch.device("cpu")

    # Sample clean (state, emission) pairs once
    states = matter.reset_state(n, dev)
    seeds = torch.randn(n, matter.seed_dim, device=dev)
    actions = torch.randn(n, matter.action_dim, device=dev) * 0.5
    with torch.no_grad():
        next_state, emission, _ = matter(states, seeds, actions)
    s = next_state.cpu().numpy()
    y = emission.cpu().numpy()

    # Sweep noise levels
    levels = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00])
    rng = np.random.default_rng(0)

    r2_gauss = []
    r2_sigdep = []
    eff_sigma_gauss = []
    eff_sigma_sigdep = []
    for c in levels:
        # Gaussian: y' = y + c*N(0, I)
        eps_g = rng.standard_normal(y.shape)
        y_g = y + c * eps_g
        r2_gauss.append(linear_probe_r2(y_g, s))
        eff_sigma_gauss.append(c)
        # Signal-dependent: y' = y + c*|y|*N(0, I) — Weber-law style
        eps_s = rng.standard_normal(y.shape)
        y_s = y + c * np.abs(y) * eps_s
        r2_sigdep.append(linear_probe_r2(y_s, s))
        # Effective sigma per-channel (mean of c*|y|)
        eff_sigma_sigdep.append(float((c * np.abs(y)).mean()))

    # === Figure ===
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    ax.plot(levels, r2_gauss, "o-", label="Gaussian (paper)", color="#1f77b4", linewidth=1.8)
    ax.plot(levels, r2_sigdep, "s--", label="Signal-dep (Weber)", color="#d62728", linewidth=1.8)
    ax.set_xlabel("noise coefficient c")
    ax.set_ylabel(r"linear-probe $R^2(\mathrm{state}\,|\,\mathrm{obs})$")
    ax.set_title("(A) Recon ceiling vs noise model")
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)

    ax = axes[1]
    ax.plot(eff_sigma_gauss, r2_gauss, "o-", label="Gaussian", color="#1f77b4", linewidth=1.8)
    ax.plot(eff_sigma_sigdep, r2_sigdep, "s--", label="Signal-dep", color="#d62728", linewidth=1.8)
    ax.set_xlabel(r"effective noise $\sigma$ (per-channel mean)")
    ax.set_ylabel(r"$R^2(\mathrm{state}\,|\,\mathrm{obs})$")
    ax.set_title("(B) Matched on effective noise level")
    ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(0, 1.05)

    plt.tight_layout()
    out_png = Path("results/obj029_signal_dep_noise.png")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"saved: {out_png}")

    # Numerics
    out_json = Path("results/obj029_signal_dep_noise.json")
    out_json.write_text(json.dumps({
        "n_samples": n,
        "levels": levels.tolist(),
        "r2_gaussian": r2_gauss,
        "r2_signal_dep": r2_sigdep,
        "effective_sigma_signal_dep": eff_sigma_sigdep,
        "max_abs_diff": float(max(abs(g - s) for g, s in zip(r2_gauss, r2_sigdep))),
    }, indent=2))
    print(f"saved: {out_json}")

    # Verdict
    diffs = [abs(g - s) for g, s in zip(r2_gauss, r2_sigdep)]
    print(f"\nMax |R²_gauss - R²_sigdep| across levels: {max(diffs):.4f}")
    print(f"Mean abs diff: {np.mean(diffs):.4f}")
    print(f"At c=0.5:  Gaussian R² = {r2_gauss[6]:.3f}, Signal-dep R² = {r2_sigdep[6]:.3f}")
    print(f"At c=0.10: Gaussian R² = {r2_gauss[2]:.3f}, Signal-dep R² = {r2_sigdep[2]:.3f}")


if __name__ == "__main__":
    main()
