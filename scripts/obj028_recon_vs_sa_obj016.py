"""obj-028: Dreamer-style recon-loss vs SA on the obj-016 wide grid.

Audit T28: Reviewer A asks whether SA adds signal beyond reconstruction
loss. obj-025 T4 answered ΔR²=0.004 on the obj-024 narrow grid (where
sensory ∈ {2,4,8,16} compressed dynamic range). The headline grid is
obj-016: 245 configs across 7 perception conditions × 5 embed × 7 seeds.

For each perception condition, compute the linear-probe ceiling
R²(state | processed_obs). For VAE conditions this is already in
ci_at_scale.json's vae_probe_results. For oracle/noise/raw_emission,
compute it via fresh sampling.

Then per-config (level × embed × seed) we have:
  - C_i (= SA on this grid)
  - avg_dist_last100  (task error)
  - recon_R²          (ceiling for the level)

Compare:
  - r(SA, dist)         (full-grid correlation)
  - r(recon_R², dist)
  - partial r(SA | recon_R², dist)
  - ΔR² of multiple regression on dist with/without SA.
"""
import json
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import torch
from worldnn.matter import RockPushMatter
from worldnn.environment import EnvironmentVAE


def linear_probe_r2(x: np.ndarray, y: np.ndarray) -> float:
    """R² predicting y from x via linear regression (per-dim averaged)."""
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    beta, *_ = np.linalg.lstsq(x_aug, y, rcond=None)
    y_pred = x_aug @ beta
    ss_res = ((y - y_pred) ** 2).sum(axis=0)
    ss_tot = ((y - y.mean(axis=0)) ** 2).sum(axis=0)
    r2_per_dim = 1.0 - ss_res / (ss_tot + 1e-12)
    return float(r2_per_dim.mean())


def sample_state_obs(matter, perception_fn, n=2000, device="cpu"):
    """Sample n (state, obs) pairs through a perception function."""
    dev = torch.device(device)
    states = matter.reset_state(n, dev)
    seeds = torch.randn(n, matter.seed_dim, device=dev)
    actions = torch.randn(n, matter.action_dim, device=dev) * 0.5
    with torch.no_grad():
        next_state, emission, _ = matter(states, seeds, actions)
    obs = perception_fn(next_state, emission)
    return next_state.cpu().numpy(), obs.cpu().numpy()


def compute_recon_ceilings(seed=42, n=2000):
    torch.manual_seed(seed)
    np.random.seed(seed)

    matter = RockPushMatter(emission_dim=8, action_dim=2, seed_dim=4)
    ceilings = {}

    # Oracle: direct state observation
    s, _ = sample_state_obs(matter, lambda ns, e: ns, n)
    ceilings["oracle"] = linear_probe_r2(s, s)

    # Oracle + noise 0.1
    s, _ = sample_state_obs(matter, lambda ns, e: ns + 0.1 * torch.randn_like(ns), n)
    s2, obs = sample_state_obs(matter,
                               lambda ns, e: ns + 0.1 * torch.randn_like(ns),
                               n)
    ceilings["oracle_noise0.1"] = linear_probe_r2(obs, s2)

    # Oracle + noise 0.5
    s2, obs = sample_state_obs(matter,
                               lambda ns, e: ns + 0.5 * torch.randn_like(ns),
                               n)
    ceilings["oracle_noise0.5"] = linear_probe_r2(obs, s2)

    # Raw emission (8D, no VAE)
    s2, obs = sample_state_obs(matter, lambda ns, e: e, n)
    ceilings["raw_emission"] = linear_probe_r2(obs, s2)

    return ceilings


def partial_corr(x, y, z):
    x, y, z = np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    z_aug = np.concatenate([z, np.ones((z.shape[0], 1))], axis=1)
    bx, *_ = np.linalg.lstsq(z_aug, x, rcond=None)
    rx = x - z_aug @ bx
    by, *_ = np.linalg.lstsq(z_aug, y, rcond=None)
    ry = y - z_aug @ by
    if rx.std() < 1e-10 or ry.std() < 1e-10:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def r2_mult(X, y):
    X_aug = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    b, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
    pred = X_aug @ b
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return float(1.0 - ss_res / ss_tot)


def main():
    d = json.load(open("results/ci_at_scale.json"))
    rows = d["results"]
    vae_probes = d["vae_probe_results"]

    # Compute recon ceilings for non-VAE perception conditions
    print("Computing recon ceilings for non-VAE conditions...")
    extra = compute_recon_ceilings()
    print("  oracle              R²:", round(extra["oracle"], 3))
    print("  oracle_noise0.1     R²:", round(extra["oracle_noise0.1"], 3))
    print("  oracle_noise0.5     R²:", round(extra["oracle_noise0.5"], 3))
    print("  raw_emission        R²:", round(extra["raw_emission"], 3))
    print("  vae_mu_lat8 (cached)R²:", round(vae_probes["8"]["r2_mean"], 3))
    print("  vae_mu_lat16(cached)R²:", round(vae_probes["16"]["r2_mean"], 3))
    print("  vae_mu_lat32(cached)R²:", round(vae_probes["32"]["r2_mean"], 3))

    recon_r2_by_level = {
        "oracle":           extra["oracle"],
        "oracle_noise0.1":  extra["oracle_noise0.1"],
        "oracle_noise0.5":  extra["oracle_noise0.5"],
        "raw_emission":     extra["raw_emission"],
        "vae_mu_lat8":      vae_probes["8"]["r2_mean"],
        "vae_mu_lat16":     vae_probes["16"]["r2_mean"],
        "vae_mu_lat32":     vae_probes["32"]["r2_mean"],
    }

    # Build per-config arrays — exclude random baseline
    configs = [r for r in rows if r["level"] in recon_r2_by_level]
    print(f"\nUsable configs: {len(configs)} (random baselines dropped)")

    sa = np.array([r["C_i"] for r in configs])
    dist = np.array([r["avg_dist_last100"] for r in configs])
    recon = np.array([recon_r2_by_level[r["level"]] for r in configs])

    print(f"\n--- Per-config correlations with task distance ---")
    print(f"  r(SA, dist)        = {np.corrcoef(sa, dist)[0,1]:+.4f}")
    print(f"  r(recon_R², dist)  = {np.corrcoef(recon, dist)[0,1]:+.4f}")
    print(f"  r(SA, recon_R²)    = {np.corrcoef(sa, recon)[0,1]:+.4f}")

    pr_sa = partial_corr(sa, dist, recon)
    pr_re = partial_corr(recon, dist, sa)
    print(f"  partial r(SA, dist | recon_R²) = {pr_sa:+.4f}")
    print(f"  partial r(recon_R², dist | SA) = {pr_re:+.4f}")

    r2_recon_only = r2_mult(recon.reshape(-1, 1), dist)
    r2_both = r2_mult(np.stack([recon, sa], axis=1), dist)
    delta = r2_both - r2_recon_only
    print(f"\n--- Multiple regression on dist ---")
    print(f"  R² recon only      = {r2_recon_only:.4f}")
    print(f"  R² recon + SA      = {r2_both:.4f}")
    print(f"  ΔR² from adding SA = {delta:+.4f}")

    # Save numerics
    out = Path("results/obj028_recon_vs_sa_obj016.json")
    out.write_text(json.dumps({
        "n_configs": len(configs),
        "recon_r2_by_level": recon_r2_by_level,
        "r_SA_dist": float(np.corrcoef(sa, dist)[0,1]),
        "r_recon_dist": float(np.corrcoef(recon, dist)[0,1]),
        "r_SA_recon": float(np.corrcoef(sa, recon)[0,1]),
        "partial_r_SA_given_recon": pr_sa,
        "partial_r_recon_given_SA": pr_re,
        "R2_recon_only": r2_recon_only,
        "R2_recon_plus_SA": r2_both,
        "deltaR2_from_SA": delta,
    }, indent=2))
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
