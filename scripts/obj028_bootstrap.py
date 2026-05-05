"""Bootstrap a 95% CI on the obj-028 ΔR² = +0.374 headline number.

Resample 245 configs with replacement; for each resample compute
R² of (recon-only) and (recon + SA), report ΔR² distribution.
"""
import json
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


def load_obj028_arrays():
    """Re-derive per-config (sa, dist, recon) arrays for bootstrap."""
    obj016 = json.load(open("results/ci_at_scale.json"))
    rd = json.load(open("results/obj028_recon_vs_sa_obj016.json"))
    recon_by_level = rd["recon_r2_by_level"]
    valid = [r for r in obj016["results"] if r["level"] in recon_by_level]
    sa = np.array([r["C_i"] for r in valid])
    dist = np.array([r["avg_dist_last100"] for r in valid])
    recon = np.array([recon_by_level[r["level"]] for r in valid])
    return sa, dist, recon


def r2_mult(X, y):
    Xa = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    b, *_ = np.linalg.lstsq(Xa, y, rcond=None)
    pred = Xa @ b
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return 1.0 - ss_res / ss_tot


def main():
    sa, dist, recon = load_obj028_arrays()
    n = len(sa)
    rng = np.random.default_rng(0)
    n_boot = 10_000

    point_recon = r2_mult(recon.reshape(-1, 1), dist)
    point_both = r2_mult(np.stack([recon, sa], axis=1), dist)
    point_delta = point_both - point_recon

    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        sa_b, dist_b, recon_b = sa[idx], dist[idx], recon[idx]
        # Skip degenerate resamples
        if recon_b.std() < 1e-9:
            deltas[i] = np.nan
            continue
        r2_r = r2_mult(recon_b.reshape(-1, 1), dist_b)
        r2_b = r2_mult(np.stack([recon_b, sa_b], axis=1), dist_b)
        deltas[i] = r2_b - r2_r

    deltas = deltas[~np.isnan(deltas)]
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    p_positive = float((deltas > 0).mean())
    print(f"n bootstrap = {len(deltas)}")
    print(f"point ΔR²       = {point_delta:+.4f}")
    print(f"bootstrap mean  = {deltas.mean():+.4f}")
    print(f"95% CI          = [{lo:+.4f}, {hi:+.4f}]")
    print(f"P(ΔR² > 0)      = {p_positive:.4f}")

    out = Path("results/obj028_bootstrap.json")
    out.write_text(json.dumps({
        "n_configs": n,
        "n_bootstrap": int(len(deltas)),
        "point_R2_recon_only": float(point_recon),
        "point_R2_recon_plus_SA": float(point_both),
        "point_deltaR2": float(point_delta),
        "bootstrap_mean_deltaR2": float(deltas.mean()),
        "ci_95_lo": float(lo),
        "ci_95_hi": float(hi),
        "P_deltaR2_positive": p_positive,
    }, indent=2))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
