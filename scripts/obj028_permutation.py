"""Permutation test for r(SA, dist) and ΔR²(SA over recon) on obj-016.

Bootstrap (D25) tells us how stable the +0.374 effect is under resampling.
Permutation tells us how often a *random* SA-label assignment would match
the observed effect. Different test, complementary information.
"""
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from obj028_bootstrap import load_obj028_arrays, r2_mult


def main():
    sa, dist, recon = load_obj028_arrays()
    n = len(sa)
    rng = np.random.default_rng(0)
    n_perm = 10_000

    obs_r = float(np.corrcoef(sa, dist)[0, 1])
    obs_delta = (r2_mult(np.stack([recon, sa], axis=1), dist)
                 - r2_mult(recon.reshape(-1, 1), dist))

    perm_r = np.empty(n_perm)
    perm_delta = np.empty(n_perm)
    for i in range(n_perm):
        sa_p = sa[rng.permutation(n)]
        perm_r[i] = float(np.corrcoef(sa_p, dist)[0, 1])
        perm_delta[i] = (r2_mult(np.stack([recon, sa_p], axis=1), dist)
                         - r2_mult(recon.reshape(-1, 1), dist))

    p_r = float((np.abs(perm_r) >= abs(obs_r)).mean())
    p_delta = float((perm_delta >= obs_delta).mean())

    print(f"observed r(SA, dist)     = {obs_r:+.4f}")
    print(f"observed ΔR²             = {obs_delta:+.4f}")
    print(f"permutation p (|r|>=obs) = {p_r:.6f}")
    print(f"permutation p (ΔR²>=obs) = {p_delta:.6f}")
    print(f"max |perm_r|             = {np.max(np.abs(perm_r)):.4f}")
    print(f"max perm_delta           = {perm_delta.max():.4f}")

    out = Path("results/obj028_permutation.json")
    out.write_text(json.dumps({
        "n_configs": n,
        "n_permutations": n_perm,
        "observed_r": obs_r,
        "observed_deltaR2": float(obs_delta),
        "permutation_p_r": p_r,
        "permutation_p_deltaR2": p_delta,
        "max_abs_perm_r": float(np.max(np.abs(perm_r))),
        "max_perm_deltaR2": float(perm_delta.max()),
    }, indent=2))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
