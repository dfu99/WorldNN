"""Render the bootstrap distribution of ΔR² for the obj-028 finding."""
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Re-run bootstrap to get full distribution (not just CI)
sys.path.insert(0, str(Path(__file__).parent))
from obj028_bootstrap import load_obj028_arrays, r2_mult

sa, dist, recon = load_obj028_arrays()
n = len(sa)
rng = np.random.default_rng(0)
n_boot = 10_000

deltas = np.empty(n_boot)
for i in range(n_boot):
    idx = rng.integers(0, n, n)
    sa_b, dist_b, recon_b = sa[idx], dist[idx], recon[idx]
    if recon_b.std() < 1e-9:
        deltas[i] = np.nan
        continue
    r2_r = r2_mult(recon_b.reshape(-1, 1), dist_b)
    r2_b = r2_mult(np.stack([recon_b, sa_b], axis=1), dist_b)
    deltas[i] = r2_b - r2_r

deltas = deltas[~np.isnan(deltas)]
lo, hi = np.quantile(deltas, [0.025, 0.975])
mean = deltas.mean()
point = r2_mult(np.stack([recon, sa], axis=1), dist) - r2_mult(recon.reshape(-1, 1), dist)

fig, ax = plt.subplots(figsize=(7, 3.8))
counts, bins, patches = ax.hist(deltas, bins=60, color="#1f77b4",
                                 edgecolor="black", alpha=0.7)
ax.axvline(point, color="#d62728", linewidth=2.5, label=f"point = +{point:.3f}")
ax.axvline(lo, color="#ff7f0e", linestyle="--", linewidth=1.8,
           label=f"95% CI = [+{lo:.3f}, +{hi:.3f}]")
ax.axvline(hi, color="#ff7f0e", linestyle="--", linewidth=1.8)
ax.axvline(0, color="gray", linestyle=":", linewidth=1.2, label="ΔR² = 0")
# Reference: obj-024 narrow-grid baseline
ax.axvline(0.004, color="purple", linestyle="-.", linewidth=1.2,
           label="obj-024 narrow-grid baseline (+0.004)")
ax.set_xlabel(r"$\Delta R^2$ (SA over recon-loss; obj-016 wide grid)")
ax.set_ylabel("bootstrap count")
ax.set_title("obj-028: bootstrap distribution of ΔR²(SA over recon) on n=245")
ax.legend(loc="upper right", fontsize=8)
ax.grid(alpha=0.3)
plt.tight_layout()
out = Path("results/obj028_bootstrap_dist.png")
plt.savefig(out, dpi=200, bbox_inches="tight")
print(f"saved: {out}")
print(f"n_bootstrap={len(deltas)}  point={point:+.4f}  95% CI=[{lo:+.4f},{hi:+.4f}]")
