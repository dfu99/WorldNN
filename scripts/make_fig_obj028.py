"""Standalone obj-028 figure: SA vs recon-loss head-to-head on obj-016."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

with open("results/obj028_recon_vs_sa_obj016.json") as f:
    d = json.load(f)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
fig.suptitle("obj-028: SA vs reconstruction-loss on obj-016 (n=245)",
             fontsize=12, fontweight="bold")

# Panel A: per-condition recon ceilings (bar chart)
ax = axes[0]
levels = ["oracle", "raw_emission", "vae_mu_lat32", "vae_mu_lat16",
          "oracle_noise0.1", "vae_mu_lat8", "oracle_noise0.5"]
r2_per_level = d["recon_r2_by_level"]
vals = [r2_per_level[L] for L in levels]
colors = ["#2ca02c", "#1f77b4", "#1f77b4", "#1f77b4",
          "#9467bd", "#1f77b4", "#d62728"]
bars = ax.barh(range(len(levels)), vals, color=colors, edgecolor="black")
ax.set_yticks(range(len(levels)))
ax.set_yticklabels([L.replace("_", " ") for L in levels], fontsize=9)
ax.invert_yaxis()
ax.set_xlim(0, 1.05)
ax.set_xlabel(r"linear-probe $R^2(\mathrm{state}\,|\,\mathrm{obs})$")
ax.set_title("(A) Per-perception recon ceiling")
for b, v in zip(bars, vals):
    ax.text(v + 0.02, b.get_y() + b.get_height()/2, f"{v:.2f}",
            va="center", fontsize=9)

# Panel B: predictor strength comparison
ax = axes[1]
labels = [
    r"$r$(SA, dist)",
    r"$r$(recon, dist)",
    r"partial $r$(SA $\mid$ recon)",
    r"partial $r$(recon $\mid$ SA)",
    r"$\Delta R^2$ from SA",
]
vals_p = [
    d["r_SA_dist"],
    d["r_recon_dist"],
    d["partial_r_SA_given_recon"],
    d["partial_r_recon_given_SA"],
    d["deltaR2_from_SA"],
]
clrs = ["#d62728", "#1f77b4", "#d62728", "#1f77b4", "#2ca02c"]
y = np.arange(len(labels))
ax.barh(y, vals_p, color=clrs, edgecolor="black")
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=9)
ax.invert_yaxis()
ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
ax.set_xlim(-0.85, 0.85)
ax.set_title("(B) SA adds $\\Delta R^2 = +0.374$ over recon")
for i, v in enumerate(vals_p):
    ax.text(v + (0.02 if v >= 0 else -0.02), i, f"{v:+.3f}",
            va="center", ha="left" if v >= 0 else "right", fontsize=9,
            fontweight="bold")

plt.tight_layout()
out = Path("figures/obj028_recon_vs_sa.png")
out.parent.mkdir(exist_ok=True)
plt.savefig(out, dpi=200, bbox_inches="tight")
print(f"saved: {out}")
