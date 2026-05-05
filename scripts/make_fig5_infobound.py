"""Render fig5_infobound.{png,pdf} for paper §5.7 from obj025/obj024 data.

Two panels: (A) linear-probe R²(S|obs) and Gaussian-MI vs sensory_dim, with
peak-SA overlay; (B) Gaussian-MI vs peak SA scatter showing r=0.975."""

import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path("paper/neurips2026/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open("results/obj025_mi_vs_sensory.json") as f:
    rd = json.load(f)
with open("results/sensory_capacity_checkpoint.json") as f:
    obj024 = json.load(f)

sensory_dims = rd["sensory_dims"]
r2 = [rd["linear_probe_R2"][str(s)] for s in sensory_dims]
gauss_mi = [rd["Gaussian_MI_nats"][str(s)] for s in sensory_dims]

cells = defaultdict(list)
for r in obj024:
    cells[(r["sensory_dim"], r["embedding_dim"])].append(r["SA"])
peak_sa = []
for sd in sensory_dims:
    means = [np.mean(cells[(sd, ed)]) for ed in [2, 4, 8, 16, 32] if (sd, ed) in cells]
    peak_sa.append(max(means))

r_corr = float(np.corrcoef(gauss_mi, peak_sa)[0, 1])

fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4))
plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8})

# Panel A
ax = axes[0]
sd_arr = np.array(sensory_dims)
ax2 = ax.twinx()
l1 = ax.plot(sd_arr, r2, "o-", color="#1f77b4", label="linear $R^2(S \\mid \\mathrm{obs})$",
             markersize=6, linewidth=1.8)
l2 = ax2.plot(sd_arr, gauss_mi, "s--", color="#2ca02c",
              label="Gaussian-MI $I(S; \\mathrm{obs})$ (nats)",
              markersize=6, linewidth=1.8)
ax.set_xscale("log", base=2)
ax.set_xticks(sensory_dims); ax.set_xticklabels(sensory_dims)
ax.set_xlabel("sensory_dim")
ax.set_ylabel(r"linear $R^2$", color="#1f77b4")
ax2.set_ylabel("Gaussian-MI (nats)", color="#2ca02c")
ax.tick_params(axis="y", labelcolor="#1f77b4")
ax2.tick_params(axis="y", labelcolor="#2ca02c")
ax.set_ylim(0, 1.05)
ax.set_title("(A) Information available about state")
ax.grid(alpha=0.3)
lines = l1 + l2
ax.legend(lines, [l.get_label() for l in lines], loc="lower right", fontsize=8)

# Panel B
ax = axes[1]
ax.plot(gauss_mi, peak_sa, "s-", color="#d62728", linewidth=2, markersize=8)
for sd, x, y in zip(sensory_dims, gauss_mi, peak_sa):
    ax.annotate(f"$s={sd}$", (x, y), textcoords="offset points", xytext=(8, -3), fontsize=9)
ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
ax.set_xlabel("Gaussian-MI $I(S; \\mathrm{obs})$ (nats)")
ax.set_ylabel("peak SA achieved")
ax.set_title(f"(B) SA ceiling vs information ($r = {r_corr:.3f}$)")
ax.grid(alpha=0.3)

plt.tight_layout()
for fmt in ("png", "pdf"):
    out = OUT_DIR / f"fig5_infobound.{fmt}"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"saved: {out}")
