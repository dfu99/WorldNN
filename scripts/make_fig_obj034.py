"""obj-034 figure: SA vs OA on the slim sensory × embed=16 grid."""
import json
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

d = json.load(open("results/obj034_outcome_alignment.json"))
rows = d["results"]
by_sensory = defaultdict(list)
for r in rows:
    by_sensory[r["sensory_dim"]].append(r)
sensory_dims = sorted(by_sensory.keys())

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
fig.suptitle(
    "obj-034: Outcome Alignment (OA) — intent-vs-real per PI request",
    fontsize=12, fontweight="bold",
)

# Panel A: SA and OA vs sensory_dim
ax = axes[0]
sa_means = [np.mean([r["SA"] for r in by_sensory[sd]]) for sd in sensory_dims]
sa_std = [np.std([r["SA"] for r in by_sensory[sd]]) for sd in sensory_dims]
oa_means = [np.mean([r["OA"] for r in by_sensory[sd]]) for sd in sensory_dims]
oa_std = [np.std([r["OA"] for r in by_sensory[sd]]) for sd in sensory_dims]
ax.errorbar(sensory_dims, sa_means, yerr=sa_std, fmt="o-", linewidth=2,
            markersize=8, color="#1f77b4", capsize=4, label="SA (cos action ↔ oracle)")
ax.errorbar(sensory_dims, oa_means, yerr=oa_std, fmt="s--", linewidth=2,
            markersize=8, color="#d62728", capsize=4, label="OA (cos action ↔ Δrock)")
ax.set_xscale("log", base=2)
ax.set_xticks(sensory_dims); ax.set_xticklabels(sensory_dims)
ax.set_xlabel("sensory_dim")
ax.set_ylabel("metric value")
ax.set_title("(A) SA vs OA across sensory richness")
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)
ax.set_ylim(-0.05, 0.65)

# Panel B: SA vs OA scatter per config
ax = axes[1]
sa_all = [r["SA"] for r in rows]
oa_all = [r["OA"] for r in rows]
colors = {2: "#d62728", 4: "#ff7f0e", 8: "#2ca02c", 16: "#1f77b4"}
for r in rows:
    ax.scatter(r["SA"], r["OA"], s=80, color=colors[r["sensory_dim"]],
               edgecolor="black", linewidth=0.5, alpha=0.85)
r_corr = np.corrcoef(sa_all, oa_all)[0, 1]
ax.set_xlabel("SA per config")
ax.set_ylabel("OA per config")
ax.set_title(f"(B) per-config SA vs OA, $r = {r_corr:+.2f}$")
import matplotlib.patches as mpatches
patches = [mpatches.Patch(color=c, label=f"sensory={s}") for s, c in colors.items()]
ax.legend(handles=patches, loc="upper left", fontsize=8)
ax.grid(alpha=0.3)
# Identity line for reference
mn, mx = min(min(sa_all), min(oa_all)), max(max(sa_all), max(oa_all))
ax.plot([mn, mx], [mn, mx], "k:", alpha=0.4, linewidth=1)

plt.tight_layout()
out = Path("figures/obj034_outcome_alignment.png")
out.parent.mkdir(exist_ok=True)
plt.savefig(out, dpi=200, bbox_inches="tight")
print(f"saved: {out}")
