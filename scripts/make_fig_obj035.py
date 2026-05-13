"""obj-035 figure: SA vs supervision weight."""
import json
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

d = json.load(open("results/obj035_sparse_supervision.json"))
rows = d["results"]
by_w = defaultdict(list)
for r in rows:
    by_w[r["supervision_weight"]].append(r)
weights = sorted(by_w.keys())

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
fig.suptitle(
    "obj-035: SA vs oracle-supervision weight (sensory=16, embed=16)",
    fontsize=12, fontweight="bold",
)

# Panel A: SA vs supervision weight
ax = axes[0]
sa_means = [np.mean([r["SA"] for r in by_w[w]]) for w in weights]
sa_std = [np.std([r["SA"] for r in by_w[w]]) for w in weights]
dist_means = [np.mean([r["final_dist"] for r in by_w[w]]) for w in weights]
# offset so 0 stays visible on log
xpos = np.arange(len(weights))
ax.errorbar(xpos, sa_means, yerr=sa_std, fmt="o-", linewidth=2,
            markersize=9, color="#1f77b4", capsize=4, label="SA")
ax.set_xticks(xpos)
ax.set_xticklabels([str(w) for w in weights])
ax.set_xlabel("oracle-supervision weight $w$")
ax.set_ylabel("SA (cos action ↔ oracle)")
ax.set_title("(A) SA scales with supervision weight")
ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.6, label="learnability threshold")
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)
ax.set_ylim(0, 1.05)
for i, (m, s) in enumerate(zip(sa_means, sa_std)):
    ax.text(i, m + 0.05, f"{m:.2f}", ha="center", fontsize=9, fontweight="bold")

# Panel B: final_dist vs SA — does high SA translate to task success?
ax = axes[1]
colors = {0.0: "#d62728", 0.01: "#ff7f0e", 0.1: "#2ca02c", 1.0: "#1f77b4"}
for r in rows:
    ax.scatter(r["SA"], r["final_dist"], s=90,
               color=colors[r["supervision_weight"]],
               edgecolor="black", linewidth=0.5, alpha=0.85)
all_sa = np.array([r["SA"] for r in rows])
all_d = np.array([r["final_dist"] for r in rows])
r_sa_d = np.corrcoef(all_sa, all_d)[0, 1]
import matplotlib.patches as mpatches
patches = [mpatches.Patch(color=c, label=f"w={w}") for w, c in colors.items()]
ax.legend(handles=patches, loc="lower left", fontsize=8)
ax.set_xlabel("SA per config")
ax.set_ylabel("final rock-target distance")
ax.set_title(f"(B) Does high SA → low dist? $r = {r_sa_d:+.2f}$")
ax.grid(alpha=0.3)

plt.tight_layout()
out = Path("figures/obj035_sparse_supervision.png")
out.parent.mkdir(exist_ok=True)
plt.savefig(out, dpi=200, bbox_inches="tight")
print(f"saved: {out}")
