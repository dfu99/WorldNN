"""obj-036 figure: 1D sensory-capacity heatmap + per-sensory curves."""
import json
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

d = json.load(open("results/obj036_1d_sensory_capacity.json"))
rows = d["results"]
by_cell = defaultdict(list)
for r in rows:
    by_cell[(r["sensory_dim"], r["embedding_dim"])].append(r["SA"])
sensory_dims = sorted({k[0] for k in by_cell})
embed_dims = sorted({k[1] for k in by_cell})

# Use |SA|: the 1D action→force projection has random sign at matter init,
# so policy can be anti-aligned with action-coordinate optimal but still
# correct in force-coordinate (and clearly is, given dist drops to 0.15).
grid = np.zeros((len(sensory_dims), len(embed_dims)))
std_grid = np.zeros_like(grid)
for i, sd in enumerate(sensory_dims):
    for j, ed in enumerate(embed_dims):
        arr = np.abs(np.array(by_cell[(sd, ed)]))
        grid[i, j] = arr.mean()
        std_grid[i, j] = arr.std()

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
fig.suptitle(
    "obj-036: 1D positioning task — Reviewer E mitigation (non-manipulation)",
    fontsize=12, fontweight="bold",
)

ax = axes[0]
im = ax.imshow(grid, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1.0)
ax.set_xticks(range(len(embed_dims))); ax.set_xticklabels(embed_dims)
ax.set_yticks(range(len(sensory_dims))); ax.set_yticklabels(sensory_dims)
ax.set_xlabel("embed_dim"); ax.set_ylabel("sensory_dim")
ax.set_title("(A) mean SA across grid")
for i in range(len(sensory_dims)):
    for j in range(len(embed_dims)):
        ax.text(j, i, f"{grid[i,j]:.2f}", ha="center", va="center",
                fontsize=9, color="white" if grid[i,j] > 0.4 else "black")
plt.colorbar(im, ax=ax)

ax = axes[1]
colors = ["#d62728", "#ff7f0e", "#1f77b4"]
for i, sd in enumerate(sensory_dims):
    means = grid[i, :]
    stds = std_grid[i, :]
    ax.errorbar(embed_dims, means, yerr=stds, fmt="o-", linewidth=2,
                markersize=8, color=colors[i], capsize=4,
                label=f"sensory={sd}")
ax.set_xscale("log", base=2)
ax.set_xticks(embed_dims); ax.set_xticklabels(embed_dims)
ax.set_xlabel("embed_dim"); ax.set_ylabel("mean |SA| (1D, sign-agreement)")
ax.set_title("(B) SA vs capacity by sensory")
ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.6,
           label="learnability threshold")
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
out = Path("figures/obj036_1d_sensory_capacity.png")
out.parent.mkdir(exist_ok=True)
plt.savefig(out, dpi=200, bbox_inches="tight")
print(f"saved: {out}")

# Headline numbers
print("\nPer-sensory peak SA:")
for i, sd in enumerate(sensory_dims):
    print(f"  sensory={sd}: peak={grid[i].max():+.3f} at embed={embed_dims[grid[i].argmax()]}")
