"""Generate figures/audit-2026-05-05.png — 6-panel project status.

Panel A: Objectives.yaml coverage (claimed vs shipped vs logged)
Panel B: Reviewer risk evolution (pre-sweep → post-obj-025 → post-§5.7-figure)
Panel C: Artifact rollup (paper-cited vs orphaned vs results-only)
Panel D: Pre-submission blocker map
Panel E: KSG estimator validation — UNDERESTIMATES truth by 30-100%
Panel F: Test coverage map — components × tested?

Refined version after second-pass audit (2026-05-05 evening):
- §5.7 figure has landed → Reviewer C drops further
- KSG sanity check exposed a structural under-estimation
- draft.md/main.tex drift documented as a 7th concern
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

FIG = Path("figures/audit-2026-05-05.png")
FIG.parent.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(3, 2, figsize=(15, 16))
fig.suptitle("WorldNN Audit — 2026-05-05 (refined; NeurIPS deadline May 6)",
             fontsize=15, fontweight="bold")

# ============================================================
# Panel A: Objectives.yaml coverage
# ============================================================
ax = axes[0, 0]
ids = ["001-005", "006-008", "009-013", "014", "015", "016", "017",
       "018", "019", "020", "021", "022", "023", "024", "025", "026", "027"]
status = {
    "001-005": "logged", "006-008": "logged", "009-013": "logged",
    "014": "logged", "015": "logged", "016": "logged", "017": "logged",
    "018": "missing",                       # never assigned
    "019": "logged",                        # backfilled this session
    "020": "logged",                        # backfilled this session
    "021": "logged",                        # backfilled this session
    "022": "logged",
    "023": "missing",                       # mi_chain reference, never logged
    "024": "logged", "025": "logged", "026": "logged",
    "027": "started_killed",
}
colors = {
    "logged": "#2ca02c",
    "missing": "#d62728",
    "started_killed": "#9467bd",
}
positions = np.arange(len(ids))
bar_colors = [colors[status[i]] for i in ids]
ax.barh(positions, [1] * len(ids), color=bar_colors, edgecolor="black")
ax.set_yticks(positions)
ax.set_yticklabels([f"obj-{i}" for i in ids], fontsize=9)
ax.invert_yaxis()
ax.set_xticks([])
ax.set_title("A. Objectives.yaml Coverage (post-backfill)")
patches = [mpatches.Patch(color=c, label=k) for k, c in colors.items()]
ax.legend(handles=patches, loc="lower right", fontsize=8)

# ============================================================
# Panel B: Reviewer risk evolution
# ============================================================
ax = axes[0, 1]
reviewers = ["A", "B", "C", "D", "E"]
risk_pre = [3, 3, 3, 3, 3]                    # all High pre-sweep
risk_obj025 = [2, 1, 1, 1, 3]                  # post-obj-025 T8
risk_now = [2, 1, 1, 1, 3]                     # post-fig5 (C reaffirmed Low)
x = np.arange(len(reviewers))
w = 0.27
ax.bar(x - w, risk_pre, w, label="pre-sweep", color="#bbbbbb", edgecolor="black")
ax.bar(x, risk_obj025, w, label="post-obj-025", color="#1f77b4", edgecolor="black")
ax.bar(x + w, risk_now, w, label="post-§5.7 fig", color="#d62728", edgecolor="black")
ax.set_xticks(x)
ax.set_xticklabels(reviewers, fontsize=10)
ax.set_yticks([1, 2, 3])
ax.set_yticklabels(["Low", "Medium", "High"])
ax.set_title("B. Reviewer Risk Evolution")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)
for i in range(len(reviewers)):
    ax.text(i, risk_now[i] + 0.05, ["Low", "Med", "High"][risk_now[i] - 1],
            ha="center", fontsize=8, fontweight="bold")

# ============================================================
# Panel C: Artifact rollup
# ============================================================
ax = axes[1, 0]
items = [
    ("paper figures cited", 4, "#2ca02c"),       # fig1, fig2(NEW), fig3, fig4, fig5(NEW)
    ("paper figures orphaned", 0, "#ff7f0e"),    # fig2 was orphaned, now cited
    ("results/ analysis pngs", 35, "#1f77b4"),
    ("§5.7 figure", 1, "#2ca02c"),               # promoted this session
    ("compiled PDF (10 pp)", 1, "#2ca02c"),
    ("official .sty in repo", 0, "#d62728"),
]
labels = [a[0] for a in items]
counts = [a[1] for a in items]
cs = [a[2] for a in items]
ax.barh(np.arange(len(labels)), counts, color=cs, edgecolor="black")
ax.set_yticks(np.arange(len(labels)))
ax.set_yticklabels(labels, fontsize=10)
ax.invert_yaxis()
ax.set_xlabel("count")
ax.set_title("C. Artifact Rollup (post-fig5 promotion)")
for i, c in enumerate(counts):
    ax.text(c + 0.5, i, str(c), va="center", fontsize=9)

# ============================================================
# Panel D: Pre-submission blocker map
# ============================================================
ax = axes[1, 1]
blockers = [
    ("Download official\nneurips_2026.sty", 1.0, "#d62728", "PI ACTION"),
    ("PI review + approval", 1.0, "#9467bd", "PI ACTION"),
    ("Promote §5.7 figure", 0.0, "#2ca02c", "DONE"),
    ("Add §5.4 dynamics fig", 0.0, "#2ca02c", "DONE"),
    ("Backfill obj-019/020/021", 0.0, "#2ca02c", "DONE"),
    ("LaTeX compiles", 0.0, "#2ca02c", "DONE"),
    ("Tests pass (29/29)", 0.0, "#2ca02c", "DONE"),
    ("Verify §5.7 numbers", 0.0, "#2ca02c", "DONE"),
]
labels_d = [b[0] for b in blockers]
sev = [b[1] for b in blockers]
colors_d = [b[2] for b in blockers]
flags = [b[3] for b in blockers]
ax.barh(np.arange(len(labels_d)), sev, color=colors_d, edgecolor="black")
ax.set_yticks(np.arange(len(labels_d)))
ax.set_yticklabels(labels_d, fontsize=9)
ax.invert_yaxis()
ax.set_xlim(0, 1.4)
ax.set_xticks([])
ax.set_title("D. Pre-Submission Blocker Map")
for i, f in enumerate(flags):
    ax.text(sev[i] + 0.02, i, f, va="center", fontsize=8, fontweight="bold")

# ============================================================
# Panel E: KSG estimator validation (NEW)
# ============================================================
ax = axes[2, 0]
rhos = np.array([0.0, 0.3, 0.6, 0.9])
ksg_est = np.array([0.0000, 0.0000, 0.0000, 0.5484])  # measured this session
ksg_truth = np.array([0.0, 0.0472, 0.2231, 0.8304])
w2 = 0.35
xx = np.arange(len(rhos))
ax.bar(xx - w2/2, ksg_truth, w2, label="truth (analytic)", color="#2ca02c", edgecolor="black")
ax.bar(xx + w2/2, ksg_est, w2, label="our KSG (n=5000, k=3)", color="#d62728", edgecolor="black")
ax.set_xticks(xx)
ax.set_xticklabels([f"$\\rho={r}$" for r in rhos])
ax.set_ylabel("Mutual Information (nats)")
ax.set_title("E. KSG Sanity — Under-estimates by 30-100%")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)
for i, (e, t) in enumerate(zip(ksg_est, ksg_truth)):
    if t > 0:
        ax.text(i + w2/2, e + 0.02, f"{e/t*100:.0f}%" if e > 0 else "0%",
                ha="center", fontsize=8, color="#d62728", fontweight="bold")

# ============================================================
# Panel F: Test coverage map (NEW)
# ============================================================
ax = axes[2, 1]
components = [
    "RockPushMatter", "RockPushWorld", "Organism.forward",
    "MultiRockMatter", "ContinuousMatter", "EnvironmentVAE",
    "compute_sa_sensory", "compute_optimal_action",
    "estimate_mi_ksg", "linear_probe_r2", "gaussian_mi_from_r2",
    "compute_chain_mi",
]
covered = ["yes", "yes", "no", "no", "no", "no", "no", "no", "no", "no", "no", "no"]
color_map = {"yes": "#2ca02c", "no": "#d62728"}
y = np.arange(len(components))
ax.barh(y, [1] * len(components), color=[color_map[c] for c in covered], edgecolor="black")
ax.set_yticks(y)
ax.set_yticklabels(components, fontsize=9)
ax.invert_yaxis()
ax.set_xticks([])
ax.set_title(f"F. Test Coverage Map — {covered.count('yes')}/{len(covered)} covered")
patches_f = [mpatches.Patch(color="#2ca02c", label="tested"),
             mpatches.Patch(color="#d62728", label="untested")]
ax.legend(handles=patches_f, loc="lower right", fontsize=8)

plt.tight_layout()
plt.savefig(FIG, dpi=180, bbox_inches="tight")
print(f"saved: {FIG}")
