"""Generate the audit-2026-05-05 status figure: 4 panels covering
project completion (claimed vs shipped), reviewer risk evolution,
artifact rollup, and pre-submission blocker map."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

FIG = Path("figures/audit-2026-05-05.png")
FIG.parent.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle("WorldNN Audit — 2026-05-05 (NeurIPS 2026 deadline May 6)",
             fontsize=15, fontweight="bold")

# ============================================================
# Panel A: Claimed vs shipped (objectives.yaml gap analysis)
# ============================================================
ax = axes[0, 0]
ids = ["001-005", "006-008", "009-013", "014", "015", "016", "017",
       "018", "019", "020", "021", "022", "023", "024", "025", "026", "027"]
status = {
    "001-005": "logged",
    "006-008": "logged",
    "009-013": "logged",
    "014": "logged",
    "015": "logged",
    "016": "logged",
    "017": "logged",
    "018": "missing",      # never assigned
    "019": "shipped+commit_only",
    "020": "shipped+commit_only",
    "021": "shipped+commit_only",
    "022": "logged",
    "023": "missing",      # mi_chain reference, never logged
    "024": "logged",
    "025": "logged",
    "026": "logged",
    "027": "started_killed",
}
colors = {
    "logged": "#2ca02c",
    "shipped+commit_only": "#ff7f0e",
    "missing": "#d62728",
    "started_killed": "#9467bd",
}
positions = np.arange(len(ids))
status_colors = [colors[status[i]] for i in ids]
ax.barh(positions, [1]*len(ids), color=status_colors, edgecolor="black")
ax.set_yticks(positions)
ax.set_yticklabels([f"obj-{i}" for i in ids], fontsize=9)
ax.invert_yaxis()
ax.set_xticks([])
ax.set_title("A. Objectives.yaml Coverage")
ax.text(0.05, 0.5, "claimed in commits/planning\nbut not logged here", transform=ax.transAxes,
        rotation=90, fontsize=8, alpha=0.4)
patches = [mpatches.Patch(color=c, label=k) for k, c in colors.items()]
ax.legend(handles=patches, loc="lower right", fontsize=8)

# ============================================================
# Panel B: Reviewer risk evolution
# ============================================================
ax = axes[0, 1]
reviewers = ["A: World Model", "B: Active Inf", "C: Info Theorist",
             "D: Neuroscientist", "E: Gen-Sim Skeptic"]
risk_initial = [3, 3, 3, 3, 3]                 # all High at sweep-start (obj-025 T8)
risk_post_obj025 = [2, 1, 1, 1, 3]              # after T3-T7
risk_post_obj026 = [2, 1, 1, 1, 3]              # 2-rock floor leaves E unchanged
risk_now = [2, 1, 1, 1, 3]                      # current
x = np.arange(len(reviewers))
w = 0.25
ax.bar(x - w, risk_initial, w, label="pre-sweep", color="#bbbbbb", edgecolor="black")
ax.bar(x, risk_post_obj025, w, label="post-obj-025", color="#1f77b4", edgecolor="black")
ax.bar(x + w, risk_now, w, label="now", color="#d62728", edgecolor="black")
ax.set_xticks(x)
ax.set_xticklabels([r.split(":")[0] for r in reviewers], fontsize=10)
ax.set_yticks([1, 2, 3])
ax.set_yticklabels(["Low", "Medium", "High"])
ax.set_title("B. Reviewer Risk Evolution")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)
for i, r in enumerate(reviewers):
    ax.text(i, risk_now[i] + 0.05, ["Low", "Med", "High"][risk_now[i] - 1],
            ha="center", fontsize=8, fontweight="bold")

# ============================================================
# Panel C: Artifact rollup (figures shipped vs orphaned)
# ============================================================
ax = axes[1, 0]
artifact_categories = [
    ("paper figures referenced", 3, "#2ca02c"),     # fig1, fig3, fig4
    ("paper figures orphaned", 1, "#ff7f0e"),       # fig2_dynamics not cited
    ("results/ analysis figures", 35, "#1f77b4"),
    ("§5.7 figure missing", 1, "#d62728"),          # rate-distortion + sensory-cap
    ("compiled PDF (10 pages)", 1, "#2ca02c"),
    ("official neurips_2026.sty", 0, "#d62728"),    # placeholder only
]
labels = [a[0] for a in artifact_categories]
counts = [a[1] for a in artifact_categories]
colors_c = [a[2] for a in artifact_categories]
ax.barh(np.arange(len(labels)), counts, color=colors_c, edgecolor="black")
ax.set_yticks(np.arange(len(labels)))
ax.set_yticklabels(labels, fontsize=10)
ax.invert_yaxis()
ax.set_xlabel("count")
ax.set_title("C. Artifact Rollup (paper-relevant)")
for i, c in enumerate(counts):
    ax.text(c + 0.5, i, str(c), va="center", fontsize=9)

# ============================================================
# Panel D: Pre-submission blocker map
# ============================================================
ax = axes[1, 1]
blockers = [
    ("Download official\nneurips_2026.sty", 1, "#d62728", "BLOCKER"),
    ("Verify page count\n(currently 10 PDF pages)", 0.6, "#ff7f0e", "REVIEW"),
    ("Promote obj-024/025\nfigure to §5.7", 0.4, "#ff7f0e", "OPTIONAL"),
    ("PI review + approval", 1, "#9467bd", "OUTSIDE-AGENT"),
    ("LaTeX compiles", 0, "#2ca02c", "RESOLVED"),
    ("Tests pass (29/29)", 0, "#2ca02c", "RESOLVED"),
    ("Bibliography (11 refs)", 0, "#2ca02c", "RESOLVED"),
    ("WD_BLACK mirror", 0, "#2ca02c", "RESOLVED"),
]
labels_d = [b[0] for b in blockers]
sev = [b[1] for b in blockers]
colors_d = [b[2] for b in blockers]
flags = [b[3] for b in blockers]
positions_d = np.arange(len(labels_d))
ax.barh(positions_d, sev, color=colors_d, edgecolor="black")
ax.set_yticks(positions_d)
ax.set_yticklabels(labels_d, fontsize=9)
ax.invert_yaxis()
ax.set_xlim(0, 1.3)
ax.set_xticks([])
ax.set_title("D. Pre-Submission Blocker Map (deadline May 6)")
for i, f in enumerate(flags):
    ax.text(sev[i] + 0.02, i, f, va="center", fontsize=8, fontweight="bold")

plt.tight_layout()
plt.savefig(FIG, dpi=180, bbox_inches="tight")
print(f"saved: {FIG}")
