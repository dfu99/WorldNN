"""
obj-025 T6: Biological calibration for sensory_dim.

Reviewer D (Sensorimotor Neuroscientist) asks: "Is your 'embedding size' supposed
to be cortical neurons, working memory slots, or something else? Pin it down."

Our sensory_dim ∈ {2,4,8,16} needs to be anchored against real biological
channel counts so reviewers can judge whether our regime is plausible.

References for channel counts (conservative literature estimates):
  • Olfactory receptor classes (human): ~400
  • Olfactory receptor classes (mouse): ~1000
  • Retinal ganglion cells (each eye): ~10^6
  • Optic nerve fibers: ~1.2 × 10^6
  • Visual cortex (V1) column types: ~100-200
  • Cochlear hair cells: ~15,000
  • Auditory nerve fibers: ~30,000
  • Cochlear frequency channels (CB): ~24 critical bands
  • Vestibular hair cells (per ear): ~33,000 (6 functional channels: 3 SCC + 2 otolith)
  • Proprioceptive muscle spindles (human): ~10^5
  • Proprioceptive functional joints: ~100
  • Cutaneous mechanoreceptors (fingertip): ~2500 receptors, ~4 channel types
  • Haptic vibrotactile channels (robotics): 1-10
  • Chemical gradients (paramecium): ~1-2
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def main():
    # Biological reference channels (log10 scale)
    # Tuples: (name, functional_channels, category)
    bio_channels = [
        ("Paramecium chemotaxis",     2,      "bacteria/proto"),
        ("Vestibular (functional)",   6,      "balance"),
        ("Haptic vibrotactile",      10,      "robotics"),
        ("Auditory critical bands",  24,      "hearing"),
        ("Cutaneous types × digit",  16,      "touch"),
        ("V1 orientation columns",   180,     "vision"),
        ("Olfactory classes (human)", 400,    "smell"),
        ("Olfactory classes (mouse)", 1000,   "smell"),
        ("Proprioceptive joints",     100,    "body"),
        ("Retinal ganglion types",    20,     "vision"),
        ("Cochlear hair cells",      15000,   "hearing"),
        ("Optic nerve fibers",       1.2e6,   "vision"),
        ("Auditory nerve fibers",    30000,   "hearing"),
    ]

    # Our sensory_dim range
    ours = [2, 4, 8, 16]

    # === Figure ===
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    fig.suptitle(
        "obj-025 T6: Biological Calibration of sensory_dim\n"
        "Our 2–16 channel range sits at the lower end of biological sensing",
        fontsize=13, fontweight="bold",
    )

    # Panel A: log-scale channel count comparison
    ax = axes[0]
    bio_channels_sorted = sorted(bio_channels, key=lambda x: x[1])
    names = [b[0] for b in bio_channels_sorted]
    counts = np.array([b[1] for b in bio_channels_sorted])
    cats = [b[2] for b in bio_channels_sorted]

    cat_colors = {
        "bacteria/proto": "#9467bd",
        "balance":        "#2ca02c",
        "robotics":       "#8c564b",
        "hearing":        "#ff7f0e",
        "touch":          "#e377c2",
        "vision":         "#1f77b4",
        "smell":          "#bcbd22",
        "body":           "#17becf",
    }
    colors = [cat_colors[c] for c in cats]
    ax.barh(range(len(names)), counts, color=colors, edgecolor="black", alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Functional channels (log scale)")
    ax.set_title("A. Channel Counts Across Biological Systems")
    ax.grid(alpha=0.3, axis="x")

    # Overlay our range as shaded band
    ax.axvspan(ours[0], ours[-1], color="red", alpha=0.15, label=f"obj-024 range: {ours[0]}–{ours[-1]}")
    ax.legend(loc="lower right")

    # Panel B: our sweep as points plus neighboring biological analogs
    ax = axes[1]
    ours_arr = np.array(ours)
    ax.scatter(ours_arr, [1] * len(ours_arr), s=160, color="red", edgecolor="black",
               zorder=3, label="obj-024 sensory_dim")
    for sd in ours_arr:
        ax.annotate(str(sd), (sd, 1), textcoords="offset points", xytext=(0, -22),
                    ha="center", fontsize=10, fontweight="bold", color="red")

    bio_plot = [
        (2,   "Paramecium\nchemotaxis"),
        (6,   "Vestibular\n(3 SCC + 3 oto)"),
        (10,  "Haptic\n(robotics)"),
        (16,  "Cutaneous\n× digit"),
        (24,  "Auditory\ncritical bands"),
        (180, "V1 orientation\ncolumns"),
        (1000, "Mouse olfactory\nclasses"),
    ]
    for x, n in bio_plot:
        ax.scatter(x, 2, s=90, color="#1f77b4", edgecolor="black", zorder=2)
        ax.annotate(n, (x, 2), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8)

    ax.set_xscale("log")
    ax.set_xlim(1, 2000)
    ax.set_ylim(0.5, 2.8)
    ax.set_yticks([1, 2])
    ax.set_yticklabels(["Our sweep", "Biology"])
    ax.set_xlabel("Channel count (log scale)")
    ax.set_title("B. Where obj-024 Sits on the Biological Scale")
    ax.grid(alpha=0.3, axis="x")
    ax.legend(loc="upper right")

    # Annotation: what sensory_dim=16 corresponds to
    ax.text(0.5, 0.05, "sensory_dim=16 ≈ vibrotactile haptic OR cutaneous-types per digit — "
            "well below the richness of vision, audition, or olfaction.",
            transform=ax.transAxes, fontsize=9, style="italic",
            bbox=dict(boxstyle="round", facecolor="lightyellow", edgecolor="gray"))

    plt.tight_layout()
    out = Path("results/obj025_bio_calibration.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")

    # Save numerics
    out_json = Path("results/obj025_bio_calibration.json")
    with out_json.open("w") as f:
        json.dump({
            "sensory_dims_tested": ours,
            "biological_channel_counts": [
                {"name": n, "functional_channels": int(c) if c == int(c) else c,
                 "category": cat} for (n, c, cat) in bio_channels
            ],
            "interpretation": {
                "sensory_dim=2":  "paramecium chemotaxis level — 1-2 chemical gradients",
                "sensory_dim=4":  "minimal vestibular: hair cell groups",
                "sensory_dim=8":  "basic proprioception or coarse touch",
                "sensory_dim=16": "vibrotactile haptic or cutaneous-types-per-digit",
                "compared_to_vision": "obj-024 max (16) is ~10^5 below optic nerve fiber count",
                "compared_to_audition": "obj-024 max (16) is ~10^3 below cochlear hair cells, comparable to critical bands (24)",
            },
        }, f, indent=2)
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
