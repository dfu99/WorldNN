"""
obj-025 T7: Markov blanket diagram linking SA to active inference.

Draws a schematic showing the WorldNN architecture as an active-inference
Markov blanket, with SA measuring posterior alignment between internal
state and true state via action similarity.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np


def main():
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Boxes: external, sensory, internal, active
    def box(x, y, w, h, label, sublabel, color, edge="black"):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                           facecolor=color, edgecolor=edge, linewidth=1.8)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h * 0.65, label, ha="center", va="center", fontsize=13, fontweight="bold")
        ax.text(x + w / 2, y + h * 0.3, sublabel, ha="center", va="center", fontsize=9, style="italic")

    box(0.5, 3.0, 2.8, 2.0, "External s", "matter state\n[rock_x, rock_y, org_x, org_y]", "#ffcccc")
    box(4.0, 3.0, 2.8, 2.0, "Sensory o", "obs = emission[:sensory_dim]\nI(s; o) gates learnability", "#cce5ff")
    box(7.5, 3.0, 2.8, 2.0, "Internal μ", "embedding\ndim = embedding_dim", "#d4edda")
    box(11.0, 3.0, 2.8, 2.0, "Active a", "policy output\n2D action", "#fff3cd")

    # Arrows forward
    def arrow(x1, y1, x2, y2, label, label_above=True, color="black"):
        a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->", mutation_scale=20,
                            color=color, linewidth=1.6)
        ax.add_patch(a)
        yl = (y1 + y2) / 2 + (0.25 if label_above else -0.35)
        ax.text((x1 + x2) / 2, yl, label, ha="center", fontsize=9, color=color)

    arrow(3.3, 4.0, 4.0, 4.0, "channel\nencoding")
    arrow(6.8, 4.0, 7.5, 4.0, "encoder\nq(s|μ)")
    arrow(10.3, 4.0, 11.0, 4.0, "policy\na(μ)")

    # Feedback arrow: a → s
    fb = FancyArrowPatch((12.4, 2.9), (1.9, 2.9),
                         arrowstyle="->", mutation_scale=20, color="#666",
                         connectionstyle="arc3,rad=-0.35", linewidth=1.6)
    ax.add_patch(fb)
    ax.text(7.1, 1.3, "environment: action perturbs state", ha="center", fontsize=10, color="#333", style="italic")

    # Oracle action box (top)
    ax.add_patch(FancyBboxPatch((10.8, 6.3), 3.0, 1.2, boxstyle="round,pad=0.08",
                                 facecolor="#f8d7da", edgecolor="black", linewidth=1.5))
    ax.text(12.3, 7.05, "Oracle a°(s)", ha="center", fontsize=12, fontweight="bold")
    ax.text(12.3, 6.65, "optimal action given full state", ha="center", fontsize=8, style="italic")

    # SA computation arrow
    ax.plot([12.4, 12.4], [5.1, 6.3], color="red", linewidth=1.5, linestyle="--")
    ax.plot([13.0, 13.0], [5.1, 6.3], color="red", linewidth=1.5, linestyle="--")
    ax.text(13.5, 5.7, "cosine\nsimilarity", ha="left", fontsize=9, color="red")

    # SA caption
    ax.text(7, 7.3,
            "SA ≡ E_s[⟨a(μ(o(s))), a°(s)⟩] — a bounded estimator of −KL[q(s|μ) ∥ p(s|o)]",
            ha="center", fontsize=12, fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="lightyellow", edgecolor="gray"))

    # Markov blanket label
    ax.text(5.4, 0.5, "Markov blanket partition: {o, a} separates internal μ from external s",
            ha="center", fontsize=10, style="italic", color="#333")

    # Free energy equation
    ax.text(7, 5.6,
            r"$F = D_{KL}[q(s|\mu) \,\|\, p(s|o)] - \ln p(o)$   →   minimized by matching $\mu$ to posterior",
            ha="center", fontsize=10, color="#444")

    plt.title("obj-025 T7: SA as a Proxy for Variational Free Energy Minimization",
              fontsize=13, fontweight="bold", pad=12)

    for out_path in [
        Path("results/obj025_markov_blanket.png"),
        Path("paper/neurips2026/figures/fig6_markov_blanket.png"),
        Path("paper/neurips2026/figures/fig6_markov_blanket.pdf"),
    ]:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
