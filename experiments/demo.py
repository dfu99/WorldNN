#!/usr/bin/env python3
"""Interactive demo: The Paramecium and the Chemical Source.

A concrete, interpretable scenario for the WorldNN framework.

SCENARIO: Chemotaxis as Perception-Action Loop
=============================================

Matter:  A chemical source with binary state — ACTIVE (emitting) or INACTIVE.
         The paramecium's goal is to make the source ACTIVE by depositing
         enzymes at the right location.

Channel: Chemical diffusion — molecules spread through the medium.
         Signal decays with distance, subject to Brownian noise.

Environment: The aqueous medium — temperature, viscosity, and other
         dissolved chemicals add noise and attenuate the signal.

Organism: A paramecium with:
         - Chemoreceptors (sensory input) — detect chemical gradient
         - Internal signaling cascade (embedding) — limited processing
         - Cilia (motor output) — move toward source + deposit enzymes

The key question: how sensitive must the paramecium's chemoreceptors be
(embedding capacity) to reliably activate the chemical source, given the
noise in chemical diffusion?

This maps directly to our framework:
- S = {ACTIVE, INACTIVE} — binary matter state
- X = chemical emission concentration
- Y = diffused concentration at paramecium's location (noisy)
- Z = environment latent (what reaches the cell membrane)
- E = internal representation (signaling cascade state)
- A = movement + enzyme deposition
"""

import sys
import os
from pathlib import Path

import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from worldnn.world import World
from worldnn.train import train_environment, train_organism


def run_demo(results_dir: str = "results"):
    """Run the paramecium chemotaxis demo."""
    os.makedirs(results_dir, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    device = torch.device("cpu")

    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 60)
    print("  WorldNN Demo: Paramecium Chemotaxis")
    print("  Binary state flip through lossy perception-action loop")
    print("=" * 60)

    # ── Create world with chemotaxis-inspired parameters ──
    world = World(
        emission_dim=4,      # chemical concentrations (different molecules)
        channel_dim=4,       # what reaches the medium after diffusion
        env_latent_dim=2,    # compressed by aqueous medium
        embedding_dim=4,     # paramecium's internal signaling state
        action_dim=2,        # movement direction + enzyme output
        seed_dim=4,          # stochastic molecular dynamics
        channel_noise=0.3,   # Brownian diffusion noise
        channel_bandwidth=0.8,  # signal attenuation
        flip_difficulty=1.0,
    ).to(device)

    # ── Phase 1: Environment learns to encode chemical signals ──
    print("\n[Phase 1] Training environment VAE (medium learns signal structure)...")
    env_losses = train_environment(world, n_steps=500, batch_size=256, device=device)
    print(f"  Final VAE loss: {env_losses[-1]:.4f}")

    # ── Phase 2: Organism learns to flip matter state ──
    print("\n[Phase 2] Training paramecium (learning chemotaxis + enzyme action)...")
    metrics = train_organism(
        world, n_episodes=500, steps_per_episode=10, batch_size=512,
        lr=1e-3, device=device
    )
    final_success = np.mean(metrics["success_rates"][-30:])
    print(f"  Final success rate: {final_success:.1%}")

    # ── Visualize a single episode ──
    print("\n[Visualizing] Running demo episode...")
    world.eval()
    with torch.no_grad():
        traj = world.run_episode(1, 20, target_state=1.0, device=device)

    # Extract single-batch trajectory
    states = [s.item() for s in traj["states"]]
    emissions = torch.stack(traj["emissions"]).squeeze(1).numpy()
    z_latents = torch.stack(traj["z_latents"]).squeeze(1).numpy()
    embeddings = torch.stack(traj["embeddings"]).squeeze(1).numpy()
    actions = torch.stack(traj["actions"]).squeeze(1).numpy()
    rewards = [r.item() for r in traj["rewards"]]

    # ── Figure 1: Episode timeline ──
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(4, 2, figure=fig, hspace=0.4, wspace=0.3)

    # State over time
    ax1 = fig.add_subplot(gs[0, :])
    ax1.step(range(len(states)), states, where="mid", linewidth=2, color="#2196F3")
    ax1.fill_between(range(len(states)), states, alpha=0.2, step="mid", color="#2196F3")
    ax1.set_ylabel("Matter State")
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(["INACTIVE", "ACTIVE"])
    ax1.set_title("Chemical Source State Over Time", fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Emissions (what the source puts out)
    ax2 = fig.add_subplot(gs[1, 0])
    for i in range(min(4, emissions.shape[1])):
        ax2.plot(emissions[:, i], alpha=0.7, label=f"Mol {i+1}")
    ax2.set_ylabel("Concentration")
    ax2.set_title("Chemical Emissions (Matter Output)")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    # Latent (what reaches paramecium)
    ax3 = fig.add_subplot(gs[1, 1])
    for i in range(z_latents.shape[1]):
        ax3.plot(z_latents[:, i], alpha=0.7, label=f"z{i+1}")
    ax3.set_ylabel("Latent Value")
    ax3.set_title("Environment Latent (After Diffusion)")
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.3)

    # Embedding (paramecium's internal state)
    ax4 = fig.add_subplot(gs[2, 0])
    im = ax4.imshow(embeddings.T, aspect="auto", cmap="coolwarm", interpolation="nearest")
    ax4.set_ylabel("Embedding Dim")
    ax4.set_xlabel("Time Step")
    ax4.set_title("Paramecium Internal State (Embedding)")
    plt.colorbar(im, ax=ax4, shrink=0.8)

    # Actions
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(actions[:, 0], label="Movement", linewidth=2, color="#FF5722")
    ax5.plot(actions[:, 1], label="Enzyme", linewidth=2, color="#4CAF50")
    ax5.set_ylabel("Action Value")
    ax5.set_xlabel("Time Step")
    ax5.set_title("Paramecium Actions")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Rewards
    ax6 = fig.add_subplot(gs[3, :])
    colors = ["#4CAF50" if r > 0 else "#F44336" for r in rewards]
    ax6.bar(range(len(rewards)), rewards, color=colors, alpha=0.7)
    ax6.set_ylabel("Reward")
    ax6.set_xlabel("Time Step")
    ax6.set_title("Reward Signal (Green=Target State Achieved)")
    ax6.grid(True, alpha=0.3)

    plt.suptitle("WorldNN Demo: Paramecium Chemotaxis Episode", fontsize=14, fontweight="bold", y=1.01)
    plt.savefig(f"{results_dir}/demo_episode.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {results_dir}/demo_episode.png")

    # ── Figure 2: Training progress ──
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # VAE training
    axes[0].plot(env_losses, color="#9C27B0", alpha=0.5)
    # Smoothed
    window = 20
    if len(env_losses) >= window:
        smoothed = np.convolve(env_losses, np.ones(window) / window, mode="valid")
        axes[0].plot(range(window - 1, len(env_losses)), smoothed, color="#9C27B0", linewidth=2)
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("VAE Loss")
    axes[0].set_title("Phase 1: Environment VAE Training")
    axes[0].grid(True, alpha=0.3)

    # Reward curve
    axes[1].plot(metrics["rewards"], color="#2196F3", alpha=0.3)
    if len(metrics["rewards"]) >= window:
        smoothed = np.convolve(metrics["rewards"], np.ones(window) / window, mode="valid")
        axes[1].plot(range(window - 1, len(metrics["rewards"])), smoothed, color="#2196F3", linewidth=2)
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Avg Reward")
    axes[1].set_title("Phase 2: Organism Reward")
    axes[1].grid(True, alpha=0.3)

    # Success rate
    axes[2].plot(metrics["success_rates"], color="#4CAF50", alpha=0.3)
    if len(metrics["success_rates"]) >= window:
        smoothed = np.convolve(metrics["success_rates"], np.ones(window) / window, mode="valid")
        axes[2].plot(range(window - 1, len(metrics["success_rates"])), smoothed, color="#4CAF50", linewidth=2)
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Success Rate")
    axes[2].set_title("Phase 2: 1-Bit Flip Success")
    axes[2].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random baseline")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{results_dir}/demo_training.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {results_dir}/demo_training.png")

    # ── Figure 3: Architecture diagram ──
    create_architecture_diagram(f"{results_dir}/architecture.png")
    print(f"  Saved: {results_dir}/architecture.png")

    print(f"\nDone! All outputs in {results_dir}/")


def create_architecture_diagram(path: str):
    """Create a detailed architecture diagram showing all ML blocks."""
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.set_aspect("equal")
    ax.axis("off")

    # Color scheme
    c_matter = "#E3F2FD"
    c_channel = "#FFF3E0"
    c_env = "#E8F5E9"
    c_org = "#FCE4EC"
    c_nn = "#BBDEFB"

    def draw_block(x, y, w, h, color, label, sublabel="", border="#333"):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.1",
            facecolor=color, edgecolor=border, linewidth=2
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2 + 0.15, label, ha="center", va="center",
                fontsize=10, fontweight="bold")
        if sublabel:
            ax.text(x + w / 2, y + h / 2 - 0.25, sublabel, ha="center", va="center",
                    fontsize=7, color="#555", style="italic")

    def draw_nn_block(x, y, w, h, label, details):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.05",
            facecolor=c_nn, edgecolor="#1565C0", linewidth=1.5
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2 + 0.2, label, ha="center", va="center",
                fontsize=8, fontweight="bold", color="#1565C0")
        ax.text(x + w / 2, y + h / 2 - 0.15, details, ha="center", va="center",
                fontsize=6, color="#555", family="monospace")

    def arrow(x1, y1, x2, y2, label="", color="#333", style="-"):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color=color, linewidth=1.5, linestyle=style)
        )
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx, my + 0.2, label, ha="center", va="bottom", fontsize=7, color=color)

    # ── Matter Block ──
    draw_block(0.5, 3, 3.5, 3.5, c_matter, "MATTER", "Binary Mealy Machine")
    draw_nn_block(1.0, 5.2, 2.5, 0.8, "Emission", "X = state_pattern + seed·W")
    draw_nn_block(1.0, 4.2, 2.5, 0.7, "Transition", "P(flip) = σ(2−‖A−target‖²)")
    ax.text(2.25, 3.5, "S ∈ {0,1}  |  X ∈ ℝ⁴  |  target ∈ ℝ²", ha="center", fontsize=7, color="#666")

    # ── Channel Block ──
    draw_block(5, 4, 2.5, 1.5, c_channel, "CHANNEL", "Fixed (not learned)")
    ax.text(6.25, 4.3, f"Y = WX + ε,  ε~N(0,σ²)", ha="center", fontsize=7, color="#666")

    # ── Environment Block ──
    draw_block(8.5, 2.5, 3.5, 4.5, c_env, "", "")
    ax.text(10.25, 6.6, "ENVIRONMENT VAE", ha="center", fontsize=10, fontweight="bold")
    draw_nn_block(9.0, 5.5, 2.5, 0.8, "Encoder", "MLP(4→32→32→μ,σ)")
    ax.text(10.25, 5.2, "z = μ + σ·ε  (reparameterize)", ha="center", fontsize=7, color="#666")
    draw_nn_block(9.0, 4.2, 2.5, 0.7, "Latent z", f"z ∈ ℝ^lat_dim")
    draw_nn_block(9.0, 3.2, 2.5, 0.7, "Decoder", "MLP(lat→32→32→4)")
    ax.text(10.25, 2.8, "Action transform: MLP(2→32→2)", ha="center", fontsize=7, color="#666")

    # ── Organism Block ──
    draw_block(12.5, 2.5, 3, 4.5, c_org, "", "")
    ax.text(14.0, 6.6, "ORGANISM", ha="center", fontsize=10, fontweight="bold")
    draw_nn_block(12.8, 5.5, 2.4, 0.7, "Sensory Filter", "MLP(lat→32→lat)·σ")
    draw_nn_block(12.8, 4.5, 2.4, 0.7, "Encoder", "MLP(lat→32→emb)·tanh")
    ax.text(14.0, 4.2, "E ∈ [-1,1]^emb_dim  ← BOTTLENECK", ha="center",
            fontsize=7, color="#C62828", fontweight="bold")
    draw_nn_block(12.8, 3.5, 2.4, 0.7, "Policy", "MLP(emb→32→2)")
    draw_nn_block(12.8, 2.7, 2.4, 0.6, "Value Head", "MLP(emb→32→1)")

    # ── Arrows ──
    arrow(4.0, 5.5, 5.0, 5.0, "Emission X")
    arrow(7.5, 5.0, 8.5, 5.9, "Channel Y")
    arrow(12.0, 4.7, 12.5, 5.3, "Latent z")

    # Action feedback (curved, going back)
    ax.annotate(
        "", xy=(4.0, 3.5), xytext=(12.5, 3.0),
        arrowprops=dict(
            arrowstyle="-|>", color="#F44336", linewidth=2,
            connectionstyle="arc3,rad=0.3", linestyle="--"
        )
    )
    ax.text(8.0, 1.8, "Action A (propagated back through environment)",
            ha="center", fontsize=8, color="#F44336", fontweight="bold")

    # Random seed input
    ax.annotate(
        "", xy=(2.25, 6.5), xytext=(2.25, 7.5),
        arrowprops=dict(arrowstyle="-|>", color="#FF9800", linewidth=2)
    )
    ax.text(2.25, 7.7, "Random Seed", ha="center", fontsize=9, fontweight="bold", color="#FF9800")

    # Information flow label
    ax.text(8.0, 8.2, "Information Flow: S → X → Y → z → E → A → S'",
            ha="center", fontsize=11, fontweight="bold", color="#333",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF9C4", edgecolor="#FBC02D"))

    ax.text(8.0, 0.5,
            "Data Processing Inequality: I(S;E) ≤ I(S;z) ≤ I(S;Y) ≤ I(S;X)\n"
            "Core Question: min dim(E) s.t. P(S'=target | A(E)) > threshold",
            ha="center", fontsize=9, color="#555", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#999"))

    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


if __name__ == "__main__":
    run_demo("results")
