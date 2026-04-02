"""obj-022: Expanded SA proxy validation across perception conditions.

Validates three oracle-free SA proxies across a grid that spans the full
SA range (0.1 to 0.9+) by including both oracle and VAE perception:

  Perception: [oracle, vae_mu_lat16]
  Embedding:  [2, 8, 32]
  Seeds:      [42, 123, 456, 789, 1337]
  Training:   500 episodes (PPO, batch=256)
  = 30 configs total

Proxies:
  A: Prediction consistency (forward model error)
  B: Action stability under observation noise
  C: Value-action alignment (finite differences)

Runs on CPU. ~2-3 hours expected.
"""

import sys
import json
import time
import math
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from worldnn.matter import RockPushMatter
from worldnn.organism import Organism
from worldnn.world import RockPushWorld
from worldnn.train import train_environment_rockpush
from coordination_quality import compute_optimal_action, measure_coordination_quality
from perception_ladder import train_organism_with_perception
from sa_proxy import (
    proxy_a_prediction_consistency,
    proxy_b_action_stability,
    proxy_c_value_action_alignment,
    proxy_d_action_entropy,
    proxy_e_policy_consistency,
)


def run_config(perception, embed_dim, seed, n_episodes=500, device="cpu"):
    """Train organism and measure true SA + all proxies."""
    torch.manual_seed(seed)
    dev = torch.device(device)

    matter = RockPushMatter(emission_dim=8, action_dim=2, seed_dim=4).to(dev)

    if perception == "oracle":
        organism = Organism(sensory_dim=4, embedding_dim=embed_dim, action_dim=2).to(dev)
        perception_fn = lambda state, emission: state
        sensory_dim = 4
    else:
        # VAE perception with mu (deterministic)
        world = RockPushWorld(
            emission_dim=8, channel_dim=8, env_latent_dim=16,
            embedding_dim=embed_dim, action_dim=2, seed_dim=4,
            channel_noise=0.01, target_x=0.8, target_y=0.8,
        ).to(dev)
        world.use_mu = True

        # Pre-train VAE
        vae_losses = train_environment_rockpush(
            world, n_steps=1500, batch_size=256, device=dev
        )

        matter = world.matter
        organism = world.organism
        env = world.environment
        ch = world.channel

        def perception_fn(state, emission):
            with torch.no_grad():
                ch_out = ch(emission)
                mu, _ = env.encode(ch_out)
            return mu

        sensory_dim = 16

    # Train organism
    t0 = time.time()
    metrics = train_organism_with_perception(
        matter, organism, perception_fn,
        target_x=0.8, target_y=0.8,
        n_episodes=n_episodes, batch_size=256, device=device,
    )
    train_time = time.time() - t0

    # True SA
    sa = measure_coordination_quality(
        organism, perception_fn, matter, n_samples=2000, device=device,
    )

    # All five proxies
    pa = proxy_a_prediction_consistency(
        organism, matter, perception_fn, n_samples=2000, device=device,
    )
    pb = proxy_b_action_stability(
        organism, matter, perception_fn, n_samples=2000, device=device,
    )
    pc = proxy_c_value_action_alignment(
        organism, matter, perception_fn, n_samples=2000, device=device,
    )
    pd = proxy_d_action_entropy(
        organism, matter, perception_fn, n_samples=2000, device=device,
    )
    pe = proxy_e_policy_consistency(
        organism, matter, perception_fn, n_samples=2000, device=device,
    )

    n_tail = min(50, len(metrics["rock_distance"]))
    avg_dist = sum(metrics["rock_distance"][-n_tail:]) / n_tail

    return {
        "perception": perception,
        "embedding_dim": embed_dim,
        "seed": seed,
        "avg_dist": avg_dist,
        "train_time_s": train_time,
        "true_SA": sa["C_i"],
        "true_SA_std": sa["C_i_std"],
        "embedding_utilization": sa["embedding_utilization"],
        **pa, **pb, **pc, **pd, **pe,
    }


def analyze_and_plot(results, results_dir):
    """Compute correlations and generate publication figure."""
    import numpy as np

    valid = [r for r in results if "error" not in r]
    if len(valid) < 5:
        print("Not enough results for analysis.")
        return

    true_sa = np.array([r["true_SA"] for r in valid])
    dists = np.array([r["avg_dist"] for r in valid])
    pa_vals = np.array([r["proxy_a_value"] for r in valid])
    pb_vals = np.array([r["proxy_b_value"] for r in valid])
    pc_vals = np.array([r["proxy_c_value"] for r in valid])
    pd_vals = np.array([r.get("proxy_d_value", 0) for r in valid])
    pe_vals = np.array([r.get("proxy_e_value", 0) for r in valid])

    print(f"\n{'='*60}")
    print(f"SA Proxy Validation (n={len(valid)} configs)")
    print(f"{'='*60}")
    print(f"True SA range: [{true_sa.min():.3f}, {true_sa.max():.3f}]")
    print(f"True SA vs distance: r = {np.corrcoef(true_sa, dists)[0,1]:.3f}")

    proxy_data = [
        ("A: Prediction consistency", pa_vals),
        ("B: Action stability", pb_vals),
        ("C: Value-action alignment", pc_vals),
        ("D: Action entropy", pd_vals),
        ("E: Policy consistency", pe_vals),
    ]

    best_proxy = None
    best_r = -1

    for name, vals in proxy_data:
        r_sa = np.corrcoef(vals, true_sa)[0, 1]
        r_dist = np.corrcoef(vals, dists)[0, 1]
        print(f"\n  {name}:")
        print(f"    r(proxy, true SA) = {r_sa:+.3f}")
        print(f"    r(proxy, dist)    = {r_dist:+.3f}")
        if abs(r_sa) > best_r:
            best_r = abs(r_sa)
            best_proxy = name

    print(f"\n  Best proxy: {best_proxy} (|r| = {best_r:.3f})")

    # Generate publication figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 5, figsize=(25, 5))

        colors = {"oracle": "#2196F3", "vae_mu_lat16": "#FF5722"}
        markers = {2: "o", 8: "s", 32: "D"}

        proxy_labels = [
            "Proxy A:\nPrediction Consistency",
            "Proxy B:\nAction Stability",
            "Proxy C:\nValue-Action Alignment",
            "Proxy D:\nAction Entropy",
            "Proxy E:\nPolicy Consistency",
        ]

        for ax, (name, vals), label in zip(axes, proxy_data, proxy_labels):

            for r_item in valid:
                c = colors.get(r_item["perception"], "gray")
                m = markers.get(r_item["embedding_dim"], "o")
                proxy_key_letter = name[0].lower()
                ax.scatter(r_item["true_SA"],
                          r_item.get(f"proxy_{proxy_key_letter}_value", 0),
                          color=c, marker=m, s=60, alpha=0.7,
                          edgecolors="black", linewidth=0.5)

            # Fit line
            r_val = np.corrcoef(vals, true_sa)[0, 1]
            z = np.polyfit(true_sa, vals, 1)
            x_range = np.linspace(true_sa.min(), true_sa.max(), 100)
            ax.plot(x_range, np.polyval(z, x_range), "k--", alpha=0.4)

            ax.set_xlabel("True SA", fontsize=11)
            ax.set_ylabel("Proxy Value", fontsize=11)
            ax.set_title(f"{label}\nr = {r_val:+.3f}", fontsize=11)

        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#2196F3',
                   markersize=8, label='Oracle'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF5722',
                   markersize=8, label='VAE lat=16'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                   markersize=6, label='emb=2'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
                   markersize=6, label='emb=8'),
            Line2D([0], [0], marker='D', color='w', markerfacecolor='gray',
                   markersize=6, label='emb=32'),
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=5,
                   fontsize=9, bbox_to_anchor=(0.5, -0.02))

        fig.suptitle("Oracle-Free SA Proxy Candidates vs True SA", fontsize=13, y=1.02)
        plt.tight_layout()
        out = results_dir / "obj022_sa_proxy.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"\nSaved figure: {out}")
    except Exception as e:
        print(f"Plot failed: {e}")
        import traceback; traceback.print_exc()

    # Also generate a 2-panel figure: best proxy vs SA, and best proxy vs distance
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        # Find best proxy key
        proxy_keys = {"A": "proxy_a_value", "B": "proxy_b_value", "C": "proxy_c_value"}
        best_key_letter = best_proxy[0]
        best_key = proxy_keys[best_key_letter]
        best_vals = np.array([r[best_key] for r in valid])

        for r in valid:
            c = colors.get(r["perception"], "gray")
            m = markers.get(r["embedding_dim"], "o")
            ax1.scatter(r["true_SA"], r[best_key],
                       color=c, marker=m, s=80, alpha=0.7,
                       edgecolors="black", linewidth=0.5)
            ax2.scatter(r[best_key], r["avg_dist"],
                       color=c, marker=m, s=80, alpha=0.7,
                       edgecolors="black", linewidth=0.5)

        # Fit lines
        r1 = np.corrcoef(true_sa, best_vals)[0, 1]
        z1 = np.polyfit(true_sa, best_vals, 1)
        x1 = np.linspace(true_sa.min(), true_sa.max(), 100)
        ax1.plot(x1, np.polyval(z1, x1), "k--", alpha=0.4)
        ax1.set_xlabel("True SA (oracle-required)", fontsize=11)
        ax1.set_ylabel(f"Proxy {best_key_letter} (oracle-free)", fontsize=11)
        ax1.set_title(f"Proxy {best_key_letter} vs True SA\nr = {r1:+.3f}", fontsize=12)

        r2 = np.corrcoef(best_vals, dists)[0, 1]
        z2 = np.polyfit(best_vals, dists, 1)
        x2 = np.linspace(best_vals.min(), best_vals.max(), 100)
        ax2.plot(x2, np.polyval(z2, x2), "k--", alpha=0.4)
        ax2.set_xlabel(f"Proxy {best_key_letter} (oracle-free)", fontsize=11)
        ax2.set_ylabel("Task Performance (distance)", fontsize=11)
        ax2.set_title(f"Proxy {best_key_letter} vs Distance\nr = {r2:+.3f}", fontsize=12)

        fig.legend(handles=legend_elements[:2], loc='lower center', ncol=2,
                   fontsize=9, bbox_to_anchor=(0.5, -0.02))
        fig.suptitle(f"Best Oracle-Free Proxy: {best_proxy}", fontsize=13, y=1.02)
        plt.tight_layout()
        out2 = results_dir / "obj022_sa_proxy_best.png"
        plt.savefig(out2, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved figure: {out2}")
    except Exception as e:
        print(f"Best-proxy plot failed: {e}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    suffix = "_gpu" if device == "cuda" else ""
    checkpoint_path = results_dir / f"sa_proxy_expanded{suffix}_checkpoint.json"
    results_path = results_dir / f"sa_proxy_expanded{suffix}.json"

    embed_dims = [2, 8, 32]
    seeds = [42, 123, 456, 789, 1337]
    perceptions = ["oracle", "vae_mu_lat16"]

    configs = []
    for perc in perceptions:
        for emb in embed_dims:
            for s in seeds:
                configs.append((perc, emb, s))

    # Resume from checkpoint
    completed = []
    completed_keys = set()
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            completed = json.load(f)
        for r in completed:
            completed_keys.add((r["perception"], r["embedding_dim"], r["seed"]))
        print(f"Resuming: {len(completed)}/{len(configs)} done")

    total = len(configs)
    remaining = total - len(completed_keys)
    print(f"obj-022 proxy sweep: {total} configs ({remaining} remaining)")
    t0 = time.time()

    for perc, emb, s in configs:
        key = (perc, emb, s)
        if key in completed_keys:
            continue

        idx = len(completed) + 1
        print(f"[{idx}/{total}] {perc}, emb={emb}, seed={s}",
              end=" ... ", flush=True)

        try:
            result = run_config(perc, emb, s, n_episodes=500, device=device)
            completed.append(result)
            completed_keys.add(key)
            print(f"SA={result['true_SA']:.3f}, dist={result['avg_dist']:.3f}, "
                  f"pA={result['proxy_a_value']:.4f}, "
                  f"pB={result['proxy_b_value']:.4f}, "
                  f"pC={result['proxy_c_value']:.4f} "
                  f"({result['train_time_s']:.0f}s)")
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback; traceback.print_exc()
            completed.append({
                "perception": perc, "embedding_dim": emb,
                "seed": s, "error": str(e),
            })
            completed_keys.add(key)

        # Checkpoint every 3 configs
        if len(completed) % 3 == 0:
            with open(checkpoint_path, "w") as f:
                json.dump(completed, f, indent=2)
            elapsed = time.time() - t0
            done_this = len(completed) - (total - remaining)
            if done_this > 0:
                rate = done_this / elapsed
                left = remaining - done_this
                if rate > 0:
                    eta = left / rate / 60
                    print(f"  [checkpoint] ETA: {eta:.0f} min")

    # Save final results
    elapsed = time.time() - t0
    with open(results_path, "w") as f:
        json.dump(completed, f, indent=2)
    with open(checkpoint_path, "w") as f:
        json.dump(completed, f, indent=2)

    n_ok = len([r for r in completed if "error" not in r])
    print(f"\nDone: {n_ok}/{total} in {elapsed/60:.1f} min")

    analyze_and_plot(completed, results_dir)


if __name__ == "__main__":
    main()
