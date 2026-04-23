"""obj-016: C_i at scale — addresses all critique points.

Scaled experiment to test robustness of C_i findings:
1. 7 seeds (up from 3) for statistical power
2. Random baseline included (untrained organism)
3. All 7 perception conditions from obj-014
4. Explicit success threshold defined vs random baseline
5. Error bars computed for all metrics

Grid:
  perception: [oracle, oracle_noise0.1, oracle_noise0.5, raw_emission,
               vae_mu_lat8, vae_mu_lat16, vae_mu_lat32]
  embedding_dim: [2, 4, 8, 16, 32]
  seeds: [42, 123, 456, 789, 1337, 2024, 3141]
  = 7 × 5 × 7 = 245 configs

Plus 35 random baseline configs (untrained, 5 embed × 7 seeds).
Total: 280 configs.
"""

import os
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
from worldnn.world import RockPushWorld
from worldnn.matter import RockPushMatter
from worldnn.organism import Organism
from worldnn.train import train_environment_rockpush
from perception_ladder import train_organism_with_perception, probe_vae_latent
from coordination_quality import compute_optimal_action, measure_coordination_quality


def measure_random_baseline(matter, sensory_dim, embed_dim, seed,
                             perception_fn, device="cpu"):
    """Measure C_i and distance for an UNTRAINED organism."""
    torch.manual_seed(seed)
    dev = torch.device(device)

    organism = Organism(
        sensory_dim=sensory_dim, embedding_dim=embed_dim, action_dim=2,
    ).to(dev)

    # Measure C_i without any training
    ci_metrics = measure_coordination_quality(
        organism, perception_fn, matter, device=device,
    )

    # Measure distance with random policy
    organism.eval()
    target = torch.tensor([0.8, 0.8], device=dev)
    dists = []
    with torch.no_grad():
        for _ in range(20):  # 20 rollouts
            state = matter.reset_state(256, dev)
            for t in range(20):
                seed_t = torch.randn(256, matter.seed_dim, device=dev)
                action = torch.randn(256, 2, device=dev) * 0.3
                state, _, _ = matter(state, seed_t, action)
            rock_pos = state[:, :2]
            dists.append(torch.norm(rock_pos - target, dim=-1).mean().item())

    return {
        "level": "random_baseline",
        "sensory_dim": sensory_dim,
        "embedding_dim": embed_dim,
        "seed": seed,
        "avg_dist_last100": sum(dists) / len(dists),
        "elapsed_s": 0,
        **ci_metrics,
    }


def run_config(level_name, matter, sensory_dim, embed_dim, seed,
               perception_fn, device="cpu", n_episodes=500):
    """Run one config with C_i measurement."""
    torch.manual_seed(seed)
    dev = torch.device(device)

    organism = Organism(
        sensory_dim=sensory_dim, embedding_dim=embed_dim, action_dim=2,
    ).to(dev)

    t0 = time.time()
    metrics = train_organism_with_perception(
        matter, organism, perception_fn,
        target_x=0.8, target_y=0.8,
        n_episodes=n_episodes, device=device,
    )
    elapsed = time.time() - t0

    ci_metrics = measure_coordination_quality(
        organism, perception_fn, matter, device=device,
    )

    n_tail = min(100, len(metrics["rock_distance"]))
    avg_dist = sum(metrics["rock_distance"][-n_tail:]) / n_tail
    avg_contact = sum(metrics["contact_rate"][-n_tail:]) / n_tail

    return {
        "level": level_name,
        "sensory_dim": sensory_dim,
        "embedding_dim": embed_dim,
        "seed": seed,
        "avg_dist_last100": avg_dist,
        "avg_contact_last100": avg_contact,
        "elapsed_s": elapsed,
        **ci_metrics,
    }


def main():
    device = os.environ.get("WORLDNN_DEVICE", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("WORLDNN_DEVICE=cuda set but CUDA unavailable")
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    checkpoint_path = results_dir / "ci_at_scale_checkpoint.json"
    results_path = results_dir / "ci_at_scale.json"

    embed_dims = [2, 4, 8, 16, 32]
    seeds = [42, 123, 456, 789, 1337, 2024, 3141]
    dev = torch.device(device)

    # Shared matter
    torch.manual_seed(0)
    matter = RockPushMatter(
        emission_dim=8, action_dim=2, seed_dim=4,
    ).to(dev)

    # Resume
    completed = []
    completed_keys = set()
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            completed = json.load(f)
        for r in completed:
            completed_keys.add((r["level"], r["embedding_dim"], r["seed"]))
        print(f"Resuming: {len(completed)} done")

    # ── Build all levels ──
    levels = []

    # Random baseline (untrained)
    def oracle_fn(state, emission):
        return state
    for emb in embed_dims:
        for s in seeds:
            levels.append(("random_baseline", 4, emb, s, oracle_fn, True))

    # Oracle
    for emb in embed_dims:
        for s in seeds:
            levels.append(("oracle", 4, emb, s, oracle_fn, False))

    # Oracle + noise
    for noise_std in [0.1, 0.5]:
        def make_noisy_fn(std):
            def fn(state, emission):
                return state + torch.randn_like(state) * std
            return fn
        for emb in embed_dims:
            for s in seeds:
                levels.append((
                    f"oracle_noise{noise_std}", 4, emb, s,
                    make_noisy_fn(noise_std), False,
                ))

    # Raw emission
    def emission_fn(state, emission):
        return emission
    for emb in embed_dims:
        for s in seeds:
            levels.append(("raw_emission", 8, emb, s, emission_fn, False))

    # VAE mu at different latent dims
    vae_probe_results = {}
    for lat_dim in [8, 16, 32]:
        torch.manual_seed(0)
        world = RockPushWorld(
            emission_dim=8, channel_dim=8, env_latent_dim=lat_dim,
            embedding_dim=8, action_dim=2, seed_dim=4,
            channel_noise=0.01, target_x=0.8, target_y=0.8,
        ).to(dev)
        world.matter = matter

        print(f"Pre-training VAE (lat={lat_dim})...", end=" ", flush=True)
        vae_losses = train_environment_rockpush(world, n_steps=1500, device=dev)
        print(f"loss={vae_losses[-1]:.4f}")

        probe = probe_vae_latent(world, device=device)
        vae_probe_results[lat_dim] = probe
        print(f"  Probe R²: mean={probe['r2_mean']:.3f}")

        env = world.environment
        ch = world.channel
        def make_vae_fn(env_module, ch_module):
            def fn(state, emission):
                with torch.no_grad():
                    ch_out = ch_module(emission)
                    mu, _ = env_module.encode(ch_out)
                return mu
            return fn
        vae_fn = make_vae_fn(env, ch)
        for emb in embed_dims:
            for s in seeds:
                levels.append((f"vae_mu_lat{lat_dim}", lat_dim, emb, s, vae_fn, False))

    total = len(levels)
    print(f"\nobj-016 at-scale sweep: {total} configs")
    t0 = time.time()

    for level_name, sensory_dim, emb, s, perc_fn, is_baseline in levels:
        key = (level_name, emb, s)
        if key in completed_keys:
            continue

        idx = len(completed) + 1
        print(f"[{idx}/{total}] {level_name}, emb={emb}, seed={s}",
              end=" ... ", flush=True)

        try:
            if is_baseline:
                result = measure_random_baseline(
                    matter, sensory_dim, emb, s, perc_fn, device=device,
                )
            else:
                result = run_config(
                    level_name, matter, sensory_dim, emb, s, perc_fn,
                    device=device,
                )
            completed.append(result)
            completed_keys.add(key)
            print(f"dist={result['avg_dist_last100']:.3f}, "
                  f"C_i={result['C_i']:.3f} "
                  f"({result['elapsed_s']:.0f}s)")
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback; traceback.print_exc()
            completed.append({
                "level": level_name, "sensory_dim": sensory_dim,
                "embedding_dim": emb, "seed": s, "error": str(e),
            })
            completed_keys.add(key)

        if len(completed) % 7 == 0:
            with open(checkpoint_path, "w") as f:
                json.dump(completed, f, indent=2)
            elapsed = time.time() - t0
            done_this_run = sum(1 for r in completed if (r["level"], r["embedding_dim"], r["seed"]) not in set())
            remaining = total - len(completed_keys)
            done_count = len(completed_keys) - (total - len(levels))
            if done_count > 0 and elapsed > 0:
                rate = done_count / elapsed
                if rate > 0:
                    print(f"  [checkpoint] {len(completed_keys)}/{total}, "
                          f"ETA: {remaining/rate/60:.0f} min")

    elapsed = time.time() - t0

    output = {
        "results": completed,
        "vae_probe_results": {str(k): v for k, v in vae_probe_results.items()},
        "total_configs": total,
        "elapsed_seconds": elapsed,
    }
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    with open(checkpoint_path, "w") as f:
        json.dump(completed, f, indent=2)

    n_ok = len([r for r in completed if "error" not in r])
    print(f"\nDone: {n_ok}/{total} in {elapsed/60:.1f} min")

    print_summary(completed)

    try:
        generate_plot(completed, vae_probe_results, results_dir)
    except Exception as e:
        print(f"Plot failed: {e}")
        import traceback; traceback.print_exc()


def print_summary(completed):
    import numpy as np

    valid = [r for r in completed if "error" not in r]

    # Random baseline stats
    baselines = [r for r in valid if r["level"] == "random_baseline"]
    if baselines:
        bl_dist = np.mean([r["avg_dist_last100"] for r in baselines])
        bl_ci = np.mean([r["C_i"] for r in baselines])
        print(f"\n=== Random Baseline ===")
        print(f"  dist={bl_dist:.3f} ± {np.std([r['avg_dist_last100'] for r in baselines]):.3f}")
        print(f"  C_i={bl_ci:.3f} ± {np.std([r['C_i'] for r in baselines]):.3f}")

    # Trained configs
    trained = [r for r in valid if r["level"] != "random_baseline"]
    print(f"\n=== Summary by level (n={len(trained)}) ===")
    by_level = defaultdict(list)
    for r in trained:
        by_level[r["level"]].append(r)

    for level in sorted(by_level):
        rs = by_level[level]
        dists = [r["avg_dist_last100"] for r in rs]
        cis = [r["C_i"] for r in rs]
        print(f"  {level:20s}: n={len(rs):>2}, "
              f"dist={np.mean(dists):.3f}±{np.std(dists):.3f}, "
              f"C_i={np.mean(cis):.3f}±{np.std(cis):.3f}")

    # Overall correlation (trained only)
    if len(trained) >= 10:
        dists = [r["avg_dist_last100"] for r in trained]
        cis = [r["C_i"] for r in trained]
        corr = np.corrcoef(dists, cis)[0, 1]
        print(f"\n  Overall r = {corr:.3f} (n={len(trained)})")

    # Between-level vs within-level decomposition
    print("\n=== Correlation Decomposition ===")
    level_means_d, level_means_c = [], []
    within_corrs = []
    for level, rs in by_level.items():
        dists = [r["avg_dist_last100"] for r in rs]
        cis = [r["C_i"] for r in rs]
        level_means_d.append(np.mean(dists))
        level_means_c.append(np.mean(cis))
        if len(rs) >= 5:
            within_corrs.append(np.corrcoef(dists, cis)[0, 1])

    if len(level_means_d) >= 3:
        between_r = np.corrcoef(level_means_d, level_means_c)[0, 1]
        print(f"  Between-level r = {between_r:.3f} (n={len(level_means_d)} levels)")
    if within_corrs:
        print(f"  Within-level mean r = {np.mean(within_corrs):.3f} "
              f"(range {min(within_corrs):.3f} to {max(within_corrs):.3f}, "
              f"n={len(within_corrs)} levels)")

    # Threshold analysis with defined success criterion
    if baselines:
        bl_dist = np.mean([r["avg_dist_last100"] for r in baselines])
        success_thresh = bl_dist - 0.02  # Must beat random by at least 0.02
        print(f"\n=== Threshold Analysis (success = dist < {success_thresh:.3f}) ===")
        for ci_thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            above = [r for r in trained if r["C_i"] >= ci_thresh]
            below = [r for r in trained if r["C_i"] < ci_thresh]
            a_success = len([r for r in above if r["avg_dist_last100"] < success_thresh])
            b_success = len([r for r in below if r["avg_dist_last100"] < success_thresh])
            a_rate = a_success / max(len(above), 1)
            b_rate = b_success / max(len(below), 1)
            print(f"  C_i >= {ci_thresh}: {len(above):>3} configs, "
                  f"{a_success:>3} succeed ({a_rate:.0%})")


def generate_plot(completed, probe_results, results_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    valid = [r for r in completed if "error" not in r]
    baselines = [r for r in valid if r["level"] == "random_baseline"]
    trained = [r for r in valid if r["level"] != "random_baseline"]
    if not trained:
        return

    bl_dist = np.mean([r["avg_dist_last100"] for r in baselines]) if baselines else 0.5

    level_colors = {
        "oracle": "#1f77b4",
        "oracle_noise0.1": "#6baed6",
        "oracle_noise0.5": "#bdd7e7",
        "raw_emission": "#2ca02c",
        "vae_mu_lat8": "#ff7f0e",
        "vae_mu_lat16": "#d62728",
        "vae_mu_lat32": "#9467bd",
        "random_baseline": "#888888",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel 1: THE KEY SCATTER — C_i vs Distance
    ax = axes[0, 0]
    for r in trained:
        color = level_colors.get(r["level"], "gray")
        ax.scatter(r["C_i"], r["avg_dist_last100"], color=color,
                  s=r["embedding_dim"] * 2.5, alpha=0.5,
                  edgecolors="black", linewidth=0.2)

    # Random baseline band
    if baselines:
        bl_cis = [r["C_i"] for r in baselines]
        bl_dists = [r["avg_dist_last100"] for r in baselines]
        ax.axhspan(np.mean(bl_dists) - np.std(bl_dists),
                   np.mean(bl_dists) + np.std(bl_dists),
                   alpha=0.15, color="gray", label="Random baseline ±1σ")

    # Fit line
    all_ci = [r["C_i"] for r in trained]
    all_dist = [r["avg_dist_last100"] for r in trained]
    corr = np.corrcoef(all_ci, all_dist)[0, 1]
    z_fit = np.polyfit(all_ci, all_dist, 1)
    x_range = np.linspace(min(all_ci), max(all_ci), 100)
    ax.plot(x_range, np.polyval(z_fit, x_range), "k--", alpha=0.4)
    ax.text(0.05, 0.95, f"r = {corr:.3f}\nn = {len(trained)}",
            transform=ax.transAxes, fontsize=11, fontweight="bold",
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    for level in level_colors:
        if level != "random_baseline" and any(r["level"] == level for r in trained):
            ax.scatter([], [], color=level_colors[level], label=level, s=30)
    ax.set_xlabel("Coordination Quality C_i")
    ax.set_ylabel("Rock-Target Distance")
    ax.set_title(f"C_i Predicts Performance (n={len(trained)}, 7 seeds)")
    ax.legend(fontsize=6, loc="upper right", ncol=2)

    # Panel 2: Between-level vs within-level
    ax = axes[0, 1]
    by_level = defaultdict(list)
    for r in trained:
        by_level[r["level"]].append(r)

    level_names, level_mean_ci, level_mean_dist, level_within_r = [], [], [], []
    for level, rs in sorted(by_level.items()):
        dists = [r["avg_dist_last100"] for r in rs]
        cis = [r["C_i"] for r in rs]
        level_names.append(level)
        level_mean_ci.append(np.mean(cis))
        level_mean_dist.append(np.mean(dists))
        if len(rs) >= 5:
            level_within_r.append(np.corrcoef(dists, cis)[0, 1])
        else:
            level_within_r.append(np.nan)

    # Scatter of level means
    for i, (name, ci, dist) in enumerate(zip(level_names, level_mean_ci, level_mean_dist)):
        color = level_colors.get(name, "gray")
        ax.scatter(ci, dist, color=color, s=100, edgecolors="black",
                  linewidth=1, zorder=5, label=name)
    if len(level_mean_ci) >= 3:
        between_r = np.corrcoef(level_mean_ci, level_mean_dist)[0, 1]
        z_fit2 = np.polyfit(level_mean_ci, level_mean_dist, 1)
        x2 = np.linspace(min(level_mean_ci), max(level_mean_ci), 50)
        ax.plot(x2, np.polyval(z_fit2, x2), "k--", alpha=0.4)
        ax.text(0.05, 0.95, f"Between-level r = {between_r:.3f}",
                transform=ax.transAxes, fontsize=10, fontweight="bold",
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.set_xlabel("Mean C_i per condition")
    ax.set_ylabel("Mean distance per condition")
    ax.set_title("Between-Level Correlation (level means)")
    ax.legend(fontsize=6, loc="upper right")

    # Panel 3: Threshold analysis bar chart
    ax = axes[1, 0]
    success_thresh = bl_dist - 0.02 if baselines else 0.48
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    rates = []
    ns = []
    for t in thresholds:
        above = [r for r in trained if r["C_i"] >= t]
        if above:
            rate = len([r for r in above if r["avg_dist_last100"] < success_thresh]) / len(above)
            rates.append(rate)
            ns.append(len(above))
        else:
            rates.append(0)
            ns.append(0)
    bars = ax.bar([f"≥{t}" for t in thresholds], rates, color="steelblue", alpha=0.85)
    for bar, rate, n in zip(bars, rates, ns):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{rate:.0%}\n(n={n})", ha="center", fontsize=8)
    ax.set_ylabel("Learning Success Rate")
    ax.set_xlabel("C_i Threshold")
    ax.set_title(f"Success = dist < {success_thresh:.3f} (random baseline - 0.02)")
    ax.set_ylim(0, 1.15)

    # Panel 4: Within-level correlations
    ax = axes[1, 1]
    valid_within = [(n, r) for n, r in zip(level_names, level_within_r) if not np.isnan(r)]
    if valid_within:
        names_w, rs_w = zip(*valid_within)
        colors_w = [level_colors.get(n, "gray") for n in names_w]
        bars = ax.barh(range(len(names_w)), rs_w, color=colors_w, alpha=0.85)
        ax.set_yticks(range(len(names_w)))
        ax.set_yticklabels(names_w, fontsize=8)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Within-Level r(C_i, distance)")
        ax.set_title(f"Within-Level Correlations (mean={np.nanmean(rs_w):.3f})")
        for i, r in enumerate(rs_w):
            ax.text(r + 0.02 if r >= 0 else r - 0.08, i, f"{r:.2f}", va="center", fontsize=9)

    fig.suptitle(f"obj-016: C_i At Scale — {len(trained)} configs, 7 seeds, random baseline",
                 fontsize=14)
    plt.tight_layout()
    out = results_dir / "obj016_ci_at_scale.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
