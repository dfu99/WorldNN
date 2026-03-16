"""Oracle vs VAE comparison plot for rock-push (obj-009).

Loads oracle results (oracle_expanded.json + oracle_gpu.json) and
VAE results (vae_comparison.json), produces a multi-panel figure
showing the perception bottleneck effect.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_oracle():
    """Load and merge oracle results."""
    results_dir = Path(__file__).parent.parent / "results"
    all_results = []

    for fname in ["oracle_expanded.json", "oracle_gpu.json"]:
        p = results_dir / fname
        if p.exists():
            with open(p) as f:
                data = json.load(f)
            all_results.extend(data)

    return all_results


def load_vae():
    """Load VAE comparison results."""
    results_dir = Path(__file__).parent.parent / "results"
    p = results_dir / "vae_comparison.json"
    if not p.exists():
        p = results_dir / "vae_comparison_checkpoint.json"
    if not p.exists():
        return []
    with open(p) as f:
        return json.load(f)


def get_dist(r):
    """Get distance metric from a result dict, handling different key names."""
    for k in ["avg_dist_last100", "avg_dist_last50", "converged_distance", "final_dist"]:
        if k in r:
            return r[k]
    return None


def get_contact(r):
    """Get contact metric from a result dict."""
    for k in ["avg_contact_last100", "avg_contact_last50", "final_contact"]:
        if k in r:
            return r[k]
    return None


def aggregate(results, metric_fn=None):
    """Group by embed dim, return {embed: (mean, std, n, values)}."""
    if metric_fn is None:
        metric_fn = get_dist
    by_emb = defaultdict(list)
    for r in results:
        if "error" in r:
            continue
        emb = r.get("embedding_dim", r.get("embed"))
        val = metric_fn(r)
        if val is not None:
            by_emb[emb].append(val)
    out = {}
    for emb, vals in sorted(by_emb.items()):
        out[emb] = (np.mean(vals), np.std(vals), len(vals), vals)
    return out


def success_rate(results, threshold=0.45):
    """Fraction of seeds with avg_dist < threshold."""
    by_emb = defaultdict(list)
    for r in results:
        if "error" in r:
            continue
        emb = r.get("embedding_dim", r.get("embed"))
        val = get_dist(r)
        if val is not None:
            by_emb[emb].append(val)
    out = {}
    for emb, vals in sorted(by_emb.items()):
        out[emb] = sum(1 for v in vals if v < threshold) / len(vals)
    return out


def main():
    oracle = load_oracle()
    vae_all = load_vae()

    if not oracle:
        print("No oracle results found!")
        return
    if not vae_all:
        print("No VAE results found!")
        return

    # Split VAE by noise level
    vae_low = [r for r in vae_all if r.get("channel_noise", 0) <= 0.02]
    vae_mod = [r for r in vae_all if 0.05 < r.get("channel_noise", 0) < 0.5]

    oracle_agg = aggregate(oracle)
    vae_low_agg = aggregate(vae_low) if vae_low else {}
    vae_mod_agg = aggregate(vae_mod) if vae_mod else {}

    oracle_success = success_rate(oracle)
    vae_low_success = success_rate(vae_low) if vae_low else {}
    vae_mod_success = success_rate(vae_mod) if vae_mod else {}

    embed_dims = sorted(oracle_agg.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ── Panel 1: Mean distance by embed dim ──
    ax = axes[0, 0]
    x = np.arange(len(embed_dims))
    width = 0.25

    oracle_means = [oracle_agg[e][0] for e in embed_dims]
    oracle_stds = [oracle_agg[e][1] for e in embed_dims]
    bars1 = ax.bar(x - width, oracle_means, width, yerr=oracle_stds,
                   label="Oracle (direct state)", color="steelblue", alpha=0.85, capsize=3)

    if vae_low_agg:
        vae_l_means = [vae_low_agg.get(e, (np.nan, 0, 0, []))[0] for e in embed_dims]
        vae_l_stds = [vae_low_agg.get(e, (np.nan, 0, 0, []))[1] for e in embed_dims]
        bars2 = ax.bar(x, vae_l_means, width, yerr=vae_l_stds,
                       label="VAE (noise=0.01)", color="coral", alpha=0.85, capsize=3)

    if vae_mod_agg:
        vae_m_means = [vae_mod_agg.get(e, (np.nan, 0, 0, []))[0] for e in embed_dims]
        vae_m_stds = [vae_mod_agg.get(e, (np.nan, 0, 0, []))[1] for e in embed_dims]
        bars3 = ax.bar(x + width, vae_m_means, width, yerr=vae_m_stds,
                       label="VAE (noise=0.1)", color="goldenrod", alpha=0.85, capsize=3)

    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Random baseline")
    ax.set_xticks(x)
    ax.set_xticklabels([str(e) for e in embed_dims])
    ax.set_xlabel("Embedding Dimension")
    ax.set_ylabel("Rock-Target Distance (lower = better)")
    ax.set_title("Perception Bottleneck: Oracle vs VAE")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 0.6)

    # ── Panel 2: Success rate ──
    ax = axes[0, 1]
    oracle_sr = [oracle_success.get(e, 0) for e in embed_dims]
    ax.plot(embed_dims, oracle_sr, "o-", label="Oracle", color="steelblue", linewidth=2)

    if vae_low_success:
        vae_l_sr = [vae_low_success.get(e, 0) for e in embed_dims]
        ax.plot(embed_dims, vae_l_sr, "s-", label="VAE (noise=0.01)", color="coral", linewidth=2)

    if vae_mod_success:
        vae_m_sr = [vae_mod_success.get(e, 0) for e in embed_dims]
        ax.plot(embed_dims, vae_m_sr, "^-", label="VAE (noise=0.1)", color="goldenrod", linewidth=2)

    ax.set_xlabel("Embedding Dimension")
    ax.set_ylabel("Success Rate (dist < 0.45)")
    ax.set_title("Reliability: How Often Does Learning Succeed?")
    ax.set_xscale("log", base=2)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 3: Individual seed scatter ──
    ax = axes[1, 0]
    np.random.seed(42)  # reproducible jitter
    for r in oracle:
        if "error" in r:
            continue
        emb = r.get("embedding_dim", r.get("embed"))
        val = get_dist(r)
        if val is not None:
            ax.scatter(emb + np.random.normal(0, 0.3), val,
                       color="steelblue", alpha=0.5, s=30, edgecolors="none")

    for r in vae_low:
        if "error" in r:
            continue
        emb = r.get("embedding_dim", r.get("embed"))
        val = get_dist(r)
        if val is not None:
            ax.scatter(emb + np.random.normal(0, 0.3), val,
                       color="coral", alpha=0.5, s=30, edgecolors="none")

    for r in vae_mod:
        if "error" in r:
            continue
        emb = r.get("embedding_dim", r.get("embed"))
        val = get_dist(r)
        if val is not None:
            ax.scatter(emb + np.random.normal(0, 0.3), val,
                       color="goldenrod", alpha=0.5, s=30, edgecolors="none")

    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Embedding Dimension")
    ax.set_ylabel("Rock-Target Distance")
    ax.set_title("Per-Seed Results (jittered)")
    ax.set_xscale("log", base=2)

    # ── Panel 4: VAE loss by embed dim (if available) ──
    ax = axes[1, 1]
    if vae_all and "vae_final_loss" in vae_all[0]:
        vae_losses = defaultdict(list)
        for r in vae_all:
            if "error" not in r and "vae_final_loss" in r:
                emb = r.get("embedding_dim", r.get("embed"))
                vae_losses[emb].append(r["vae_final_loss"])
        if vae_losses:
            embs_v = sorted(vae_losses.keys())
            means_v = [np.mean(vae_losses[e]) for e in embs_v]
            stds_v = [np.std(vae_losses[e]) for e in embs_v]
            ax.bar(range(len(embs_v)), means_v, yerr=stds_v,
                   tick_label=[str(e) for e in embs_v],
                   color="mediumpurple", alpha=0.8, capsize=3)
            ax.set_xlabel("Embedding Dimension")
            ax.set_ylabel("VAE Final Loss")
            ax.set_title("VAE Reconstruction Quality")
        else:
            ax.text(0.5, 0.5, "No VAE loss data", ha="center", va="center",
                    transform=ax.transAxes)
    else:
        # Gap analysis: how much does VAE cost?
        ax.text(0.5, 0.5, "Waiting for VAE results...", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")

    fig.suptitle("Rock-Push (obj-009): Quantifying the Perception Bottleneck\n"
                 "Oracle = direct 4D state | VAE = channel→VAE→latent",
                 fontsize=13)
    plt.tight_layout()

    results_dir = Path(__file__).parent.parent / "results"
    out_path = results_dir / "oracle_vs_vae_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

    # Print summary table
    print("\n=== Summary ===")
    print(f"{'Embed':>6} | {'Oracle':>12} | {'VAE low':>12} | {'VAE mod':>12} | {'Gap (low)':>10}")
    print("-" * 65)
    for e in embed_dims:
        o = oracle_agg[e][0]
        vl = vae_low_agg.get(e, (np.nan,))[0] if vae_low_agg else np.nan
        vm = vae_mod_agg.get(e, (np.nan,))[0] if vae_mod_agg else np.nan
        gap = vl - o if not np.isnan(vl) else np.nan
        print(f"{e:>6} | {o:>12.3f} | {vl:>12.3f} | {vm:>12.3f} | {gap:>10.3f}")


if __name__ == "__main__":
    main()
