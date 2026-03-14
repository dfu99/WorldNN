#!/usr/bin/env python3
"""Analyze stochastic resonance results from obj-006.

Produces:
  1. Resonance curves with 95% CI (PPO & REINFORCE × env_lat 1 & 2)
  2. Heatmap: noise × algorithm, colored by success rate
  3. Statistical tests: is the resonance peak significantly > endpoints?
  4. Effect size analysis: Cohen's d for peak vs. low-noise
  5. Algorithm comparison: PPO advantage as function of noise
  6. Summary table printed to stdout

Run:
  python experiments/analyze_stochastic_resonance.py [--results-dir results]
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import math


def _ttest_ind_greater(a: list[float], b: list[float]) -> tuple[float, float]:
    """Welch's t-test (one-sided, alternative='greater'). Returns (t_stat, p_value)."""
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    n1, n2 = len(a), len(b)
    m1, m2 = np.mean(a), np.mean(b)
    v1, v2 = np.var(a, ddof=1), np.var(b, ddof=1)
    se = np.sqrt(v1 / n1 + v2 / n2)
    if se == 0:
        return (0.0, 0.5)
    t = (m1 - m2) / se
    # Welch-Satterthwaite degrees of freedom
    num = (v1 / n1 + v2 / n2) ** 2
    den = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
    df = num / den if den > 0 else 1
    # Approximate one-sided p-value using normal for large df, otherwise use
    # a simple incomplete beta approximation
    # For our purposes (df >= 4, moderate t), normal approx is fine
    p = 0.5 * math.erfc(t / math.sqrt(2))  # one-sided P(T > t) under H0
    return (float(t), float(p))


def load_results(results_dir: str) -> dict:
    """Load stochastic resonance results JSON."""
    path = Path(results_dir) / "stochastic_resonance_results.json"
    if not path.exists():
        # Try checkpoint
        path = Path(results_dir) / "stochastic_resonance_checkpoint.json"
    if not path.exists():
        print(f"No results found in {results_dir}/")
        print("Expected: stochastic_resonance_results.json or stochastic_resonance_checkpoint.json")
        sys.exit(1)

    with open(path) as f:
        data = json.load(f)
    print(f"Loaded {len(data['resonance_results'])} resonance results from {path}")
    if data.get("timeout_results"):
        print(f"  + {len(data['timeout_results'])} timeout rerun results")
    return data


def group_by(results: list[dict], keys: list[str]) -> dict:
    """Group results by a combination of keys."""
    groups = defaultdict(list)
    for r in results:
        key = tuple(r[k] for k in keys)
        groups[key].append(r["final_success"])
    return groups


def compute_stats(values: list[float]) -> dict:
    """Compute mean, std, 95% CI, and n for a list of values."""
    arr = np.array(values)
    n = len(arr)
    mean = np.mean(arr)
    std = np.std(arr, ddof=1) if n > 1 else 0.0
    se = std / np.sqrt(n) if n > 1 else 0.0
    ci95 = 1.96 * se
    return {"mean": mean, "std": std, "se": se, "ci95": ci95, "n": n}


def cohens_d(group1: list[float], group2: list[float]) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    m1, m2 = np.mean(group1), np.mean(group2)
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (m1 - m2) / pooled_std


def find_resonance_peak(groups: dict, algo: str, lat: int) -> tuple:
    """Find the noise level with highest mean success for given algo/lat."""
    best_noise, best_mean = None, -1
    for (noise, l, a), vals in groups.items():
        if a == algo and l == lat:
            m = np.mean(vals)
            if m > best_mean:
                best_mean = m
                best_noise = noise
    return best_noise, best_mean


def statistical_tests(results: list[dict]) -> dict:
    """Run statistical tests for stochastic resonance."""
    groups = group_by(results, ["channel_noise", "env_latent_dim", "algorithm"])
    tests = {}

    for algo in ["ppo", "reinforce"]:
        for lat in [1, 2]:
            peak_noise, peak_mean = find_resonance_peak(groups, algo, lat)
            if peak_noise is None:
                continue

            peak_vals = groups.get((peak_noise, lat, algo), [])
            # Compare peak to lowest noise (0.01)
            low_vals = groups.get((0.01, lat, algo), [])
            # Compare peak to highest noise (2.0)
            high_vals = groups.get((2.0, lat, algo), [])

            key = f"{algo}_lat{lat}"
            tests[key] = {
                "peak_noise": peak_noise,
                "peak_mean": np.mean(peak_vals),
                "peak_std": np.std(peak_vals, ddof=1) if len(peak_vals) > 1 else 0,
            }

            if low_vals and peak_vals and len(peak_vals) > 1 and len(low_vals) > 1:
                t_stat, p_val = _ttest_ind_greater(peak_vals, low_vals)
                tests[key]["vs_low"] = {
                    "t": t_stat, "p": p_val,
                    "d": cohens_d(peak_vals, low_vals),
                    "low_mean": np.mean(low_vals),
                }

            if high_vals and peak_vals and len(peak_vals) > 1 and len(high_vals) > 1:
                t_stat, p_val = _ttest_ind_greater(peak_vals, high_vals)
                tests[key]["vs_high"] = {
                    "t": t_stat, "p": p_val,
                    "d": cohens_d(peak_vals, high_vals),
                    "high_mean": np.mean(high_vals),
                }

    return tests


def plot_resonance_curves(results: list[dict], results_dir: str):
    """Publication-quality resonance curves with 95% CI."""
    groups = group_by(results, ["channel_noise", "env_latent_dim", "algorithm"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    colors = {"ppo": "#1565C0", "reinforce": "#E65100"}
    markers = {"ppo": "o", "reinforce": "s"}

    for ax, lat in zip(axes, [1, 2]):
        for algo in ["ppo", "reinforce"]:
            noise_levels = sorted(set(
                k[0] for k in groups if k[1] == lat and k[2] == algo
            ))
            if not noise_levels:
                continue

            means, ci_lo, ci_hi, noises = [], [], [], []
            for noise in noise_levels:
                vals = groups.get((noise, lat, algo), [])
                if not vals:
                    continue
                s = compute_stats(vals)
                means.append(s["mean"])
                ci_lo.append(s["mean"] - s["ci95"])
                ci_hi.append(s["mean"] + s["ci95"])
                noises.append(noise)

            means = np.array(means)
            ci_lo = np.array(ci_lo)
            ci_hi = np.array(ci_hi)

            ax.plot(noises, means, f"-{markers[algo]}", color=colors[algo],
                    label=algo.upper(), linewidth=2, markersize=6, zorder=3)
            ax.fill_between(noises, ci_lo, ci_hi, color=colors[algo], alpha=0.15, zorder=2)

        # Mark resonance peak for PPO at lat=1
        if lat == 1:
            peak_noise, peak_mean = find_resonance_peak(groups, "ppo", 1)
            if peak_noise and peak_noise not in (0.01, 2.0):  # Only annotate if not an endpoint
                ax.annotate(
                    f"Peak: σ={peak_noise}\n{peak_mean:.1%}",
                    xy=(peak_noise, peak_mean),
                    xytext=(peak_noise * 3, peak_mean - 0.15),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
                    fontsize=10, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF9C4", alpha=0.9,
                              edgecolor="#F9A825"),
                )

        ax.set_xlabel("Channel Noise σ", fontsize=12)
        if lat == 1:
            ax.set_ylabel("Success Rate", fontsize=12)
        ax.set_title(f"env_latent_dim = {lat}", fontsize=13)
        ax.legend(fontsize=11, loc="lower left")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xscale("log")
        ax.set_xlim(0.008, 2.5)

    fig.suptitle(
        "Stochastic Resonance in Perception-Action Loops\n"
        "Success rate vs. channel noise (embed_dim=4, 5 seeds, 95% CI)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    out = f"{results_dir}/sr_resonance_curves.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_heatmap(results: list[dict], results_dir: str):
    """Heatmap: noise × condition, colored by mean success."""
    groups = group_by(results, ["channel_noise", "env_latent_dim", "algorithm"])

    conditions = [("ppo", 1), ("reinforce", 1), ("ppo", 2), ("reinforce", 2)]
    cond_labels = ["PPO\nlat=1", "REINFORCE\nlat=1", "PPO\nlat=2", "REINFORCE\nlat=2"]
    noise_levels = sorted(set(k[0] for k in groups))

    matrix = np.full((len(conditions), len(noise_levels)), np.nan)
    for i, (algo, lat) in enumerate(conditions):
        for j, noise in enumerate(noise_levels):
            vals = groups.get((noise, lat, algo), [])
            if vals:
                matrix[i, j] = np.mean(vals)

    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_xticks(range(len(noise_levels)))
    ax.set_xticklabels([f"{n:.2f}" for n in noise_levels], rotation=45, ha="right")
    ax.set_yticks(range(len(conditions)))
    ax.set_yticklabels(cond_labels)
    ax.set_xlabel("Channel Noise σ", fontsize=12)

    # Annotate cells
    for i in range(len(conditions)):
        for j in range(len(noise_levels)):
            if not np.isnan(matrix[i, j]):
                color = "white" if matrix[i, j] < 0.4 or matrix[i, j] > 0.85 else "black"
                ax.text(j, i, f"{matrix[i, j]:.0%}", ha="center", va="center",
                        fontsize=8, fontweight="bold", color=color)

    plt.colorbar(im, ax=ax, label="Success Rate", shrink=0.8)
    ax.set_title("Success Rate Heatmap: All Conditions", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = f"{results_dir}/sr_heatmap.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_ppo_advantage(results: list[dict], results_dir: str):
    """PPO advantage (PPO - REINFORCE) as function of noise, by env_lat."""
    groups = group_by(results, ["channel_noise", "env_latent_dim", "algorithm"])

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {1: "#7B1FA2", 2: "#388E3C"}

    for lat in [1, 2]:
        noise_levels = sorted(set(
            k[0] for k in groups if k[1] == lat and k[2] == "ppo"
        ))
        diffs, ci_lo, ci_hi, noises = [], [], [], []

        for noise in noise_levels:
            ppo_vals = groups.get((noise, lat, "ppo"), [])
            rf_vals = groups.get((noise, lat, "reinforce"), [])
            if not ppo_vals or not rf_vals:
                continue

            # Bootstrap the difference
            ppo_arr = np.array(ppo_vals)
            rf_arr = np.array(rf_vals)
            diff = np.mean(ppo_arr) - np.mean(rf_arr)
            # Propagate SE
            se = np.sqrt(
                (np.std(ppo_arr, ddof=1)**2 / len(ppo_arr)) +
                (np.std(rf_arr, ddof=1)**2 / len(rf_arr))
            )
            diffs.append(diff)
            ci_lo.append(diff - 1.96 * se)
            ci_hi.append(diff + 1.96 * se)
            noises.append(noise)

        if diffs:
            ax.plot(noises, diffs, "-o", color=colors[lat], linewidth=2,
                    markersize=6, label=f"env_lat={lat}")
            ax.fill_between(noises, ci_lo, ci_hi, color=colors[lat], alpha=0.15)

    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5, linewidth=1)
    ax.set_xlabel("Channel Noise σ", fontsize=12)
    ax.set_ylabel("PPO − REINFORCE (Δ success rate)", fontsize=12)
    ax.set_title(
        "PPO Advantage Over REINFORCE vs. Noise Level\n"
        "Does the optimizer interact with stochastic resonance?",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    plt.tight_layout()
    out = f"{results_dir}/sr_ppo_advantage.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_variance_analysis(results: list[dict], results_dir: str):
    """Plot how variance across seeds changes with noise level."""
    groups = group_by(results, ["channel_noise", "env_latent_dim", "algorithm"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    colors = {"ppo": "#1565C0", "reinforce": "#E65100"}

    for ax, lat in zip(axes, [1, 2]):
        for algo in ["ppo", "reinforce"]:
            noise_levels = sorted(set(
                k[0] for k in groups if k[1] == lat and k[2] == algo
            ))
            stds, noises = [], []
            for noise in noise_levels:
                vals = groups.get((noise, lat, algo), [])
                if len(vals) > 1:
                    stds.append(np.std(vals, ddof=1))
                    noises.append(noise)

            if stds:
                ax.plot(noises, stds, f"-o", color=colors[algo], label=algo.upper(),
                        linewidth=2, markersize=6)

        ax.set_xlabel("Channel Noise σ", fontsize=12)
        if lat == 1:
            ax.set_ylabel("Std Dev of Success Rate (across seeds)", fontsize=12)
        ax.set_title(f"env_latent_dim = {lat}", fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")

    fig.suptitle(
        "Outcome Variability vs. Noise Level\n"
        "Does noise increase or decrease reliability?",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    out = f"{results_dir}/sr_variance.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_composite(results: list[dict], tests: dict, results_dir: str):
    """Four-panel composite figure for the paper/report."""
    groups = group_by(results, ["channel_noise", "env_latent_dim", "algorithm"])

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # Panel A: Resonance curves (lat=1 only, both algos)
    ax_a = fig.add_subplot(gs[0, 0])
    colors = {"ppo": "#1565C0", "reinforce": "#E65100"}
    markers = {"ppo": "o", "reinforce": "s"}

    for algo in ["ppo", "reinforce"]:
        noise_levels = sorted(set(k[0] for k in groups if k[1] == 1 and k[2] == algo))
        means, ci95s, noises = [], [], []
        for noise in noise_levels:
            vals = groups.get((noise, 1, algo), [])
            if vals:
                s = compute_stats(vals)
                means.append(s["mean"])
                ci95s.append(s["ci95"])
                noises.append(noise)
        means, ci95s = np.array(means), np.array(ci95s)
        ax_a.plot(noises, means, f"-{markers[algo]}", color=colors[algo],
                  label=algo.upper(), linewidth=2, markersize=5)
        ax_a.fill_between(noises, means - ci95s, means + ci95s, color=colors[algo], alpha=0.12)

    ax_a.set_xlabel("Channel Noise σ")
    ax_a.set_ylabel("Success Rate")
    ax_a.set_title("A. Stochastic Resonance (env_lat=1)", fontweight="bold")
    ax_a.legend(fontsize=10)
    ax_a.grid(True, alpha=0.3)
    ax_a.set_xscale("log")
    ax_a.set_ylim(-0.05, 1.05)

    # Panel B: Control (lat=2, both algos)
    ax_b = fig.add_subplot(gs[0, 1])
    for algo in ["ppo", "reinforce"]:
        noise_levels = sorted(set(k[0] for k in groups if k[1] == 2 and k[2] == algo))
        means, ci95s, noises = [], [], []
        for noise in noise_levels:
            vals = groups.get((noise, 2, algo), [])
            if vals:
                s = compute_stats(vals)
                means.append(s["mean"])
                ci95s.append(s["ci95"])
                noises.append(noise)
        means, ci95s = np.array(means), np.array(ci95s)
        ax_b.plot(noises, means, f"-{markers[algo]}", color=colors[algo],
                  label=algo.upper(), linewidth=2, markersize=5)
        ax_b.fill_between(noises, means - ci95s, means + ci95s, color=colors[algo], alpha=0.12)

    ax_b.set_xlabel("Channel Noise σ")
    ax_b.set_ylabel("Success Rate")
    ax_b.set_title("B. Control (env_lat=2, no resonance)", fontweight="bold")
    ax_b.legend(fontsize=10)
    ax_b.grid(True, alpha=0.3)
    ax_b.set_xscale("log")
    ax_b.set_ylim(-0.05, 1.05)

    # Panel C: PPO advantage
    ax_c = fig.add_subplot(gs[1, 0])
    lat_colors = {1: "#7B1FA2", 2: "#388E3C"}
    for lat in [1, 2]:
        noise_levels = sorted(set(k[0] for k in groups if k[1] == lat and k[2] == "ppo"))
        diffs, ses, noises = [], [], []
        for noise in noise_levels:
            ppo = groups.get((noise, lat, "ppo"), [])
            rf = groups.get((noise, lat, "reinforce"), [])
            if ppo and rf:
                diff = np.mean(ppo) - np.mean(rf)
                se = np.sqrt(
                    (np.std(ppo, ddof=1)**2 / len(ppo)) +
                    (np.std(rf, ddof=1)**2 / len(rf))
                )
                diffs.append(diff)
                ses.append(1.96 * se)
                noises.append(noise)
        diffs, ses = np.array(diffs), np.array(ses)
        if len(diffs):
            ax_c.plot(noises, diffs, "-o", color=lat_colors[lat], linewidth=2,
                      markersize=5, label=f"env_lat={lat}")
            ax_c.fill_between(noises, diffs - ses, diffs + ses, color=lat_colors[lat], alpha=0.12)

    ax_c.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    ax_c.set_xlabel("Channel Noise σ")
    ax_c.set_ylabel("PPO − REINFORCE")
    ax_c.set_title("C. PPO Advantage vs. Noise", fontweight="bold")
    ax_c.legend(fontsize=10)
    ax_c.grid(True, alpha=0.3)
    ax_c.set_xscale("log")

    # Panel D: Statistical summary table
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.axis("off")

    table_data = []
    headers = ["Condition", "Peak σ", "Peak %", "vs Low\np-val", "Cohen's d"]
    for key in ["ppo_lat1", "reinforce_lat1", "ppo_lat2", "reinforce_lat2"]:
        t = tests.get(key, {})
        if not t:
            continue
        label = key.replace("_", " ").replace("lat", "lat=").upper()
        peak = f"{t.get('peak_noise', '?')}"
        peak_pct = f"{t.get('peak_mean', 0):.1%}"
        vs_low = t.get("vs_low", {})
        p_str = f"{vs_low.get('p', 1):.4f}" if vs_low else "—"
        d_str = f"{vs_low.get('d', 0):.2f}" if vs_low else "—"
        table_data.append([label, peak, peak_pct, p_str, d_str])

    if table_data:
        table = ax_d.table(
            cellText=table_data, colLabels=headers,
            loc="center", cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor("#E3F2FD")
                cell.set_fontsize(10)
            cell.set_edgecolor("#BDBDBD")
    ax_d.set_title("D. Statistical Tests (peak vs. σ=0.01)", fontweight="bold", pad=20)

    fig.suptitle(
        "Stochastic Resonance in Perception-Action Loops (obj-006)",
        fontsize=15, fontweight="bold", y=0.98,
    )
    out = f"{results_dir}/sr_composite.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def print_summary(results: list[dict], tests: dict):
    """Print a text summary of findings."""
    groups = group_by(results, ["channel_noise", "env_latent_dim", "algorithm"])

    print("\n" + "=" * 70)
    print("STOCHASTIC RESONANCE ANALYSIS — OBJ-006 SUMMARY")
    print("=" * 70)

    for algo in ["ppo", "reinforce"]:
        for lat in [1, 2]:
            key = f"{algo}_lat{lat}"
            t = tests.get(key)
            if not t:
                continue

            print(f"\n--- {algo.upper()} | env_lat={lat} ---")
            print(f"  Peak noise:     σ = {t['peak_noise']}")
            print(f"  Peak success:   {t['peak_mean']:.1%} ± {t.get('peak_std', 0):.1%}")

            vs_low = t.get("vs_low")
            if vs_low:
                sig = "***" if vs_low["p"] < 0.001 else "**" if vs_low["p"] < 0.01 else "*" if vs_low["p"] < 0.05 else "n.s."
                print(f"  vs σ=0.01:      {vs_low['low_mean']:.1%} → {t['peak_mean']:.1%}  "
                      f"(t={vs_low['t']:.2f}, p={vs_low['p']:.4f} {sig}, d={vs_low['d']:.2f})")

            vs_high = t.get("vs_high")
            if vs_high:
                sig = "***" if vs_high["p"] < 0.001 else "**" if vs_high["p"] < 0.01 else "*" if vs_high["p"] < 0.05 else "n.s."
                print(f"  vs σ=2.0:       {vs_high['high_mean']:.1%} → {t['peak_mean']:.1%}  "
                      f"(t={vs_high['t']:.2f}, p={vs_high['p']:.4f} {sig}, d={vs_high['d']:.2f})")

    # Check for resonance: is peak at an interior point (not endpoint)?
    print("\n--- RESONANCE VERDICT ---")
    for algo in ["ppo", "reinforce"]:
        t = tests.get(f"{algo}_lat1")
        if t and t["peak_noise"] not in (0.01, 2.0):
            vs_low = t.get("vs_low", {})
            if vs_low and vs_low.get("p", 1) < 0.05:
                print(f"  {algo.upper()} lat=1: YES — significant resonance peak at σ={t['peak_noise']}")
            else:
                print(f"  {algo.upper()} lat=1: SUGGESTIVE — peak at σ={t['peak_noise']} but not significant")
        elif t:
            print(f"  {algo.upper()} lat=1: NO — peak at endpoint σ={t['peak_noise']}")

    t2 = tests.get("ppo_lat2")
    if t2:
        if t2["peak_noise"] in (0.01, 0.05):
            print(f"  PPO lat=2: NO resonance (monotonic decrease as expected)")
        else:
            print(f"  PPO lat=2: UNEXPECTED — peak at σ={t2['peak_noise']}")

    print("=" * 70)


def analyze_timeout_reruns(data: dict):
    """Print timeout rerun results if available."""
    timeout = data.get("timeout_results", [])
    if not timeout:
        return

    print("\n--- TIMEOUT RERUNS (noise=2.0, env_lat=4, 800 episodes) ---")
    for r in timeout:
        print(f"  embed={r['embedding_dim']}: {r['final_success']:.1%}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze stochastic resonance results")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    data = load_results(args.results_dir)
    results = data["resonance_results"]

    if not results:
        print("No resonance results to analyze.")
        return

    # Statistical tests
    tests = statistical_tests(results)

    # Print summary
    print_summary(results, tests)
    analyze_timeout_reruns(data)

    # Generate all plots
    print("\nGenerating plots...")
    plot_resonance_curves(results, args.results_dir)
    plot_heatmap(results, args.results_dir)
    plot_ppo_advantage(results, args.results_dir)
    plot_variance_analysis(results, args.results_dir)
    plot_composite(results, tests, args.results_dir)

    # Save tests as JSON for reference
    tests_path = f"{args.results_dir}/sr_statistical_tests.json"
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    import copy
    serializable = copy.deepcopy(tests)
    for key, val in serializable.items():
        for k2, v2 in val.items():
            if isinstance(v2, dict):
                for k3 in v2:
                    v2[k3] = convert(v2[k3])
            else:
                val[k2] = convert(v2)

    with open(tests_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Saved {tests_path}")

    print("\nDone! Generated 5 figures + 1 JSON summary.")


if __name__ == "__main__":
    main()
