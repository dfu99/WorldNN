"""Generate publication-quality figure for SA proxy validation (obj-022)."""

import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def load_results():
    """Load the best available results (GPU > CPU)."""
    results_dir = Path(__file__).parent.parent / "results"
    for name in ["sa_proxy_expanded_gpu.json", "sa_proxy_expanded.json",
                  "sa_proxy_expanded_gpu_checkpoint.json",
                  "sa_proxy_expanded_checkpoint.json"]:
        p = results_dir / name
        if p.exists():
            data = json.load(open(p))
            valid = [r for r in data if "error" not in r]
            if len(valid) >= 15:
                print(f"Loaded {len(valid)} results from {name}")
                return valid
    raise FileNotFoundError("No proxy results found")


def main():
    results_dir = Path(__file__).parent.parent / "results"
    valid = load_results()

    # Check which proxies are available
    has_d = any("proxy_d_value" in r for r in valid)

    colors = {"oracle": "#2196F3", "vae_mu_lat16": "#FF5722"}
    markers = {2: "o", 8: "s", 32: "D"}
    perc_labels = {"oracle": "Oracle (4D state)", "vae_mu_lat16": "VAE $\\mu$ (16D latent)"}

    # ===== Figure 1: 3-panel proxy comparison =====
    proxy_list = [
        ("A: Prediction\nConsistency", "proxy_a_value"),
        ("D: Action\nVariance", "proxy_d_value") if has_d else ("B: Action\nStability", "proxy_b_value"),
        ("E: Policy\nConsistency", "proxy_e_value") if has_d else ("C: Value-Action\nAlignment", "proxy_c_value"),
    ]

    true_sa = np.array([r["true_SA"] for r in valid])
    dists = np.array([r["avg_dist"] for r in valid])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for ax, (pname, pkey) in zip(axes, proxy_list):
        vals = np.array([r.get(pkey, 0) for r in valid])

        for r in valid:
            c = colors.get(r["perception"], "gray")
            m = markers.get(r["embedding_dim"], "o")
            ax.scatter(r.get(pkey, 0), r["avg_dist"],
                      color=c, marker=m, s=60, alpha=0.7,
                      edgecolors="black", linewidth=0.5)

        # Fit line
        r_val = np.corrcoef(vals, dists)[0, 1]
        z = np.polyfit(vals, dists, 1)
        x_range = np.linspace(vals.min(), vals.max(), 100)
        ax.plot(x_range, np.polyval(z, x_range), "k--", alpha=0.4, linewidth=1.5)

        ax.set_xlabel(f"Proxy {pname}", fontsize=10)
        ax.set_ylabel("Rock-Target Distance", fontsize=10)
        ax.text(0.05, 0.95, f"$r = {r_val:+.3f}$", transform=ax.transAxes,
                fontsize=11, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

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
               fontsize=9, bbox_to_anchor=(0.5, -0.06))

    fig.suptitle("Oracle-Free Proxy Candidates vs Task Performance", fontsize=13)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    out1 = results_dir / "obj022_proxy_comparison.png"
    plt.savefig(out1, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out1}")

    # ===== Figure 2: Best proxy (action_var) deep dive =====
    if not has_d:
        print("No proxy D data, skipping deep dive figure")
        return

    action_var = np.array([r["proxy_d_action_var"] for r in valid])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel 1: action_var vs true SA
    for r in valid:
        c = colors.get(r["perception"], "gray")
        m = markers.get(r["embedding_dim"], "o")
        ax1.scatter(r["proxy_d_action_var"], r["true_SA"],
                   color=c, marker=m, s=70, alpha=0.7,
                   edgecolors="black", linewidth=0.5)

    r_full = np.corrcoef(action_var, true_sa)[0, 1]
    z = np.polyfit(action_var, true_sa, 1)
    x_range = np.linspace(action_var.min(), action_var.max(), 100)
    ax1.plot(x_range, np.polyval(z, x_range), "k--", alpha=0.4, linewidth=1.5)
    ax1.set_xlabel("Action Variance (oracle-free)", fontsize=11)
    ax1.set_ylabel("True SA (oracle-required)", fontsize=11)
    ax1.set_title("Proxy vs True Metric", fontsize=12)
    ax1.text(0.05, 0.95, f"$r = {r_full:+.3f}$", transform=ax1.transAxes,
             fontsize=11, verticalalignment="top",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Panel 2: action_var vs distance (the useful comparison)
    for r in valid:
        c = colors.get(r["perception"], "gray")
        m = markers.get(r["embedding_dim"], "o")
        ax2.scatter(r["proxy_d_action_var"], r["avg_dist"],
                   color=c, marker=m, s=70, alpha=0.7,
                   edgecolors="black", linewidth=0.5)

    r_dist = np.corrcoef(action_var, dists)[0, 1]
    z = np.polyfit(action_var, dists, 1)
    x_range = np.linspace(action_var.min(), action_var.max(), 100)
    ax2.plot(x_range, np.polyval(z, x_range), "k--", alpha=0.4, linewidth=1.5)
    ax2.set_xlabel("Action Variance (oracle-free)", fontsize=11)
    ax2.set_ylabel("Rock-Target Distance", fontsize=11)
    ax2.set_title("Proxy vs Task Performance", fontsize=12)
    ax2.text(0.05, 0.95, f"$r = {r_dist:+.3f}$", transform=ax2.transAxes,
             fontsize=11, verticalalignment="top",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Within-perception annotations
    for perc_name, perc_key in [("Oracle", "oracle"), ("VAE", "vae_mu_lat16")]:
        items = [r for r in valid if r["perception"] == perc_key]
        av = np.array([r["proxy_d_action_var"] for r in items])
        dd = np.array([r["avg_dist"] for r in items])
        r_within = np.corrcoef(av, dd)[0, 1]

    fig.legend(handles=legend_elements[:2], loc='lower center', ncol=2,
               fontsize=9, bbox_to_anchor=(0.5, -0.04))

    fig.suptitle("Action Variance as Oracle-Free SA Proxy", fontsize=13)
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    out2 = results_dir / "obj022_proxy_best.png"
    plt.savefig(out2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out2}")

    # Print summary table
    print("\n=== SUMMARY TABLE ===")
    print(f"{'Proxy':<30} {'r(SA)':>8} {'r(dist)':>8} {'Oracle r(dist)':>14} {'VAE r(dist)':>12}")
    print("-" * 75)
    for pname, pkey in [
        ("A: Pred Consistency", "proxy_a_value"),
        ("B: Action Stability", "proxy_b_value"),
        ("C: Value-Action Align", "proxy_c_value"),
        ("D: Action Variance", "proxy_d_value"),
        ("D.1: Var only", "proxy_d_action_var"),
        ("D.2: Magnitude only", "proxy_d_mag"),
        ("E: Policy Consistency", "proxy_e_value"),
    ]:
        vals = np.array([r.get(pkey, 0) for r in valid])
        r_sa = np.corrcoef(vals, true_sa)[0, 1]
        r_d = np.corrcoef(vals, dists)[0, 1]

        # Within-perception
        oracle_items = [r for r in valid if r["perception"] == "oracle"]
        vae_items = [r for r in valid if r["perception"] == "vae_mu_lat16"]
        ov = np.array([r.get(pkey, 0) for r in oracle_items])
        od = np.array([r["avg_dist"] for r in oracle_items])
        vv = np.array([r.get(pkey, 0) for r in vae_items])
        vd = np.array([r["avg_dist"] for r in vae_items])
        r_o = np.corrcoef(ov, od)[0, 1] if len(oracle_items) > 5 else float('nan')
        r_v = np.corrcoef(vv, vd)[0, 1] if len(vae_items) > 5 else float('nan')

        print(f"{pname:<30} {r_sa:+8.3f} {r_d:+8.3f} {r_o:+14.3f} {r_v:+12.3f}")


if __name__ == "__main__":
    main()
