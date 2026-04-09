"""obj-023: Mutual Information Chain — I(S;X), I(S;Y), I(S;Z), I(S;E) across perception conditions.

Computes the information-theoretic chain at each stage of the perception
pipeline for all 7 perception conditions used in the paper. This addresses
Gershman & Griffiths' reviewer objection: show where information dies
in the chain and relate it to SA.

Perception conditions (matching paper):
  - oracle (direct 4D state)
  - oracle+noise(0.1)
  - oracle+noise(0.5)
  - linear_proj (random 4→4 matrix)
  - vae_mu_lat16
  - vae_mu_lat8
  - vae_mu_lat4

For each condition × embed_dim × seed, we:
  1. Train the organism (reuse perception_ladder infrastructure)
  2. Collect (state, emission, channel_out, z, embedding) tuples
  3. Compute I(S;X), I(S;Y), I(S;Z), I(S;E) via KSG estimator
  4. Also compute SA for correlation with I(S;E)

Output: results/mi_chain.json + results/mi_chain.png
"""

import sys
import json
import time
import math
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from worldnn.world import RockPushWorld
from worldnn.matter import RockPushMatter
from worldnn.organism import Organism
from worldnn.channels import Channel
from worldnn.train import train_environment_rockpush
from worldnn.utils import estimate_mi_ksg

# Reuse training from perception_ladder
from perception_ladder import train_organism_with_perception


def compute_mi_chain(
    matter, organism, perception_fn, chain_components,
    n_samples=2000, device="cuda",
):
    """Compute MI at each stage of the perception chain.

    chain_components: dict with optional keys 'channel', 'environment'
    to compute intermediate MI. If None, only I(S;X) and I(S;E) are computed.

    Returns dict with I(S;X), I(S;Y), I(S;Z), I(S;E).
    """
    dev = torch.device(device)
    organism.eval()

    states, emissions, channel_outs, z_latents, embeddings = [], [], [], [], []

    with torch.no_grad():
        batch = 256
        for _ in range(n_samples // batch + 1):
            state = matter.reset_state(batch, dev)
            seed = torch.randn(batch, matter.seed_dim, device=dev)
            action = torch.randn(batch, 2, device=dev) * 0.1
            next_state, emission, _ = matter(state, seed, action)

            states.append(next_state.cpu())
            emissions.append(emission.cpu())

            # Channel output
            if 'channel' in chain_components:
                ch_out = chain_components['channel'](emission)
                channel_outs.append(ch_out.cpu())
            else:
                channel_outs.append(emission.cpu())  # no channel = identity

            # Environment latent
            if 'environment' in chain_components:
                ch_out = chain_components['channel'](emission) if 'channel' in chain_components else emission
                mu, _ = chain_components['environment'].encode(ch_out)
                z_latents.append(mu.cpu())
            else:
                z_latents.append(torch.zeros(batch, 1))  # placeholder

            # Organism embedding
            obs = perception_fn(next_state, emission)
            _, emb, _ = organism(obs)
            embeddings.append(emb.cpu())

    states = torch.cat(states)[:n_samples].numpy()
    emissions = torch.cat(emissions)[:n_samples].numpy()
    channel_outs = torch.cat(channel_outs)[:n_samples].numpy()
    z_latents = torch.cat(z_latents)[:n_samples].numpy()
    embeddings = torch.cat(embeddings)[:n_samples].numpy()

    # Use first state dim (rock_x) as the "hidden state" scalar
    # Also compute for full state vector
    s_scalar = states[:, 0:1]  # rock_x position
    s_full = states  # full 4D state

    result = {}
    # MI with full state
    result["I(S;X)"] = estimate_mi_ksg(s_full, emissions)
    result["I(S;Y)"] = estimate_mi_ksg(s_full, channel_outs)
    if 'environment' in chain_components:
        result["I(S;Z)"] = estimate_mi_ksg(s_full, z_latents)
    else:
        result["I(S;Z)"] = result["I(S;Y)"]  # no env = passthrough
    result["I(S;E)"] = estimate_mi_ksg(s_full, embeddings)

    # MI with scalar state (for cleaner interpretation)
    result["I(S1;E)"] = estimate_mi_ksg(s_scalar, embeddings)

    return result


def compute_sa(matter, organism, perception_fn, n_samples=1000, device="cuda"):
    """Compute SA = cos(a_learned, a_optimal) across samples."""
    dev = torch.device(device)
    organism.eval()

    cos_sims = []
    with torch.no_grad():
        batch = 256
        for _ in range(n_samples // batch + 1):
            state = matter.reset_state(batch, dev)
            seed = torch.randn(batch, matter.seed_dim, device=dev)
            action = torch.randn(batch, 2, device=dev) * 0.1
            next_state, emission, _ = matter(state, seed, action)

            # Optimal action: direction from organism to rock → target
            target = torch.tensor([0.8, 0.8], device=dev)
            rock_pos = next_state[:, :2]
            org_pos = next_state[:, 2:4]
            a_optimal = (rock_pos - org_pos)
            a_optimal = a_optimal / (a_optimal.norm(dim=-1, keepdim=True) + 1e-8)

            # Learned action under degraded perception
            obs = perception_fn(next_state, emission)
            a_learned, _, _ = organism(obs)
            a_learned = a_learned / (a_learned.norm(dim=-1, keepdim=True) + 1e-8)

            cos_sim = (a_learned * a_optimal).sum(dim=-1)
            cos_sims.append(cos_sim.cpu())

    cos_sims = torch.cat(cos_sims)[:n_samples]
    return cos_sims.mean().item()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    checkpoint_path = results_dir / "mi_chain_checkpoint.json"
    results_path = results_dir / "mi_chain.json"

    embed_dims = [8, 16, 32]
    seeds = [42, 123, 456]
    dev = torch.device(device)

    # Create shared matter
    torch.manual_seed(0)
    matter = RockPushMatter(
        emission_dim=8, action_dim=2, seed_dim=4,
    ).to(dev)

    # Pre-train VAEs
    vaes = {}
    for lat_dim in [4, 8, 16]:
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
        vaes[lat_dim] = (world.environment, world.channel)

    # Define perception conditions with chain components
    conditions = []

    # Oracle conditions (no channel/env)
    def oracle_fn(state, emission):
        return state
    for emb in embed_dims:
        for s in seeds:
            conditions.append({
                "name": "oracle", "sensory_dim": 4, "embed_dim": emb,
                "seed": s, "perception_fn": oracle_fn,
                "chain": {},
            })

    # Oracle + noise
    for noise_std in [0.1, 0.5]:
        def make_noisy(std):
            def fn(state, emission):
                return state + torch.randn_like(state) * std
            return fn
        for emb in embed_dims:
            for s in seeds:
                conditions.append({
                    "name": f"oracle+noise({noise_std})",
                    "sensory_dim": 4, "embed_dim": emb, "seed": s,
                    "perception_fn": make_noisy(noise_std),
                    "chain": {},
                })

    # Linear projection
    torch.manual_seed(99)
    proj_matrix = torch.randn(4, 4, device=dev) * 0.5
    proj_bias = torch.randn(4, device=dev) * 0.2
    def linear_proj_fn(state, emission):
        return state @ proj_matrix + proj_bias
    for emb in embed_dims:
        for s in seeds:
            conditions.append({
                "name": "linear_proj", "sensory_dim": 4, "embed_dim": emb,
                "seed": s, "perception_fn": linear_proj_fn,
                "chain": {},
            })

    # VAE conditions
    for lat_dim in [4, 8, 16]:
        env, ch = vaes[lat_dim]
        def make_vae_fn(env_m, ch_m):
            def fn(state, emission):
                with torch.no_grad():
                    ch_out = ch_m(emission)
                    mu, _ = env_m.encode(ch_out)
                return mu
            return fn
        vae_fn = make_vae_fn(env, ch)
        for emb in embed_dims:
            for s in seeds:
                conditions.append({
                    "name": f"vae_mu_lat{lat_dim}",
                    "sensory_dim": lat_dim, "embed_dim": emb, "seed": s,
                    "perception_fn": vae_fn,
                    "chain": {"channel": ch, "environment": env},
                })

    # Resume from checkpoint
    completed = []
    completed_keys = set()
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            completed = json.load(f)
        for r in completed:
            completed_keys.add((r["name"], r["embed_dim"], r["seed"]))
        print(f"Resuming: {len(completed)} done")

    total = len(conditions)
    print(f"\nMI chain: {total} configs")
    t0 = time.time()

    for i, cfg in enumerate(conditions):
        key = (cfg["name"], cfg["embed_dim"], cfg["seed"])
        if key in completed_keys:
            continue

        idx = len(completed) + 1
        print(f"[{idx}/{total}] {cfg['name']}, emb={cfg['embed_dim']}, seed={cfg['seed']}",
              end=" ... ", flush=True)

        try:
            torch.manual_seed(cfg["seed"])
            organism = Organism(
                sensory_dim=cfg["sensory_dim"],
                embedding_dim=cfg["embed_dim"],
                action_dim=2,
            ).to(dev)

            # Train
            metrics = train_organism_with_perception(
                matter, organism, cfg["perception_fn"],
                target_x=0.8, target_y=0.8,
                n_episodes=500, device=device,
            )

            # Compute MI chain
            mi = compute_mi_chain(
                matter, organism, cfg["perception_fn"],
                cfg["chain"], n_samples=2000, device=device,
            )

            # Compute SA
            sa = compute_sa(
                matter, organism, cfg["perception_fn"],
                n_samples=1000, device=device,
            )

            n_tail = min(100, len(metrics["rock_distance"]))
            avg_dist = sum(metrics["rock_distance"][-n_tail:]) / n_tail

            result = {
                "name": cfg["name"],
                "embed_dim": cfg["embed_dim"],
                "seed": cfg["seed"],
                "avg_dist": avg_dist,
                "SA": sa,
                **mi,
            }
            completed.append(result)
            print(f"SA={sa:.3f}, I(S;E)={mi['I(S;E)']:.3f}, dist={avg_dist:.3f}")

        except Exception as e:
            print(f"FAILED: {e}")
            import traceback; traceback.print_exc()
            completed.append({
                "name": cfg["name"], "embed_dim": cfg["embed_dim"],
                "seed": cfg["seed"], "error": str(e),
            })

        # Checkpoint every 3 configs
        if len(completed) % 3 == 0:
            with open(checkpoint_path, "w") as f:
                json.dump(completed, f, indent=2)

    elapsed = time.time() - t0

    # Save final
    with open(results_path, "w") as f:
        json.dump({"results": completed, "elapsed_s": elapsed}, f, indent=2)
    with open(checkpoint_path, "w") as f:
        json.dump(completed, f, indent=2)

    print(f"\nDone: {len(completed)}/{total} in {elapsed/60:.1f} min")

    # Summary
    print("\n=== MI Chain Summary ===")
    by_cond = defaultdict(list)
    for r in completed:
        if "error" not in r:
            by_cond[r["name"]].append(r)

    print(f"{'Condition':25s} {'I(S;X)':>8s} {'I(S;Y)':>8s} {'I(S;Z)':>8s} {'I(S;E)':>8s} {'SA':>8s} {'Dist':>8s}")
    for cond in ["oracle", "oracle+noise(0.1)", "oracle+noise(0.5)",
                 "linear_proj", "vae_mu_lat16", "vae_mu_lat8", "vae_mu_lat4"]:
        if cond in by_cond:
            rs = by_cond[cond]
            print(f"{cond:25s} "
                  f"{np.mean([r['I(S;X)'] for r in rs]):8.3f} "
                  f"{np.mean([r['I(S;Y)'] for r in rs]):8.3f} "
                  f"{np.mean([r['I(S;Z)'] for r in rs]):8.3f} "
                  f"{np.mean([r['I(S;E)'] for r in rs]):8.3f} "
                  f"{np.mean([r['SA'] for r in rs]):8.3f} "
                  f"{np.mean([r['avg_dist'] for r in rs]):8.3f}")

    # Generate plot
    try:
        generate_plot(completed, results_dir)
    except Exception as e:
        print(f"Plot failed: {e}")
        import traceback; traceback.print_exc()


def generate_plot(results, results_dir):
    """Generate MI chain visualization: 3 panels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = [r for r in results if "error" not in r]
    if not valid:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Color map for conditions
    cmap = {
        "oracle": "#2196F3",
        "oracle+noise(0.1)": "#64B5F6",
        "oracle+noise(0.5)": "#90CAF9",
        "linear_proj": "#FFC107",
        "vae_mu_lat16": "#FF9800",
        "vae_mu_lat8": "#FF5722",
        "vae_mu_lat4": "#D32F2F",
    }

    # Panel 1: MI chain waterfall (embed=32 average)
    ax = axes[0]
    stages = ["I(S;X)", "I(S;Y)", "I(S;Z)", "I(S;E)"]
    stage_labels = ["Emission\nI(S;X)", "Channel\nI(S;Y)", "Env Latent\nI(S;Z)", "Embedding\nI(S;E)"]
    cond_order = ["oracle", "oracle+noise(0.1)", "oracle+noise(0.5)",
                  "linear_proj", "vae_mu_lat16", "vae_mu_lat8", "vae_mu_lat4"]

    by_cond = defaultdict(list)
    for r in valid:
        if r["embed_dim"] == 32:
            by_cond[r["name"]].append(r)

    for cond in cond_order:
        if cond in by_cond:
            rs = by_cond[cond]
            mi_vals = [np.mean([r[s] for r in rs]) for s in stages]
            ax.plot(range(len(stages)), mi_vals, 'o-', label=cond,
                    color=cmap.get(cond, "gray"), linewidth=2, markersize=6)

    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stage_labels, fontsize=9)
    ax.set_ylabel("Mutual Information I(S; ·) [nats]")
    ax.set_title("Information Chain (embed=32)")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)

    # Panel 2: I(S;E) vs SA scatter
    ax = axes[1]
    for r in valid:
        c = cmap.get(r["name"], "gray")
        ax.scatter(r["I(S;E)"], r["SA"], color=c, alpha=0.6, s=40,
                  edgecolors="white", linewidth=0.5)

    # Compute correlation
    ise = [r["I(S;E)"] for r in valid]
    sa = [r["SA"] for r in valid]
    from scipy.stats import pearsonr
    r_val, p_val = pearsonr(ise, sa)
    ax.set_xlabel("I(S;E) [nats]")
    ax.set_ylabel("Sensorimotor Alignment (SA)")
    ax.set_title(f"I(S;E) vs SA (r={r_val:.3f}, p={p_val:.1e})")
    ax.grid(alpha=0.3)

    # Add legend manually
    for cond in cond_order:
        ax.scatter([], [], color=cmap.get(cond, "gray"), label=cond, s=40)
    ax.legend(fontsize=7, loc="lower right")

    # Panel 3: I(S;E) vs task performance (distance)
    ax = axes[2]
    for r in valid:
        c = cmap.get(r["name"], "gray")
        ax.scatter(r["I(S;E)"], r["avg_dist"], color=c, alpha=0.6, s=40,
                  edgecolors="white", linewidth=0.5)

    r_val2, p_val2 = pearsonr(ise, [r["avg_dist"] for r in valid])
    ax.set_xlabel("I(S;E) [nats]")
    ax.set_ylabel("Rock-Target Distance (lower = better)")
    ax.set_title(f"I(S;E) vs Performance (r={r_val2:.3f}, p={p_val2:.1e})")
    ax.grid(alpha=0.3)

    for cond in cond_order:
        ax.scatter([], [], color=cmap.get(cond, "gray"), label=cond, s=40)
    ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("obj-023: Mutual Information Chain Across Perception Conditions", fontsize=14)
    plt.tight_layout()
    out = results_dir / "mi_chain.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
