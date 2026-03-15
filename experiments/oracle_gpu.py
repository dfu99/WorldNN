"""Oracle baseline on GPU — embed=8,16,32 with 5 seeds.

Supplements the CPU-run expanded oracle (embed=2,4 done).
"""
import json
import sys
import time
import numpy as np
import torch

sys.path.insert(0, "src")
from worldnn.matter import RockPushMatter
from worldnn.train import train_oracle_ppo_rockpush


EMBED_DIMS = [8, 16, 32]
SEEDS = [42, 123, 456, 789, 1337]
N_EPISODES = 500
BATCH_SIZE = 512
RESULTS_FILE = "results/oracle_gpu.json"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    results = []
    total = len(EMBED_DIMS) * len(SEEDS)
    idx = 0

    for embed in EMBED_DIMS:
        for seed in SEEDS:
            idx += 1
            torch.manual_seed(seed)
            np.random.seed(seed)

            matter = RockPushMatter().to(device)
            t0 = time.time()
            metrics = train_oracle_ppo_rockpush(
                matter,
                embedding_dim=embed,
                n_episodes=N_EPISODES,
                steps_per_episode=20,
                batch_size=BATCH_SIZE,
                device=device,
            )
            elapsed = time.time() - t0

            avg_dist = np.mean(metrics["rock_distance"][-50:])
            avg_contact = np.mean(metrics["contact_rate"][-50:])

            result = {
                "embed": embed,
                "seed": seed,
                "final_dist": round(metrics["rock_distance"][-1], 4),
                "avg_dist_last50": round(avg_dist, 4),
                "final_contact": round(metrics["contact_rate"][-1], 4),
                "avg_contact_last50": round(avg_contact, 4),
                "elapsed_s": round(elapsed, 1),
            }
            results.append(result)
            print(
                f"[{idx}/{total}] embed={embed}, seed={seed} | "
                f"dist={avg_dist:.3f} contact={avg_contact:.3f} ({elapsed:.0f}s)",
                flush=True,
            )

            with open(RESULTS_FILE, "w") as f:
                json.dump(results, f, indent=2)

    print("\n=== Summary ===")
    for embed in EMBED_DIMS:
        dists = [r["avg_dist_last50"] for r in results if r["embed"] == embed]
        contacts = [r["avg_contact_last50"] for r in results if r["embed"] == embed]
        print(
            f"embed={embed:2d}: dist={np.mean(dists):.3f}±{np.std(dists):.3f}, "
            f"contact={np.mean(contacts):.3f}±{np.std(contacts):.3f}"
        )


if __name__ == "__main__":
    main()
