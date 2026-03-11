# Planning

## Current Priority: Perturbation Study Analysis

First milestone is complete — the framework is built, the demo works,
and the perturbation study is running. Next focus is analyzing results
and refining the experiment.

### Active

1. **Re-run perturbation study with PPO** — obj-004 showed PPO fixes
   the 1D learning failure (4.6% → 87.3%). The original 75-config sweep
   used REINFORCE, so env_lat=1 results reflect optimizer limits, not
   information bounds. Need to re-sweep with PPO to get true capacity
   curves. (Requires GPU for full 75-config sweep.)

### Next Steps

2. **Add predictive processing** — Organism predicts next observation
   and uses prediction error as additional learning signal.

4. **Continuous state spaces** — Extend matter beyond binary to
   continuous state (e.g., rock position), requiring finer control.

5. **Rock-pushing scenario** — Organism infers rock position via light,
   applies force, senses gravitational/friction feedback. Tests
   multi-channel perception + force-based action.

6. **NN-based matter** — Replace explicit physics with learned Mealy
   machine for more complex matter dynamics.

7. **Theoretical bounds** — Derive analytical min embedding dim as
   function of channel capacity and environment compression.

## Open Questions

- ANSWERED: REINFORCE fails on 1D due to gradient variance. PPO fixes it.
  Open: does the fix hold across all noise levels and embedding dims?
- How does the number of perception-action cycles affect success?
  (Currently fixed at 10 steps per episode.)
- Should the organism also do unsupervised world-model learning
  alongside RL?
- Does the learnability gap (separable but unlearnable) appear at
  other dimensionalities, or is 1D uniquely problematic?

## Recently Completed

- [2026-03-10] Project scaffolding: pyproject.toml, src/, tests/, experiments/
- [2026-03-10] Core framework: Matter, Channel, EnvironmentVAE, Organism, World
- [2026-03-10] Two-phase training: VAE pre-training + REINFORCE with Gaussian policy
- [2026-03-10] Paramecium chemotaxis demo — 85.6% success rate
- [2026-03-10] Architecture diagram with ML block details
- [2026-03-10] 14 unit tests passing
- [2026-03-10] Perturbation study launched (75 configurations)
- [2026-03-10] README with overview, architecture, results, future directions
- [2026-03-10] Latent failure analysis: H3 confirmed — env_lat=1 is RL bottleneck, not info bottleneck
- [2026-03-10] PPO fix: 4.6% → 87.3% success on env_lat=1 (19x improvement)
