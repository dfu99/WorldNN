# Planning

## Current Priority: Investigate Stochastic Resonance & Advance Framework

PPO sweep is complete. Two milestones done: (1) framework built and demo
working, (2) perturbation study analyzed with PPO. The surprising stochastic
resonance finding at env_lat=1 opens a new research direction.

### Active

1. **Investigate stochastic resonance at env_lat=1** (obj-006)
   PACE job 4930637 (8hr, RTX 6000): 265 configs, ~3h in / 8h.
   Analysis pipeline ready (`experiments/analyze_stochastic_resonance.py`).

2. **Predictive processing experiment** (obj-007)
   PredictiveOrganism implemented: predicts next z, uses prediction
   error as auxiliary loss alongside PPO. 96 configs: noise × env_lat ×
   pred_coef × 3 seeds. PACE job 4932201 submitted (6hr, RTX 6000).

### Next Steps

3. **Continuous state spaces** — Extend matter beyond binary to
   continuous state (e.g., rock position), requiring finer control.

4. **Rock-pushing scenario** — Organism infers rock position via light,
   applies force, senses gravitational/friction feedback. Tests
   multi-channel perception + force-based action.

5. **NN-based matter** — Replace explicit physics with learned Mealy
   machine for more complex matter dynamics.

6. **Refine theoretical bounds** — Current Fano bounds are loose (predict
   dim_min=1 everywhere). Need tighter bounds that account for optimizer
   learnability, not just information availability.

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
- [2026-03-10] PPO trainer refactored into train.py; PPO sweep script ready
- [2026-03-13] PPO perturbation sweep (obj-005): 70/75 configs, stochastic resonance discovered at env_lat=1
- [2026-03-14] Analysis pipeline for stochastic resonance (obj-006): 5-figure suite with statistical tests
- [2026-03-14] Predictive processing implementation: PredictiveOrganism + PPO+Prediction trainer + PACE job 4932201
- [2026-03-14] Theoretical bounds (obj-007-theory): Fano/channel capacity analysis, 6-panel figure, key insight: bottleneck is learnability not information
