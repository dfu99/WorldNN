# Planning

## Current Priority: Synthesize findings and design higher-complexity tasks

Three PACE experiments completed (obj-006, obj-007, obj-008). Key takeaway:
for 1-bit/1D tasks, neither organism capacity nor predictive processing
matter — the bottleneck is purely optimizer quality (PPO vs REINFORCE).
The framework needs higher-complexity tasks to find where capacity limits
actually bite.

### Active

1. **Rock-pushing experiment** (obj-009)
   RockPushMatter + RockPushWorld implemented (4D state, multi-channel
   emissions, contact physics). 180 configs: noise × env_lat × embed_dim
   × 3 seeds. PACE job 4952915 submitted (8hr, RTX 6000). 29 tests passing.

2. **NN-based matter** — Replace explicit physics with learned Mealy machine
   for more complex matter dynamics. Could create tasks where embedding dim
   genuinely matters.

3. **Increase task complexity** — The current 1-bit / 1D tasks are too
   simple to stress organism capacity. Need multi-bit state, multi-step
   planning, or multi-object coordination to find the capacity boundary.

### Next Steps

4. **Refine learnability model** — GSNR model predicted resonance peak at
   σ=0.44; actual data shows flat response (no real resonance). Model needs
   recalibration with obj-006 data.

5. **Multi-step planning** — Vary episode length (currently 10 steps).
   Longer horizons may reveal capacity requirements.

## Open Questions

- ANSWERED: Stochastic resonance at env_lat=1 was seed variance, not a
  real effect. PPO response to noise is nearly flat until σ>0.5.
- ANSWERED: Predictive processing (next-state prediction) does not help
  for simple tasks. Binary state-flip is too easy for world models.
- ANSWERED: Embedding dim doesn't matter for continuous tasks either.
  1D position tracking is still too low-dimensional.
- How complex must the task be before embedding dimension matters?
- Would multi-object or multi-bit state spaces create a real capacity
  bottleneck?
- Does episode length (perception-action cycles) interact with capacity?

## Recently Completed

- [2026-03-15] obj-006: Stochastic resonance debunked — PPO flat ~0.82 across noise 0.01-0.5, no real peak
- [2026-03-15] obj-007: Predictive processing has zero effect (0.811 with vs 0.811 without)
- [2026-03-15] obj-008: Embedding dim negligible for continuous tasks (0.232-0.236 mean distance)
- [2026-03-14] Analysis pipeline for stochastic resonance (obj-006): 5-figure suite with statistical tests
- [2026-03-14] Predictive processing implementation: PredictiveOrganism + PPO+Prediction trainer
- [2026-03-14] Theoretical bounds (obj-007-theory): Fano/channel capacity analysis, bottleneck is learnability not information
- [2026-03-14] Continuous state spaces (obj-008-impl): ContinuousMatter, ContinuousWorld, PPO trainer
- [2026-03-14] Learnability bounds (obj-007-learn): GSNR model explains REINFORCE failure and dim irrelevance
- [2026-03-13] PPO perturbation sweep (obj-005): 70/75 configs
- [2026-03-10] Core framework, demos, perturbation study, latent failure analysis, PPO fix
