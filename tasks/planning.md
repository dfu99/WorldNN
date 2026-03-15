# Planning

## Current Priority: Synthesize findings and design higher-complexity tasks

Three PACE experiments completed (obj-006, obj-007, obj-008). Key takeaway:
for 1-bit/1D tasks, neither organism capacity nor predictive processing
matter — the bottleneck is purely optimizer quality (PPO vs REINFORCE).
The framework needs higher-complexity tasks to find where capacity limits
actually bite.

### Active

1. **Rock-pushing experiment** (obj-009)
   **Oracle baseline COMPLETE (5 seeds × 5 embed dims):**
   embed=2: 0.501 (0/5 success) | embed=4: 0.448 (1/5) | embed=8: 0.356 (3/5)
   embed=16: 0.422 (4/5) | **embed=32: 0.306 (5/5 success, best)**
   Capacity monotonically improves reliability. embed=32 achieves genuine
   spatial control (best seed: dist=0.200, 63.5% contact).
   PACE job 4956171: Full VAE pipeline still running (~0.49 everywhere).
   **Next: compare oracle vs VAE to quantify perception bottleneck.**

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
- ANSWERED: Rock-push (4D state) shows clear capacity effect with oracle
  perception. embed=32 optimal (0.306, 100% success), embed=2 random (0.501).
  Reliability increases monotonically with capacity.
- How does VAE quality modulate the capacity effect? (VAE pipeline shows
  no learning at all — too lossy for spatial tasks)
- How complex must the task be before embedding dimension matters?
- Would multi-object or multi-bit state spaces create a real capacity
  bottleneck?
- Does episode length (perception-action cycles) interact with capacity?

## Recently Completed

- [2026-03-15] obj-009-oracle: Oracle baseline — embed dim matters! embed=8 best (0.395 vs 0.50 random)
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
