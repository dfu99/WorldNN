# Planning

## Current Priority: Fix the perception pipeline or redesign experiment

The oracle vs VAE comparison (obj-010) is the most important finding yet:
the VAE **completely destroys** the capacity effect. Oracle shows embed=32
achieves 0.306, but VAE shows 0.502 (random) at ALL embed dims, even with
minimal channel noise (0.01). The VAE latent doesn't preserve spatial info.

**Key insight:** Before studying organism capacity limits, we must ensure
the perception pipeline preserves enough task-relevant information.

### Active

1. **Diagnose WHY the VAE fails for spatial tasks** (obj-011)
   The VAE final loss is ~0.11, which seems reasonable, but the organism
   can't use the latent for directional control. Possible causes:
   - VAE latent doesn't encode positional info (only energy/magnitude)
   - Emission design doesn't contain enough state info
   - VAE pre-training on random rollouts doesn't learn task-relevant features
   - env_latent_dim=4 is too small for 8D emissions with 4D state
   **Next steps:** Analyze VAE latent quality (probe for state recovery),
   try higher env_latent_dim, or redesign emissions to be more informative.

2. **Design perception-quality ladder** — Create a series of perception
   conditions between oracle (direct state) and full VAE: e.g., oracle +
   noise, oracle + linear compression, VAE with larger latent, etc.
   Find the threshold where capacity effects emerge/disappear.

3. **NN-based matter** — Replace explicit physics with learned Mealy machine
   for more complex matter dynamics.

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

- [2026-03-16] obj-010: VAE vs oracle comparison — VAE kills all capacity effects (0.502 everywhere vs oracle 0.306)
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
