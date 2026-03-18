# Planning

## Current Priority: Run full pipeline with adequate perception

The perception ladder (obj-011) diagnosed the VAE failure and mapped the
full transition. Root cause: VAE lat=4 compresses 8D→4D, losing 77% of
state variance (rock_y R²=0.044). Raw emission works fine (0.438).

**Key findings from the ladder:**
- Oracle: 0.387 (emb=32) — best possible, strong capacity effect
- Raw emission (8D): 0.438 — organism CAN learn from emissions
- VAE lat=16: 0.460 (emb=32) — adequate perception, capacity effect present
- VAE lat=4: 0.498 — broken (R²=0.229)
- Noise sensitivity sharp: oracle+noise(0.5) kills all learning

### Active

1. **Coordination quality C_i + mu fix** (obj-013) — SUBMITTED TO PACE (job 5027120)
   Fixed VAE pipeline: added `use_mu` flag to RockPushWorld + train loop.
   New experiment measures C_i (cosine similarity with optimal action) after
   training. 50 configs: oracle + VAE(mu) lat=16 × embed=[2,4,8,16,32] × 5 seeds.
   Awaiting PACE results. When done: fetch results, generate plot, assess
   whether C_i predicts task performance across conditions.

2. **Characterize the perception-capacity interaction** — The ladder shows
   capacity gap shrinks as perception degrades: oracle (+0.098), emission
   (+0.015), VAE lat=16 (+0.033). Is there a quantitative relationship
   between probe R² and the capacity gap?

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
- ANSWERED: VAE lat=4 destroys spatial info (R²=0.229, rock_y R²=0.044).
  lat=16 preserves 82% and enables learning (0.460). Raw emission works (0.438).
  The capacity effect scales with perception quality: oracle gap=+0.098,
  emission gap=+0.015, VAE lat=16 gap=+0.033, VAE lat=4 gap=+0.007.
- How complex must the task be before embedding dimension matters?
- Would multi-object or multi-bit state spaces create a real capacity
  bottleneck?
- Does episode length (perception-action cycles) interact with capacity?
- THEORETICAL: The sensory-motor alignment framework (rotation matrices
  and projections) unifies WorldNN and vaural. The organism's embedding
  layer IS the learned alignment operator $R$. See tasks/research.md.

## Recently Completed

- [2026-03-17] Formalized PI's sensory-motor alignment intuition (cat experiment, rotation matrices, neurotrophic constraints) in tasks/research.md. Assessed project fit: Intuition 1 bridges WorldNN+vaural, Intuition 2 (hierarchical LLM race condition) is standalone.

- [2026-03-16] obj-012: Oracle vs VAE lat=16 sweep — oracle capacity gap +0.115, standard VAE pipeline zero learning (z vs mu)
- [2026-03-16] obj-011: Perception ladder — VAE lat=4 root cause (R²=0.229), raw emission works (0.438), lat=16 enables learning (0.460)
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
