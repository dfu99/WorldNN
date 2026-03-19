# Planning

## Singular Result: C_i Predicts Sensorimotor Learning (r = -0.87)

**Living paper**: `paper/draft.md`

obj-013 established the headline: coordination quality C_i (cosine
alignment between learned policy and optimal action) predicts task
performance with r = -0.867. Sharp threshold: C_i ≥ 0.6 → 100% success,
C_i < 0.5 → 97% failure. Capacity and perception are multiplicative —
you need both to cross the threshold.

### Active

1. **Expanded C_i sweep** (obj-014) — Add more perception conditions to
   strengthen the C_i curve: raw emission (8D), oracle+noise(0.1, 0.5),
   VAE lat=4, VAE lat=32. Same embed_dim sweep. This populates the
   scatter plot with more data points and tests whether the r = -0.87
   and the threshold hold across ALL conditions, not just oracle vs one
   VAE setting.

2. **C_i dynamics during training** — Measure C_i every 50 episodes
   during training (not just post-hoc). Shows the *trajectory* of
   alignment learning. Predicts: oracle+large embed should show rapid
   C_i rise, while VAE conditions plateau early.

### Backlog (not distractions, but lower priority than the paper)

3. Multi-object / higher-dim task to stress capacity further
4. Cross-validate C_i in vaural (emitter alignment) and CorticalNN
   (per-layer alignment)
5. NN-based matter for more complex dynamics

### Parked (distractions removed)

- ~~GSNR learnability model recalibration~~ — superseded by C_i
- ~~Multi-step planning / episode length~~ — not relevant to the paper
- ~~Perception-capacity interaction characterization~~ — C_i IS this

## Open Questions

- ANSWERED: C_i predicts performance with r=-0.867. Threshold at ~0.5-0.6.
- ANSWERED: Capacity and perception are multiplicative, not additive.
- Does the C_i threshold generalize to other tasks?
- Can C_i dynamics during training predict final performance early?
- Does the r = -0.87 hold when we add more perception conditions?

## Recently Completed

- [2026-03-18] obj-013: C_i coordination quality r=-0.867, threshold at 0.5-0.6. VAE mu fix insufficient — VAE lat=16 caps C_i at 0.46.
- [2026-03-18] Cross-posted Intuition 1 to vaural and CorticalNN research docs
- [2026-03-17] Formalized PI's sensory-motor alignment intuition + prior art assessment
- [2026-03-16] obj-012: Oracle capacity gap +0.115, VAE pipeline zero learning (z vs mu)
- [2026-03-16] obj-011: Perception ladder — VAE diagnosis, raw emission works
- [2026-03-16] obj-010: VAE vs oracle comparison
- [2026-03-15] obj-009: Oracle baseline — capacity effect confirmed
- [2026-03-15] obj-006/007/008: Stochastic resonance, predictive processing, continuous tasks
