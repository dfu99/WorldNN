# Planning

## Singular Result: C_i Predicts Sensorimotor Learning (r = -0.74)

**Living paper**: `paper/draft.md`

The perception-action loop is asymmetric: organisms perceive through
lossy channels but act directly on true state. C_i measures this bridge.
Across 105 configs and 7 perception conditions, r = -0.735. Threshold:
C_i ≥ 0.8 → 100% success, C_i < 0.5 → 85% failure.

### Active

1. **C_i dynamics during training** (obj-015) — Measure C_i every 50
   episodes during training (not just post-hoc). Shows the *trajectory*
   of alignment learning. Key prediction: oracle+large embed should show
   rapid C_i rise crossing threshold early, while VAE conditions plateau
   below threshold. This would strengthen the paper by showing C_i is
   not just a post-hoc diagnostic but a real-time predictor.

### Backlog

2. Higher-dimensional task (multi-object) to stress capacity further
3. Cross-validate C_i in vaural (emitter alignment) and CorticalNN
   (per-layer alignment in bio-derived networks)
4. Investigate C_i estimation without access to optimal action
   (contrastive/self-supervised proxies)

## Open Questions

- ANSWERED: C_i predicts performance (r=-0.735 across 7 conditions)
- ANSWERED: Capacity × perception is multiplicative, not additive
- ANSWERED: Perception-action asymmetry is the core framing
- Does C_i trajectory during training predict final success early?
- Can C_i be estimated without knowing the optimal action?
- Does the threshold generalize to higher-dimensional tasks?

## Recently Completed

- [2026-03-20] Paper draft rewritten with perception-action asymmetry framing
- [2026-03-20] obj-014: Expanded C_i sweep — 105 configs, 7 conditions, r=-0.735
- [2026-03-18] obj-013: C_i coordination quality r=-0.867, threshold at 0.5-0.6
- [2026-03-18] Cross-posted Intuition 1 to vaural and CorticalNN
- [2026-03-17] Formalized sensory-motor alignment intuition + prior art assessment
- [2026-03-16] obj-012: Oracle capacity gap +0.115, VAE pipeline zero learning
- [2026-03-16] obj-011: Perception ladder — VAE diagnosis
- [2026-03-16] obj-010: VAE vs oracle comparison
- [2026-03-15] obj-009: Oracle baseline — capacity effect confirmed
