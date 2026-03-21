# Planning

## Singular Result: C_i Predicts Sensorimotor Learning (r = -0.74)

**Living paper**: `paper/draft.md`

The perception-action loop is asymmetric: organisms perceive through
lossy channels but act directly on true state. C_i measures this bridge.
Across 105 configs and 7 perception conditions, r = -0.735. Threshold:
C_i ≥ 0.8 → 100% success, C_i < 0.5 → 85% failure.

### Active

1. **C_i dynamics during training** (obj-015) — SUBMITTED TO PACE (job 5264477)
   27 configs: oracle/emission/VAE × emb=2/8/32 × 3 seeds. Measures C_i
   every 50 episodes. Awaiting results.

2. **Address obj-014 critique** — See `tasks/critique_obj014.md`. Key fixes:
   - Fix threshold table (row-shift error, wrong counts)
   - Define success criterion with random-action baseline
   - Decompose correlation (between-level r=-0.894, within-level r=-0.47)
   - Increase seeds to 5 for key conditions
   - Address oracle_noise0.5 C_i reversal
   - Distinguish C_i from policy cosine similarity in imitation learning

### Backlog

3. Higher-dimensional task (multi-object) to stress capacity further
4. Cross-validate C_i in vaural and CorticalNN
5. C_i estimation without access to optimal action

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
