# Planning

## Target: ICLR 2027 (October 2026 deadline)

**Living paper**: `paper/draft.md`
**Head Scientist critique**: `tasks/head-scientist-critique.md`
**Progress tracker**: `tasks/head-scientist-progress.md`

Sensorimotor alignment (C_i) predicts learning success across asymmetric
perception-action loops. At scale (245 configs, 7 seeds): r = -0.724,
interaction p = 5×10⁻⁹, within-level mean r = -0.582.

### Active — Head Scientist Priorities

1. **Second task — FIX REQUIRED** (obj-017 completed but NOBODY LEARNED)
   Multi-rock push: 3 rocks, 8D state. All conditions at random baseline
   (~0.493 vs 0.489). r = -0.194. C_i at emb=64 reaches ~0.44 (some
   alignment) but doesn't translate to performance. Suspected causes:
   (a) 500 episodes insufficient for 8D, (b) hidden_size=32 bottleneck,
   (c) reward too diffuse. MUST FIX AND RE-RUN before main-track.

2. **Self-supervised C_i proxy** — makes metric practical beyond simulation.
   Options: prediction error as proxy, contrastive alignment, value function
   gradient magnitude. Needs design + experiments.

3. **Rename + framing fixes** in paper draft:
   - "coordination quality" → "sensorimotor alignment" (avoid MARL confusion)
   - Lead with blind cat hook
   - Cite Pinto et al., DreamerV3, quasimetric RL

### Completed from Head Scientist Critique

- [x] Formal interaction test: F(1,241)=34.2, p=5×10⁻⁹ (obj-016)
- [x] Seeds increased to 7 (obj-016, 280 configs)
- [x] Random baseline: dist=0.516±0.003, C_i=0.003±0.022
- [x] Success criterion: dist < 0.511 (baseline - 2σ)
- [x] Correlation decomposition: between r=-0.878, within mean r=-0.582
- [x] C_i dynamics (obj-015): slope r=-0.705 predicts final success

### Backlog

4. Cosine vs L2 ablation — action_magnitude slightly outperforms C_i (r=-0.739 vs -0.724). Investigate.
5. Cross-validate in vaural and CorticalNN

## Open Questions

- ANSWERED: Interaction is significant (p=5×10⁻⁹ with 7 seeds)
- ANSWERED: Within-level r improved to -0.58 (was -0.47 with 3 seeds)
- ANSWERED: C_i slope predicts final success (r=-0.705)
- oracle_noise0.5 reversal persists — boundary condition, not a bug
- Action magnitude (r=-0.739) slightly outperforms cosine C_i (r=-0.724) — why?
- Does the threshold generalize to 8D+ tasks?

## Recently Completed

- [2026-03-22] obj-016: At-scale (245+35 configs, 7 seeds), interaction p=5e-9, within r=-0.58
- [2026-03-22] Formal interaction test on existing data (Head Scientist exp #1)
- [2026-03-21] obj-015: C_i dynamics — slope r=-0.705 predicts success
- [2026-03-20] obj-014: Expanded C_i sweep — 105 configs, r=-0.735
- [2026-03-20] Paper rewritten with perception-action asymmetry framing
- [2026-03-18] obj-013: C_i metric introduced, r=-0.867
- [2026-03-17] Sensory-motor alignment formalization + prior art
- [2026-03-16] obj-012: Oracle capacity gap +0.115, VAE pipeline broken
