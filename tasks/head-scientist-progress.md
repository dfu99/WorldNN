# Head Scientist Critique — Progress Tracker

## Experiments (priority order from critique)

### 1. Formal interaction test ✅ DONE
- **obj-014 (105 configs)**: F(1,101) = 15.6, p = 7.9×10⁻⁵
- **obj-016 (245 configs, 7 seeds)**: F(1,241) = 34.2, p = 5.0×10⁻⁹
- Interaction β = -0.00354 (negative: capacity helps MORE with better perception)
- ΔR² from interaction = 0.077

### 2. Increase seeds to 5+ ✅ DONE (obj-016)
- Increased from 3 to 7 seeds per condition
- Within-level mean r improved: -0.47 (3 seeds) → -0.58 (7 seeds)
- oracle_noise0.5 reversal persists (+0.463) — this is a genuine
  boundary condition where perception is too degraded for C_i to be meaningful
- All other 6 conditions show r between -0.72 and -0.80

### 3. Second task (8D+ locomotion/manipulation) — NOT YET STARTED
- Non-negotiable for main track
- Need to design and implement
- Major PACE job

### 4. C_i dynamics during training ✅ DONE (obj-015)
- C_i slope in first 200 episodes predicts final success: r = -0.705
- Early snapshot (ep=100) weaker: r = -0.232
- The RATE of alignment improvement is more predictive than any single snapshot

### 5. Self-supervised proxy for C_i — NOT YET STARTED
- Required for practical relevance beyond simulation

## Rigor Gaps

### Fixed
- [x] Success criterion defined: dist < baseline - 2σ = 0.511
- [x] Random baseline measured: dist = 0.516 ± 0.003, C_i = 0.003 ± 0.022
- [x] Seeds increased to 7
- [x] Formal interaction test with p-value
- [x] Correlation decomposition (between r=-0.878, within mean r=-0.582)

### Partially Fixed
- [ ] Threshold table: C_i≥0.5 → 98% success (n=53), C_i≥0.6 → 100% (n=18).
      Better than before but still thin at high end (n=2 above 0.8).
- [ ] Metric ablation: cosine (r=-0.724) vs action_magnitude (r=-0.739) —
      action magnitude is slightly better! Need to investigate.
- [ ] oracle_noise0.5 reversal: persists with 7 seeds. Explained as boundary
      condition but needs explicit discussion in paper.

### Not Yet Fixed
- [ ] Only one task (rock-push)
- [ ] C_i is post-hoc (no proxy estimator)
- [ ] No ablation of cosine vs L2 vs info gain (only compared to action_magnitude)

## Framing Fixes
- [ ] Rename "coordination quality" → "sensorimotor alignment"
- [ ] Lead with blind cat hook more aggressively
- [ ] Cite Pinto et al., DreamerV3, quasimetric RL in related work
