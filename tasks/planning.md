# Planning

## Target: ICLR 2027 (October 2026 deadline)

**Living paper**: `paper/draft.md`
**Head Scientist critique**: `tasks/head-scientist-critique.md`
**Progress tracker**: `tasks/head-scientist-progress.md`

Sensorimotor alignment (SA) predicts learning success across asymmetric
perception-action loops. At scale (245 configs, 7 seeds): r = -0.724,
interaction p = 5×10⁻⁹, within-level mean r = -0.582.

### Active — Remaining Blockers

1. **Fetch multi-rock v2 results** — PACE job 5453852 (RTX 6000) still
   pending. When it completes: fetch, analyze, run interaction test,
   update paper §5.5 with second-task results. THIS IS THE ICLR BLOCKER.
   If it fails again, consider reducing to 2 rocks (6D) as fallback.

2. **Self-supervised SA proxy** — Design needed. Options: (a) prediction
   error as proxy (train world model, measure prediction accuracy),
   (b) value function gradient magnitude, (c) contrastive alignment with
   augmented observations. Nice-to-have for main track, not strictly
   required.

### All Completed from Head Scientist Critique

- [x] Formal interaction test: F(1,241)=34.2, p=5×10⁻⁹ (obj-016)
- [x] Seeds increased to 7 (obj-016, 280 configs)
- [x] Random baseline: dist=0.516±0.003, SA=0.003±0.022
- [x] Success criterion: dist < 0.511 (baseline - 2σ)
- [x] Correlation decomposition: between r=-0.878, within r=-0.582
- [x] SA dynamics (obj-015): slope r=-0.705, time-to-threshold, paper §5.4
- [x] Metric ablation: 6 metrics, cosine F=104.8, mag-weighted r=-0.893
- [x] Framing: renamed SA, blind cat hook, related work §2, imitation learning distinction
- [x] Reversal analysis: oracle_noise0.5 explained as boundary artifact, paper §6
- [x] Multi-rock diagnosed + fixed + re-submitted (job 5453852)

## Open Questions

- ANSWERED: Interaction significant (p=5e-9)
- ANSWERED: Within-level r=-0.58 (improved with 7 seeds)
- ANSWERED: SA slope predicts success (r=-0.705)
- ANSWERED: Reversal = boundary artifact (distance std=0.004)
- ANSWERED: Mag-weighted SA outperforms cosine overall (r=-0.893 vs -0.724)
- Does the SA threshold generalize to 8D? (awaiting PACE)

## Recently Completed

- [2026-03-24] AFK session: 7/8 tasks completed. Metric ablation, framing overhaul, SA dynamics pub figure, reversal analysis, progress report. Multi-rock fetch blocked on PACE.
- [2026-03-24] Multi-rock diagnosis: 500ep insufficient → 1000ep, hidden=64, worst-rock reward
- [2026-03-22] obj-016: At-scale (280 configs, 7 seeds), interaction p=5e-9
- [2026-03-21] obj-015: SA dynamics — slope r=-0.705
- [2026-03-20] obj-014: Expanded sweep — 105 configs, r=-0.735
- [2026-03-18] obj-013: SA metric introduced, r=-0.867
- [2026-03-17] Sensory-motor alignment formalization
- [2026-03-16] obj-012/011/010: Perception ladder, VAE diagnosis, oracle baseline
