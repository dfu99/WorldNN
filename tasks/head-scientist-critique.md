# Head Scientist Critique — WorldNN
## Target: ICLR 2027 (October 2026 deadline)

**VERDICT: Do not submit now. Revise first. Strong workshop paper; main-track with second task.**

## Novelty Assessment

C_i (cosine alignment between degraded-perception policy and optimal-state action) fills a genuine gap. No prior work defines this specific metric across a designed perception-degradation ladder.

Closest prior work:
- **Asymmetric Actor-Critic** (Pinto et al.; arxiv:2512.01188): trains with privileged state, evaluates without. C_i measures the residual alignment — different and more diagnostic.
- **DreamerV3 / IRIS prediction probes**: implicitly measure something like C_i during imagination rollouts. C_i is computed in real state space, not latent — this distinction is the crux but draft §3.2 buries it.
- **Quasimetric RL**: measures directional asymmetry in value space; C_i does it in action space.

**No paper defines cosine alignment between a degraded-perception policy and optimal-state action across a designed perception-degradation ladder. That gap is genuine.**

## Rigor Gaps (must fix)

1. **Only one task** (rock-push, 4D/2D) — threshold values likely task-specific. **A second task is non-negotiable for main track.**
2. **3 seeds/condition**, C_i≥0.8 bucket has only N=5 — one unlucky seed would collapse "100% learn"
3. **C_i is post-hoc** (requires optimal action a*(s)) — no proxy estimator proposed
4. **No ablation of metric definition** — why cosine vs L2? vs information gain?
5. **Threshold claim is correlational, not causal** — needs factorial ANOVA or interaction term regression

## Framing Fix

- **Rename** "coordination quality" → "sensorimotor alignment" to avoid MARL confusion
- **Lead with "blind cat" hook** more aggressively — it's the paper's strongest narrative device
- State novelty as: "we explicitly design the perception-action gap and measure alignment across it"

## Experiments Needed (priority order)

1. **Formal interaction test** (existing data, no PACE) — log-linear model with perception × capacity interaction term and p-value. DO THIS FIRST.
2. **Increase seeds to 5** for threshold-critical conditions (C_i ≥ 0.7–0.8 bucket). Moderate PACE cost.
3. **Second task** (8D+ locomotion or object manipulation) — non-negotiable for main track. Major PACE job.
4. **C_i dynamics during training** — does early-training C_i predict final success? Makes C_i actionable as a training-time diagnostic.
5. **Self-supervised proxy for C_i** without oracle access — makes the metric practically relevant beyond simulation.

## Action Plan

Start with #1 (can be done on existing data, CPU only) and #2 (moderate PACE job). Then design #3 (second task) as a PACE submission. Update `paper/draft.md` and `tasks/planning.md` as you go.
