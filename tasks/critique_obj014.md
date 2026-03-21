# Internal Critique: obj-014 Results (2026-03-21)

## Strengths

- Core conceptual contribution (perception-action asymmetry, C_i as bridge metric) is clear and well-motivated
- Controlled simulation independently varying perception/capacity is methodologically sound
- Data processing inequality chain provides principled theoretical basis
- "Blind cat" analogy maps correctly to oracle_noise0.5 results
- Paper is honest about limitations

## Weaknesses

1. **Threshold table in paper Section 4.2 has wrong counts.** Raw data shows only 2 records with C_i ≥ 0.8, not 5 as claimed. Row-shift error — must fix.

2. **Success/failure criterion undefined.** "100% learning success" never defines what distance counts as success. Using dist < 0.45 gives 10/105 successes; dist < 0.47 gives 22. Binary success rates require explicit threshold.

3. **r = -0.735 is largely a between-condition effect.** Between 7 level means: r = -0.894. Within individual conditions: mean r = -0.472. One condition (oracle_noise0.5) shows r = +0.591 (reversal). The overall r reflects cluster structure, not a smooth within-condition relationship.

## Robustness Concerns

1. **Oracle dominates high-C_i range.** All top C_i values come from oracle. No VAE condition reaches C_i > 0.58. The metric may just be recapitulating perception quality ordering.

2. **Only 3 seeds per condition.** Mean within-condition std = 0.068, with 23% of conditions std > 0.1. Apparent trends could be noise.

3. **Near-floor performance throughout.** Only 10/105 configs (9.5%) achieve dist < 0.45. The paper is largely studying the failure regime.

4. **oracle_noise0.5 reversal unexplained.** Positive r within this condition — higher C_i associated with WORSE performance. Red flag.

## Novelty Assessment

- C_i (cosine to optimal action) is not substantively different from behavioral cloning alignment scores or action-prediction accuracy in imitation learning
- The FRAMING (asymmetric loop, bridge metric) is the novel part, not the formula
- The simulation apparatus has no close analog — that's genuine novelty
- Must preempt the "this is just policy cosine similarity with a new name" objection

## Statistical Concerns

1. r = -0.735 with n=105 but effective df much lower (7 conditions × 5 × 3)
2. "Sharp threshold" based on 2 data points above C_i=0.8 — premature
3. C_i_std within runs averages 0.590 (nearly as large as mean of 0.415)
4. No confidence intervals anywhere in the paper

## Actionable Improvements

1. **Fix threshold table** — recount from raw data
2. **Define success criterion** explicitly with random-action baseline
3. **Decompose correlation** — report between-level r, within-level r, color-coded scatter
4. **Add random baseline** — what dist does a random policy achieve?
5. **Increase to 5 seeds** for key comparisons
6. **Address oracle_noise0.5 reversal** explicitly
7. **Add error bars** to all tables
8. **Distinguish C_i from policy cosine similarity** in related work section
