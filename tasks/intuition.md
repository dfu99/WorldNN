# Project Intuition

## One-line claim

Two coupled claims govern effective action in a perception-action loop:
(1) within a regime where the sensory chain preserves task-relevant
information, the symmetry-action (SA) metric over the agent's internal
embedding predicts task success better than direct supervision metrics;
(2) the regime itself is defined by an information-theoretic bound —
the achievable SA ceiling scales with I(state; observation), and model
capacity cannot substitute for information absent from the input.

## Why we believe it

**Primary evidence (SA as predictor):** In a 245-config × 7-seed sweep on a
1-rock manipulation task, the magnitude-weighted SA metric correlates
r=-0.893 with task error; a formal interaction test gives F=34.2, p=5×10⁻⁹.
Direct head-to-head against reconstruction-loss (obj-028, 2026-05-05) on
the same 245-config grid: SA r=-0.724 vs recon r=-0.436; partial
r(SA | recon)=-0.679; SA adds ΔR²=+0.374 over recon-loss alone in
multiple regression on task distance. The result transfers: 2-rock 6D
gives r=-0.728; SA dynamics during training (r=-0.705) predict trajectory;
transfer across physics (94-106%) and appearance (89-110% per-config /
89-106% per-cell-mean) holds. The oracle-free proxy (action variance
r=-0.82) means SA can be estimated from agent behavior alone, without
ground-truth state.

**Secondary evidence (sensory-capacity tradeoff):** obj-024 (100 configs,
sensory_dim ∈ {2,4,8,16} × embed_dim ∈ {2,4,8,16,32} × 5 seeds, 800 episodes
each) shows the SA ceiling scales with sensory richness: peak SA by sensory
dim is 0.069, 0.042, 0.105, 0.234 for sensory=2,4,8,16 respectively. The
substitution effect exists — sensory=16/embed=2 (SA=0.033) outperforms
sensory=2/embed=32 (SA=-0.007) — but is modest in magnitude. The cleanest
finding is a floor effect: when sensory_dim ≤ 4, no amount of model capacity
rescues performance (SA ≈ 0 across embed ∈ [2,32]). This is consistent with
the Data Processing Inequality: capacity cannot recover information absent
from the input.

## What would falsify it

**Primary claim (SA as predictor):** Falsified if a controlled experiment
manipulates SA without manipulating reconstruction loss (or vice versa)
and SA fails to predict task success in the manipulated direction. The
obj-024 ΔR² = 0.004 result already shows that on compressed dynamic-range
grids, SA does not add signal beyond input-information predictors — this
is a scope boundary we have disclosed in §7.5 rather than a full
falsification, because the wide-dynamic-range result (r=-0.724, 245
configs) still stands. Full falsification would require: (a) wide-dynamic
cross-task regime where SA does NOT predict; (b) successful intervention
that changes SA without changing recon and performance tracks recon, not SA.

**Sensory-capacity claim (info-bound):** obj-025 T3 makes this the
STRONGER claim — Gaussian-MI(S; obs) vs peak SA correlates at r=0.975
on obj-024. Falsified if (a) the correlation weakens below r<0.7 on a
second task family (obj-026 2-rock replicate in progress), (b) the
floor effect disappears with radically different hyperparameters, or (c)
longer training reveals a phase transition where sensory=2 eventually
does learn with embed=32 (which would imply the information-theoretic
ceiling was a convergence artifact). Power for the substitution effect
at n=5 is 0.69 (underpowered); full confirmation needs n≥13 per cell.

**Task-similarity risk (Reviewer E):** obj-026 (2-rock sensory-capacity
replicate, 2026-04-21, 60 configs on RunPod A4500) did NOT confirm the
substitution pattern. Peak SA dropped from 0.234 (1-rock) to 0.098
(2-rock); overall mean dist = 0.501 (near-random), consistent with a
floor effect mirroring obj-017 (3-rock). The reward dilution across two
objects exceeded the 800-episode PPO training budget. This is a scope
restriction, not a contradiction — obj-024's within-regime finding
stands, but the information-bound claim is currently only supported on
tasks PPO can solve within compute budget. Full falsification would
require obj-026 pattern persisting with much longer training (≥2000
episodes, n=13 seeds), which would imply 2-rock has no sensory-capacity
substitution even after convergence.

## Target panel and venue

- *Panel*: see `tasks/review-panel.yaml`
- *Venue*: NeurIPS 2026 (May 6 deadline)
- *Why this venue*: NeurIPS is the right home for the cross-cutting
  world-models + information-theory + embodied-cognition framing; the
  paper draft is structurally complete and ready for PI review. The
  sensory-capacity finding, if robust, strengthens the info-theoretic
  pillar and directly addresses Reviewer C's anticipated rate-distortion
  concerns.

## Open questions

- Is SA a special case of the information bottleneck? If I(Z; task_label)
  determines task success, is mag-weighted SA just an empirical estimator
  of this MI?
- Why does the boundary artifact appear in the reversal analysis? Is it
  signaling a degenerate task structure, or revealing a failure mode of SA?
- Does the bottleneck-width sweep (d ∈ {2,...,64}) confirm the rate-distortion
  prediction that SA degrades smoothly with width, or do we see a phase
  transition?
- For the sensory-capacity tradeoff: what is I(S; obs) at each sensory_dim,
  and does it quantitatively match the SA ceiling? (obj-025 T3)
- Does the substitution effect emerge cleanly with longer training, or is
  the obj-024 result near the asymptote?
