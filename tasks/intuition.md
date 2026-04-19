# Project Intuition

## One-line claim

In a perception-action loop where matter is observable only through lossy
energy channels, a measurable symmetry-action (SA) metric over the agent's
internal embedding predicts task success better than direct supervision
metrics — and the SA-achievable ceiling is gated by the information content
of the sensory channel, not by model capacity.

## Why we believe it

**Primary evidence (SA as predictor):** In a 245-config × 7-seed sweep on a
1-rock manipulation task, the magnitude-weighted SA metric correlates
r=-0.893 with task error, and a formal interaction test (F=34.2, p=5×10⁻⁹)
establishes SA as a non-trivial predictor beyond reconstruction loss. The
result transfers: 2-rock 6D gives r=-0.728; SA dynamics during training
(r=-0.705) predict trajectory; transfer across physics (93-106%) and
appearance (89-102%) holds. The oracle-free proxy (action variance r=-0.82)
means SA can be estimated from agent behavior alone, without ground-truth
state.

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

**Primary claim:** Falsified if a controlled experiment manipulates SA without
manipulating reconstruction loss (or vice versa) and SA fails to predict
task success in the manipulated direction. Currently SA and recon are
correlated through training dynamics — disentangling them requires an
intervention. The 3-rock multi-rock result already gives r=-0.300 (much
weaker), warning that the metric may not scale to higher-dimensional
manipulation. Equally falsifying: if SA-vs-success on a qualitatively
different task family (locomotion, dexterous grasping, navigation) gives
r > -0.3, the claim is restricted to manipulation-style tasks.

**Sensory-capacity claim:** Falsified if (a) the substitution effect
disappears with longer training (the effect could be a PPO convergence
artifact rather than an information bound), (b) sensory=2/embed=32 can be
made to learn with different hyperparameters (defeating the "information
floor" narrative), or (c) the SA ceiling at sensory=16 fails to exceed
sensory=2 on a second task family. Current effect size is small enough
(Cohen's d ≈ 0.3-0.5 range, pending bootstrap analysis) that sensitivity
to task structure is a live risk.

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
