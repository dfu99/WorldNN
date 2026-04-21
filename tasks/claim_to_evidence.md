# Claim-to-Evidence Map (2026-04-19)

Maps our current evidence to each reviewer's anticipated weakness.
Compiled from obj-001 through obj-025 (obj-025 covers T3–T7 analyses).

## Core claim (from tasks/intuition.md)

The SA metric over the organism's internal embedding predicts task success
better than direct supervision metrics — and the SA-achievable ceiling is
gated by the information content of the sensory channel, not by model
capacity.

## Relevant literature (2025-2026)

1. *Higher Embedding Dimension Creates a Stronger World Model for a Simple
   Sorting Task* (2025, arXiv:2510.18315) — Direct parallel to our embedding
   bottleneck claim. Finds embedding-dim effects "level out after approximately
   30," consistent with our obj-016 finding that capacity effects diminish at
   embed_dim ≥ 16 in the rock-push task.

2. *Interpreting neural computations by examining intrinsic and embedding
   dimensionality of neural activity* (Jazayeri & Ostojic 2022, arXiv:2107.04084,
   still widely cited in 2025 surveys) — "The intrinsic dimensionality of neural
   activity is determined by three sources: incoming stimuli, ongoing movements,
   and latent variables." Supports our decomposition of SA into sensory-driven
   vs internally-generated components.

3. *Efficient Vision-Language-Action Models for Embodied Manipulation: A
   Systematic Survey* (2025, arXiv:2510.17111) — Identifies "fusion of cross-modal
   perception becomes key to break through bottlenecks" as a main 2025 research
   direction. Aligns our sensory-capacity tradeoff with the survey's framing:
   cross-modal richness drives learnability.

4. *BC-IB: Behavior Cloning with Information Bottleneck for Embodied Manipulation*
   (cited in 2025 surveys) — Integrates information bottleneck into policy
   learning to compress task-irrelevant information. Our SA metric can be
   interpreted as the complementary objective: measuring how much task-relevant
   structure survives the bottleneck.

### Literature sweep conclusion

No 2025-2026 paper directly tests the sensory-capacity substitution hypothesis
with controlled sensory_dim sweeps. Our obj-024 is a novel empirical test. The
floor effect (sensory ≤ 4 → no learning regardless of capacity) aligns with the
general BC-IB direction without being a duplicate contribution. The rate-
distortion framing (obj-025 T3) is the newest-flavored contribution — no
direct competitor.

## Per-reviewer evidence mapping

### Reviewer A — World Model / Model-Based RL (Ha / Hafner)

*Signature question:* "Doesn't Dreamer's reconstruction loss tell you the
same thing as SA?"

*Our evidence:*
- Primary (obj-016, 245 configs, 7 perception × 5 embed × 7 seeds): SA
  r=-0.724 with task error; F=34.2, p=5e-9 interaction test; SA's
  mag-weighted variant r=-0.893 — a dynamic range where recon-loss-only
  predictors could not explain this much variance.
- Counter-result we must disclose (obj-025 T4): On the obj-024 grid where
  dynamic range is compressed (sensory ≤ 4 cannot learn), SA adds only
  ΔR²=0.004 over input-side predictors. *Framing:* SA captures structural
  alignment ACROSS perception richness; when perception is uniformly
  impoverished, SA collapses to information availability. This is not a
  failure but a specification of where SA is informative.

*Action items for paper:* Explicitly state the regime where SA adds signal
vs where it does not. Reference obj-016 as the primary evidence; cite T4
as a scope boundary.

### Reviewer B — Embodied Cognition / Active Inference (Clark / Friston)

*Signature question:* "Where is the predictive component?... Your agent
only acts to maximize task reward."

*Our evidence:*
- obj-025 T7 (tasks/theory_notes/active_inference.md) derives SA as a
  bounded estimator of −KL[q(s|μ) ∥ p(s|o)], connecting to variational
  free energy.
- obj-024 floor effect is predicted by the active-inference framework:
  when I(s;o) falls below task-required MI, no internal capacity can
  reduce F. We now have the empirical curve.
- Our Markov blanket mapping is explicit: sensory_dim = blanket width;
  embedding_dim = internal-state dim. Figure at results/obj025_markov_blanket.png.

*Action items for paper:* Add 1 paragraph in Discussion citing Friston 2010
and referencing the Markov blanket figure.

### Reviewer C — Information Theorist (Tishby / Still)

*Signature question:* "Define your doubt as I(X;W) - I(Z;W). Estimate it
with KSG. Show me the curve."

*Our evidence:*
- obj-025 T3 (results/obj025_mi_vs_sensory.png): Linear-probe R²(S|obs) =
  0.153, 0.275, 0.821, 1.000 for sensory = {2,4,8,16}. Gaussian-MI =
  0.33, 0.64, 3.44, 27.6 nats. Correlation with peak SA: *r = 0.975*.
- This IS the rate-distortion curve Reviewer C demands. KSG alone
  underestimates in 16D (known issue); we report KSG + normalized
  Gaussian-MI for robustness.
- obj-014 chain-MI (I(S;X) → I(S;Y) → I(S;Z) → I(S;E)) already
  demonstrates monotonic DPI for the full pipeline.

*Action items for paper:* Promote T3 rate-distortion figure to main text
or an appendix; frame as "quantifying the information-theoretic floor."

### Reviewer D — Sensorimotor Neuroscientist (Wolpert / Kording)

*Signature question:* "Is your 'embedding size' supposed to be cortical
neurons, working memory slots, or something else? Pin it down."

*Our evidence:*
- obj-025 T6 (results/obj025_bio_calibration.png): Our sensory_dim range
  (2–16) sits at the *low end* of biological sensing — comparable to
  paramecium chemotaxis (2), vestibular (6), haptic (10), cutaneous types
  per digit (16). Well below audition (10^4 cochlear hair cells), vision
  (10^6 optic nerve fibers).
- Embedding dim (2–32) maps best to working-memory-slot analogs
  (Baddeley-style capacity), not cortical-column counts.

*Action items for paper:* Add the biological calibration table to the
Methods section. Rename framing from "cortical neurons" to "functional
channels / WM slots" throughout paper.

### Reviewer E — Generative Simulation Skeptic (Ganguli / Anandkumar)

*Signature question:* "Your second task (2-rock 6D) gives r=-0.728, very
close to 1-rock r=-0.724. Is this transfer or just task similarity? Why
is 3-rock r=-0.300 a footnote?"

*Our evidence:*
- obj-021 (2-rock 6D): r=-0.728 on the original 2-rock PPO setup with
  oracle/VAE perception — replicates 1-rock SA-predicts-success pattern.
- obj-019 physics transfer: 93-106% retention across physics variants.
- obj-020 appearance transfer: 89-102% retention.
- obj-017 (3-rock): r=-0.300 due to floor effect (dist 0.46-0.50 across
  all conditions). Task-complexity scaling limit, disclosed as limitation.
- **obj-026 (2-rock sensory-capacity replicate, 2026-04-21): FLOOR EFFECT
  — the sensory-capacity substitution pattern did NOT replicate on 2-rock.
  Overall mean dist = 0.501 (near-random); peak SA dropped from 0.234
  (1-rock) to 0.098 (2-rock); the monotonic sensory-richness ceiling
  disappeared.** This is consistent with obj-017 (3-rock floor): reward
  dilution across objects exceeds the 800-episode PPO training budget for
  *sensory-limited* perception. It does NOT contradict obj-024 (the
  within-regime SA/capacity pattern on 1-rock stands), but it DOES
  restrict the information-bound claim's empirical support to tasks
  that are learnable within the training budget.

*Action items for paper:*
- Explicitly acknowledge the obj-026 floor effect in §5.7 and §7.5.
- State the scope of the sensory-capacity claim: demonstrated on 1-rock
  4D manipulation; replicates on 2-rock only when PPO converges.
- Either run a longer-training 2-rock replicate (n≥13 seeds, ≥2000
  episodes) OR frame obj-024 as single-task evidence and cite obj-021's
  transfer numbers as the cross-task backup.
- Acceptable framing: "We observe the information-theoretic bound
  empirically on one task; generalization to higher-object-count tasks
  requires training budget scaling that exceeded our compute allocation."

## Publication readiness summary

| Reviewer | Risk level | Primary evidence | Additional needed? |
|:----|:----|:----|:----|
| A (World Model) | Medium | obj-016 (r=-0.724) | Scope-boundary disclosure (T4, done) |
| B (Active Inference) | Low | T7 theory note | Paragraph in §7 (done in draft) |
| C (Info Theorist) | Low | T3 rate-distortion r=0.975 | Promote T3 to main text (done §5.7) |
| D (Neuroscientist) | Low | T6 calibration | Fold table into Methods |
| E (Generative Sim) | *Still High* | obj-021 + transfer; obj-026 floor | Longer 2-rock training OR scope-restrict claim to 1-rock |

*Highest residual risk:* Reviewer E, unchanged after obj-026. The 2-rock
sensory-capacity replicate hit a floor effect, providing NO positive
confirmation of the substitution pattern beyond 1-rock. Options:
(a) Scope-restrict the information-bound claim to 1-rock and lean on
obj-021 for cross-task SA; (b) Rerun obj-026 with ≥2000 episodes and n=13+
seeds (compute-heavy); (c) Run an intermediate task (e.g., 1-rock with
randomized target) to probe whether substitution is task-width-specific.
