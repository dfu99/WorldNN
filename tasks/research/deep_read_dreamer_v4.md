# Deep Read: Dreamer V4 (Hafner, Yan et al., 2509.24527, 2025)

## (1) What they do

Dreamer V4 is a scalable world-model agent. Third generation of the Dreamer
line. A block-causal transformer is used for both the tokenizer and the
dynamics model. The agent is trained *inside* the world model (imagination),
and for Minecraft diamond-retrieval it learned entirely offline without
environment interaction. Scale: 2500 hours of video with only 100 hours of
paired action labels; retains 80% of action-conditioned generation accuracy.

Two technical moves:
1. *Shortcut forcing* — extends diffusion forcing and shortcut models to
   enable few-step (4 vs 64) denoising of latent states.
2. *X-prediction* — direct prediction of clean frames stabilizes long
   rollouts.

## (2) What WorldNN has that they don't

- *SA metric.* Dreamer V4 has no policy-alignment diagnostic comparable to
  SA. Its evaluation is per-task reward. If you had oracle actions, you
  could compute SA on Dreamer agents and likely see it collapse at
  perception-limit regimes just as we do.
- *Rate-distortion argument.* We have an explicit I(S; obs) vs peak-SA
  correlation (obj-025 T3, r=0.975). Dreamer V4 offers no information-
  theoretic bound on its agents' achievable performance.
- *Sensory-capacity substitution test.* Our obj-024 directly varies input
  channels vs model capacity on the same task. Dreamer V4 holds both
  variable.

## (3) What they have that WorldNN could adopt

- *Transformer backbone.* Scale matters. Our MLP organism may hit ceilings
  that a transformer would not; revisit after scope is clearer.
- *Sparse action supervision.* 4% paired actions → 80% accuracy. This
  directly maps to a WorldNN ablation: train with N% of oracle-action
  pairs and measure SA.
- *Offline video training.* Our organism must interact to learn; Dreamer
  V4 shows the dynamics can be largely learned from observation. Candidate
  direction for reducing compute on RunPod.
- *Shortcut forcing / X-prediction.* Not relevant to our scale but noted
  for future.

## (4) Citation position: orthogonal support

Threat? No — Dreamer V4 and WorldNN answer different questions. Dreamer is
"can we build a better world model at scale?"; WorldNN is "how does SA
relate to task-learnability under controlled information loss?"

Support? Yes — Dreamer V4 is the canonical modern example of action-
conditioned latent dynamics. We cite it in §5.7 and §7.5 as the "large-
scale instantiation of the formalism our organism implements at small
scale."

Orthogonal? Primarily. Their scale claim does not falsify our regime claim;
our regime claim does not falsify their scale claim.
