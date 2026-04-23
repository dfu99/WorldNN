# Deep Read: The Missing Reward — Active Inference in the Era of Experience (2508.05619, 2025)

## (1) What they do

Conceptual paper (*no empirical experiments*). Proposes LLM + Active Inference
fusion. LLM functions as amortized inference engine for Expected Free Energy
minimization. The pseudocode (Algorithm 1) is:
1. LLM generates candidate policies
2. LLM predicts future states/observations under each policy
3. Compute information gain and preference alignment
4. EFE = −(Information Gain) − (Preference Alignment)
5. Select policy minimizing EFE

Prediction error is defined as *surprise*: negative log-probability of an
observation given the agent's model. Variational Free Energy combines model
complexity and prediction accuracy. When a lab-assistant scenario observes
an unexpected pH indicator, VFE jumps from 0.5 to 3.2.

Core framing: traditional RL asks "maximize reward"; AIF asks "confirm
predictions and reach expected states." No external reward — safety lives in
the C matrix (preferences).

## (2) What WorldNN has that they don't

- *Empirical results.* They're a vision paper; we have 245-config / 100-config
  / 60-config empirical grids.
- *Measurable structural correlate.* SA is computable per-config. EFE in
  their pseudocode depends on LLM-generated rollouts — expensive and noisy.
- *Bounded estimator argument.* Our obj-025 T7 theory note *proves* SA is a
  bounded estimator of -F under policy-deterministic conditions. The paper
  argues for EFE as a framework without supplying the behavioral correlate.

## (3) What they have that WorldNN could adopt

- *EFE as policy objective.* Our PPO reward can be reformulated as EFE
  minimization with the appropriate preference matrix. An ablation:
  reward-based PPO vs EFE-based PPO, compare SA trajectories.
- *Amortized inference via a generative model.* Rather than explicit
  rollouts, use a generative predictor. We have PredictiveOrganism (from
  obj-009) but never attached it to policy learning.
- *Preference matrix C formalism.* Our reward is scalar; a C matrix would
  let us specify goal-state distributions more richly (e.g., multi-modal
  preferences for 2-rock tasks).

## (4) Citation position: support, orthogonal framing

Threat? No — they have no empirical results to conflict with.
Support? Strong — they provide the theoretical substrate our obj-025 T7
note extends. Cite in Discussion §7.
Orthogonal? Their vision is LLM-based agents; ours is small structured
agents. They could cite SA as "the diagnostic AIF has been missing."

## Action items

- Write a 1-paragraph discussion pointing to this paper in §7.
- Keep an EFE-based PPO variant as a candidate future-work ablation.
