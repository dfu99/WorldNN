# Deep Read: ACT-JEPA — Joint-Embedding Predictive Architecture for Policy Learning (2501.14622, 2025)

## (1) What they do

Hybrid imitation-learning + self-supervised-learning architecture. End-to-
end training jointly predicts:
1. Action sequences
2. Latent observation sequences

Operates in JEPA latent space, using a single predictor head conditioned on
both past states and candidate actions. Outperforms strongest baselines in
all environments tested, with up to 40% improvement in "world model
understanding" and 10% higher task success rate.

Core claim: standard IL has "underdeveloped world models" because it trains
only on action prediction. Adding a latent-state prediction objective shares
representation learning between policy and dynamics.

## (2) What WorldNN has that they don't

- *Information-loss chain.* ACT-JEPA assumes full observation; WorldNN
  controls what gets through (channel → environment → sensory_dim slice).
- *Explicit SA diagnostic.* Their 10% task-success improvement is reported
  only as reward uplift; they have no metric to attribute the improvement
  to representation-level structural change.
- *Perception-capacity decomposition.* Our obj-024 grid. ACT-JEPA doesn't
  decompose where its improvement comes from (better perception? better
  policy? better prediction-supervision?).

## (3) What they have that WorldNN could adopt

- *Joint action+observation prediction loss.* Our PPO organism predicts
  only the policy; PredictiveOrganism predicts observations but isn't
  attached. An ACT-JEPA-style joint loss is a clean ablation. Expected
  result: SA rises faster in early training (their "40% better world
  model understanding" would translate to a better SA slope, which we
  measured as r=-0.705 predictor of final performance).
- *End-to-end training of predictor+policy.* They share representation.
  We train organism policy-head and value-head jointly but not a
  prediction head.

## (4) Citation position: adjacent, adoptable

Threat? No — they are an architecture improvement, not a diagnostic.
Support? Moderate — they demonstrate the value of joint prediction+action
losses, matching our claim that action-aware representations beat raw
recon.
Orthogonal? They could be a concrete ablation variant of WorldNN (see
action item).

## Action items

- Queue as a candidate future experiment: "ACT-JEPA-style joint loss in
  WorldNN, compare SA trajectories vs PPO-only baseline."
- Cite in Related Work §2 under JEPA-family world models.
