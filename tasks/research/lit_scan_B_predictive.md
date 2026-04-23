# Lit Scan B — Predictive Processing / Active Inference / Free Energy (2024-2026)

Queries: "active inference agent reinforcement learning 2025 embodied",
"predictive coding neural network agent 2025 hierarchical prediction error".

## Papers collected

### 1. The Missing Reward: Active Inference in the Era of Experience (2508.05619, 2025)
LLM + active-inference fusion. "Era of Experience" vision: agents generate
their own training signal and interpret it through a principled free-energy
lens. Expected Free Energy (EFE) minimization replaces unbounded reward
maximization. Bounded rationality given mathematical formalism.
**Relevance:** Matches our obj-025 T7 theory note. Their EFE replaces reward;
ours uses PPO reward. WorldNN could add an EFE-minimization variant as
ablation. Their framing would support a paper reframe to "SA as EFE estimator."

### 2. On Predictive Planning and Counterfactual Learning in Active Inference (2403.12417, 2024)
Combines predictive planning with counterfactual learning under active
inference. Formalism for rolling out alternative policies and scoring by EFE.
**Relevance:** Closer to Dreamer's latent-imagination but grounded in active
inference. WorldNN could test whether SA improves under counterfactual-aware
training.

### 3. Active Predictive Coding (Neural Computation 2024, 36:1)
Unifying neural model for active perception, compositional learning, and
hierarchical planning. Explicitly models perception as action-conditioned
prediction error minimization.
**Relevance:** Parallel formalism to ours (organism as active-inference
agent). Their "active perception" is decision about WHERE to look; our
"sensory_dim" choice about HOW MANY channels is structurally similar.

### 4. Introduction to Predictive Coding Networks for ML (2506.06332, 2025)
Survey/intro of predictive coding for ML. Layers of latent variables,
top-down predictions, bottom-up errors. Attention to training-depth
limitations (>5-7 layers degrade due to imbalanced errors).
**Relevance:** Good citation for the predictive-coding framing in Related
Work. Depth limit is interesting for scaling our organism.

### 5. PrediRep: Hierarchical Predictive Coding (ScienceDirect 2025,
S089360802500125X)
Unsupervised convolutional-recurrent network with hierarchical R/E units.
Differences (prediction errors) propagate up; predictions flow down.
**Relevance:** Concrete implementation of predictive coding at scale. Shows
how to instantiate a predictive organism with multi-level structure.

### 6. Predictive Coding Light (Nature Communications 2025)
s41467-025-64234-z. Predictive coding framework applied broadly; accessible
overview for citation.
**Relevance:** Nature-tier citation for Related Work.

### 7. Energy optimization induces predictive-coding properties in
multi-compartment spiking networks (PMC12180623, 2025)
Spiking neural networks spontaneously develop predictive-coding behavior
when trained to minimize energy. Biologically grounded.
**Relevance:** Links our free-energy theory note to biological plausibility.

### 8. PyHGF (library)
Python library for dynamic probabilistic networks that approximate Bayesian
inference via prediction + precision-weighted prediction errors.
**Relevance:** Tool to consider; not a paper.

## Methodological moves unexplored by WorldNN

1. *Expected Free Energy policy selection* (EFE vs reward). Our organism
   uses PPO reward. An EFE-minimizing variant could be a head-to-head.
2. *Hierarchical predictive coding* (PrediRep): multi-level R/E units.
   WorldNN has a single-level embedding; multi-level could test whether
   the SA ceiling rises with hierarchical prediction.
3. *Precision-weighted prediction errors* (PyHGF-style): SA treats all
   errors equally; precision weighting could shape the SA trajectory.
4. *Counterfactual rollouts* (2403.12417): our organism has no explicit
   rollout; Dreamer V4 and these active-inference papers do.

## SA-analogues in these papers

- Expected Free Energy (EFE) — both papers use this as the policy-selection
  objective. SA is a behavioral correlate; EFE is the theoretical target
  SA is (provably, per obj-025 T7) a bounded estimator of.
- Prediction error (all predictive-coding work) — complementary to SA. SA
  is a policy-space quantity; prediction error is an observation/latent-
  space quantity. These are orthogonal as per our response to the PI.
