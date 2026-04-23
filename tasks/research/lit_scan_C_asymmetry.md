# Lit Scan C — Structural Asymmetry / Sensorimotor Contingency (2024-2026)

Queries: "sensorimotor contingency deep learning 2025 neural network embodiment",
"structural asymmetry perception action embodied learning 2025 metric".

## Papers collected

### 1. Embodied Sensorimotor Control: computational modeling of neural movement control (2509.14360, 2025; PMC12458597)
Reviews deep RL for embodied control. Argues RL-trained agents exhibit
neural population activity closely resembling posterior parietal cortex
(PPC). State-value and policy representations enrich with learning.
**Relevance to WorldNN:** They benchmark agents against biology; we
benchmark SA against task reward. Their PPC-similarity is analogous to our
"learned embedding resembles state manifold" — could retrofit as SA
validation.

### 2. Deep Sensorimotor Control by Imitating Predictive Models of Human Motion (2508.18691, 2025)
Trains sensorimotor policies with RL by imitating predicted human motions
conditioned on past robot states. Imitation + prediction fusion.
**Relevance:** Action-conditioning matches our setup; they assume a
human-motion teacher, we use oracle actions. Structurally similar learning
signal.

### 3. Embodied Intelligence: A Synergy of Morphology, Action, Perception and Learning (ACM CS 2025; Liu, Guo, Cangelosi)
Comprehensive survey. Argues structural characteristics reduce optimization
space for action generation, improving feasibility and explainability.
Benchmarks: Habitat, ManiSkill.
**Relevance:** Primary survey for Related Work. Our sensory-capacity
tradeoff is an example of "structural characteristics reducing optimization
space."

### 4. Continuous sensorimotor transformation enhances robustness of neural dynamics to perturbation (PMC11968799, 2025)
Macaque motor cortex continuously transforms sensory inputs into actions;
this continuous transformation enhances robustness to perturbations.
**Relevance:** Biological support for SA-as-transformation-alignment. If
the transformation is continuous and disrupted by perception failure, SA
should correlate with biological robustness measures.

### 5. Neural Brain: A Neuroscience-inspired Framework for Embodied Agents (2505.07634, 2025)
Hierarchical perception, closed-loop sensorimotor control, modular cognitive
reasoning, neuroplastic memory adaptation, neuromorphic computation.
**Relevance:** Framework paper; good for discussion of where WorldNN sits
relative to modern neuroscience-inspired agent architectures.

### 6. Representation learning in artificial/biological neural networks underlying sensorimotor integration (Science Advances 2023, still 2025-cited)
Direct comparison of ANN vs biological representations for sensorimotor
tasks.
**Relevance:** Establishes the precedent for comparing learned representations
to neural data — suggests a validation direction for SA.

### 7. Embodied AI: A Survey on the Evolution from Perceptive to Behavioral Intelligence (SmartBot 2025)
Evolution narrative: perception-first → behavior-first agents.
**Relevance:** Citation for "perception-action asymmetry" framing in
Related Work.

### 8. Toward the next frontier of embodied AI (oaepublish 2025, ir.2025.44)
Next-frontier viewpoint on embodied AI, including structural priors.

## SA-analogue metrics in these papers

- *Neural similarity scores* (Representation Learning in ANN/Biology, 2023):
  compare ANN activations to neural recordings. Structurally analogous to
  SA but target is biological activity, not oracle actions.
- *Task-success rates on manipulation benchmarks* (ManiSkill, Habitat):
  blunt performance measures. SA is a more structural probe.
- *Information-theoretic transfer bounds* (Sensorimotor Representation
  Learning survey 2021, cited): information-bottleneck-style analyses of
  representation learning. Our obj-025 T3 rate-distortion argument
  parallels this.

## Methodological moves unexplored by WorldNN

1. *Neural-similarity validation*: compare organism embeddings to macaque
   PPC data (2509.14360). Requires biological data. Future work.
2. *Continuous-transformation robustness metric* (2505.07634): measure how
   SA degrades under input perturbations.
3. *Morphological variation* (ACM CS survey): WorldNN uses one fixed
   organism body. Morphology-vs-SA sweep is unexplored.
4. *Imitation-from-predictive-motion-model* (2508.18691): adds a teacher
   signal derived from motion prediction. Our supervision comes from PPO
   reward; this alternative is untested.
