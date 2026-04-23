# Lit Scan A — World Models & Latent Dynamics (2024-2026)

Queries: "world model latent dynamics embodied agent 2025", "Dreamer V4 PlaNet
successor action conditioned latent dynamics 2025", "JEPA Joint Embedding
Predictive Architecture action conditioning robotics 2025".

## Papers collected

### 1. Dreamer V4 — Scalable World Model Agents (Hafner, Yan et al., 2509.24527, 2025)
Third generation of Dreamer. Block-causal transformer architecture for both
tokenizer and dynamics model. Learns accurate action conditioning from 2500h
of video with only 100h of paired actions (80% action-conditioned generation
accuracy without full supervision). Uses *shortcut forcing* for 4-step
denoising of latent states; X-prediction for long-rollout stability.
**Relevance to WorldNN:** Their "action-conditioned latent dynamics" IS the
canonical formalism. WorldNN's SA metric is measuring something orthogonal
(policy-action alignment, not latent rollout accuracy). Dreamer's evaluation
is per-task reward across 150 tasks; ours is per-config SA correlation. They
could compute something like SA on their agents if they had oracle access;
they'd see it collapse under information-starved regimes like we do.

### 2. V-JEPA 2-AC (Meta AI, 2506.09985, 2025)
Action-conditioned Joint-Embedding Predictive Architecture. Post-trained on
<62h of unlabeled Droid robot videos. Deployed zero-shot on Franka arms for
pick-and-place with image goals. Predictor uses block-causal attention;
trained with teacher-forcing + rollout losses. Pretrained encoders frozen;
only the action-conditioned predictor is trained for conditional embedding
prediction.
**Relevance:** Direct comparison point for "organism with capacity but no
action grounding" — their encoder is frozen (analogous to oracle perception
with compressed channels), and only the action-conditioning predictor is
trained. Their ablations on action-supervision amount could inform our
sensory-capacity substitution experiments. They do NOT analyze a SA-like
quantity; gap we could fill.

### 3. ACT-JEPA (2501.14622, 2025)
Unifies imitation learning and self-supervised learning. Jointly predicts
action sequences AND latent observation sequences.
**Relevance:** They couple policy learning with world-model learning via a
shared predictor. Our organism has a separate policy head (from obj-024) and
a predictive head exists (PredictiveOrganism) but we've never tied them. An
ACT-JEPA-inspired joint-loss variant could be a WorldNN ablation testing
whether action prediction improves SA.

### 4. A Comprehensive Survey on World Models for Embodied AI (2510.16732, 2025)
Three families: RSSM-based (Dreamer), JEPA-based, Transformer-based.
RSSM enhances predictive capabilities by learning temporal dynamics via
visual inputs, enabling latent trajectory optimization.
**Relevance:** Good citation for the Related Work section of our paper.
Situates WorldNN squarely in the "RSSM-like with action-conditioning" family.

### 5. MoWM — Mixture-of-World-Models (2509.21797, 2025)
Modulates low-level pixel-world-model features with a latent-space world
model for global temporal dynamics. Hybrid integration enhances
action-relevant signals while preserving visual details.
**Relevance:** Orthogonal to WorldNN — they're tackling multi-resolution
world models. But the "action-relevant signals" framing maps to our SA
ceiling idea: how much action-relevant information survives the chain.

### 6. Embodied AI Agents — Modeling the World (2506.22355, 2025)
Survey of embodied agents' world-model approaches.

### 7. A review of embodied intelligence systems (Frontiers 2025)
Three-layer framework: multimodal perception, world modeling, structured
strategies.
**Relevance:** Matches our three-layer architecture (matter → environment →
organism). Good framing citation.

### 8. A Survey of Embodied World Models (Shang et al., Tsinghua, 2025)
Comprehensive survey. Complements 2510.16732.

## Methodological moves unexplored by WorldNN

1. *Block-causal transformer* as dynamics backbone (Dreamer V4). We use
   small MLPs; scaling to transformer might shift embedding_dim dependence.
2. *Shortcut forcing* / few-step diffusion rollout (Dreamer V4) — irrelevant
   for our task but noteworthy for scaling.
3. *Action-supervision sparsity* — Dreamer V4 gets 80% from 4% paired
   action data; analogue in our setup: train with partial action labels.
4. *Joint action-observation prediction loss* (ACT-JEPA) — couples the
   prediction-error signal to the policy in a way our current setup does
   not.
5. *Frozen encoder + trained predictor* (V-JEPA 2-AC) — tests whether
   action conditioning can be learned without touching perception. An
   ablation we have not run.
6. *Teacher-forcing + rollout losses jointly* (V-JEPA 2-AC) — we only use
   per-step reward; rollout-stability losses may improve SA dynamics.

## 1-sentence relevance summaries

| Paper | Relevance |
|-------|-----------|
| Dreamer V4 | Canonical action-conditioned world model; would benefit from SA-style diagnostic |
| V-JEPA 2-AC | Frozen-encoder ablation WorldNN hasn't run; useful baseline |
| ACT-JEPA | Joint policy+prediction loss is a WorldNN ablation candidate |
| 2510.16732 survey | Cite in Related Work |
| MoWM | Action-relevant-signals framing aligns with SA ceiling |
| 2506.22355 survey | Cite in Related Work |
| Frontiers 2025 review | Three-layer architecture citation |
| Shang et al. survey | Cite alongside 2510.16732 |
