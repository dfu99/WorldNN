# Sensorimotor Alignment Predicts Learning in Asymmetric Perception-Action Loops

**Living draft — last updated 2026-03-24**

## Abstract

In 1970, Blakemore and Cooper raised kittens in environments with
restricted visual input. The kittens went functionally blind — not
because their eyes failed, but because their visual cortex could not
align what it saw with what the body experienced. The sensory pathway
was pruned.

We formalize this phenomenon. Embodied agents face a fundamental
asymmetry: they perceive the world through lossy channels but act on
it directly. We introduce *sensorimotor alignment* (SA) — the cosine
alignment between an agent's learned policy and the optimal action on
the true physical state — and show it predicts task performance with
r = −0.72 across 245 independently controlled conditions (7 perception
modes × 5 capacity levels × 7 seeds; F(1,241) = 34.2, p = 5 × 10⁻⁹
for the perception × capacity interaction). A learnability threshold
emerges: SA ≥ 0.5 yields 98% success; SA < 0.3 yields 22%.

The key finding: perception quality and organism capacity are
*multiplicative*. Increasing capacity cannot compensate for inadequate
perception, because additional representational power is useless when
the incoming signal carries insufficient information about the true
state that actions affect. The organism is computationally blind — like
Blakemore and Cooper's kittens.

## 1. Introduction

In 1970, Blakemore and Cooper raised kittens in cylinders painted with
only vertical or only horizontal stripes. After five months, the
kittens could not see orientations they had never experienced — neurons
in their visual cortex had lost selectivity for the missing angles.
The visual pathway that could not align with motor experience was
pruned (Blakemore & Cooper, 1970).

This is not merely a curiosity of developmental neuroscience. It
reveals a general principle: sensory pathways that fail to achieve
*alignment* between perception and action within a critical window
are eliminated. The brain does not maintain channels that carry no
actionable information.

We study this principle computationally. In our simulation, an
organism must push a rock to a target — but it never sees the rock
directly. It perceives the world through a chain of lossy
transformations (emission → noise → compression), while its actions
operate directly on the true physical state. The perception-action
loop is fundamentally asymmetric:

```
Perception:  true state → emission → medium → encoding → internal model
                          (lossy)    (noisy)   (compressed)  (bounded)

Action:      motor command → directly modifies true state
```

We define *sensorimotor alignment* (SA): the cosine similarity between
the organism's learned action (based on degraded perception) and the
analytically optimal action (computed from true state). SA measures
the bridge between what the organism sees and what it must do.

Across 245 conditions varying perception quality and organism capacity,
SA predicts task performance (r = −0.72), exhibits a learnability
threshold (SA ≥ 0.5 → 98% success), and reveals a significant
multiplicative interaction between perception and capacity
(p = 5 × 10⁻⁹). When perception is too degraded, no amount of
capacity helps — the organism is computationally blind.

## 2. Related Work

**Asymmetric Actor-Critic** (Pinto et al., 2018) trains RL agents
with privileged state information during training but deploys with
observations only. SA is complementary: rather than using privileged
info as a training signal, we use it as a *diagnostic* — measuring
the residual alignment gap between what the policy achieves under
degraded perception and what it could achieve with full state access.

**DreamerV3** (Hafner et al., 2023) and **IRIS** (Micheli et al.,
2023) learn world models and implicitly measure prediction quality
in latent space. SA differs in that it is computed in *true state
space*, not latent space — this distinction matters because the
action operates on true state, not on the model's latent
representation.

**Quasimetric RL** (Wang et al., 2023) measures directional
asymmetries in value function space. SA measures asymmetry in *action
space* — the gap between the policy's action direction under degraded
perception and the optimal direction under full state observation.

**Policy cosine similarity in imitation learning** is superficially
similar to SA. The key difference: in imitation learning, the expert
and learner share the same observation. SA specifically measures
alignment *across the perception gap* — the expert sees true state
while the learner sees degraded observations. This is what makes SA
diagnostic of the perception-action asymmetry.

**Information bottleneck** (Tishby et al., 2000) characterizes optimal
compression-prediction tradeoffs. **Active inference** (Friston, 2010)
frames the perception-action loop as free energy minimization. SA
operationalizes both: low SA ↔ high free energy ↔ lossy compression
that destroys action-relevant information. But SA provides a concrete,
computable number rather than a theoretical bound.

**Critical period neuroscience** (Blakemore & Cooper, 1970; Rossi et
al., 1999; Huang et al., 1999): BDNF-dependent critical periods set
a biological timeout on sensory-motor alignment. If alignment (SA) is
not achieved within the viability window, the pathway is pruned. Our
simulation reproduces this: the oracle+noise(0.5) condition shows
SA ≈ 0.14, permanently below threshold — the computational blind cat.

**Gap filled by this work**: No prior paper defines cosine alignment
between a degraded-perception policy and optimal-state action across a
designed perception-degradation ladder with independently controllable
perception quality and organism capacity. That gap is genuine.

## 3. The WorldNN Simulation

### 3.1 The asymmetric loop

WorldNN models the full asymmetric perception-action loop:

```
┌──────────────────── PERCEPTION (lossy) ────────────────────┐
│                                                             │
│  Matter ──► Emission ──► Channel ──► Environment ──► Organism
│  (true       (8D          (adds       (VAE            (sensory
│   state)      projection)  noise)      compression)    filter →
│                                                        embedding →
│                                                        policy)
│                                                             │
│  Matter ◄──────────────── ACTION (direct) ──────────────────┘
│  (true state                  2D motor command
│   modified directly)          operates on real physics
└─────────────────────────────────────────────────────────────┘
```

The critical asymmetry: the organism's observation passes through
four lossy stages. Its action bypasses all of them — it directly
modifies the true physical state. The rock's atoms do not care how
the organism perceived them. They respond to the actual force applied.

**Matter**: Rock-push task. True state = [rock_x, rock_y, org_x, org_y].
The rock has physical properties (position, inertia, contact radius)
that determine its response to force. These properties are never
directly observable.

**Perception chain**: Matter emits 8D signals (a fixed projection of
true state — analogous to light patterns). A channel adds noise
(atmospheric distortion). A VAE environment compresses the signal
(sensory apparatus limitations). The organism receives the final
compressed representation.

**Action**: The organism outputs a 2D motor command. This acts
*directly on the matter's true state* — not on the perceived state,
not through the VAE in reverse. The action channel is clean.

### 3.2 Independently controlled variables

| Variable | Range | What it controls |
|----------|-------|-----------------|
| Perception mode | Oracle, oracle+noise, raw emission, VAE μ | How degraded the percept is |
| env_latent_dim | 8, 16, 32 | VAE compression severity |
| channel_noise | 0.01 – 0.5 | Signal corruption |
| embedding_dim | 2, 4, 8, 16, 32 | Organism's internal capacity |

The oracle condition (direct true state observation) eliminates the
perception chain entirely — the surgeon sees clearly. This provides
the upper bound on what any perception mode can achieve.

## 4. Sensorimotor Alignment

### 4.1 Definition

The organism must bridge degraded perception to correct action on true
state. We measure this bridge directly.

After training, for each sampled true state s:

$$\text{SA} = \mathbb{E}_s\left[\cos\left(\pi_\theta(o(s)),\ a^*(s)\right)\right]$$

where:
- π_θ(o(s)) is the organism's action given its degraded observation
- a*(s) is the analytically optimal action computed from the true state
  (move toward rock if far, push rock toward target if near)

SA = 1: the organism's action perfectly matches what it should do,
despite only seeing a degraded version.
SA = 0: the organism's action is orthogonal to optimal — it cannot
bridge the perception-action gap.

### 4.2 Metric ablation: why cosine alignment?

We compared six candidate metrics as predictors of task performance
on the at-scale dataset (n=245):

| Metric | Overall r | Within-level r | Interaction F |
|--------|----------|----------------|--------------|
| **mag-weighted SA** (cos × ‖a‖) | **−0.893** | **−0.714** | 98.6 |
| action magnitude ‖a‖ | −0.739 | −0.582 | 39.4 |
| |SA| (abs cosine) | −0.727 | −0.588 | 101.7 |
| SA (cosine) | −0.724 | −0.582 | **104.8** |
| positive fraction (SA > 0) | −0.699 | −0.568 | 108.5 |
| embedding utilization | −0.499 | −0.573 | 12.2 |

**Magnitude-weighted SA** — the product of directional alignment and
action magnitude — is the strongest overall predictor (r = −0.893).
The organism must both point in the right direction AND push with
sufficient force. Plain cosine SA has the strongest interaction signal
(F = 104.8) because scale-invariance isolates the directional component.

We report cosine SA as the primary metric (interpretability, interaction
sensitivity) and magnitude-weighted SA as a complementary predictor.

Note: SA is superficially similar to cosine action-prediction accuracy
used in imitation learning. The critical distinction is that in
imitation learning, expert and learner share the same observation
space. SA measures alignment *across the perception gap* — the optimal
action is computed from true state, while the policy acts on degraded
observations. This asymmetry is the core of the metric.

### 4.3 Theoretical grounding

SA operationalizes the sensory-motor alignment framework: perception
modalities provide embeddings e_i that must be *aligned* via learned
operators R_i into a representation from which motor commands are
projected: a = W_m R(e). SA measures the quality of that alignment.

Connections:
- **Friston's FEP**: low SA ↔ high free energy ↔ poor generative model
- **CCA/CLIP**: R_i operators are what cross-modal alignment learns
- **Critical periods**: if SA < ε for duration τ (neurotrophic
  viability window), the pathway is pruned — the biological blind cat

## 5. Results

All results from the at-scale experiment: 245 trained configs + 35
random baselines. 7 perception conditions × 5 embedding dimensions ×
7 seeds per condition. Random baseline (untrained organism):
dist = 0.516 ± 0.003, SA = 0.003 ± 0.022. Success defined as
dist < 0.511 (baseline − 2σ).

### 5.1 SA predicts task performance across 7 perception conditions

| Perception mode | Mean dist ↓ | Mean SA ↑ | Success rate |
|----------------|-------------|------------|-------------|
| Oracle (direct true state) | 0.480 ± 0.050 | 0.483 ± 0.142 | 69% (24/35) |
| Oracle + noise σ=0.1 | 0.488 ± 0.042 | 0.452 ± 0.138 | 63% (22/35) |
| Raw emission (8D, no VAE) | 0.466 ± 0.039 | 0.489 ± 0.090 | 89% (31/35) |
| VAE μ lat=32 | 0.498 ± 0.022 | 0.452 ± 0.066 | 63% (22/35) |
| VAE μ lat=16 | 0.499 ± 0.021 | 0.440 ± 0.076 | 57% (20/35) |
| VAE μ lat=8 | 0.508 ± 0.014 | 0.286 ± 0.057 | 37% (13/35) |
| Oracle + noise σ=0.5 | 0.525 ± 0.004 | 0.141 ± 0.077 | 0% (0/35) |

Overall Pearson correlation: **r = −0.724** (n=245, p < 10⁻⁴⁰).

**Correlation decomposition**: Between the 7 condition means,
r = −0.878. Within individual conditions, mean r = −0.582 (range:
−0.801 to +0.463). Six of seven conditions show strong within-level
correlation (r < −0.7); oracle+noise(0.5) is the exception (see §6).

### 5.2 Learnability threshold

| SA threshold | N above | Success rate |
|-------------|---------|-------------|
| ≥ 0.3 | 187 | 68% |
| ≥ 0.4 | 124 | **88%** |
| ≥ 0.5 | 53 | **98%** |
| ≥ 0.6 | 18 | **100%** |
| < 0.3 | 58 | 22% |

SA ≥ 0.5 yields 98% success (52/53). SA ≥ 0.6 yields 100% (18/18).
The transition from failure to success occurs in the SA = 0.3–0.5
range, with the sharpest inflection at 0.4.

### 5.3 Formal interaction test: capacity × perception

A log-linear regression model predicts rock-target distance from
perception quality (ordinal, 0–6), log₂(embedding_dim), and their
interaction:

$$\text{dist} = \beta_0 + \beta_p \cdot \text{perc} + \beta_c \cdot \log_2(\text{emb}) + \beta_{pc} \cdot \text{perc} \times \log_2(\text{emb}) + \epsilon$$

| Coefficient | Value | Interpretation |
|------------|-------|---------------|
| β_perception | +0.003 | Main effect of perception quality |
| β_capacity | −0.001 | Main effect of embedding dimension |
| **β_interaction** | **−0.004** | Capacity helps MORE with better perception |

Full model R² = 0.458. The interaction term is highly significant:
**F(1, 241) = 34.2, p = 5.0 × 10⁻⁹**. Adding the interaction
increases R² by 0.077 over the additive model (R² = 0.381).

The negative interaction coefficient confirms that capacity and
perception are **multiplicative, not additive**: increasing embedding
dimension reduces distance more when perception quality is higher.
Actions affect the true physical state, not the percept — a larger
brain cannot help if the eyes do not convey where the rock actually is.

## 6. The Blind Cat

Blakemore & Cooper's (1970) kittens lost orientation selectivity
because the visual input could not align with motor experience within
the critical period. We observe the same phenomenon computationally.

**Stochastic VAE (z)**: When the organism receives stochastic samples
from the VAE posterior, sampling noise destroys spatial signal.
SA ≈ 0 across all conditions — zero learning, regardless of capacity.
The organism is computationally blind.

**VAE lat=8 (μ)**: Even with deterministic encoding, extreme
compression (8D→8D→8-dim latent) caps SA at ~0.29. The perception
chain does not preserve enough directional information for the
organism to determine which way to push the actual rock.

**Oracle + noise σ=0.5**: Direct state observation corrupted by heavy
noise. SA drops to 0.14 — below the learnability threshold. This
condition also shows a within-level correlation *reversal* (r = +0.46):
higher SA is associated with slightly worse performance. We interpret
this as a boundary effect: when perception is at the noise floor, the
cosine metric captures alignment with noise structure rather than task
structure. This reversal occurs only when SA is uniformly below the
learnability threshold (all 35 configs fail), so it does not
contaminate the main finding.

## 7. Discussion

### The asymmetry as the core contribution

The perception-action asymmetry is not merely a detail of our
simulation — it is the fundamental structure of embodied cognition.
SA measures the organism's ability to bridge degraded perception to
direct physical action. The entire perception chain exists to give
the organism enough information to act correctly on a reality it
can never directly observe.

### Limitations

- Rock-push is 4D state, 2D action — a second, higher-dimensional
  task is needed to test threshold generalization
- SA requires a computable optimal action (feasible in simulation,
  not in general real-world tasks) — a self-supervised proxy is needed
- The learnability threshold (SA ≈ 0.4–0.5) may be task-specific
- oracle+noise(0.5) correlation reversal at the noise floor

### Future work

- **Second task** (8D+ state) to validate threshold generalization
- **Self-supervised SA proxy** without oracle access
- SA dynamics during training (slope r = −0.705 predicts success)
- Cross-validation in vaural and CorticalNN

## Figures

1. `results/obj016_ci_at_scale.png` — At-scale (245 configs, 7 seeds,
   random baseline, correlation decomposition)
2. `results/obj015_ci_dynamics.png` — SA dynamics during training
3. `results/obj014_expanded_ci.png` — Expanded sweep (105 configs)
4. `results/obj013_coordination_quality.png` — Initial SA discovery

## References

- Blakemore, C. & Cooper, G. F. (1970). Development of the brain depends on the visual environment. Nature, 228(5270), 477-478.
- Friston, K. (2010). The free-energy principle: a unified brain theory? Nature Reviews Neuroscience, 11(2), 127-138.
- Hafner, D. et al. (2023). Mastering Diverse Domains through World Models. arXiv:2301.04104.
- Huang, Z. J. et al. (1999). BDNF regulates the maturation of inhibition and the critical period of plasticity in mouse visual cortex. Cell, 98(6), 739-755.
- Micheli, V. et al. (2023). Transformers are Sample-Efficient World Models. ICLR 2023.
- Pinto, L. et al. (2018). Asymmetric Actor Critic for Image-Based Robot Learning. RSS 2018.
- Radford, A. et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. ICML 2021.
- Rossi, F. M. et al. (1999). Monocular deprivation decreases brain-derived neurotrophic factor immunoreactivity in the rat visual cortex. Neuroscience, 90(2), 363-368.
- Tishby, N., Pereira, F. C. & Bialek, W. (2000). The information bottleneck method. arXiv:physics/0004057.
- Wang, T. et al. (2023). On the Optimal Value of the Quasimetric Entropy. NeurIPS 2023.

## Status

- [x] Core simulation with asymmetric perception-action loop
- [x] SA definition, measurement, and learnability threshold
- [x] At-scale validation: 245 configs, 7 seeds (r = −0.724)
- [x] Formal interaction test: F(1,241) = 34.2, p = 5×10⁻⁹
- [x] Random baseline and explicit success criterion
- [x] Correlation decomposition (between r=−0.878, within r=−0.582)
- [x] SA dynamics: slope r = −0.705 predicts success (obj-015)
- [x] Metric ablation: 6 metrics compared, cosine + mag-weighted best
- [x] Related work: Pinto, DreamerV3, quasimetric RL, imitation learning
- [x] Framing: renamed to "sensorimotor alignment", blind cat hook leads
- [ ] Second task (8D+) — BLOCKER for main track (re-submitted to PACE)
- [ ] Self-supervised SA proxy
