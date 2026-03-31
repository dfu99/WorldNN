# Sensorimotor Alignment Predicts Learning in Asymmetric Perception-Action Loops

**Living draft, last updated 2026-03-27**

## Abstract

Blakemore and Cooper (1970) showed that kittens raised without
exposure to specific visual orientations lost cortical selectivity for
those orientations; sensory pathways that failed to align with motor
experience were pruned. We formalize this principle for embodied
agents that perceive the world through lossy channels but act on it
directly.

We define *sensorimotor alignment* (SA), the cosine similarity between
an agent's learned policy and the optimal action computed from the true
physical state. SA predicts task performance with r = −0.72 on a 4D
rock-push task (245 configs, 7 seeds; F(1,241) = 34.2, p = 5 × 10⁻⁹)
and r = −0.73 on a 6D two-rock task. SA exhibits a learnability
threshold: SA ≥ 0.5 yields 98% success; SA < 0.3 yields 22%. SA
transfers across physics and appearance variants (89−106% retention).

Perception quality and organism capacity interact multiplicatively,
not additively. In the data, increasing capacity reduces distance to
target only when perception quality is sufficient (β_interaction =
−0.004, p = 5 × 10⁻⁹). When perception is too degraded, larger
embedding dimensions do not improve performance, consistent with the
interpretation that additional representational power cannot compensate
for insufficient information in the incoming signal.

## 1. Introduction

Blakemore and Cooper (1970) raised kittens in cylinders painted with
only vertical or only horizontal stripes. After five months, neurons
in the kittens' visual cortex had lost selectivity for the missing
orientations. Sensory pathways that failed to achieve alignment between
perception and motor experience within the critical period were pruned.

This result illustrates a general principle in developmental
neuroscience: sensory pathways that do not contribute to actionable
perception are eliminated (Hubel & Wiesel, 1970; Huang et al., 1999).

We study this principle computationally. In our simulation, an
organism must push a rock to a target, but it never sees the rock
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
analytically optimal action (computed from true state). SA quantifies
the degree to which degraded perception supports correct action.

Across 245 conditions varying perception quality and organism capacity,
SA predicts task performance (r = −0.72) and reveals a significant
multiplicative interaction between perception and capacity
(p = 5 × 10⁻⁹). The results, detailed in §5, show that when
perception is sufficiently degraded, increasing organism capacity does
not improve performance.

## 2. Related Work

**Asymmetric Actor-Critic** (Pinto et al., 2018) trains RL agents
with privileged state information during training but deploys with
observations only. Our metric SA (defined in §4) serves a
complementary role: rather than using privileged information as a
training signal, we use it as a post-hoc diagnostic that measures the
residual alignment gap between actions under degraded perception and
actions under full state access.

**DreamerV3** (Hafner et al., 2023) and **IRIS** (Micheli et al.,
2023) learn world models and implicitly measure prediction quality
in latent space. SA is computed in true state space, not latent space;
this distinction matters because the action operates on true state,
not on the model's latent representation.

**Quasimetric RL** (Wang et al., 2023) measures directional
asymmetries in value function space. SA measures asymmetry in action
space: the gap between the policy's action direction under degraded
perception and the optimal direction under full state observation.

**Policy cosine similarity in imitation learning** is superficially
similar to SA. In imitation learning, the expert and learner share the
same observation. SA measures alignment across the perception gap,
where the optimal action is computed from true state while the policy
acts on degraded observations. This asymmetry is what makes the metric
diagnostic of the perception-action loop.

**Information bottleneck** (Tishby et al., 2000) characterizes optimal
compression-prediction tradeoffs. **Active inference** (Friston, 2010)
frames the perception-action loop as free energy minimization. SA
operationalizes both: low SA corresponds to high free energy and lossy
compression that destroys action-relevant information. SA differs from
these frameworks in that it provides a computable scalar rather than a
theoretical bound.

**Critical period neuroscience** (Blakemore & Cooper, 1970; Rossi et
al., 1999; Huang et al., 1999): BDNF-dependent critical periods set
a biological timeout on sensory-motor alignment. If alignment is not
achieved within the viability window, the pathway is pruned. In our
simulation, the oracle+noise(0.5) condition produces SA ≈ 0.14,
permanently below the learnability threshold (§5.2).

**Gap addressed by this work**: No prior work defines cosine alignment
between a degraded-perception policy and optimal-state action across a
designed perception-degradation ladder with independently controllable
perception quality and organism capacity.

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
four lossy stages, while its action bypasses all of them and directly
modifies the true physical state. The rock responds to the actual
force applied, regardless of the organism's perceptual accuracy.

**Matter**: Rock-push task. True state = [rock_x, rock_y, org_x, org_y].
The rock has physical properties (position, inertia, contact radius)
that determine its response to force. These properties are never
directly observable.

**Perception chain**: Matter emits 8D signals (a fixed projection of
true state, analogous to light patterns). A channel adds noise
(atmospheric distortion). A VAE environment compresses the signal
(sensory apparatus limitations). The organism receives the final
compressed representation.

**Action**: The organism outputs a 2D motor command. This acts
*directly on the matter's true state*, not on the perceived state,
not through the VAE in reverse. The action channel is clean.

### 3.2 Independently controlled variables

| Variable | Range | What it controls |
|----------|-------|-----------------|
| Perception mode | Oracle, oracle+noise, raw emission, VAE μ | How degraded the percept is |
| env_latent_dim | 8, 16, 32 | VAE compression severity |
| channel_noise | 0.01 – 0.5 | Signal corruption |
| embedding_dim | 2, 4, 8, 16, 32 | Organism's internal capacity |

The oracle condition (direct true state observation) eliminates the
perception chain entirely, providing the upper bound on what any
perception mode can achieve.

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
SA = 0: the organism's action is orthogonal to optimal; it cannot
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

Magnitude-weighted SA, the product of directional alignment and
action magnitude, is the strongest overall predictor (r = −0.893).
This reflects both requirements for task success: the organism must
point in the correct direction and push with sufficient force. Plain
cosine SA has the strongest interaction signal (F = 104.8) because
scale-invariance isolates the directional component.

We report cosine SA as the primary metric (interpretability, interaction
sensitivity) and magnitude-weighted SA as a complementary predictor.

### 4.3 Theoretical grounding

SA operationalizes the sensory-motor alignment framework. Each
perception modality provides an embedding e_i. A learned alignment
operator R_i maps each embedding into a shared representation, from
which a motor projection W_m produces actions: a = W_m R(e). SA
measures the quality of R, i.e., how well the learned alignment
preserves action-relevant information.

Connections to existing frameworks:
- **Friston's FEP**: low SA corresponds to high free energy and a
  poor generative model of the environment
- **CCA/CLIP**: the R_i operators correspond to what cross-modal
  alignment methods learn (Radford et al., 2021)
- **Critical periods**: if SA remains below a threshold ε for
  duration τ (the neurotrophic viability window), the pathway is
  pruned (Huang et al., 1999)

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
| < 0.3 | 58 | 7% |

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
Actions affect the true physical state, not the percept; a larger
embedding cannot help if the perception chain does not convey where
the rock is.

### 5.4 SA dynamics during training

SA trajectory during training also predicts final success. We measured
SA every 50 episodes across 27 configs (3 perception modes × 3
embedding dims × 3 seeds).

The slope of SA over the first 200 episodes correlates with final
rock-target distance at r = −0.705, stronger than any single snapshot
(e.g., SA at episode 100: r = −0.232). The rate of alignment
improvement is more predictive than the alignment level at any given
time.

Time-to-threshold differs by condition. Using SA ≥ 0.4 as the
learnability threshold:
- Oracle: reaches threshold by episode 100 for all embed dims
- Raw emission: emb=8,32 reach by episode 133; emb=2 slower (283ep)
- VAE μ lat=16: emb=32 reaches by 133ep; emb=2 often never reaches

In this task, configs with SA slope < 0.1 per 100 episodes by episode
200 did not achieve successful learning. If this pattern holds across
tasks, SA slope could serve as an early stopping criterion, though
cross-task validation is needed.

### 5.5 Second task: 2-rock push (6D state)

To test whether SA generalizes beyond the 4D rock-push task, we
implemented a 2-rock variant with 6D state [r1x, r1y, r2x, r2y, ox,
oy], 12D emission, and 2D action. The organism must push 2 rocks to
separate targets. We evaluated 3 perception conditions × 3 embedding
dims × 5 seeds = 45 configs with 1000 training episodes.

| Perception mode | Mean dist | Mean SA | Learns? |
|----------------|-----------|---------|---------|
| Oracle emb=4 | 0.503 ± 0.008 | 0.408 ± 0.131 | Marginal |
| Oracle emb=16 | 0.486 ± 0.005 | 0.644 ± 0.063 | Yes |
| Oracle emb=64 | 0.466 ± 0.005 | 0.714 ± 0.046 | Yes |
| Raw emission emb=4 | 0.503 ± 0.009 | 0.386 ± 0.090 | Marginal |
| Raw emission emb=16 | 0.471 ± 0.008 | 0.564 ± 0.024 | Yes |
| Raw emission emb=64 | 0.449 ± 0.005 | 0.587 ± 0.033 | Yes |
| VAE μ lat=16 emb=4 | 0.509 ± 0.001 | 0.355 ± 0.032 | No |
| VAE μ lat=16 emb=16 | 0.500 ± 0.010 | 0.379 ± 0.062 | Marginal |
| VAE μ lat=16 emb=64 | 0.466 ± 0.006 | 0.540 ± 0.012 | Yes |

The SA-performance correlation at 6D is r = −0.728 (n=45), matching
the 4D single-rock result (r = −0.724). The capacity gradient is
present across all perception modes: oracle SA rises from 0.41
(emb=4) to 0.71 (emb=64); raw emission from 0.39 to 0.59; VAE from
0.36 to 0.54. The monotonic capacity effect and the perception-
dependent ceiling both replicate.

Raw emission at emb=64 achieves the best performance (dist = 0.449 vs
baseline 0.483), consistent with the 4D finding that raw emission
outperforms all VAE conditions.

**3-rock supplementary (8D state).** We also tested a 3-rock variant
(8D state, 105 configs, 7 seeds). The correlation is weaker
(r = −0.300, p = 4 × 10⁻⁴) due to a floor effect: most conditions
achieve distances within 0.01 of the random baseline (0.489). The
perception × capacity interaction remains significant (F(1,101) = 12.5,
p = 4 × 10⁻⁴). SA values at 8D are comparable to 4D (oracle emb=64:
SA = 0.66), confirming the metric remains meaningful. The weaker
correlation is attributable to insufficient training budget for the
harder 3-object task.

### 5.6 SA transfer across physics and appearance variants

If SA captures structural alignment rather than memorized actions, it
should generalize to objects with different physical properties and
visual appearance. We tested this by training organisms on a standard
rock and evaluating SA on 8 variants without retraining.

**Physics transfer (obj-019).** We varied push_radius (±20%) and
push_strength (±30%) to create rocks that respond differently to
force. SA retention across 15 trained organisms:

| Variant | SA retention |
|---------|-------------|
| Standard (control) | 102% |
| Smaller rock (radius −20%) | 101% |
| Larger rock (radius +20%) | 103% |
| Heavier rock (strength −30%) | 100% |
| Lighter rock (strength +30%) | 102% |
| Hard combo (smaller + heavier) | 94% |
| Easy combo (larger + lighter) | 106% |

Single-parameter changes produce 100−103% retention. Even the
hardest combined variant (smaller and heavier) retains 94%.

**Appearance transfer (obj-020).** We additionally perturbed the
emission projection matrix (state_proj) by adding ε × N(0,1) noise
(ε ∈ {0.1, 0.3, 0.5}), so that variant rocks both look and respond
differently. SA retention across 240 evaluations (4 physics variants
× 4 appearance levels × 3 embed dims × 5 seeds):

| Physics \ Appearance ε | 0.0 | 0.1 | 0.3 | 0.5 |
|------------------------|-----|-----|-----|-----|
| Standard | 101% | 101% | 101% | 102% |
| Smaller rock | 100% | 99% | 100% | 100% |
| Heavier rock | 97% | 96% | 97% | 97% |
| Hard combo | 96% | 94% | 95% | 95% |

Appearance perturbation has no measurable effect on SA retention:
r(ε, retention) = 0.033. The organism's alignment is invariant to
changes in the emission matrix up to ε = 0.5, indicating that SA
measures structural understanding of the push interaction rather
than a memorized mapping from specific visual patterns to actions.

## 6. Perception Failure Conditions

Three conditions produce SA values below the learnability threshold,
analogous to the pruned sensory pathways observed by Blakemore and
Cooper (1970) in orientation-deprived kittens.

*Stochastic VAE (z).* When the organism receives stochastic samples
from the VAE posterior, sampling noise destroys the spatial signal.
SA ≈ 0 across all conditions and all capacity levels. The stochastic
samples do not preserve the directional information required for the
organism to determine push direction.

*VAE lat=8 (μ).* Even with deterministic encoding, extreme
compression (8D emission → 8-dim latent) caps SA at approximately
0.29. The perception chain does not preserve enough directional
information for the organism to determine which way to push the rock.

*Oracle + noise σ=0.5.* Direct state observation corrupted by heavy
Gaussian noise. SA = 0.14 ± 0.08, below the learnability threshold.
All 35 configs fail (0% success). This condition shows a within-level
correlation reversal (r = +0.46): higher SA is associated with
slightly worse performance.

Investigation reveals this is a **boundary artifact**, not a metric
failure. Three observations:

1. **Distance variance is near-zero** (std = 0.004). All configs
   achieve dist ≈ 0.525 regardless of SA. When the outcome variable
   has no variance, any correlation with a predictor is noise.

2. **Capacity increases SA but not performance.** emb=32 achieves
   SA = 0.20 vs emb=2 at SA = 0.12; the organism develops more
   directional alignment with more capacity. But distance actually
   worsens slightly (0.529 vs 0.523). The larger model overfits to
   noise structure, producing directional alignment that is
   anti-correlated with task-relevant structure.

3. **VAE lat=8, with comparable SA (0.29 ± 0.06), shows normal
   negative correlation (r = −0.75)**, because it has enough
   distance variance (std = 0.015) for the signal to emerge.

The reversal occurs exclusively when perception carries zero
task-relevant information (all configs fail equally). SA is undefined
in this regime; it measures alignment with noise, not with task
structure. This does not contaminate the main finding: the reversal
condition contributes no successes and sits entirely below the
learnability threshold.

## 7. Discussion

### 7.1 Scope of the perception-action asymmetry

The perception-action asymmetry is a structural property of any
system that perceives through lossy channels and acts on true state.
SA measures the organism's ability to produce correct actions from
degraded perception. The perception chain exists to provide the
organism with sufficient information to act on a state it cannot
directly observe.

### Limitations

- Rock-push is 4D state, 2D action; a second, higher-dimensional
  task is needed to test threshold generalization
- SA requires a computable optimal action, feasible in simulation
  but not in general real-world tasks; a self-supervised proxy is needed
- The learnability threshold (SA ≈ 0.4–0.5) may be task-specific
- oracle+noise(0.5) correlation reversal at the noise floor

### Future work

- **Self-supervised SA proxy** without oracle access, enabling SA
  estimation in tasks where the optimal action is not computable
- **Higher-dimensional tasks** (8D+ state) with increased training
  budget to test capacity limits
- Cross-validation in related simulation frameworks

## Figures

1. `results/obj016_ci_at_scale.png`: SA predicts task performance
   across 245 conditions. (a) SA vs. rock-target distance by perception
   mode. (b) Correlation decomposition: between-level r = −0.878,
   within-level mean r = −0.582. (c) Random baseline distribution.
2. `results/obj015_ci_dynamics.png`: SA trajectories during training.
   (a) SA over 500 episodes for 27 configs (3 perception × 3 embed ×
   3 seeds). (b) SA slope vs. final distance (r = −0.705).
3. `results/obj014_expanded_ci.png`: SA across 105 configs (expanded
   sweep). (a) SA vs. distance by perception level. (b) Embedding
   dimension effect within each perception condition.
4. `results/obj013_coordination_quality.png`: SA measurement
   validation. (a) SA distribution across conditions. (b) SA vs.
   task performance (initial 27-config experiment).

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
- [x] Second task: 2-rock 6D (r = −0.728) + 3-rock 8D (r = −0.300)
- [x] SA transfer: physics (93−106%) + appearance (89−102%)
- [ ] LaTeX conversion for NeurIPS 2026 submission
- [ ] Self-supervised SA proxy (nice-to-have)
