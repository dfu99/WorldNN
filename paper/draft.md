# Coordination Quality Predicts Sensorimotor Learning in Asymmetric Perception-Action Loops

**Living draft — last updated 2026-03-20**

## Abstract

Embodied agents face a fundamental asymmetry: they perceive the world
through lossy channels but act on it directly. An organism sees a rock
through light scattered by atmosphere, compressed by retinal resolution,
and filtered by neural encoding — yet its hand pushes the actual rock,
not the percept. We formalize this asymmetry in a controllable
simulation and introduce *coordination quality* C_i: the cosine
alignment between an agent's learned policy and the optimal action on
the true state. Across 105 conditions spanning 7 perception modes
and 5 capacity levels, C_i predicts task performance with r = -0.74.
A sharp threshold emerges: C_i ≥ 0.8 yields 100% learning success;
C_i < 0.5 yields 85% failure. The key finding is that perception
quality and organism capacity are *multiplicative* — increasing
capacity cannot compensate for inadequate perception, because additional
representational power is useless when the incoming signal carries
insufficient information about the true state that actions affect.

## 1. Introduction

Consider a surgeon operating through a foggy window. Her hands are
inside the room — her actions are precise and direct on the patient's
tissue. But her eyes see only a degraded image through the glass. The
surgery succeeds or fails based on whether the foggy image preserves
enough spatial information to guide her precise hands.

This is the fundamental structure of all embodied cognition. An
organism never perceives matter directly. Light reflects off surfaces,
travels through atmosphere, enters the eye, is transduced by
photoreceptors, compressed by retinal ganglion cells, and encoded in
cortical representations. Each step loses information. Yet the
organism's actions — reaching, pushing, grasping — operate on the
actual physical state of matter: real atoms, real forces, real positions.

The perception-action loop is asymmetric:

```
Perception:  true state → emission → medium → sensory encoding → internal model
                          (lossy)    (noisy)   (compressed)        (bounded)

Action:      motor command → directly modifies true state
```

Prior work treats pieces of this. The information bottleneck (Tishby
et al., 2000) characterizes optimal compression. Active inference
(Friston, 2010) frames the loop as free energy minimization.
Contrastive learning (CLIP; Radford et al., 2021) aligns cross-modal
representations. Critical period neuroscience (Blakemore & Cooper,
1970) shows that sensory pathways pruned when they fail to align with
motor experience — the biological "timeout" on finding the bridge
between perception and action.

What is missing is a *single metric* that predicts whether an agent
can bridge the gap between its degraded perception and its direct
physical actions, in a setting where both perception quality and agent
capacity are independently controllable.

We present such a metric: *coordination quality* C_i.

## 2. The WorldNN Simulation

### 2.1 The asymmetric loop

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

### 2.2 Independently controlled variables

| Variable | Range | What it controls |
|----------|-------|-----------------|
| Perception mode | Oracle, oracle+noise, raw emission, VAE μ | How degraded the percept is |
| env_latent_dim | 8, 16, 32 | VAE compression severity |
| channel_noise | 0.01 – 0.5 | Signal corruption |
| embedding_dim | 2, 4, 8, 16, 32 | Organism's internal capacity |

The oracle condition (direct true state observation) eliminates the
perception chain entirely — the surgeon sees clearly. This provides
the upper bound on what any perception mode can achieve.

## 3. Coordination Quality

### 3.1 Definition

The organism must bridge degraded perception to correct action on true
state. We measure this bridge directly.

After training, for each sampled true state s:

$$C_i = \mathbb{E}_s\left[\cos\left(\pi_\theta(o(s)),\ a^*(s)\right)\right]$$

where:
- π_θ(o(s)) is the organism's action given its degraded observation
- a*(s) is the analytically optimal action computed from the true state
  (move toward rock if far, push rock toward target if near)

C_i = 1: the organism's action on true state perfectly matches what
it should do, despite only seeing a degraded version.
C_i = 0: the organism's action is orthogonal to optimal — it cannot
bridge the perception-action gap.

### 3.2 Why C_i is the right metric

The asymmetry is key. Existing metrics measure one side:

- **Probe R²** measures perception quality (how much true state
  survives the lossy chain) — but not whether the organism uses it.
- **Task reward** measures action success — but not why it succeeds
  or fails.
- **C_i** measures the bridge: does the full pipeline (lossy
  perception → bounded embedding → policy) produce actions that are
  correct *with respect to the true state it cannot see?*

### 3.3 Theoretical grounding

C_i operationalizes the sensory-motor alignment framework: perception
modalities provide embeddings e_i that must be *aligned* via learned
operators R_i into a representation from which motor commands are
projected: a = W_m R(e). C_i measures the quality of that alignment.

Connections:
- **Friston's FEP**: low C_i ↔ high free energy ↔ poor generative model
- **CCA/CLIP**: R_i operators are what cross-modal alignment learns
- **Critical periods**: if C_i < ε for duration τ (neurotrophic
  viability window), the pathway is pruned — the biological "blind cat"

## 4. Results

All results reported from the at-scale experiment (obj-016): 245
trained configs + 35 random baselines. 7 perception conditions × 5
embedding dimensions × 7 seeds per condition. Random baseline
(untrained organism): dist = 0.516 ± 0.003, C_i = 0.003 ± 0.022.
Success defined as dist < 0.511 (baseline − 2σ).

### 4.1 C_i predicts task performance across 7 perception conditions

| Perception mode | Mean dist ↓ | Mean C_i ↑ | Success rate |
|----------------|-------------|------------|-------------|
| Oracle (direct true state) | 0.480 ± 0.050 | 0.483 ± 0.142 | 69% (24/35) |
| Oracle + noise σ=0.1 | 0.488 ± 0.042 | 0.452 ± 0.138 | 63% (22/35) |
| Raw emission (8D, no VAE) | 0.466 ± 0.039 | 0.489 ± 0.090 | 89% (31/35) |
| VAE μ lat=32 | 0.498 ± 0.022 | 0.452 ± 0.066 | 63% (22/35) |
| VAE μ lat=16 | 0.499 ± 0.021 | 0.440 ± 0.076 | 57% (20/35) |
| VAE μ lat=8 | 0.508 ± 0.014 | 0.286 ± 0.057 | 37% (13/35) |
| Oracle + noise σ=0.5 | 0.525 ± 0.004 | 0.141 ± 0.077 | 0% (0/35) |

Overall Pearson correlation: **r = -0.724** (n=245, p < 10⁻⁴⁰).

**Correlation decomposition**: Between the 7 condition means,
r = −0.878. Within individual conditions, mean r = −0.582 (range:
−0.801 to +0.463). Six of seven conditions show strong within-level
correlation (r < −0.7); oracle+noise(0.5) is the exception (see §5).

### 4.2 Learnability threshold

| C_i threshold | N above | Success rate |
|--------------|---------|-------------|
| ≥ 0.3 | 187 | 68% |
| ≥ 0.4 | 124 | **88%** |
| ≥ 0.5 | 53 | **98%** |
| ≥ 0.6 | 18 | **100%** |
| < 0.3 | 58 | 22% |

C_i ≥ 0.5 yields 98% success (52/53). C_i ≥ 0.6 yields 100% (18/18).
The transition from failure to success occurs in the C_i = 0.3–0.5
range, with the sharpest inflection at 0.4.

Note: we report this as a "learnability threshold" rather than a
"sharp threshold" — the transition is a steep gradient, not a cliff.
The C_i ≥ 0.8 bucket contains only n=2, so we do not make claims
about that range.

### 4.3 Formal interaction test: capacity × perception

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
This is because actions affect the true physical state, not the
percept — a larger brain cannot help if the eyes do not convey where
the rock actually is.

## 5. The Blind Cat

Blakemore & Cooper (1970) raised kittens in restricted visual
environments. The kittens lost orientation selectivity for unseen
angles — visual cortex pathways that could not align with motor
experience were pruned.

Our simulation produces the computational equivalent. In the VAE(z)
condition (stochastic sampling, obj-012), the sampling noise destroys
spatial signal. The organism's perception carries no information
about the directional structure of the true state. No amount of
capacity can find the bridge: C_i ≈ 0 across all conditions. The
organism is computationally blind.

Even with deterministic μ (obj-013/014), VAE lat=8 caps C_i at ~0.29.
The perception is too degraded for the organism to determine which
direction to push the actual rock. It can see *something*, but what
it sees does not contain enough about the true physical state that
its actions will affect.

Oracle + noise σ=0.5 shows the same pattern: C_i drops to 0.196.
Noise on the true state itself (not just on the representation) is
equally destructive.

## 6. Discussion

### The asymmetry as the core contribution

The perception-action asymmetry is not merely a detail of our
simulation — it is the fundamental structure of embodied cognition.
Every biological organism faces it. The contribution of C_i is not
just "perception quality matters" — it is that C_i specifically
measures the organism's ability to bridge degraded perception to
direct physical action. The entire perception chain exists to give
the organism enough information to act correctly on a reality it
can never directly observe.

### Limitations

- Rock-push is 4D state, 2D action — a second, higher-dimensional
  task is needed to test threshold generalization
- C_i requires a computable optimal action (feasible in simulation,
  not in general real-world tasks) — a self-supervised proxy is needed
- The learnability threshold (C_i ≈ 0.4–0.5) may be task-specific
- oracle+noise(0.5) shows a within-level correlation reversal
  (r = +0.46) — at the noise floor, C_i becomes unreliable

### Future work

- **Second task** (8D+ state, ≥4D action) to validate threshold
  generalization — non-negotiable for main-track submission
- **Self-supervised C_i proxy** without oracle access — prediction
  error, contrastive alignment, or value gradient as estimators
- C_i dynamics during training (obj-015 shows slope r = −0.705)
- Cross-validation in vaural and CorticalNN

## Figures

1. `results/obj016_ci_at_scale.png` — At-scale (245 configs,
   7 seeds, random baseline, correlation decomposition)
2. `results/obj015_ci_dynamics.png` — C_i dynamics during training
   (slope r = −0.705)
3. `results/obj014_expanded_ci.png` — Expanded sweep (105 configs)
4. `results/obj013_coordination_quality.png` — Initial C_i discovery

## Status

- [x] Core simulation with asymmetric perception-action loop
- [x] C_i definition, measurement, and learnability threshold
- [x] At-scale validation: 245 configs, 7 seeds (r = −0.724)
- [x] Formal interaction test: F(1,241) = 34.2, p = 5×10⁻⁹
- [x] Random baseline and explicit success criterion
- [x] Correlation decomposition (between r=−0.878, within r=−0.582)
- [x] C_i dynamics: slope r = −0.705 predicts success (obj-015)
- [ ] Second task (8D+) — BLOCKER for main track
- [ ] Self-supervised C_i proxy
- [ ] Framing: rename to "sensorimotor alignment", lead with blind cat
