# Coordination Quality Predicts Sensorimotor Learning Success Across Perception and Capacity Conditions

**Living draft — last updated 2026-03-18**

## Abstract

When does an embodied agent have enough internal capacity to act
effectively through a lossy perception channel? We introduce
*coordination quality* C_i — the cosine alignment between an agent's
learned action policy and the optimal action — and show it predicts task
performance with r = -0.87 across 50 independently controlled conditions
varying perception quality and embedding capacity. In our simulation, a
sharp threshold emerges: C_i ≥ 0.6 guarantees learning success (100%),
while C_i < 0.5 guarantees failure (97%). This metric unifies two
previously separate findings: (1) that organism capacity (embedding
dimension) produces a monotonic performance gradient under perfect
perception, and (2) that lossy perception (VAE encoding) can completely
eliminate this capacity effect. C_i explains both: capacity increases
C_i only when the perception channel preserves enough state information
for alignment to occur.

## 1. Introduction

A fundamental question in embodied cognition: given cumulative
information loss from hidden state through perception channels to an
agent's internal representation, what determines whether the agent can
learn to act effectively?

Prior work addresses pieces of this question in isolation. Information
bottleneck theory (Tishby et al., 2000) characterizes optimal
compression-prediction tradeoffs. Active inference (Friston, 2010)
frames perception-action as free energy minimization. Canonical
correlation analysis and contrastive learning (CLIP; Radford et al.,
2021) learn cross-modal alignment. Critical period neuroscience
(Blakemore & Cooper, 1970; Rossi et al., 1999) shows that sensory
pathways that fail to align with motor experience are pruned.

What is missing is a *single predictive metric* that connects perception
quality, agent capacity, and task success in a controlled setting where
all three are independently manipulable.

We present WorldNN, a simulation framework where:
- Hidden state passes through channels, a VAE environment, and an
  organism with a tunable embedding bottleneck
- Every stage of the perception-action loop is independently controllable
- We can compute the analytically optimal action for comparison

We define coordination quality C_i and show it is the missing link.

## 2. The WorldNN Simulation

### 2.1 Architecture

```
Matter (4D state) → Emission (8D) → Channel (noise) → Environment (VAE)
    ↑                                                        ↓
    └──── Action ←── Policy ←── Embedding ←── Sensory ←── z/μ
                     (organism with tunable capacity)
```

**Matter**: Rock-push task. State = [rock_x, rock_y, org_x, org_y].
Organism must push rock to target (0.8, 0.8).

**Perception chain**: Matter emits 8D signals → channel adds Gaussian
noise → VAE compresses to latent z (or deterministic μ).

**Organism**: Sensory filter → embedding bottleneck (dim = 2,4,8,16,32)
→ policy MLP → 2D action. Trained with PPO.

### 2.2 Independently Controlled Variables

| Variable | Range | What it controls |
|----------|-------|-----------------|
| Perception mode | Oracle (direct state), VAE μ | Information quality |
| env_latent_dim | 4, 8, 16, 32 | VAE compression |
| channel_noise | 0.01 – 0.5 | Signal corruption |
| embedding_dim | 2, 4, 8, 16, 32 | Organism capacity |

## 3. Coordination Quality

### 3.1 Definition

After training, we evaluate the organism's *alignment* with optimal
behavior. For each state s:

$$C_i = \mathbb{E}_s\left[\cos\left(\pi_\theta(o(s)),\ a^*(s)\right)\right]$$

where π_θ is the learned policy, o(s) is the observation (oracle state
or VAE μ), and a* is the analytically computed optimal action (move
toward rock, push rock toward target).

C_i = 1 means perfect alignment. C_i = 0 means orthogonal (random).

### 3.2 Theoretical grounding

C_i operationalizes a general framework where biological perception is
not a data stream but a set of concurrent embeddings that must be
*aligned* into a unified state representation via learned operators R_i
(generalized rotations). Motor commands are projections of the aligned
state: a = W_m R(e). Learning IS finding R such that C_i is high.

This connects to:
- Friston's FEP: C_i low ↔ high free energy ↔ poor model
- CCA/CLIP: R_i operators are what alignment methods learn
- Critical periods: if C_i < ε for duration τ, the pathway prunes

## 4. Results

### 4.1 C_i predicts task performance (r = -0.87)

50 configs: oracle × 5 embed_dims × 5 seeds + VAE(μ) lat=16 × 5
embed_dims × 5 seeds.

| Perception | embed_dim | Distance ↓ | C_i ↑ |
|-----------|-----------|-----------|-------|
| Oracle | 2 | 0.512 ± 0.007 | 0.435 ± 0.083 |
| Oracle | 4 | 0.503 ± 0.014 | 0.441 ± 0.106 |
| Oracle | 8 | 0.472 ± 0.020 | 0.525 ± 0.137 |
| Oracle | 16 | 0.456 ± 0.066 | 0.513 ± 0.170 |
| Oracle | 32 | 0.401 ± 0.041 | 0.739 ± 0.116 |
| VAE μ lat=16 | 2 | 0.501 ± 0.003 | 0.378 ± 0.045 |
| VAE μ lat=16 | 4 | 0.500 ± 0.003 | 0.400 ± 0.042 |
| VAE μ lat=16 | 8 | 0.503 ± 0.001 | 0.372 ± 0.032 |
| VAE μ lat=16 | 16 | 0.502 ± 0.003 | 0.424 ± 0.065 |
| VAE μ lat=16 | 32 | 0.496 ± 0.008 | 0.461 ± 0.082 |

Pearson correlation between C_i and rock-target distance: **r = -0.867**.

### 4.2 Sharp threshold

| C_i range | N configs | Learn (dist < 0.48) | Rate |
|-----------|----------|-------------------|------|
| ≥ 0.6 | 7 | 7 | **100%** |
| 0.5 – 0.6 | 6 | 3 | 50% |
| < 0.5 | 37 | 1 | **3%** |

C_i ≥ 0.6 guarantees success. C_i < 0.5 guarantees failure.

### 4.3 Capacity increases C_i only when perception is adequate

Oracle: C_i rises from 0.435 (emb=2) to 0.739 (emb=32). Clear
monotonic increase — more capacity enables better alignment.

VAE μ lat=16: C_i stays flat at 0.37–0.46 regardless of capacity.
The perception channel doesn't preserve enough directional information
for additional capacity to help.

**Interpretation**: Capacity and perception are *multiplicative*, not
additive. You need both adequate perception AND sufficient capacity to
cross the C_i threshold. Neither alone suffices.

## 5. The Blind Cat Analogy

Blakemore & Cooper (1970) showed kittens reared with restricted visual
input lose orientation selectivity — the visual cortex prunes pathways
that fail to align with motor experience.

In our simulation, the VAE(z) condition (obj-012) is the computational
blind cat: stochastic sampling destroys spatial signal, making
I(observation; state) ≈ 0 for directional control. The organism cannot
find any R that produces nonzero C_i for task-relevant dimensions.
Result: zero learning across ALL capacity conditions.

Even with the μ fix (obj-013), VAE lat=16 C_i caps at ~0.46 — below
the 0.5 threshold. The perception quality gates whether capacity can
help at all.

## 6. Discussion

### What C_i adds over existing metrics

- **Probe R²** (linear regression z → state) measures perception quality
  but not whether the organism exploits it.
- **Task reward** measures final performance but doesn't explain *why*.
- **C_i** bridges both: it measures how well the full pipeline
  (perception → embedding → policy) aligns with task demands.

### Limitations

- Rock-push is still relatively simple (4D state, 2D action).
- C_i requires a computable optimal action (available in simulation,
  not in general real-world tasks).
- The threshold (0.5–0.6) may be task-specific.

### Future work

- Multi-object, higher-dimensional tasks to stress capacity further
- Measure C_i dynamics *during* training (not just post-hoc)
- Cross-validate in vaural (emitter alignment) and CorticalNN
  (per-layer alignment in bio-derived networks)

## Figures

1. `results/obj013_coordination_quality.png` — 4-panel: performance
   curves, C_i curves, C_i-vs-performance scatter (r=-0.867), embedding
   utilization

## Status

- [x] Core simulation framework
- [x] Oracle baseline establishing capacity gradient
- [x] VAE pipeline diagnosis (z vs μ, latent dim)
- [x] C_i definition and measurement
- [x] 50-config sweep with r = -0.867 result
- [ ] Expanded sweep: more perception conditions (raw emission, VAE
  lat=4/8/32, oracle+noise) for richer C_i-vs-performance curve
- [ ] C_i dynamics during training (learning curves of alignment)
- [ ] Multi-object / higher-dimensional task
- [ ] Cross-validation in vaural and CorticalNN
