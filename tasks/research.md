# Research

## Theoretical Framing

The perception-action loop is a chain of noisy channels:

```
S  ──f_matter──▶  X  ──f_env──▶  Y  ──f_sense──▶  Z  ──f_embed──▶  E
(hidden state)  (channels)    (observed)       (sensed)          (embedding)
```

Each `f` is a lossy transformation. By the data processing inequality,
mutual information can only decrease:

```
I(S; E) ≤ I(S; Z) ≤ I(S; Y) ≤ I(S; X)
```

The organism must choose action `A` based on `E` to flip `S` (binary).
The question: what is the minimum `dim(E)` such that
`P(S' = target | A(E))` exceeds some threshold, as a function of the
information bottlenecks at each stage?

## Sensory-Motor Alignment as Projection Learning (PI intuition, 2026-03-17)

### The cat experiment and the blind-by-neglect phenomenon

A cat raised with a headset displaying random visual noise — uncorrelated
with its proprioceptive, vestibular, and motor experience — goes blind.
The visual cortex atrophies. Interpretation: the brain searches for an
alignment between visual input and the rest of the sensorimotor loop. When
none exists, the pathway is pruned.

### Formalization: Intuition 1 — Senses as projections, not data streams

Biological perception is not a data stream that gets "processed." It is a
set of *concurrent embeddings* that must be *aligned* into a unified state
representation. Formally:

Let the organism maintain a holistic state embedding:

$$\mathbf{h} = \bigoplus_i R_i(\mathbf{e}_i) \in \mathbb{R}^D$$

where $\mathbf{e}_i \in \mathbb{R}^{d_i}$ is the embedding from sensory
modality $i$ (vision, proprioception, vestibular, etc.) and
$R_i : \mathbb{R}^{d_i} \to \mathbb{R}^D$ is a learned alignment operator
(generalized rotation + scaling into the shared space).

The holistic state $\mathbf{h}$ is then projected onto downstream task
manifolds via motor projection matrices:

$$\mathbf{a} = W_m \, \mathbf{h}$$

**Learning = finding each $R_i$** such that the projection
$W_m R_i(\mathbf{e}_i)$ has high magnitude and correct direction. Define
the *coordination quality* for modality $i$:

$$\mathcal{C}_i = \frac{\mathbb{E}\left[\langle W_m R_i(\mathbf{e}_i), \, \mathbf{a}^* \rangle\right]}{\|\mathbf{a}^*\| \cdot \mathbb{E}[\|\mathbf{e}_i\|]}$$

where $\mathbf{a}^*$ is the optimal action. This measures what fraction of
modality $i$'s information survives into useful motor commands.

**The cat experiment, formalized:**

Let $\mathbf{e}_v$ = visual embedding, $\mathbf{e}_p$ = proprioceptive.
Normally, $I(\mathbf{e}_v; \mathbf{e}_p) > 0$ because what you see
correlates with where you are and what you're doing. The brain exploits
this mutual information to learn $R_v$.

With random visual input: $I(\mathbf{e}_v; \mathbf{e}_p) \approx 0$.
No rotation $R_v$ can produce nonzero $\mathcal{C}_v$ because the visual
embedding carries no information about motor-relevant state. The
optimization landscape for $R_v$ is flat — pure noise gradients.

**Neurotrophic constraint (critical period):**

The search for $R_i$ is metabolically bounded. Define a *viability window*
$\tau_i$ set by local neurotrophic factor concentration (BDNF, NGF, etc.).
The rule:

$$\text{If } \mathcal{C}_i(t) < \epsilon \text{ for all } t \in [0, \tau_i], \text{ then pathway } i \text{ undergoes apoptosis.}$$

This is a use-it-or-lose-it threshold with a biological clock. When
learning succeeds, activity-dependent BDNF release sustains the neurons.
When it fails, trophic support withdraws and pruning follows.

**Connection to WorldNN (strong):**

The organism architecture IS this framework:
- `sensory_dim` → embedding → action IS $\mathbf{e} \to R(\mathbf{e}) \to W_m \mathbf{h}$
- The embedding layer IS learning the alignment $R$
- obj-012 result: when VAE stochastic z adds noise, $I(\mathbf{e}_v; \text{state}) \approx 0$
  and the organism cannot learn ANY $R$ — identical to the blind cat
- The capacity effect (embed_dim) = degrees of freedom available for $R$
- VAE lat=4 destroying spatial info = visual noise headset (no recoverable alignment)

**Connection to vaural (strong):**

The Emitter learning to map sound tokens → actions through a fixed random
channel IS learning $R$: finding the rotation/scaling that aligns the
sound embedding with the unknown motor-to-acoustic transform. The
Receiver is a frozen downstream projection $W_m$. The Emitter searches
for $R$ such that $W_m R(\mathbf{e}_\text{sound})$ reconstructs the input.

### Formalization: Intuition 2 — Hierarchical context race condition

**Question:** Why can't LLMs use hierarchical small-context models instead
of one large-context model?

**Setup:** Partition a document into chunks $\{c_1, \ldots, c_n\}$.

- Lower tier: $L_\theta$ produces local embeddings $z_i = L_\theta(c_i)$
- Upper tier: $H_\phi$ produces global context $g = H_\phi(z_1, \ldots, z_n)$

**The race condition:** Ideally, each local model should condition on
global context:

$$z_i^* = L_\theta(c_i \mid g)$$

But $g$ depends on the $z_i$'s:

$$g^* = H_\phi(z_1^*, \ldots, z_n^*)$$

This is a **fixed-point equation**:

$$g^* = H_\phi\big(L_\theta(c_1 \mid g^*), \ldots, L_\theta(c_n \mid g^*)\big)$$

**Why iterative refinement fails for training:**

You could try EM-style iteration:
1. $z_i^{(0)} = L_\theta(c_i)$  (context-free)
2. $g^{(k)} = H_\phi(z_1^{(k)}, \ldots, z_n^{(k)})$
3. $z_i^{(k+1)} = L_\theta(c_i \mid g^{(k)})$

Problems:
- **Representation shift:** When $L_\theta$ updates to condition on $g$,
  its output distribution shifts, invalidating $H_\phi$'s learned
  composition function. This is a form of non-stationarity analogous to
  multi-agent instability.
- **Gradient truncation:** Backprop through the iteration requires
  differentiating through the full unrolled loop. BPTT through $K$
  iterations scales as $O(K)$ in memory and often diverges.
- **No convergence guarantee:** The composed map $F(g) = H_\phi(L_\theta(c_1|g), \ldots)$
  is a deep nonlinear function. Fixed-point iteration converges only if
  $F$ is a contraction, which is not guaranteed (and unlikely for
  expressive networks).

**Why Transformers solve this differently:**

Self-attention sidesteps the hierarchy entirely. Every token attends to
every other token — there is no "lower tier" that must summarize before
a "higher tier" can integrate. The $O(n^2)$ cost of full attention IS
the cost of avoiding the race condition. The large context window is not
a limitation — it is the *solution*.

The brain similarly uses recurrent, bidirectional processing rather than
strict hierarchy. Top-down attention modulates bottom-up processing in
the same network, simultaneously. There are no "tier 1 neurons" that
must finish before "tier 2 neurons" start — it's all concurrent, which
is why biological neural networks avoid the race condition too.

**Formal impossibility (sketch):**

For any hierarchical decomposition where lower-tier models are trained
with frozen upper-tier context (or vice versa), there exists a class of
documents where the lower-tier representations are provably suboptimal:

Consider a document where paragraph $c_i$ contains the word "bank." Its
embedding depends on whether the document is about finance or rivers —
information only available globally. If $L_\theta(c_i)$ must commit to
an embedding without global context, it must either:
(a) pick one interpretation (lossy, wrong 50% of the time), or
(b) maintain both (exponential blowup in ambiguous cases).

A flat-attention model resolves this in $O(1)$ by attending to
disambiguating context elsewhere in the document.

### The perception-action asymmetry (PI insight, 2026-03-20)

The perception-action loop in WorldNN is fundamentally *asymmetric*:

- **Perception**: true state → emission → channel → VAE → observation
  (each step lossy)
- **Action**: action → directly modifies true state (NO lossy chain)

The organism perceives a degraded version of reality but acts on the
real physics. The rock doesn't respond to the organism's *perception*
of the rock — it responds to the actual force applied to its actual
atoms. The action channel is clean.

C_i measures exactly this bridge: *given degraded perception of true
state, can the organism determine the correct action on true state?*

The environment (VAE) is not a camera — it is the *medium* through
which information travels (light, sound, etc.). The organism's hand
doesn't push the medium; it pushes the real object. But it must decide
where to push based only on what the medium conveyed.

This is deeper than "perception quality matters." The contribution is:
the organism must bridge degraded perception and direct physical action.
The entire perception chain exists to give the organism enough
information to act correctly on a reality it can never directly observe.

Analogy: performing surgery through a foggy window, but your hands are
inside the room. Vision is degraded; actions are precise. C_i measures
whether degraded vision gives enough to guide precise hands.

### Prior art and novelty assessment (2026-03-17)

**Intuition 1 overlaps with:**
- Blakemore & Cooper (1970) — stripe-reared kittens lose orientation
  selectivity. The real experiment, close to (but not identical to) the
  random-noise headset version described above.
- Friston's Free Energy Principle (2010) — perception-action as KL
  divergence minimization. The cat interpretation IS this framework.
- CCA / Deep CCA / CLIP — learning projections to align cross-modal
  representations. The R_i operators are what CCA/DCCA learns.
- Churchland lab / Vyas et al. (2020, PNAS) — motor cortex uses rotational
  dynamics to separate sensory from motor representations; learning aligns
  axes (84° → 67° measured).
- BDNF / critical period neuroscience (Rossi 1999, Capsoni 1996, Huang 1999)
  — monocular deprivation decreases BDNF in contralateral visual cortex.
  Activity-dependent trophic support is established.

**What may be novel:** The unified synthesis connecting developmental
neuroscience (critical periods + BDNF), multimodal alignment math (CCA),
and our specific experimental apparatus (WorldNN capacity + vaural emitter)
under one framework with a biological deadline. No single paper combines
all three. The cross-project bridge is genuine. But it's a synthesis, not
a new theorem.

**Intuition 2 overlaps with:**
- Deep Equilibrium Models (Bai et al., NeurIPS 2019) — solve fixed-point
  equations via implicit differentiation. They show it CAN work, partially
  contradicting the impossibility claim.
- Hierarchical context merging (Song et al. 2024), In-Context Former (2024),
  Stingy Context (2025) — people ARE doing hierarchical compression.
- EM algorithm — the iterative refinement IS EM.
- ELMo → BERT — the "bank" example is the textbook motivation for
  contextual embeddings.

**What may be novel:** The "race condition" metaphor; the O(n²) = cost of
avoiding hierarchy observation. But impossibility claim needs tightening
given DEQs. Must characterize WHICH function classes break.

**Path to novelty:** Experimentally measure C_i (coordination quality) in
WorldNN/vaural and relate it to probe R² values. That's a testable,
concrete contribution rather than a re-derivation of known math.

### Project fit assessment

**Intuition 1 (sensory-motor alignment):**

This is the *theoretical bridge* between WorldNN and vaural. Both projects
are experimental instantiations of the same framework:
- WorldNN: organism learns $R$ under information loss (VAE bottleneck)
- vaural: emitter learns $R$ through fixed random channel (ActionToSignal)

Recommendation: **Formalize in WorldNN** (it has the richer experimental
apparatus for testing), **cross-reference from vaural**. The alignment
framework unifies both projects' central questions: "how does an agent
learn to act effectively through a lossy, unknown channel?" WorldNN
approaches this via the capacity/information-loss axis; vaural approaches
it via the sensorimotor coupling axis. Same math, different experiments.

**Intuition 2 (hierarchical context race condition):**

This does NOT naturally fit either project. Neither WorldNN nor vaural
has the architecture or experimental setup to test hierarchical LLM
context compression. It is a standalone theoretical observation about
*information architecture in sequence models*.

Recommendation: **Standalone research note or short paper.** It could
live in a `theory/` project, or as an appendix to a broader paper on
information processing hierarchies. Do NOT force it into WorldNN or
vaural — it would be a distraction from their focused experimental
programs. If you want to connect it eventually, the bridge would be:
"biological neural networks avoid the race condition via recurrence;
our simulation (WorldNN/vaural) similarly uses end-to-end differentiable
pipelines rather than hierarchical training."

## Related Work to Investigate

- **Noah Goodman** — Bayesian world models, probabilistic programming
  (WebPPL, Pyro). The idea that world state = posterior over observable
  quantities.
- **Information bottleneck method** (Tishby et al.) — Directly relevant:
  optimal tradeoff between compression and prediction.
- **Active inference** (Friston) — Free energy minimization as a
  unifying principle for perception and action.
- **Rate-distortion theory** — Fundamental limits on lossy compression.
- **POMDPs** — Partially observable MDPs; the organism's problem is
  essentially a POMDP where the observation function is the full
  lossy chain.
- **Embodied cognition / sensorimotor contingency theory** — How
  organisms' action repertoires shape what they can perceive and learn.

## Key Quantities to Track in Experiments

- Mutual information at each stage: I(S;X), I(S;Y), I(S;Z), I(S;E)
- Embedding dimension vs. task success rate
- Channel noise variance vs. required embedding size
- Number of action-feedback cycles needed for reliable state change
