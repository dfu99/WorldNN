# Cross-Discipline Hook: Fungal Hyphal Signaling ↔ Action-Space Asymmetry

## Choice and rationale

The PI's example hinted at fungal hyphal signal propagation as an analogue
for agent-to-agent context degradation. This memo develops that mapping
explicitly against three axes of WorldNN: network topology, information
engineering, and failure modes.

## The biological system

Mycorrhizal fungi (especially arbuscular mycorrhizal fungi and
ectomycorrhizal networks) form common mycorrhizal networks (CMNs) that
physically link root systems of neighboring plants. Signal propagation
through hyphae uses:

- *Ca²⁺ waves*: electrochemical signaling with cm-per-second propagation
  speeds.
- *Pressure pulses*: mechanical signals from cytoplasmic streaming.
- *Chemical gradients*: soluble metabolites (auxin, strigolactones).
- *Electrical potentials*: action-potential-like depolarizations observed
  in *Pleurotus* and *Schizophyllum*.

The network is topologically a directed acyclic graph *within a single
mycelium*, but across a CMN becomes a mesh with cycles and branch points.
Signal propagation is lossy: each junction reflects and attenuates; long
paths accumulate noise.

## Three structural analogies to WorldNN

### 1. Serial-chain information loss (matches obj-024 floor effect)

A hyphal segment is analogous to one link in the WorldNN perception chain.
Each segment contributes a multiplicative attenuation factor. The overall
signal that reaches a distant plant root is the product of segment
transmissions, exactly like `I(S; obs)` in WorldNN after emission →
environment → sensory filter.

**Prediction (testable in WorldNN):** if we simulate a serial chain with
*k* cascaded lossy channels, the achievable SA ceiling should fall as
*k* grows, matching the sensory_dim = 2 floor we observe. This gives
cross-domain support for the Data Processing Inequality framing we
already cite.

### 2. Single-point-of-failure at branch nodes

In a CMN mesh, "hub" hyphae connect many downstream nodes. Cutting a hub
produces disproportionate downstream loss. In WorldNN, the embedding
bottleneck (embedding_dim) is a hub: reducing it below a task-specific
threshold cascades to failure across all downstream policy signals.

**Prediction:** graph-theoretic centrality of the embedding (measured as
the fraction of downstream policy variance it mediates) should correlate
with the SA ceiling. We can compute this from activation variance
decomposition; it would be a new diagnostic complementing SA.

### 3. Action-space asymmetry ↔ directional hyphal growth

Hyphae grow directionally (apical extension toward resource gradients).
Growth is both a *sensing* mechanism (hyphal tips sample the
environment) and an *actuation* mechanism (they commit the mycelium to
one direction over others). The asymmetry is intrinsic: information
flows bidirectionally along a hypha, but growth (action) is unidirectional
and far more committal.

This mirrors WorldNN's core thesis: perception is lossy and returnable,
but action modifies the true state irreversibly. In the fungal case,
"undoing" hyphal extension is metabolically expensive; in WorldNN,
"undoing" a rock push requires new actions that have their own
perception-gated cost.

**Prediction:** the investment-asymmetry parameter (how much more it
costs to reverse an action than to observe one) should appear in any
reduced-form model of the SA ceiling. In fungi this corresponds to the
Gibbs-free-energy cost of hyphal retraction vs. extension.

## Shared information-engineering backbone

Both systems face the same meta-question: *given a noisy, lossy physical
channel that also physically constrains action, how much of the
environment can the agent model, and how does that modeling budget
trade off against action capacity?*

In fungi, the trade-off is metabolic: more sensing hyphae cost ATP that
could be spent on extension. In WorldNN, it is computational: more
embedding dimensions cost gradient-descent cycles that could be spent on
policy refinement.

Both systems evolve / train to exploit a *regime* in which the
perception-cost + action-cost sum is minimized per unit of accomplished
goal. SA (or an analogue in fungi) measures how well the agent has
landed in that regime.

## Testable prediction bridging the two

Biology has measured signal propagation failure rates in CMNs (Simard
1997, Gorzelak 2020, Babikova 2013). These predict a power-law decay of
signal intensity with network distance. WorldNN could simulate a
serial-chain extension of our current channel and test whether SA
decays with the same exponent. A matching exponent would be a striking
cross-domain validation of the information bound; mismatch would localize
the biological effect to metabolic (not purely informational) constraints.

## Why this helps the paper

Cited in a short Discussion paragraph (§7.7 or as a new §7.4a), this hook
gives:

1. A concrete, non-neuroscience biological analogue (distinguishing us
   from the usual free-energy-principle citations).
2. A falsifiable quantitative prediction (signal-decay exponent match).
3. A future-work direction (serial-chain extension) that doesn't require
   a new task family.

The analogy strengthens the "perception-action asymmetry is a general
principle of embodied information processing" framing.
