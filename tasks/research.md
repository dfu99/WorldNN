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
