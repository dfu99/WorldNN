# Planning

## Current Priority: Foundation

Establish the theoretical framework and minimal simulation before
scaling up.

### Next Steps

1. **Formalize the information-theoretic model** — Write down the
   chain of lossy transformations mathematically. Define entropy at
   each stage. Derive (or conjecture) the relationship between
   cumulative channel capacity loss and required embedding size for
   1-bit reliable state change.

2. **Build minimal binary-state simulation** — One piece of matter
   (2 states), one channel, one environment VAE, one organism. Sweep
   over noise/compression parameters and embedding sizes to find the
   empirical curve.

3. **Choose tech stack** — Likely PyTorch for NNs and VAE, plus a
   lightweight simulation harness. Decide on experiment tracking.

4. **Scale to multi-object / multi-channel** — After the binary case
   is understood, add complexity incrementally.

## Open Questions

- What loss function / training signal does the organism use? Is it
  purely reward-based (RL) or does it also do unsupervised world-model
  learning?
- How does the Mealy machine's NN architecture relate to the
  "complexity" of the matter? Is matter complexity a free parameter?
- Should the environment VAE be pre-trained and frozen, or learned
  jointly?
- How to formalize "difficulty of producing a 1-bit change" — success
  rate? expected number of attempts? mutual information between action
  and outcome?
