# Planning

## Current Priority: Perturbation Study Analysis

First milestone is complete — the framework is built, the demo works,
and the perturbation study is running. Next focus is analyzing results
and refining the experiment.

### Active

1. **Analyze perturbation study results** — 75 configs sweeping
   (noise, env_latent, embedding). Early results show env_latent_dim=1
   is a hard bottleneck (~5% success). Need full heatmaps, MI chains,
   and min-capacity curves.

2. **Investigate env_lat=1 failure mode** — Is the VAE under-trained,
   or is 1 latent dim fundamentally insufficient for binary state
   discrimination through the channel?

### Next Steps

3. **Add predictive processing** — Organism predicts next observation
   and uses prediction error as additional learning signal.

4. **Continuous state spaces** — Extend matter beyond binary to
   continuous state (e.g., rock position), requiring finer control.

5. **Rock-pushing scenario** — Organism infers rock position via light,
   applies force, senses gravitational/friction feedback. Tests
   multi-channel perception + force-based action.

6. **NN-based matter** — Replace explicit physics with learned Mealy
   machine for more complex matter dynamics.

7. **Theoretical bounds** — Derive analytical min embedding dim as
   function of channel capacity and environment compression.

## Open Questions

- Why does env_latent_dim=1 fail completely? Is it the VAE training
  or fundamental information theory?
- How does the number of perception-action cycles affect success?
  (Currently fixed at 10 steps per episode.)
- Should the organism also do unsupervised world-model learning
  alongside RL?

## Recently Completed

- [2026-03-10] Project scaffolding: pyproject.toml, src/, tests/, experiments/
- [2026-03-10] Core framework: Matter, Channel, EnvironmentVAE, Organism, World
- [2026-03-10] Two-phase training: VAE pre-training + REINFORCE with Gaussian policy
- [2026-03-10] Paramecium chemotaxis demo — 85.6% success rate
- [2026-03-10] Architecture diagram with ML block details
- [2026-03-10] 14 unit tests passing
- [2026-03-10] Perturbation study launched (75 configurations)
- [2026-03-10] README with overview, architecture, results, future directions
