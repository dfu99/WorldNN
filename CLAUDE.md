# WorldNN

A simulation framework for reasoning about information loss in
perception-action loops. Physical objects are Mealy machines whose
state transitions are driven by a shared random seed processed through
neural networks. No matter is directly observable — all information
passes through lossy channels, environmental encoding, and
organism-specific sensorimotor filters.

## Core Architecture

```
Random Seed
    │
    ▼
┌─────────┐   classical    ┌─────────────┐   channels    ┌───────────┐
│  Matter  │──mechanics────▶│ Environment │──(light,etc)─▶│ Organism  │
│ (Mealy   │                │ (VAE over   │               │           │
│  machine │◀──action───────│  channels)  │◀──action──────│ Sensori-  │
│  + NN)   │   feedback     │             │   feedback    │ motor +   │
└─────────┘                 └─────────────┘               │ Embedding │
                                                          └───────────┘
```

### Layers of the model

1. **Matter** — Mealy state machines. A global random seed is processed
   by a per-object neural network to produce deterministic classical
   mechanics (forces, positions, emissions). State is hidden; only
   outputs (energy channels) are externally visible.

2. **Channels** — The "measurable" outputs of matter: light, sound,
   heat, chemical gradients, etc. These are the only carriers of
   information between matter and observers.

3. **Environment** — A VAE that encodes/decodes channel signals,
   controlling observability. Introduces fixed compression and noise.
   Represents medium effects (atmosphere, distance, occlusion).

4. **Organism / Actor** — Any neural substrate that:
   - Samples a subset of channels (sensorimotor complex) — lossy
   - Embeds those samples into an internal representation — lossy again,
     bounded by embedding size / neural substrate capacity
   - Produces actions that propagate back through the environment to
     affect matter state

### Central question

Given a piece of matter with a binary state space, what is the
relationship between:

- The entropy / lossiness at each compression step (matter → channels →
  environment → sensorimotor → embedding)
- The minimum embedding size of the organism required to reliably induce
  a 1-bit state change in the targeted matter

This is essentially: **how much internal model capacity does an agent
need to act effectively, as a function of cumulative information loss
in the perception-action loop?**

### Inspirations

- **Noah Goodman's Bayesian world model** — the state of everything
  expressed as Bayesian probabilities conditioned on other known
  quantities and qualities.
- **Mealy machines** — classical formalism for state-dependent I/O,
  used here because matter's output depends on both its current state
  and the input (random seed / actions).
- **VAEs** — natural fit for the environment layer: learned lossy
  compression with a structured latent space.

## Task Files

| File | Consult when... |
|------|-----------------|
| `tasks/planning.md` | Starting a session, choosing what to work on |
| `tasks/lessons.md` | Before changing any subsystem |
| `tasks/research.md` | Working on theory, math, experiment design |
| `tasks/backend.md` | Implementing simulation engine, NN architecture |

## Tech Stack

TBD — to be decided after initial prototyping direction is set.

## PACE Cluster SLURM Rules

When writing SLURM scripts for the PACE cluster:

- **Account**: Always use `-A gts-yke8`
- **A100**: `--gres=gpu:A100:N` and **must** add `-C A100-80GB` constraint
- **RTX 6000**: `--gres=gpu:RTX_6000:N` (note underscore). No constraint needed.
- **H100**: `--gres=gpu:H100:N`. No constraint needed.
- **H200**: `--gres=gpu:H200:N`. No constraint needed.
- **Modules**: Always `module load cuda` for GPU jobs
- **Mail**: `--mail-type=END,FAIL` / `--mail-user=daniel.fu@emory.edu`
- **Paths**: scratch at `~/scratch/`, project storage at `~/p-yke8-0/`

## PACE Cluster SLURM Rules

When writing SLURM scripts for the PACE cluster:

- **Account**: Always use `-A gts-yke8`
- **A100**: `--gres=gpu:A100:N` and **must** add `-C A100-80GB` constraint
- **RTX 6000**: `--gres=gpu:RTX_6000:N` (note underscore). No constraint needed.
- **H100**: `--gres=gpu:H100:N`. No constraint needed.
- **H200**: `--gres=gpu:H200:N`. No constraint needed.
- **Modules**: Always `module load cuda` for GPU jobs
- **Mail**: `--mail-type=END,FAIL` / `--mail-user=daniel.fu@emory.edu`
- **Paths**: scratch at `~/scratch/`, project storage at `~/p-yke8-0/`
