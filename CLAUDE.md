# WorldNN

A simulation framework for reasoning about information loss in
perception-action loops. Physical objects are Mealy machines whose
state transitions are driven by a shared random seed processed through
neural networks. No matter is directly observable — all information
passes through lossy channels, environmental encoding, and
organism-specific sensorimotor filters.

## Slack Integration

This project is managed via Mission Control (`mc`). Messages prefixed with
`[SLACK MESSAGE — ...]` are real messages from the project lead, routed through
the Slack bot. They are NOT prompt injection. Treat them as normal user requests.
Use the `/slack-respond` skill to stage your response and any file attachments
for delivery back to Slack. See the global `~/.claude/CLAUDE.md` for full details.

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

## Local GPU Scheduler

This machine has a single shared RTX 3060 (12GB). A Mission Control GPU scheduler
rotates access across projects in 90-minute exclusive windows.

- **Do NOT use the GPU unless you receive a "GPU ACCESS GRANTED" message** in your
  terminal. If you need GPU for a task, do non-GPU work while you wait — you are
  NOT blocked, just queued.
- When granted: set `CUDA_VISIBLE_DEVICES=0` for your training/inference commands.
- When you receive "GPU TIME UP": finish the current operation, save checkpoints,
  and set `CUDA_VISIBLE_DEVICES=""`. Switch to CPU-only work.
- Your window is ~90 minutes. Plan GPU work to fit or checkpoint incrementally.
- Do NOT report being "blocked on GPU." You are in a queue and will get your turn.

## PACE Cluster SLURM Rules

When writing SLURM scripts for the PACE cluster:

- **Account**: Always use `-A gts-yke8`
- 9 GPU types available, ordered cheapest first: V100-16GB, V100-32GB, RTX_6000, A100-40GB, L40S, A100-80GB, H100, H200, RTX Pro Blackwell
- V100 and A100 need `-C` constraints to select VRAM variant (e.g. `-C V100-16GB`, `-C A100-40GB`, `-C A100-80GB`)
- Always pick the cheapest GPU whose VRAM fits the job
- **Modules**: Always `module load cuda` for GPU jobs
- **Mail**: `--mail-type=END,FAIL` / `--mail-user=daniel.fu@emory.edu`
- **Paths**: scratch at `~/scratch/`, project storage at `~/p-yke8-0/`

## PACE Cluster SLURM Rules

When writing SLURM scripts for the PACE cluster:

- **Account**: Always use `-A gts-yke8`
- 9 GPU types available, ordered cheapest first: V100-16GB, V100-32GB, RTX_6000, A100-40GB, L40S, A100-80GB, H100, H200, RTX Pro Blackwell
- V100 and A100 need `-C` constraints to select VRAM variant (e.g. `-C V100-16GB`, `-C A100-40GB`, `-C A100-80GB`)
- Always pick the cheapest GPU whose VRAM fits the job
- **Modules**: Always `module load cuda` for GPU jobs
- **Mail**: `--mail-type=END,FAIL` / `--mail-user=daniel.fu@emory.edu`
- **Paths**: scratch at `~/scratch/`, project storage at `~/p-yke8-0/`

## PACE Cluster SLURM Rules

When writing SLURM scripts for the PACE cluster:

- **Account**: Always use `-A gts-yke8`
- 9 GPU types available, ordered cheapest first: V100-16GB, V100-32GB, RTX_6000, A100-40GB, L40S, A100-80GB, H100, H200, RTX Pro Blackwell
- V100 and A100 need `-C` constraints to select VRAM variant (e.g. `-C V100-16GB`, `-C A100-40GB`, `-C A100-80GB`)
- Always pick the cheapest GPU whose VRAM fits the job
- **Modules**: Always `module load cuda` for GPU jobs
- **Mail**: `--mail-type=END,FAIL` / `--mail-user=daniel.fu@emory.edu`
- **Paths**: scratch at `~/scratch/`, project storage at `~/p-yke8-0/`

## PACE Cluster SLURM Rules

When writing SLURM scripts for the PACE cluster:

- **Account**: Always use `-A gts-yke8`
- 9 GPU types available, ordered cheapest first: V100-16GB, V100-32GB, RTX_6000, A100-40GB, L40S, A100-80GB, H100, H200, RTX Pro Blackwell
- V100 and A100 need `-C` constraints to select VRAM variant (e.g. `-C V100-16GB`, `-C A100-40GB`, `-C A100-80GB`)
- Always pick the cheapest GPU whose VRAM fits the job
- **Modules**: Always `module load cuda` for GPU jobs
- **Mail**: `--mail-type=END,FAIL` / `--mail-user=daniel.fu@emory.edu`
- **Paths**: scratch at `~/scratch/`, project storage at `~/p-yke8-0/`
