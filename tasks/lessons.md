# Lessons

## Training

- **Random frozen NN weights in matter produce unlearnable physics.**
  The first approach used a randomly initialized MLP (frozen) for matter
  transitions. The organism could never learn because the action→flip
  relationship was random noise. Fix: use explicit physics (action
  proximity to target pattern → flip probability) for interpretable,
  learnable dynamics.

- **REINFORCE needs proper Gaussian log-probs, not action-magnitude
  surrogates.** First attempt used `advantage * action.sum()` which
  has no policy gradient interpretation. Fix: sample from Normal(mean, std),
  compute log_prob, use standard REINFORCE: `-log_prob * advantage`.

- **REINFORCE fails with low-dimensional inputs; use PPO.** With env_lat=1
  (scalar z input), REINFORCE cannot learn (~5% success) despite the
  latent being 97% separable. PPO's clipped objective solves it (87%).
  The gradient variance of vanilla REINFORCE is too high when the input
  manifold is 1D. Always use PPO for low-dimensional perception spaces.

- **Reward shaping matters.** Pure binary reward (±1 for target state)
  was too sparse. Adding a small bonus for flip probability when in
  wrong state (`0.1 * flip_prob * (1 - state)`) significantly helped
  early training.

- **Stochastic resonance was seed variance, not a real effect.** The obj-005
  finding (noise=0.5 outperforms noise=0.01 at lat=1) did not replicate in
  obj-006 with 5 seeds and 13 noise levels. PPO at lat=1 is nearly flat
  (~0.82) across noise 0.01-0.5, then degrades. Always use ≥5 seeds before
  claiming non-monotonic effects.

- **PPO makes organism embedding dim almost irrelevant for binary tasks.**
  With REINFORCE, embedding_dim mattered (especially at high noise). With PPO,
  embed=1 and embed=16 perform nearly identically. The "min capacity" results
  from obj-002 were largely optimizer artifacts, not information-theoretic.

## Architecture

- **Environment VAE latent_dim=1 is NOT an information bottleneck — it's
  an RL learning bottleneck.** Deep analysis (obj-003) showed env_lat=1
  latent space is well-separated: 97.3% threshold accuracy, Cohen's d=3.83,
  overlap=0.06. The VAE *does* encode state into 1D. But the organism's
  REINFORCE policy fails to learn from a 1D input (~5% success). With
  env_lat≥2, the same policy architecture succeeds (~86%). The bottleneck
  is the RL optimization landscape, not information loss. Possible causes:
  gradient signal too noisy in 1D, policy network needs multi-dimensional
  input for stable learning, or the Gaussian policy parameterization is
  poorly suited to 1D latent inputs.

- **Organism embedding dim has diminishing returns past ~4.** For the
  binary state task with sufficient environmental information (lat≥2),
  even embedding_dim=1 achieves ~77%. The bottleneck is the environment,
  not the organism brain size — at least for this simple task.

## Infrastructure

- **Use `CUDA_VISIBLE_DEVICES=''` for CPU-only runs.** Avoids CUDA
  initialization overhead and errors when no GPU is available.

- **PACE RTX 6000 nodes enforce max 6 CPUs per GPU.** Requesting
  `--cpus-per-task=8` with 1 RTX 6000 fails with "Maximum CPU:GPU
  ratio of 6:1". Use `--cpus-per-task=6` or fewer.

- **Local scipy is broken (numpy 2.4 vs scipy 1.11).** Avoid `from scipy
  import stats` in analysis scripts. Use manual implementations for t-tests
  and other basic stats. PACE environment may differ — check there too.

- **PredictiveOrganism replaces world.organism after World init.** To use
  predictive processing, create a normal World then swap:
  `world.organism = PredictiveOrganism(sensory_dim=env_lat, ...)`. The
  forward pass is compatible but adds `forward_with_prediction()` method.

- **Fano's inequality gives trivially loose bounds for binary state tasks.**
  With H(S)=1 bit, even noisy channels preserve enough MI for the task.
  The real bottleneck is optimizer learnability (PPO vs REINFORCE), not
  information availability. Future theoretical work should model
  representation learnability, not just information content.

- **Predictive processing doesn't help simple tasks.** Adding next-state
  prediction as auxiliary loss (pred_coef from 0.01 to 1.0) had zero effect
  on binary state-flip or continuous positioning. The tasks are too simple
  for world-model learning to provide any benefit. Save predictive processing
  for multi-step, multi-object tasks.

- **Random action_transform kills directional control.** The environment's
  `propagate_action()` MLP (randomly initialized, never trained for action
  semantics) destroys directional information. For tasks requiring spatial
  control (like rock-push), pass actions directly to matter. The action
  transform only works for pattern-matching tasks (binary flip) where the
  organism finds an arbitrary target pattern.

- **Emission design determines VAE learnability.** If emissions only encode
  relative position, the VAE can't separate individual object positions.
  Emissions need to contain all state variables the organism needs. Use a
  fixed full-state projection rather than ad-hoc channel-specific encodings.

- **1-bit and 1D tasks cannot reveal capacity limits.** Both binary
  (state-flip) and continuous (1D position) tasks show no embedding dim
  effect. H(S) is too low for organism capacity to matter. Need multi-bit
  state, multi-object, or multi-step planning tasks to find where capacity
  limits actually bite.

- **3 seeds is insufficient for rock-push — use 5 minimum.** Initial oracle
  with 3 seeds suggested embed=8 optimal and embed=16 regressing. With 5 seeds,
  embed=32 is actually optimal (0.306, 5/5 success) and reliability increases
  monotonically. The high seed variance at embed=4-16 made small samples misleading.

- **Perception quality gates whether organism capacity matters at all.**
  Rock-push oracle baseline (direct state) shows clear embed dim effect
  (embed=8: 0.395 vs embed=2: 0.50). But with VAE in the loop, early data
  shows ~0.49 across all embed dims — the VAE destroys spatial info before
  the organism can use it. Implication: before studying capacity, ensure
  the perception pipeline preserves enough information for the task. Test
  with oracle first, then add perception layers incrementally.
