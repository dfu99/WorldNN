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

- **Reward shaping matters.** Pure binary reward (±1 for target state)
  was too sparse. Adding a small bonus for flip probability when in
  wrong state (`0.1 * flip_prob * (1 - state)`) significantly helped
  early training.

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
