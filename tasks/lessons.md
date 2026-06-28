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

- **Local scipy is broken (numpy 2.4 vs scipy 1.11).** Avoid `from scipy
  import stats` in analysis scripts. Use manual implementations for
  t-tests and other basic stats.

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
  Perception ladder (obj-011) quantified this precisely. Oracle shows clear
  capacity effect (emb=8→32 gap: +0.098). Raw 8D emission still shows it
  (+0.015). VAE lat=16 preserves it (+0.033). But VAE lat=4 kills it
  (+0.007, essentially zero). Root cause: VAE lat=4 only preserves 23% of
  state variance (rock_y R²=0.044). lat=16 preserves 82%. Always probe VAE
  latent quality (linear R²) before running organism training. If R²<0.5
  for any critical state variable, increase latent dim.

- **C_i (coordination quality) predicts learning with r=-0.735 across 7 conditions.**
  Cosine similarity between learned action and optimal action is the single best
  predictor of task performance. Expanded from r=-0.867 (2 conditions, 50 configs)
  to r=-0.735 (7 conditions, 105 configs). Threshold: C_i≥0.8 → 100% success,
  C_i<0.5 → 85% failure. Capacity and perception are multiplicative. Raw emission
  (no VAE, 8D) outperforms all VAE conditions — the VAE adds compression loss
  without benefit when the organism could learn from raw signals directly.

- **Standard pipeline uses stochastic z — kills learning for spatial tasks.**
  The world.step() → organism loop feeds z (sampled from VAE posterior) to the
  organism. For rock-push, z's sampling noise destroys the spatial signal.
  obj-012 showed VAE lat=16 at 0.500 (random) through standard pipeline, but
  obj-011's perception ladder got 0.460 using deterministic mu. Always use mu
  (encoder mean) for tasks requiring precise spatial information. Stochastic z
  may only work for coarse classification tasks (like binary state-flip).

- **Multi-rock (8D) needs 1000+ episodes, hidden≥64, focused reward shaping.**
  obj-017 failed (all conditions at baseline) because: (1) 500 episodes
  insufficient for 3-object sequential task, (2) hidden_size=32 bottleneck
  for 8D→64D, (3) reward diluted across 3 rocks (each contributes only 1/3).
  Fix: increase episodes, hidden size, and focus reward on worst-performing
  rock rather than averaging. CPU diagnostic confirmed even single-rock doesn't
  learn reliably at batch=64 — need batch≥256 for PPO stability.

- **torch.norm(x, -1) computes L^(-1) norm, not dim=-1.** Use torch.norm(x, dim=-1)
  for Euclidean distance along last dimension. The positional argument is the
  norm ORDER (p), not the dimension.

- **VAE env_latent_dim must be ≥ state dimensionality for spatial tasks.**
  For 4D state projected through 8D emissions, lat=4 is catastrophically
  too small. lat=8 is marginal (R²=0.453). lat=16 is adequate (R²=0.817).
  Rule of thumb: lat ≥ 2× effective state dim for the task.

## Analysis

- **KSG estimator needs normalized inputs for stable MI in >4D.** Raw
  KSG on unnormalized high-dim data (e.g., 4D state vs 16D emission)
  gives negative/clamped-zero outputs because the Chebyshev distance
  blows up with scale mismatch. Fix: z-score both x and y before KSG.
  Even then KSG struggles past ~8D. For rate-distortion analyses, pair
  KSG with a linear-probe R² → Gaussian-MI lower bound (far more
  stable and interpretable for reviewers).

- **Dynamic-range matters for correlation claims.** obj-024's per-config
  SA vs dist correlation was only r=-0.245, far below obj-016's r=-0.724.
  Reason: obj-024 had most configs near zero SA because sensory_dim ≤ 4
  cannot learn. Always report the spread of the independent variable
  when making correlation claims — a compressed range tanks r even when
  the underlying mechanism is correct.

- **Small absolute effect + very low noise = large Cohen's d.** obj-024
  substitution (+0.040 mean diff) looked "weak" in raw SA, but with the
  poor-input condition's std=0.011 vs rich-min's std=0.050, Cohen's d
  came out to +1.10 (large). Bootstrap CIs exclude zero. Always compute
  effect size, not just raw magnitude, when claiming an effect is "weak."

- **Linear-probe R² is a per-sensory-dim property, not per-config.**
  When grouping configs by design variables for R² comparisons, R² is
  shared across all configs with the same sensory_dim. Partial correlation
  with SA controls for this correctly; raw correlation does not.

- **KSG MI estimator under-reports on 2-D Gaussian by 30-100%.** Audit
  2026-05-05 D2 verified `src/worldnn/utils.py::estimate_mi_ksg` returns
  the clamped-zero output at rho≤0.6 (truth ≤0.22 nats) and recovers
  ~66% at rho=0.9. The `max(0.0, mi)` clamp masks the negative raw
  output that signals failed estimation. For paper-grade MI on
  high-dimensional y, use linear-probe Gaussian-MI (see
  `experiments/obj025_mi_vs_sensory.py`) instead. Behavior pinned by
  `TestKSGEstimator`.

- **Cross-check backfill numbers against JSONs every time.** Audit D6
  found two errors I made when backfilling obj-019/020/021 from commit
  messages alone: claimed retention range 89-102% (actual 89-110%);
  claimed oracle SA mean 0.55 (actual 0.588). Lesson: paper-cite
  prose is rounded for readability; the JSONs are the source of truth.

- **Single source of truth for the manuscript.** Audit D3 found
  `paper/draft.md` and `paper/neurips2026/main.tex` had drifted on
  multiple sections. Pick one (we picked `main.tex` for the submission
  target) and deprecate the other with an explicit DEPRECATED preamble.

- **Commit-message claims are not state.** A commit "obj-027: launched
  on RunPod" was true at commit-time but became misleading after
  preemption. Always issue a counter-commit when a launched job fails
  ("yield to halulujah; kill obj-027" pattern).

- **Bibliography orphan check is cheap and worth doing.** `radford2021learning`
  was bibitem-defined but never cited in body — caught by `grep \\\\cite`
  vs `grep \\\\bibitem`. Run before every submission.

## Writing

- **Woven citations cost body budget.** Adding 30 NeurIPS-style citations
  with inline prose weave (one or two sentences each into §3.1, §5.7,
  Related Work, §7) cost ~1-2pp of body length — pushed an 8.5-9pp draft
  to 10-11pp, breaching the NeurIPS 9-page limit. The bibliography itself
  is free (refs/appendix are unlimited at NeurIPS); the cost lives in the
  prose hooks needed to motivate each cite. If a paper is budget-bound,
  expand the bibliography but defer weaving to the appendix's Extended
  Related Work, then trim back the body integrations to one-clause
  parentheticals. Lesson surfaced during obj-037 (commit 050dd18).

## Process

- **Autonomy is the contract.** PI codifies evals and constraints; the
  agent explores solutions and executes. No approval-seeking on reversible
  in-tree actions (edits, experiments, figures, commits). No multiple-choice
  menus offered to the PI when one option is clearly better. Hard rule
  lives in `~/.claude/CLAUDE.md` under "Autonomy (no approval-seeking)".
  Confirmation is required only for: destructive ops outside the working
  tree (force-push, drop DB, send Slack/email externally, deploy), real
  money, shared infra. Everything inside the repo: act.

- **Corrections belong in code, not memory.** When the PI pushes back,
  the fix goes in a script/skill/hook/lessons.md — not just an internal
  resolution. Memory is the weakest enforcement; codify hard constraints
  in code or hooks so future sessions inherit them automatically. Example:
  the writing-style pre-commit hook at `.githooks/pre-commit` was added
  for D20 emdash-regression after one PI compliance sweep; it now
  prevents the *next* regression silently.

## Infrastructure (continued)

- **RunPod A4500 has no scheduler — gate every GPU launch on VRAM.**
  Shared A4500 node at 213.173.102.219:12447 is unmanaged. Multiple users
  can OOM each other if they launch without checking. Before any GPU
  job: estimate peak VRAM (params × 4B + optimizer ≈ 2× params + batch
  activations), then `mc runpod check` (JSON) or `mc runpod fits <gb>`
  (exit 0 if safe). Use `mc runpod await <gb> --timeout N` to block
  until free. Small PPO jobs (~2K params, batch=256) use ~0.3 GB; VAE
  pretraining (batch=512) can hit 2-4 GB.

## Research framing (added 2026-06-25)

- **Correlation-on-a-toy reads as trivial; reframe to a *limit*.** The PI
  rejected the SA result as "dressed up trivialities" — a hand-designed
  alignment score correlating with learning speed, on one toy, with small
  effects, confirming an obvious intuition. The fix is not more polish: it
  is to convert the central claim from a measured correlation into a
  *provable bound* (here, a min-cut over the perception-action loop, with
  an architecture-independence impossibility result). Reviewers forgive a
  toy that *demonstrates* a proven limit; they do not forgive a toy whose
  only output is a correlation. When a result feels incremental, ask:
  "what is the theorem this is empirical evidence *for*?"

- **Test a reframe on existing data before spending compute.** Before
  committing to the min-cut reframe I correlated |SA| against
  min(sensory, embedding) on the already-collected 100-cell grid — zero
  new compute, ~20 min. It returned r=0.67 vs 0.47/0.39 single-axis (the
  min-cut fingerprint), which justified the pivot *and* revealed the toy
  sits near its controllability floor (memory rarely binds). A cheap
  re-analysis of existing results can both validate a new framing and
  diagnose the next experiment's design.

- **Do not claim novelty without an adversarial literature check.** Twice
  now the PI has called the central result trivial ("dressed up
  trivialities", then "sleight of hand"). The obj-039 bound was sold as
  "nobody has that"; in fact it is a Fano + data-processing special case of
  the Information Bottleneck, information-theoretic control (Tatikonda–
  Mitter), empowerment (Klyubin), and bounded-rationality rate-distortion
  (Genewein–Braun). Before writing "novel" or "nobody has", name the three
  closest prior works and state explicitly what is left after subtracting
  them. If nothing is left, say so.
- **A fancier formalism is not a contribution; relabeling is not novelty.**
  Calling a constraint I(·;·) ≤ C "internal memory capacity" instead of
  "channel capacity" adds nothing — data-processing treats every edge
  identically. Reframing a trivial correlation into a trivial bound is the
  same defect with more LaTeX. The test for a real contribution: is there a
  *measurement or claim that would be false under the prior framework*? If
  the new framing makes no different prediction, it is cosmetic.
