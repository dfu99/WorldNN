# Experiment Plan — Follow-on Experiments (2026-04-23)

Compute snapshot at plan time: `mc runpod check` returned 44.4 GB free of
45 GB (0% util). Ample headroom for the experiments below.

Five proposed experiments, ranked by expected marginal contribution.

---

## E1. Asymmetry-scaling curve at larger capacity (HIGHEST MARGINAL VALUE)

### Hypothesis
The substitution effect (sensory-richness ↔ embedding capacity) observed
on obj-024 at embed_dim ∈ {2,4,8,16,32} either persists, flattens, or
saturates at larger capacity. Current data cannot distinguish.

### Minimal design
- Task: 1-rock rock-push (same as obj-024)
- sensory_dim ∈ {2, 8, 16} (drop 4 to save compute; keep extremes)
- embed_dim ∈ {16, 32, 64, 128, 256} (extends obj-024's 32 cap)
- n_seeds = 8 per cell (power-analysis target: n=13 for d=1.10 at α=0.05,
  but n=8 gives acceptable 0.75 power)
- 800 episodes per config
- Total configs: 3 × 5 × 8 = 120

### Compute
- Per-config: ~1.5 GB VRAM (PPO at embed=256, batch=256)
- Peak concurrent: if we run 1-at-a-time, ~1.5 GB
- Verified fits: `mc runpod fits 3` → exit 0 (3 GB gives headroom)
- Estimated wall-clock: ~3-4 hours on A4500 (config ≈ 90 s median)

### Falsifier
If peak SA at sensory=16 continues monotonically upward past embed=128,
the floor interpretation ("capacity saturates around 16-32") is falsified;
we would need to revise the info-bound claim's implicit capacity scale.

If peak SA at sensory=2 rises above 0.2 at embed=256, the rate-distortion
claim (obj-025 T3 r=0.975) would be inconsistent — low I(S;obs) should
not yield high SA regardless of capacity.

### Marginal contribution
Settles the "capacity saturates" assumption in §5.7 and §7.5. Direct
response to Reviewer A's "does more capacity eventually win?" line of
critique. At 120 configs, cheapest meaningful extension of obj-024.

---

## E2. Outcome Alignment (OA) metric validation

### Hypothesis
The PI-proposed "intent vs real" metric — cos(push_direction,
Δrock) — is correlated with SA on the obj-024 grid and independently
predicts task performance. If yes, it becomes a *cheap oracle-free*
alternative to the action-variance proxy from obj-022.

### Minimal design
- Use existing obj-024 checkpoint (100 configs)
- Re-run each trained organism in eval mode, log per-step (action,
  rock_state_t, rock_state_t+1)
- Compute OA = E[cos(action_xy, Δrock_xy)] for contact steps
- Compare OA vs SA correlation, OA vs dist correlation
- Generate 2-panel figure

### Compute
- CPU-only, <1 hour. No RunPod needed.
- `mc runpod fits 0` trivially succeeds.

### Falsifier
If OA ≈ 0 or |corr(OA, dist)| < 0.3, it's not a useful proxy. If
corr(OA, SA) > 0.9, it's redundant with SA (not a useful addition).

### Marginal contribution
Responds directly to PI's "intent vs real" framing. Cheap. Could replace
or complement the action-variance proxy in §5.8.

---

## E3. Action-space causal intervention

### Hypothesis
SA captures *structural* alignment that should degrade gracefully when
the action-space is rotated. If we apply a fixed orthogonal transform R
to the organism's action output at eval time (a_out = R · a_raw), SA
should drop by exactly the angle of R, not collapse to random.

### Minimal design
- Re-use obj-024's trained organisms (15 of them: best + worst cells)
- For each: evaluate under R_θ ∈ {rotation 30°, 60°, 90°, reflection}
- Measure SA under each perturbation
- Expect: SA_rotated ≈ cos(θ) × SA_original

### Compute
- CPU-only eval, <30 minutes. No RunPod.
- 15 × 4 = 60 SA evaluations.

### Falsifier
If SA under 30° rotation already collapses to noise (|SA_rotated|<0.05),
the metric is coordinate-frame-specific rather than structural. Would
weaken the "structural alignment" framing in §4.3.

### Marginal contribution
Direct test of the "structural vs. coordinate-frame-locked" claim in §4.3.
Cheap, high-information.

---

## E4. SA slope early-stop validation (cross-task)

### Hypothesis
obj-015 established SA slope < 0.1 per 100 ep at ep=200 predicts final
failure on 1-rock. Does the same threshold work on (a) 2-rock, (b) the
1-rock-with-randomized-target variant mentioned as Option (c) in the
Reviewer E mitigation discussion?

### Minimal design
- Re-use obj-026 2-rock data (60 configs, already has SA trajectories?)
- If not: re-evaluate organism checkpoints to extract SA at ep={100,200,500,800}
- Test: does SA slope at ep=200 predict final_dist on 2-rock?

### Compute
- If trajectories already saved: CPU analysis only.
- If not: re-train or re-evaluate. Estimated 2 GB × 60 configs = ~2 hours
  on A4500.
- `mc runpod fits 3` → exit 0.

### Falsifier
If the slope-threshold does NOT predict failure on 2-rock (i.e., some
configs with low slope still converge, or some with high slope still
fail), the diagnostic is 1-rock-specific. Would be honest disclosure.

### Marginal contribution
Validates the practical "SA as training diagnostic" claim in §7.3. If it
holds, we have a deployable rule-of-thumb. If it breaks, we disclose the
scope.

---

## E5. Multi-task breadth — 1D continuous positioning sensory-capacity

### Hypothesis
The sensory-capacity substitution pattern observed on 1-rock manipulation
holds on a *qualitatively different* task family (1D continuous
positioning, not push-based). Address Reviewer E task-similarity concern
by showing the pattern on non-rock-push task.

### Minimal design
- Task: ContinuousMatter (already in matter.py) — 1D position control
- sensory_dim ∈ {1, 2, 4} (emission_dim is 4 for ContinuousMatter)
- embed_dim ∈ {2, 4, 8, 16, 32}
- n_seeds = 5 per cell
- 500 episodes per config
- Total: 3 × 5 × 5 = 75 configs

### Compute
- Per-config ~0.2 GB VRAM (smaller task).
- Wall-clock: ~2 hours on A4500.
- `mc runpod fits 1` → exit 0.

### Falsifier
If the monotonic sensory-richness ceiling does NOT appear on 1D
positioning, the substitution pattern is manipulation-specific. Would
narrow the info-bound claim.

### Marginal contribution
Strongest candidate for moving Reviewer E risk from High to Medium. A
qualitatively different task, not just a scale-up.

---

## Ranked recommendation

1. **E1** (asymmetry-scaling) — addresses Reviewer A directly, closes
   the capacity-ceiling gap in obj-024.
2. **E5** (multi-task breadth) — addresses Reviewer E, small compute.
3. **E2** (OA metric) — cheap, responds to PI's direct ask.
4. **E3** (action-space intervention) — cheap, shores up §4.3 framing.
5. **E4** (SA-slope cross-task) — useful but incremental.

E2 and E3 can run on CPU tonight. E1 and E5 can be chained on RunPod
with memory gating. E4 depends on whether we have obj-026 per-episode
SA trajectories saved.
