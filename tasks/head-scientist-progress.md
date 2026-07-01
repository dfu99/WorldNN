# Head Scientist Critique — Progress Tracker

> **Note (2026-06): PACE access was permanently removed.** PACE jobs referenced
> below are dead. Any remaining 8D/second-task compute goes to RunPod via
> `mc runpod`. Current live roadmap is in `tasks/planning.md` (theory angle,
> awaiting PI direction before further compute).

## Experiments (priority order from critique)

### 1. Formal interaction test ✅ DONE
- **obj-016 (245 configs, 7 seeds)**: F(1,241) = 34.2, p = 5.0×10⁻⁹
- Interaction β = -0.004 (capacity helps MORE with better perception)
- ΔR² from interaction = 0.077

### 2. Increase seeds to 5+ ✅ DONE (obj-016)
- Increased to 7 seeds per condition (280 total configs)
- Within-level mean r improved: -0.47 (3 seeds) → -0.58 (7 seeds)
- All 6 non-floor conditions show r between -0.72 and -0.80

### 3. Second task (8D) ⏳ IN PROGRESS (job 5453852)
- First attempt (obj-017 v1, 500ep, hidden=32): FAILED — all at baseline
- Diagnosed: insufficient episodes, hidden bottleneck, diluted reward
- Fixed: 1000ep, hidden=64, worst-rock focused reward
- Re-submitted to PACE RTX 6000 (job 5453852) — **PACE access removed 2026-06; this job is dead.** Re-run on RunPod (`mc runpod`) if the 8D second task is still needed, or drop per current planning.md.

### 4. SA dynamics during training ✅ DONE (obj-015)
- SA slope (first 200ep) predicts final success: r = -0.705
- Time-to-threshold: oracle 100ep, raw emission 133ep, VAE 267ep
- Paper §5.4 added with practical early-stopping implication

### 5. Self-supervised proxy for SA — NOT YET STARTED

## Rigor Gaps

### Fixed
- [x] Success criterion: dist < 0.511 (baseline - 2σ)
- [x] Random baseline: dist = 0.516 ± 0.003, SA = 0.003 ± 0.022
- [x] Seeds increased to 7
- [x] Formal interaction test with p-value
- [x] Correlation decomposition (between r=-0.878, within r=-0.582)
- [x] Metric ablation: 6 metrics, cosine best interaction (F=104.8),
      mag-weighted best overall (r=-0.893)
- [x] oracle_noise0.5 reversal: explained as boundary artifact
      (distance std=0.004, all configs fail equally)
- [x] SA dynamics: slope r=-0.705, time-to-threshold analysis

### Not Yet Fixed
- [ ] Only one task validated (8D re-run was pending on PACE; PACE removed 2026-06 — re-scope to RunPod via `mc runpod` or drop)
- [ ] SA is post-hoc (no self-supervised proxy estimator)

## Framing Fixes — ALL DONE
- [x] Renamed "coordination quality" → "sensorimotor alignment" (SA)
- [x] Blind cat hook leads abstract and §1
- [x] Related work §2: Pinto, DreamerV3, quasimetric RL, imitation learning
- [x] §4.2 distinguishes SA from policy cosine similarity
