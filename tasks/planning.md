# Planning

## Target: NeurIPS 2026 (May 6, 2026 deadline)

**Paper draft**: `paper/draft.md`
**LaTeX**: `paper/neurips2026/main.tex`
**Figures**: `paper/neurips2026/figures/`

### Status: Ready for PI review before submission.

All experimental, analytical, and writing tasks are complete. The paper
is structurally finished with LaTeX conversion, camera-ready figures,
expanded discussion, conclusion, and oracle-free proxy estimation section.

### Pre-submission checklist

- [x] 4 PDF figures at 300 DPI in paper/neurips2026/figures/
- [x] ~561 lines LaTeX, estimated 8.5-9 pages (within NeurIPS 9-page limit)
- [x] 11 bibliography entries, all citations resolved
- [x] Anonymous author placeholder
- [x] Supplementary materials section with code URL
- [x] Placeholder neurips_2026.sty (replace with official before submission)
- [ ] Download official neurips_2026.sty from neurips.cc
- [ ] Compile LaTeX and verify rendering
- [ ] PI review and approval

### All completed work

- [x] SA metric (r=-0.724, 245 configs, 7 seeds)
- [x] Formal interaction test (F=34.2, p=5e-9)
- [x] Metric ablation (6 metrics, mag-weighted SA r=-0.893)
- [x] SA dynamics (slope r=-0.705)
- [x] SA transfer: physics (93-106%) + appearance (89-102%)
- [x] Reversal analysis (boundary artifact)
- [x] Framing overhaul (SA rename, blind cat, related work)
- [x] Academic writing revision (48 violations fixed)
- [x] Second task: 2-rock 6D (r=-0.728)
- [x] Multi-rock 3-rock analysis (r=-0.300, supplementary)
- [x] Main 6-panel figure + 2 supplementary figures
- [x] Numbers pass (threshold table corrected)
- [x] Abstract polished (~160 words)
- [x] Introduction with 5 numbered findings + outline
- [x] Discussion expanded (6 subsections)
- [x] Conclusion paragraph
- [x] LaTeX conversion (527 lines)
- [x] Camera-ready figures (300 DPI, PNG+PDF)
- [x] Transfer section (§5.6)
- [x] Pre-submission checklist
- [x] Oracle-free SA proxy (obj-022): action variance r=-0.82

## Active AFK goal (2026-04-23 03:07 — iter-3)

Deep research sweep COMPLETE. 4 parts across 8 tasks (T17-T24). Artifacts
at tasks/research/ (10 memos). Synthesis memo is sweep_memo_2026-04-23.md.

## Standing directives

- **halulujah priority (highest).** If `mc runpod check` shows any
  halulujah process running or queued, YIELD — checkpoint cleanly, kill
  own job, wait. Re-check before resuming. dippy-WAN is lowest priority.
- **Keep A40 productive** when halulujah is idle. Read
  `development/status/runpod-active.json` for address. Always gate via
  `mc runpod check/fits/await` before launching. Never hardcode.
- **WD_BLACK diff/mirror** at every milestone — `rsync -a --delete` per
  subdir (`results/`, `figures/`, `paper/`) into
  `/media/dan/WD_BLACK/claude/backups/WorldNN/<sub>/`. No dated snapshots.
  Scope: experimental outputs only, never code/caches. See memory.

### Currently SUBSCRIBED (event-driven, no polling)
- **obj-027 asymmetry-scaling** — killed at config 2/90 when halulujah
  started. Subscribed to pod via `mc runpod subscribe WorldNN "..."`
  (queue position 8 of 8 at last check: dippy-WAN, conformers, haro,
  RL-Arm, --help, FIND-SNP, difftoken, WorldNN). Will be woken via `mc
  send` on release. On wake: re-check `mc runpod fits 5` and either
  submit or resubscribe.
- Re-launch command: `mc runpod sync WorldNN && mc runpod exec "cd
  ~/projects/WorldNN && WORLDNN_DEVICE=cuda nohup python3
  experiments/obj027_asymmetry_scaling.py > /tmp/obj027.log 2>&1 &"`.
  Script writes checkpoint every 5 configs — config 2 results (seed=42
  and 123 at sensory=2/embed=16: SA=0.009, 0.027) were NOT persisted, so
  resume re-does them. Cost: ~1 min. No action needed beyond relaunch.
  Must `mc runpod release` when obj-027 finishes to wake next subscriber.

### Queued (ready to launch after obj-027)
- **obj-028 1D sensory-capacity** (75 configs on ContinuousMatter, 500 ep,
  batch=1024). Script written at experiments/obj028_1d_sensory_capacity.py.
  Estimated VRAM 2 GB; `mc runpod fits 2` before launch.

### Backlog
- **obj-029 PPO hyperparameter sweep** — lr ∈ {1e-4, 3e-4, 1e-3} × entropy_coef
  ∈ {0.001, 0.01, 0.05} × clip_eps ∈ {0.1, 0.2, 0.3} × 3 seeds on obj-024's
  peak cell (sensory=16, embed=16). 27×3 = 81 configs at batch=1024. Never
  done; could reveal whether obj-024 was optimally tuned. Est. 4 GB.
- **obj-030 Outcome Alignment** — requires adding OA logging to the training
  loop (current scripts don't save trajectories). Rerun 1 cell of obj-024
  with OA instrumentation. Responds to PI's "intent vs real" question.
  Est. 1 GB CPU/GPU.
- **obj-031 Sparse-action supervision** (Dreamer V4-inspired) — train with
  only N% of oracle-action pairs, measure SA at N ∈ {1%, 5%, 25%, 100%}.
  Tests whether our SA ceiling holds under weaker supervision. Est. 2 GB.

## Recently Completed

- [2026-04-23] AFK iter-3: Deep research sweep — 24 recent papers catalogued, 3 deep-reads, current-state review, 5 ranked experiments, fungal-hyphae cross-discipline memo
- [2026-04-22] Compliance sweep: PACE scrub, silent-GPU-grab fix, manuscript emdash/hedge cleanup
- [2026-04-21] obj-026: 2-rock sensory-capacity replicate floor effect (peak SA 0.098, Reviewer E risk HIGH)
- [2026-04-19] obj-025: AFK review-panel sprint — rate-distortion (r=0.975 MI vs peak SA), bootstrap CIs (Cohen's d=1.10 substitution), biological calibration, SA-as-free-energy, reviewer risk map
- [2026-04-19] obj-024: Sensory-capacity tradeoff (100 configs, peak SA=0.234 at sensory=16/embed=16, floor effect confirms DPI, substitution effect statistically significant with d=1.10)
- [2026-04-01] Oracle-free proxy experiment (obj-022): action variance r=-0.82 with distance
- [2026-03-31] Pre-submission checklist, LaTeX figures + discussion update
- [2026-03-31] Camera-ready figures at 300 DPI
- [2026-03-31] LaTeX conversion: paper/neurips2026/main.tex
- [2026-03-31] Abstract + introduction polished for NeurIPS
- [2026-03-31] §5.6 SA transfer (physics + appearance)
- [2026-03-31] Numbers pass (threshold table 22% → 7%)
- [2026-03-31] §5.5 updated with 2-rock as primary second task
