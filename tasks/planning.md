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

## Active AFK goal (2026-04-19 13:19 — refill)

Previous iteration (obj-025, 03:24) COMPLETE. New iteration focused on:
(a) Reviewer E residual risk — 2-rock sensory-capacity replicate on RunPod A4500
(b) Paper integration — fold obj-024/obj-025 results into draft with scope caveats
(c) Discriminating extension — longer-training test for substitution-vs-floor

RunPod A4500 now available for >10min jobs: `mc runpod sync WorldNN` / fetch.

## Next priority

T9-T15 complete. T16 pending obj-026 (2-rock sensory-capacity) data from
RunPod — still running at config 1/60 as of 13:55. When checkpoint has
30+ configs: run obj026_seed_robustness.py adapted for 2-rock, generate
1-rock vs 2-rock comparison figure, update claim_to_evidence.md with
Reviewer E status.

## Recently Completed

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
