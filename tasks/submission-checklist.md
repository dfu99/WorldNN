# NeurIPS 2026 Submission — Run of Show

For the PI. The audit (`tasks/audit-2026-05-05.md`) has done everything
agent-doable. This file is the linear sequence of remaining steps when
the official style file lands.

## Step 1 — Drop the official `.sty` into the repo (PI, ~5 min)

Source options (any one):

- `https://neurips.cc` Author Information page (login required)
- Overleaf NeurIPS 2026 template (`bjdwqfdkyftc`) — copy
  `neurips_2026.sty` from there
- A colleague's existing checkout

Target path: `paper/neurips2026/neurips_2026.sty` (overwrites the
placeholder stub currently in the repo).

## Step 2 — Recompile (agent-doable; ~1 min if `.sty` is in place)

```
cd paper/neurips2026
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Verify:
- `main.pdf` is generated.
- Page count: body should be ≤9 pages, references unlimited. With
  the placeholder stub the body fits in 9 pages and refs occupy pages
  9-11 (total 11). The official style may shift typography; if body
  overflows page 9, the §7.5 SA-vs-recon paragraph is the most
  trimmable (its CI/permutation sentence could collapse to one
  semicolon-separated clause).
- All `\ref{}` resolve (no `??` in the PDF).

## Step 3 — Final visual scan (PI, ~10 min)

Recommended skim order:

1. **Abstract.** Now cites the headline ΔR²=+0.374 finding (D29).
2. **§5.7 Information-theoretic bound.** Figure 5
   (`fig5_infobound.pdf`) is the rate-distortion curve, $r=0.975$. Read
   the table.
3. **§7.5 Limitations.** Now contains:
   - Standard limitations paragraph (3-rock floor, oracle+noise reversal)
   - "Noise model sensitivity" paragraph (D18)
   - "SA versus reconstruction-loss" paragraph (D16/D25/D27 — the
     headline)
   - "Cross-task generalization" paragraph (D24 — Reviewer E
     forward-prediction)
4. **References.** 14 entries; verify all are in expected venues.

The **rebuttal-letter draft** (`tasks/rebuttal_letter_draft.md`) is the
single document that summarizes how every reviewer concern is
addressed. Read it if you want the audit's view of the paper from a
reviewer's perspective without scrolling 1100 lines of audit.md.

## Step 4 — Submit to OpenReview (PI)

Create OpenReview submission, upload `main.pdf`, supplementary
materials (the repo has a permanent code URL in the supplementary
section), confirm anonymous-author placeholder is in place.

## Recovery if something goes wrong

| Symptom | Most likely cause | Fix |
|----|----|----|
| `main.pdf` body overflows page 9 | Official `.sty` has tighter line-height than placeholder | Trim the §7.5 SA-vs-recon CI sentence |
| `??` in compiled PDF | Missing `\label{}` or stale `.aux` | `rm main.aux` and recompile fresh |
| Bibliography keys not found | Typo in `\citep{}` | Check against `\bibitem` definitions |
| Figure path error | Filename mismatch | All figures live in `paper/neurips2026/figures/`; verify each `\includegraphics` path |

## Snapshot of what the audit produced

For the record. After 16 audit passes the agent shipped:

- 9 paper-text edits (abstract, §2, §3.2, §5.4, §5.7×3, §7.4, §7.5×3,
  §Supp Materials)
- 5 paper figures (fig1-fig5, plus fig6 in supplementary)
- 17 new tests (46/46 passing, 12/12 surfaces covered)
- 1 pre-commit hook (`.githooks/pre-commit`)
- 3 new objectives.yaml entries (obj-027 audit, obj-028 Dreamer h2h,
  obj-029 signal-dep noise)
- 1 rebuttal-letter draft
- 1 navigation TOC for the audit document itself

The single residual blocker has been the official style file. Once
that lands, the four steps above ship the submission.
