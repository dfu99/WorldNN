# Planning

## Target: ICLR 2027 (October 2026 deadline)

**Living paper**: `paper/draft.md`
**Main figure**: `results/fig_main.png`

### Status: Workshop-ready. Main-track with minor additions.

The ICLR blocker (second task) is resolved. obj-021 (2-rock, 6D state)
shows r = -0.728, matching 4D single-rock (r = -0.724). SA generalizes
across tasks, physics variants (93-106%), and appearance variants (89-102%).

### Remaining for main-track submission

1. **Add 2-rock results to paper** — Update §5.5 to include obj-021
   (r=-0.728) alongside the 3-rock results. The 2-rock is the cleaner
   second task; 3-rock is supplementary.

2. **SA proxy GPU validation** — Rerun experiments/sa_proxy.py with
   proper training (500ep, batch=256, GPU). CPU validation was inconclusive
   due to insufficient training. Nice-to-have, not blocking.

3. **Final pass**: update paper status checklist, regenerate main figure
   with 2-rock panel, ensure all numbers match latest data.

### All Completed

- [x] SA metric (r=-0.724, 245 configs, 7 seeds)
- [x] Formal interaction test (F=34.2, p=5e-9)
- [x] Metric ablation (6 metrics, mag-weighted SA r=-0.893)
- [x] SA dynamics (slope r=-0.705)
- [x] SA transfer: physics (93-106%) + appearance (89-102%)
- [x] Reversal analysis (boundary artifact)
- [x] Framing overhaul (SA rename, blind cat, related work)
- [x] Academic writing revision (48 violations fixed)
- [x] Second task: 2-rock 6D (r=-0.728) — ICLR BLOCKER RESOLVED
- [x] Multi-rock 3-rock analysis (r=-0.300, honest §5.5)
- [x] Main 6-panel figure

## Recently Completed

- [2026-03-29] obj-021: 2-rock (6D) r=-0.728 — ICLR second task resolved
- [2026-03-29] obj-020: SA transfer with appearance variation (89-102%)
- [2026-03-29] Multi-rock v2 analysis + paper §5.5
- [2026-03-29] Main 6-panel publication figure
- [2026-03-27] Academic writing revision (48 violations)
- [2026-03-25] obj-019: SA transfer (93-106% physics retention)
- [2026-03-24] Framing overhaul, metric ablation, SA dynamics figure
- [2026-03-22] obj-016: At-scale (280 configs, 7 seeds), p=5e-9
