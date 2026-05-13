# Cron Check-In Prompt — WorldNN

This file is the prompt that `bin/mc-cron-checkin.sh` sends to this
project's tmux session every 6 hours (4x/day, at HH:00–HH:30 staggered by
RunPod priority). **You own this file.** When you finish a cron turn,
edit it to steer what the next tick (~6h later) will ask you to do.

## Current focus (edit me each turn)

*Updated 2026-05-13 cron tick:* obj-034 (Outcome Alignment), obj-035
(sparse oracle-supervision), and obj-036 (1D-positioning Reviewer E
mitigation) all logged with figures.

Next-tick highest-EV item: **paper integration** of those three
findings into the live draft. Concrete steps for the next cron:

1. Read `tasks/objectives.yaml` for obj-034/035/036 entries to refresh.
2. Add a short paragraph to `paper/neurips2026/main.tex` §7 (Discussion)
   citing OA (r=0.75 with SA, saturates earlier) and noting the §7.5
   pure-PPO threshold-crossing finding from obj-035.
3. Add a §7.5 Limitations sentence: "On easier task families (1D
   continuous positioning, obj-036), the SA framework saturates without
   showing the rate-distortion structure; the substitution effect is
   visible only when the task has multiple bottlenecks."
4. Recompile + verify ≤9 body pages.
5. Commit + WD_BLACK mirror sync.

This is CPU-only, ungated on RunPod, paper-grade. Fallback if some
deeper issue: pick obj-033 PPO hparam sweep (~4h CPU).

## Standing rules

- Re-read `CLAUDE.md` and `tasks/planning.md` if you need to ground.
- Per the autonomy rule (`~/.claude/CLAUDE.md` § Autonomy), do not ask
  the PI for direction on reversible in-repo work — pick the higher-EV
  option and execute, defend after.
- Per the evals rule (`~/.claude/CLAUDE.md` § Evals), do not unwire the
  tripwire hook in `.claude/settings.json`.
- If you need RunPod GPU and it is occupied, subscribe via
  `mc runpod subscribe WorldNN "<note>"` and continue on a CPU-friendly
  sub-task in the meantime. (RunPod priority: halulujah > FIND-SNP > others.)
- Visualize results before claiming done (rule 6 in global CLAUDE.md).
