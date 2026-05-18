# Cron Check-In Prompt — WorldNN

This file is the prompt that `bin/mc-cron-checkin.sh` sends to this
project's tmux session every 6 hours (4x/day, at HH:00–HH:30 staggered by
RunPod priority). **You own this file.** When you finish a cron turn,
edit it to steer what the next tick (~6h later) will ask you to do.

## Current focus (edit me each turn)

*Updated 2026-05-15 cron tick:* Paper integration of obj-034/035/036
landed (commit 66d9d1c, §7.5 +3 paragraphs). T25 closed post-deadline
(commit at 2026-05-15, see /tmp/mc-task-done/WorldNN/done.txt).

Next-tick options (autonomy contract — pick highest EV):

1. **obj-033 PPO hparam sweep** (~4h CPU, ungated). lr × entropy × clip
   × 3 seeds on obj-024's peak cell. Tests whether obj-024 was
   optimally tuned. Closes a backlog item. Pick this if no fresher
   target appears.

2. **WD_BLACK mirror sync** (last sync 2026-05-13; new artifacts:
   obj-036 + paper integration commit). ~30 sec.

3. **Reviewer-rebuttal cross-check pass 2**: now that §7.5 has 3 new
   paragraphs, re-verify tasks/rebuttal_letter_draft.md still aligns
   with the paper text. ~10 min.

Pick (2) first as housekeeping, then start (1) in foreground.

Standing rules unchanged.

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
