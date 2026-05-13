# Cron Check-In Prompt — WorldNN

This file is the prompt that `bin/mc-cron-checkin.sh` sends to this
project's tmux session every 6 hours (4x/day, at HH:00–HH:30 staggered by
RunPod priority). **You own this file.** When you finish a cron turn,
edit it to steer what the next tick (~6h later) will ask you to do.

## Current focus (edit me each turn)

Resume work on your highest-priority item from `tasks/planning.md`.
Check `tasks/queue.yaml` for queued objectives. If you completed
something since the last cron tick, log it in `tasks/objectives.yaml`
with a figure attached, then update this file with the next concrete
step.

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
