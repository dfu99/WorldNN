# Honest reassessment: the controllability bound is not a contribution

_2026-06-27 — after PI pushback on obj-039._

## What the PI said

Three challenges to the min-cut "novelty":
1. Is "nobody has that" because it's untrue / irrelevant / nonsensical, or
   because it genuinely hasn't been done?
2. The Information Bottleneck literature already studies this — it's not a
   triviality in there.
3. Saying the limit is *internal memory size* rather than *communication
   channel* is a sleight of hand.

He is right on all three.

## The concession (precise)

- **Q1 — it has been done.** The bound is a Fano + data-processing exercise.
  The surrounding ideas are all established:
  - Information Bottleneck (Tishby–Pereira–Bialek; deep variational IB) —
    compression-vs-relevance in learned representations.
  - Information-theoretic control (Tatikonda–Mitter) — control under
    communication constraints; empowerment (Klyubin) — action→future-sensor
    channel capacity.
  - Rate-distortion / bounded-rationality agents (Genewein–Braun; Tishby,
    "information theory of decisions and actions") — *capacity-limited
    controllers*, i.e. exactly "internal memory as the bottleneck."
  "Nobody has that" was an overclaim made without a literature check.

- **Q2 — IB subsumes it.** The compression-relevance tradeoff is the *core*
  IB result and is deep. Rederiving a DPI bound adds nothing to it.

- **Q3 — relabeling.** In the static, single-step setting, "memory capacity
  C" and "channel capacity C" are the identical constraint I(·;·) ≤ C; DPI
  treats every edge alike. "Internal memory" is a cosmetic rename. The only
  non-cosmetic version is *temporal* (memory accumulates across steps; a
  channel is per-use) — and that is covered by finite-memory control and
  sequential rate-distortion.

## Net

The theoretical-bound framing is **dead as a contribution**. obj-038/039
were a correction from "trivial correlation" into "trivial-but-fancier
bound" — same defect, more LaTeX.

## The one candidate that is empirical, not theoretical

**Achievability gap under learning.** The bound is an *upper* limit. A
channel's capacity is reachable by ideal coding; an SGD-trained embedding
may systematically under-use its capacity. Open empirical question:

> How far below the information limit does a *learned* (PPO) agent sit, and
> does that gap depend on *which* edge binds — a passive channel (sensory
> sampling, VAE) vs. a learned register (the embedding)?

If the gap is structured by bottleneck type, that is the one place
"memory ≠ channel" is non-vacuous, and it is exactly what a simulator can
measure and pure theory cannot. **Status: candidate only.** Must do an
adversarial literature check (RL achievability vs information limits;
representation-learning capacity utilization) *before* claiming any novelty
— the failure mode that just bit us.

## Decision posture

Theory angle closed. Awaiting PI direction. Two live paths:
(a) test the achievability-gap question (cheap-ish; mostly re-analysis +
one instrumented sweep), or
(b) accept that WorldNN may lack a novel core and wind down / repurpose.
The PI is "still circling" what makes this impactful and likely has an
intuition the agent is missing — sync before spending more compute.
