# Reframe: from a correlation to a controllability bound

_2026-06-25 — response to PI: "dressed up trivialities, not impactful research."_

## The trivial core (the critique is correct)

- **The headline is a correlation.** We defined a sensorimotor-alignment
  score and showed it tracks learning speed (r ≈ -0.72). A skeptic reads
  this as: "you built a difficulty proxy and showed it correlates with
  difficulty." Circular-flavored, unsurprising.
- **One bespoke toy** (rock-push), **small effects** (peak alignment
  ≈ 0.23; substitution effect d ≈ 1.1). External validity is thin — one
  reviewer flagged single-environment as high risk and we never closed it.
- **The motivating intuition is obvious**: more information lost in the
  loop → more internal capacity needed. Quantifying the obvious on a toy
  is incremental, not a contribution.

These are real. No amount of polish fixes a correlation-on-a-toy.

## The reframe that makes it a contribution

Stop reporting a correlation. **State a limit, prove it, verify it.**

Model the whole perception-action loop as a graph of lossy channels:

```
matter state → channels → environment VAE → sensors → embedding   (perception)
embedding → actuator → environment → matter                       (control)
```

Each edge has an information capacity (bits/step). 

**Target theorem.** The agent can reliably induce the 1-bit target
transition (success ≥ 1-ε) **iff** the *min-cut* of this graph between
"matter state" and "agent decision" carries at least the task's required
bits H_task(ε). Below the min-cut, control is impossible for *any* agent.

### Two consequences — neither trivial, both testable

1. **Substitution becomes a theorem, not an observation.** Sensory
   bandwidth and embedding capacity are two edges on the perception side;
   below the min-cut they trade off at an **exchange rate fixed by the
   channel capacities** — "1 bit of sensing is worth k bits of memory,"
   with k *derived*, not fitted. We already see a noisy hint of this
   trade-off. The law turns it into a predicted number.

2. **Architecture independence → an impossibility result.** The bound is
   over channel capacities, so it holds for *any* architecture. Show that
   no agent — bigger net, RNN, transformer — beats the min-cut floor.
   An impossibility result is far harder to dismiss as a toy artifact than
   a correlation is.

## Why this is novel

Information limits of control exist in the literature: Touchette & Lloyd
(information-theoretic limits of control), Tatikonda–Mitter (control under
communication constraints), empowerment (Klyubin), rate-distortion theory
of decision-making (Sims; Tishby). In *all* of them the bottleneck is a
**communication channel rate**. The open slice:

- the bottleneck is the agent's **internal representation capacity**
  (embedding size), not a channel rate; and
- the channel is a **learned** lossy map (the VAE) we can measure
  end-to-end.

A min-cut law over a learned perception-action graph, with a *verified*
perceptual↔representational exchange rate, is genuinely unclaimed.

## What it takes to claim it — and where it dies

| Step | Falsifier |
|------|-----------|
| Derive the bound for the binary-transition task (H_task small, ε clean) | Agents reliably control *below* the predicted min-cut → bound wrong/loose |
| Measure each edge's capacity (we already estimate MI along the chain) | Measured exchange rate k ≠ predicted k → substitution isn't min-cut |
| Sweep architectures at fixed capacities | A bigger/different net beats the floor → not a fundamental limit |
| One transfer to a non-toy channel (e.g. real sensory-substitution bandwidth) | Doesn't transfer → single-environment critique stands |

## The cheapest possible test (no GPU)

Derive the bound for the flip-the-bit task and check whether the
**substitution data we already have** matches the predicted exchange rate.
Pure theory + re-analysis of existing mutual-information estimates. If it
matches, we have a law. If not, we killed the reframe for the price of a
memo — before investing in new runs.

## Recommendation

Start the cheap test. It is reversible, costs no compute, and decides
whether the whole reframe has legs. If the PI's own thinking is heading
somewhere else, redirect before the derivation hardens.
