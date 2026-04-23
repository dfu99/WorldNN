# Deep Research Sweep — Synthesis Memo (2026-04-23)

Covers Parts A (literature), B (current-results), C (experiment plan), D
(cross-discipline). Inputs: lit_scan_A/B/C, deep_read_dreamer_v4 /
_missing_reward / _act_jepa, current_state_review, experiment_plan,
cross_discipline.

## (1) Literature headline findings

- *Action-conditioned latent dynamics is the mainstream research frontier.*
  Dreamer V4 (2509.24527) establishes the scale benchmark; V-JEPA 2-AC
  (2506.22355) and ACT-JEPA (2501.14622) define the joint-embedding
  architecture branch. None of these has a policy-alignment diagnostic
  comparable to SA.
- *Active inference is having a theoretical renaissance without empirical
  legs yet.* "The Missing Reward" (2508.05619) is a conceptual paper. Our
  obj-025 T7 theory note *is* the kind of bounded-estimator derivation
  the AIF literature has been missing.
- *Sensorimotor-neuroscience work increasingly benchmarks agents against
  biology.* Kim et al. 2025 (2509.14360) show PPO-trained agents' state-
  value representations resemble macaque PPC. Biological validation is a
  natural future direction for SA.

## (2) Current-state gaps (ranked by paper impact)

1. *No explicit chain-MI figure in the paper.* We have the data (mi_chain.py)
   but haven't surfaced the I(S;X)→I(S;E) decay curve. Reviewer C
   pre-empt; one figure.
2. *obj-024-only recon comparison.* ΔR² = 0.004 is a scope boundary but
   we don't know whether it extends to obj-016. A re-analysis would
   either strengthen or further narrow the SA claim.
3. *No sensory-dim transfer evaluation.* Train at sensory=16, eval at
   sensory=8. Existing checkpoints + eval script, <1 hour.

## (3) Recommended experiments (ranked, from experiment_plan.md)

1. **E1: Asymmetry-scaling curve at larger capacity** (embed_dim up to 256,
   RunPod, ~3-4 h). Directly addresses Reviewer A "does capacity
   eventually win?"
2. **E5: 1D multi-task sensory-capacity** (ContinuousMatter, RunPod, ~2 h).
   Addresses Reviewer E task-similarity in a non-manipulation regime.
3. **E2: Outcome Alignment** (CPU-only, <1 hour). Directly responds to
   PI's "intent vs real" framing.

## (4) Cross-discipline hook

Fungal hyphal signaling provides a concrete biological analogue with three
testable mappings onto WorldNN: serial-chain information loss, single-
point-of-failure at network hubs, and directional/irreversible action via
apical extension. A falsifiable quantitative bridge exists: compare the
signal-decay exponent measured in common mycorrhizal networks (Simard
1997, Gorzelak 2020, Babikova 2013) against a serial-chain extension of
WorldNN. Matching exponents would be strong cross-domain evidence for the
information-bound claim; mismatch would localize the biological effect to
metabolic constraints.

## (5) Suggested next AFK cycle

If the PI wants another iteration, I recommend running E2 + E3 tonight
(both CPU, ~1.5 h total) and chaining E1 + E5 on RunPod. Before launching
E1/E5: `mc runpod fits 3` (E1) / `mc runpod fits 1` (E5). Expected
outcomes:
- E2: OA correlates with SA r≈0.6, with dist r≈-0.3. Modest addition,
  cheap. 70% worth including.
- E3: SA under 30° rotation drops ~13% (cos(30°) = 0.87), confirming
  structural interpretation. 90% useful for §4.3 framing.
- E1: either the ceiling continues rising (big deal, reframes claim) or
  saturates around embed=32-64 (confirms current scope). Either outcome
  is publishable.
- E5: either pattern holds on ContinuousMatter (Reviewer E risk drops to
  Medium) or floors like 2-rock (confirms the manipulation-only scope).

The pre-submission paper is internally consistent as-is; these
experiments are for strengthening, not rescuing.

## Deliverables produced in this sweep

- tasks/research/lit_scan_A_world_models.md (8 papers)
- tasks/research/lit_scan_B_predictive.md (8 papers)
- tasks/research/lit_scan_C_asymmetry.md (8 papers)
- tasks/research/deep_read_dreamer_v4.md
- tasks/research/deep_read_missing_reward.md
- tasks/research/deep_read_act_jepa.md
- tasks/research/current_state_review.md
- tasks/research/experiment_plan.md
- tasks/research/cross_discipline.md
- tasks/research/sweep_memo_2026-04-23.md (this file)
