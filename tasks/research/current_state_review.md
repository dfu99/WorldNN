# WorldNN Current-State Review (2026-04-23)

Artifacts reviewed: `paper/draft.md`, `paper/neurips2026/main.tex`,
`experiments/{rock_push.py, predictive_processing.py,
sensory_capacity_tradeoff.py, mi_chain.py, ci_at_scale.py}`,
`tasks/{claim_to_evidence.md, intuition.md, objectives.yaml}`.

## (a) Unclaimed adjacent hypotheses

1. *SA slope as early-stop criterion holds across compute budgets.*
   obj-015 shows SA slope r=-0.705 predicts final performance at n_ep=500.
   We have not tested whether the same cutoff (slope < 0.1 per 100 ep at
   ep=200) holds at 1600 or 800 episodes. This is a direct empirical
   claim we could make at low compute.

2. *SA transfer survives action-space perturbations.*
   obj-019/020 test physics/appearance transfer but keep the action space
   identical. If we rotate or rescale the action vector (e.g., multiply
   by a fixed orthogonal matrix), does SA retention stay high? This
   measures whether SA truly tracks *structural* alignment or is
   coordinate-frame-locked.

3. *Prediction-head vs policy-head alignment (Outcome Alignment).*
   From the PI conversation: we measure SA in action space, but "intent
   vs real" in outcome space is unmeasured. We have the data to compute a
   cheap oracle-free variant (cos(action_xy, Δrock_xy) when in contact).
   Would correlate with SA if both track the same underlying structure.

4. *Action-variance proxy extrapolates below the learnability threshold.*
   obj-022 validates the action-variance proxy on the main grid but does
   not test whether it still correlates with performance in the SA<0.3
   regime. We can re-analyze existing data.

5. *Task-timescale dependence of SA.*
   SA is measured on the final policy. No work on how SA evolves within
   a single episode (from episode-start exploration to episode-end
   exploitation). A within-episode SA trace could reveal whether the
   organism uses one consistent strategy or two.

6. *Perception-chain noise vs policy-stochasticity trade.*
   We fix PPO's exploration noise schedule. Perception noise (σ=0.1,0.5
   in oracle+noise) and policy noise could interact; SA may become
   measurable only in a narrow band of their product.

## (b) Reviewer-visible weak spots (beyond claim_to_evidence.md)

1. *Single organism morphology.* Our organism is a fixed MLP with
   sensory_filter → encoder → embedding → policy. ACM CS Embodied
   Intelligence survey (2025) emphasizes morphology-as-structural-prior.
   A reviewer could argue: your capacity claim is architecture-specific.

2. *No comparison to a Dreamer-family world model on the same task.*
   Dreamer V4 exists. Reviewer A could demand a head-to-head: would
   Dreamer's latent reconstruction loss predict our task performance as
   well as SA?

3. *Linear-probe recon R² vs SA comparison is on obj-024 only.* Should
   be on obj-016 too (the primary grid) for full disclosure. Currently we
   claim ΔR²=0.004 on obj-024; an obj-016 number would either strengthen
   or further scope-restrict the SA advantage claim.

4. *No explicit information-chain-loss figure.* obj-023 mi_chain.py
   computes I(S; X), I(S; Y), I(S; Z), I(S; E) but the paper doesn't
   show the chain-decay curve. A 1-panel addition to §5.7 would be a
   strong pre-empt for Reviewer C.

5. *Transfer results (§5.6) don't cover sensory-dim transfer.* Train with
   sensory=16, evaluate at sensory=8. Does SA drop gracefully or sharply?
   obj-024 data has the training side; we need eval at transfer.

6. *No ablation of the SA normalization choice.* mag-weighted SA vs
   plain-cosine is mentioned in §4.2 but not explicitly defended beyond
   a correlation-comparison table.

## (c) Scaling gaps

1. *Organism size ceiling.* Our embed_dim caps at 32 (≈2K params).
   Dreamer V4 operates at millions of parameters. We have not run
   embed_dim=128 or 256 to test whether the substitution effect persists
   or vanishes at larger capacity.

2. *Episode count.* 800 episodes is the current cap. obj-026's floor
   effect on 2-rock suggests some tasks need ≥2000 ep. Until we run
   longer, Reviewer E will stay High.

3. *Multi-task grid.* We have 1-rock (primary), 2-rock (obj-021
   with r=-0.728 in the SA-primary setting, but obj-026 floor in the
   sensory-capacity setting), and 3-rock (floor). No task-diversity
   evidence: all rock-push. Adding a non-push task (e.g., 1D continuous
   positioning with sensory sweep) would broaden the claim.

4. *Batch-size sensitivity.* Our batch=256 was set for local RTX 3060.
   PPO stability is known to depend on batch size (see obj-017 lesson).
   Repro grid at batch={128, 512} on RunPod A4500 would be quick.

5. *Seed count per cell.* obj-024 has n=5, obj-026 has n=3. Power
   analysis (obj-025 T14) says d=1.10 substitution needs n=13 for 80%
   power — we are under-sampled. A targeted n=13 run at 4 cells
   (rich-min, poor-max, ceiling, floor) would close this.

6. *Compute transparency.* Paper doesn't report wall-clock time or VRAM
   per config. obj-025 T14 framing ("compute-heavy") is qualitative.
   Reviewers increasingly expect compute disclosures.
