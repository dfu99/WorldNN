# Rebuttal Letter Draft (NeurIPS 2026)

*Drafted 2026-05-05 from `tasks/audit-2026-05-05.md` and
`tasks/claim_to_evidence.md`. Anchor each response to a specific paper
section + figure + table to give the reviewer something concrete.
This is the response we would write if the listed signature questions
were raised in a real review cycle. Comments inside `{...}` are notes
to ourselves, not for the reviewer.*

---

## To Reviewer A (World Model / Model-Based RL Researcher)

We address each of your signature concerns in turn.

**Mealy machine versus POMDP framing (§3.1).** We adopt the Mealy
formalism not to claim novelty over POMDPs but to make the
*perception-action asymmetry* explicit at the type level: matter's
output depends on both the hidden state and the input (random seed plus
action), and the only access an organism has to that state is through
explicit lossy channels. A POMDP's observation function $O(o \mid s, a)$
hides this same asymmetry but treats it as a black box; our framing
keeps the chain decomposable so each link can be perturbed in isolation
(perception ladder, §5.1). All our SA results would translate cleanly
into the POMDP language; the reverse mapping is more cumbersome.

**Reconstruction-loss versus SA (§7.5).** This is the most direct
challenge to our contribution and we have made it the focus of our final
analysis. On the at-scale 245-config grid (7 perception levels × 5
embed × 7 seeds), per-perception linear-probe recon ceilings range from
$R^2 = 0.19$ (oracle+noise(0.5)) to $R^2 = 1.00$ (oracle). Per-config
recon correlates with task distance at $r = -0.436$, weaker than SA's
$r = -0.724$. After controlling for recon, SA still correlates with
distance at partial $r = -0.679$; recon's residual partial correlation
*flips sign* to $+0.285$. Multiple regression on distance gives
$R^2_{\text{recon}} = 0.190$ versus $R^2_{\text{recon+SA}} = 0.563$, so SA
adds $\Delta R^2 = +0.374$ over recon (95% bootstrap CI
$[+0.253, +0.489]$; permutation $p < 10^{-4}$). On a deliberately
narrow grid (obj-024, sensory $\in \{2,4,8,16\}$, no perception
variation), $\Delta R^2$ collapses to $+0.004$. This is the cleanest
characterization we can make: SA captures structural alignment that
varies *across* perception conditions; on grids where perception is
uniformly degraded, recon and SA carry the same signal.

**DMC suite head-to-head.** A direct DMC benchmark would require
fitting Dreamer at our scales, which is outside this paper's scope. The
recon-vs-SA delta on our wide grid is the directly equivalent
measurement, and we report it with the rigor any DMC comparison would
demand: bootstrap CI, permutation test, and an explicit narrow-grid
control.

---

## To Reviewer B (Embodied Cognition Theorist)

**Predictive component.** SA is itself a behavioral correlate of
posterior alignment under variational free energy. The §7 paragraph
citing Friston (2010) sketches the bound; the supplementary
`tasks/theory_notes/active_inference.md` gives the full derivation
showing SA is an empirically-tractable lower bound on
$-\mathrm{KL}[q(s \mid \mu) \| p(s \mid o)]$ when policies are
deterministic in the limit. Action variance (oracle-free SA proxy,
§7) provides the same diagnostic without requiring oracle access — the
"closed-loop dynamics with prediction-action coupling" you want is
exactly the closed obj-022 result.

**Markov blanket formalization.** We map the four blanket components
explicitly in §3.1 (external $s$ → sensory $o$ → internal $\mu$ →
active $a$ → environment closure). The diagram in supplementary
Figure 6 makes the mapping graphical. The width of the sensory
component is `sensory_dim`; the internal-state dim is `embedding_dim`;
the rate-distortion result in §5.7 shows that capacity in $\mu$
cannot recover information absent from $o$, which is exactly the
information-theoretic content of the Markov-blanket constraint.

**Free energy connection.** Cited in §2 Related Work and operationalized
across §5.7 (rate-distortion bound) and §7 (SA as bounded
free-energy estimator).

---

## To Reviewer C (Information Theorist)

**KSG estimate of $I(X;W) - I(Z;W)$.** §5.7 + Figure 5: linear-probe
$R^2(s \mid \text{obs})$ rises monotonically with `sensory_dim`
(0.15, 0.28, 0.82, 1.00), corresponding to Gaussian-MI lower bounds of
0.33, 0.64, 3.44, 27.6 nats. The peak SA achievable at each
`sensory_dim` correlates with the Gaussian-MI lower bound at $r = 0.975$.
We use the Gaussian lower bound rather than KSG itself because (audit
D2) our KSG implementation under-reports MI by 30-100% in the
medium-correlation regime; the linear-probe path gives a tight
analytically-grounded bound that matches the rate-distortion shape you
would expect from KSG when it works. The under-estimation is pinned by
a unit test (`tests/test_components.py::TestKSGEstimator`) so future
refactors cannot silently shift the paper numbers.

**Minimum embedding size versus channel capacity.** They are related but
not identical: channel capacity is $\sup_{p(x)} I(X; Y)$, an upper bound
on what *any* downstream policy could learn; minimum embedding size is
the dimension of $\mu$ at which task performance crosses a threshold
under our specific policy class (PPO with MLP organism). Our experiments
operationalize the latter; the former is implicit in the rate-distortion
ceiling we report.

**Action variance versus value-function gradient.** §7 explicitly
addresses this. Action variance is $\mathrm{Var}[a(o)]$ across observed
states; value-function gradient is $\nabla_{\!o} V(o)$. They coincide
only when the policy is the greedy maximizer of a single-step value
function, which is not how PPO's stochastic policy is constructed and
not what we observe empirically. The $r = -0.82$ correlation reflects
that organisms whose actions show structured variation across observed
states are the ones that learned an effective policy; near-constant
output indicates a degenerate solution.

---

## To Reviewer D (Sensorimotor Neuroscientist)

**Embedding size: cortical neurons or working-memory slots?**
§3.2 (after Table 1) frames `embedding_dim` as a working-memory-slot
analogue, not a literal cortical-neuron count. The relevant biological
referent is the size of the internal state buffer that mediates between
perception and action.

**Signal-dependent versus Gaussian noise.** §7.5 (Noise model
sensitivity) reports the result of a sensitivity analysis: under
$\sigma_i \propto |y_i|$ (Weber-style) versus additive Gaussian on the
same matter dynamics, recon ceilings coincide within $\pm 0.05 \, R^2$
when matched on effective per-channel $\sigma$. Signal-dependent noise
is *easier* to decode at matched coefficient because low-amplitude
channels emit proportionally less corruption; the Gaussian model is the
conservative choice and the conclusion is invariant.

**Animal experiment.** §7.4 states a falsifiable prediction: in a
behavioral animal model, monocular deprivation during the
BDNF-dependent critical period should drive SA in visuomotor reaching
below the learnability threshold (analogous to our oracle+noise(0.5)
regime). The dose-response variable is the integrated below-threshold SA
duration, not the absolute SA endpoint; this distinguishes the
SA-as-diagnostic framing from naive critical-period cutoff theory.

---

## To Reviewer E (Generative Simulation Skeptic)

We anticipated these concerns and address them honestly.

**2-rock $r = -0.728$ versus 1-rock $r = -0.724$ — transfer or task
similarity?** The 2-rock task is structurally distinct: the organism
must select between two pushable objects, the reward signal is averaged
across both rocks, and the optimal policy must switch attention. The
matching correlation is evidence of *transfer of the SA mechanism*, not
of fixed task structure. Our §5.6 transfer experiments make this
explicit: 94-106% retention across physics variants and 89-110% across
appearance variants on rocks with both different physics *and* different
visual signatures. If SA were a memorized task-specific mapping, the
appearance-variant retention should drop with $\epsilon$; we measure
$r(\epsilon, \text{retention}) = 0.033$, which is statistically
indistinguishable from zero invariance.

**Why is 3-rock $r = -0.300$ a footnote, and why does 2-rock
sensory-capacity hit a floor?** §7.5 discloses both. 3-rock at 1000
episodes hits a budget floor (distance 0.46-0.50 across all conditions,
no organism learns reliably even with oracle perception); the SA
correlation is then measuring within-floor noise. 2-rock at 800 episodes
in the sensory-capacity sweep (obj-026) shows the same floor. The
correct read is *task-budget scaling*, not metric failure: SA is
informative when the task is in a regime where some organisms succeed
and others do not. Our cross-task generalization paragraph (§7.5) makes
the prediction explicit: SA on a different task family should follow
the rate-distortion mapping in Figure 5; the prefactor calibration is
the open empirical question.

**Boundary-artifact reversal — does SA measure degenerate failure
modes?** §6 documents this as a feature, not a bug. The reversal
($r = +0.46$ within oracle+noise(0.5)) appears *only when distance has
near-zero variance* (std = 0.004); when all configs fail equally, any
correlation is noise. SA is undefined at the noise floor by design.
This is documented as a scope condition, not hidden.

---

## What we have not yet addressed

A small number of items remain post-camera-ready:

1. *Direct DMC head-to-head with Dreamer V3.* The recon-vs-SA delta on
   our wide grid is the directly equivalent measurement; a DMC-scale
   experiment is future work.
2. *Cross-task SA ceiling on a non-manipulation task family.* §7.5
   states the prediction; experimental confirmation is queued for the
   post-submission revision (`tasks/queue.yaml::T32` 1D positioning).
3. *Multi-bit task at higher capacity.* obj-031 (asymmetry-scaling at
   embed up to 512) was launched on RunPod but interrupted by a higher-
   priority co-tenant; results pending pod availability.

We commit to running these as a revision package if accepted.
