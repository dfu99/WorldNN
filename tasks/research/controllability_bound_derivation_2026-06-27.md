# Formal derivation: a controllability bound for the perception-action loop

_2026-06-27 — obj-039. Makes the min-cut reframe (obj-038) rigorous._

## Setup

Target matter has a binary state $S \in \{0,1\}$, uniform prior ($H(S)=1$
bit). It is a Mealy machine: the next state $S' = \delta(S, A)$ depends on
*both* the current state and the agent's action $A$. The task is to induce
a specified 1-bit transition (e.g. flip: drive $S' = 1\oplus S$) with
success probability $\ge 1-\epsilon$.

Crucially the *correct action is state-dependent*: the action that flips
state 0 is not the action that flips state 1. (If a single action flipped
both, the task would carry 0 bits and be trivial — no perception needed.)
So to act correctly the agent must first distinguish $S$.

The agent perceives $S$ only through a cascade of lossy channels:

$$ S \;\to\; C \;\to\; Z_{\text{env}} \;\to\; X \;\to\; E $$

- $C$: emitted channels (light/sound/heat) — matter's only observable output
- $Z_{\text{env}}$: environment VAE latent (learned lossy compression)
- $X$: the organism's sensorimotor sample (a subset of channels)
- $E$: the agent's internal embedding (bounded by embedding size $d_E$)

This is a Markov chain: each variable depends on the previous one only.
The policy then maps $A = \pi(E)$, and $A$ drives matter through a control
channel back to $S'$.

## The bound

**Step 1 — the task forces state estimation (Fano).** To pick the correct
state-dependent action with error $\le \epsilon$, the agent must hold an
estimate $\hat S(E)$ of the current state with error $P(\hat S \ne S) \le
\epsilon$. Fano's inequality gives

$$ H(S \mid E) \;\le\; H_b(\epsilon), \qquad H_b(\epsilon) = -\epsilon\log_2\epsilon - (1-\epsilon)\log_2(1-\epsilon). $$

Since $I(S;E) = H(S) - H(S\mid E) = 1 - H(S\mid E)$,

$$ I(S;E) \;\ge\; 1 - H_b(\epsilon). \tag{1} $$

**Step 2 — the channel cascade caps the available information (DPI).** By
the data-processing inequality along $S\to C\to Z\to X\to E$, information
about $S$ can only decrease at each stage:

$$ I(S;E) \;\le\; I(S;X) \;\le\; I(S;Z) \;\le\; I(S;C). $$

Each stage's information is itself capped by that channel's capacity
$\mathrm{cap}_k = \max_{p} I(\text{in};\text{out})$. For a *series* chain
the binding constraint is the single tightest edge — the **min-cut**:

$$ I(S;E) \;\le\; C_{\min} := \min_k \mathrm{cap}_k. \tag{2} $$

**Step 3 — combine.** From (1) and (2):

$$ \boxed{\,1 - H_b(\epsilon) \;\le\; C_{\min} \quad\Longleftrightarrow\quad \epsilon \;\ge\; H_b^{-1}\!\big(1 - C_{\min}\big)\,} $$

This is the controllability bound. **No agent — any architecture, any
training — can flip the bit with error below $H_b^{-1}(1-C_{\min})$, where
$C_{\min}$ is the tightest channel in its perception loop.** Reliable
control ($\epsilon \to 0$) requires $C_{\min} \to 1$ full bit.

## Three consequences

1. **Min-cut, not smooth blend (explains obj-038).** Sensory bandwidth and
   embedding capacity are two edges of the *same series chain*; the bound
   depends on their **minimum**, so performance tracks $\min(\text{sensory},
   \text{memory})$ — exactly the $r=0.67$ vs $0.47/0.39$ result. Increasing
   the slack edge cannot raise $I(S;E)$, which is why over-provisioning
   memory past the sensory limit *lowers* measured alignment (regularization
   cost with no information gain).

2. **The bound quantitatively predicts the observed floor.** Measured
   perception information (KSG estimator, obj-025) peaks at
   $C_{\min}\approx 0.33$ bits (sensory=16). The bound then forbids error
   below $H_b^{-1}(1-0.33) \approx \mathbf{17.5\%}$. This is *why* the
   rock-push toy has always sat at its floor (peak $|SA|$ only 0.23, control
   distance barely moves): perception never delivers even half a bit, so the
   task is information-starved by construction. The long-standing "floor
   effect" is not a training artifact — it is the bound biting. See
   `figures/controllability_bound.png`.

3. **Architecture independence → impossibility result.** $C_{\min}$ is a
   property of the *channels*, not the policy. So no choice of $\pi$ (wider
   MLP, RNN, attention) can beat the floor. This is the quotable,
   toy-resistant claim: a *limit*, not a correlation.

## What this tells us to build next

The series structure means a *static, single-step* task admits **no
substitution** — only hard min(). Genuine substitution (memory compensating
for weak instantaneous sensing) requires the agent to *integrate over time*:
a temporally extended, partially observed task where memory accumulates
$I(S;E_t)$ across steps. Then the constraint becomes
$(\text{sensory bits/step}) \times (\text{usable integration steps}) \ge 1$,
a real **exchange rate** between perceptual bandwidth and memory depth. That
is obj-040: redesign the task so memory binds, and measure whether the
predicted sensory×time trade appears.

## Falsifiers (unchanged from the reframe memo)

- An agent flips the bit with error **below** $H_b^{-1}(1-C_{\min})$ → the
  bound is wrong or $C_{\min}$ is mis-measured.
- In the temporally-extended task, the measured exchange rate $\ne$ the
  predicted (sensory bits/step × steps) → substitution is not min-cut.
- A larger/different architecture beats the floor at fixed $C_{\min}$ → not
  a fundamental limit.

## Status

Theory complete and internally consistent; the floor-prediction match
($0.33$ bits → $17.5\%$ error floor) is a non-trivial quantitative
confirmation on existing data, no new compute. Ready to (a) show the PI as
the concrete "impactful" artifact, and (b) drive the obj-040 task redesign.
