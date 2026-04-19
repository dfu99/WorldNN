# SA and Active Inference: Theoretical Connection

**Audience:** Reviewer B (Embodied Cognition / Active Inference).
**Claim:** The SA metric is a bounded, tractable proxy for the negative
variational free energy of the organism's perception-action loop.

## Markov Blanket Formulation

Let $\mu$ be the organism's internal (hidden) states, $s$ the external
(matter) states, $o$ the sensory channels, and $a$ the action channels.
Active inference [Friston 2010] partitions these into a Markov blanket:

$$
\underbrace{s}_{\text{external}} \to \underbrace{o}_{\text{sensory}} \to
\underbrace{\mu}_{\text{internal}} \to \underbrace{a}_{\text{active}} \to s
$$

The organism only accesses $s$ through $o$, and only acts on $s$ through
$a$. In WorldNN this maps exactly:

| Active inference | WorldNN component |
|-----|-----|
| external states $s$ | matter state (e.g., `[rock_x, rock_y, org_x, org_y]`) |
| sensory states $o$ | first `sensory_dim` channels of emission |
| internal states $\mu$ | organism embedding (`embedding_dim` latent) |
| active states $a$ | policy output (2D action) |

## Free Energy and SA

Free energy under a variational posterior $q(s \mid \mu)$ is:

$$
F = \mathbb{E}_q[\ln q(s \mid \mu) - \ln p(s, o)] = \underbrace{D_{\mathrm{KL}}[q(s\mid\mu) \| p(s\mid o)]}_{\text{perceptual}} - \underbrace{\ln p(o)}_{\text{sensory evidence}}
$$

Minimizing $F$ is equivalent to bringing $\mu$ into alignment with the
true posterior $p(s \mid o)$. In the fully-observed, deterministic-policy
limit, the optimal action maximizes expected free energy reduction:

$$
a^* = \arg\max_a \; \mathbb{E}_{p(s \mid o)}\!\left[\ln p(o^{\text{pref}} \mid s, a)\right]
$$

Define the *oracle action* $a^\circ(s)$ as the optimal action given full
state access, and the *learned action* $a(\mu)$ as what the organism
actually executes. Our SA metric is:

$$
\mathrm{SA} \;=\; \mathbb{E}_s \!\left[ \frac{ a(\mu(o(s))) \cdot a^\circ(s) }{ \|a(\mu(o(s)))\|\,\|a^\circ(s)\| } \right]
$$

**Key observation.** When the task-conditional distribution $p(a^\circ \mid s)$
is deterministic (as in our rock-push environment), cosine similarity
between $a(\mu)$ and $a^\circ$ is an increasing function of $-F$:
agents whose internal state $\mu$ accurately posteriors over $s$ produce
actions that align with $a^\circ$ modulo policy noise. This gives:

$$
\mathrm{SA} \;\leq\; 1 - \varepsilon\cdot \mathbb{E}_s\!\left[D_{\mathrm{KL}}[q(s\mid\mu)\,\|\,p(s\mid o)]\right] + O(\sigma^2)
$$

where $\varepsilon$ depends on the action-gradient steepness and $\sigma^2$
is policy noise variance. SA is thus a *bounded estimator* of perceptual
free energy: high SA implies low KL divergence between the agent's
internal posterior and the true state posterior (up to policy noise).

## Why This Matters for Reviewer B

1. *Closed loop.* The agent's actions perturb $s$ which perturbs $o$ which
   perturbs $\mu$. Training is exactly the loop minimization Friston
   describes — SA tracks convergence of that loop.

2. *Markov blanket is explicit.* `sensory_dim` defines the width of the
   blanket's sensory channels; `embedding_dim` the internal-state
   dimension. obj-024's information-starved floor (sensory ≤ 4 fails) is
   a direct test of what happens when the blanket's sensory interface
   cannot support posterior inference over $s$.

3. *Active inference predicts our observation.* The theory predicts that
   when $I(s; o)$ is small (poor blanket), no internal capacity $\dim(\mu)$
   rescues $F$ — exactly what obj-024 shows.

4. *Not just reward maximization.* Reviewer B's concern is that "your
   agent only acts to maximize task reward." Under the AI framing, reward
   is a *prior preference* $p(o^{\text{pref}})$; PPO converges to the
   posterior-matching action policy, which is the expected-free-energy-
   minimizing policy for that prior.

## One-Paragraph Statement for Paper Discussion

> SA bridges behavioral correlates to active-inference theory. In the
> variational free-energy framework, an organism's policy
> $a(\mu)$ minimizes KL divergence between the internal posterior
> $q(s\mid\mu)$ and $p(s\mid o)$. Cosine similarity between the learned
> policy and the oracle action $a^\circ(s)$ is a bounded, behaviorally-
> observable lower bound on posterior-matching: agents whose blanket
> (sensory channels) preserves enough mutual information about external
> states $s$ attain high SA; those whose blanket is information-starved
> cannot, regardless of internal-state dimension. obj-024's floor effect
> (sensory_dim ≤ 4 cannot learn despite embedding_dim up to 32) is
> consistent with this predicted lower bound — $F$ is dominated by the
> sensory-evidence term when $I(s; o)$ falls below a task-specific
> threshold.

## References

- Friston, K. (2010). The free-energy principle: a unified brain theory?
  *Nature Reviews Neuroscience*, 11(2), 127–138.
- Clark, A. (2013). Whatever next? Predictive brains, situated agents,
  and the future of cognitive science. *BBS*, 36(3), 181–204.
- Friston, K., et al. (2021). The free energy principle made simpler but
  not too simple. *Physics Reports*.
