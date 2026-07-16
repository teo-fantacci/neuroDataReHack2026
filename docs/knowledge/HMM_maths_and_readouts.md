# The HMM: decoding, state ordering, directedness, and the ARI — the maths

Written 2026-07-14, in answer to a set of specific questions. Companion to
[`choosing_K_for_the_HMM.md`](choosing_K_for_the_HMM.md) (*why* likelihood fails) and
[`ARI_and_state_stability.md`](ARI_and_state_stability.md) (the protocol). This one is the
**mathematics and the interpretation**, including two places where the project's own language
has been misleading.

---

## 1. Decoding: what Viterbi actually computes

You asked how the decoding is done. It matters, because "the decoded state sequence" is the
scientific object — the ARI, `forward_fraction` and every state figure are computed from it.

There are **two different things** you can ask a fitted HMM, and they do not agree.

**(a) The per-bin posterior (forward–backward).** For each bin independently,

$$\gamma_t(k) \;=\; P(z_t = k \mid x_{1:T})$$

computed from the forward variable $\alpha_t(k) = P(x_{1:t}, z_t = k)$ and the backward variable
$\beta_t(k) = P(x_{t+1:T} \mid z_t = k)$ as $\gamma_t \propto \alpha_t \beta_t$. This is what
`occupancy` is built from.

**(b) The most likely *path* (Viterbi).** The single sequence maximising the joint posterior:

$$z^{*}_{1:T} \;=\; \arg\max_{z_{1:T}} \; P(z_{1:T} \mid x_{1:T})$$

by dynamic programming, in log space:

$$\delta_t(k) \;=\; \max_{j}\Big[\delta_{t-1}(j) + \log A_{jk}\Big] \;+\; \log P(x_t \mid z_t=k),
\qquad \psi_t(k) \;=\; \arg\max_j \Big[\delta_{t-1}(j) + \log A_{jk}\Big]$$

with $\delta_1(k) = \log \pi_k + \log P(x_1 \mid z_1 = k)$, then backtrack from
$\arg\max_k \delta_T(k)$ through $\psi$.

**These are not the same, and the difference is not pedantic.** Taking $\arg\max_k \gamma_t(k)$
independently at each bin can produce a path the model considers **impossible** (it can step
through a transition with $A_{jk} = 0$). Viterbi is constrained to a *legal, coherent
trajectory* — which is exactly what we want when the claim is about a **sequence**.

We use Viterbi (`hmmlearn`'s `model.predict()`). `hmmlearn`'s likelihood and Viterbi path were
validated against **dynamax**: LL agrees to $3.6\times10^{-16}$ relative, and the Viterbi paths
agree on **100 %** of bins (`scripts/validate_hmm_vs_dynamax.py`).

---

## 2. State ordering: how "forward" is defined (your question 5)

Your intuition — *"the discrete latent evolves and it changes every trial, so how can there be
an order?"* — is exactly right about the **paths**, and the answer is that the ordering is a
property of the **ensemble**, computed once after fitting.

Concretely, in `hmm.py::_order_by_time`:

1. **Decode every trial** with Viterbi → `states`, shape $(n_{\text{trials}}, n_{\text{bins}})$.
   Each trial has its own path; they differ.
2. **Average over trials** to get the occupancy matrix, shape $(K, n_{\text{bins}})$:

$$\mathrm{occ}_k(t) \;=\; \frac{1}{n_{\text{trials}}}\sum_{i=1}^{n_{\text{trials}}}
\mathbb{1}\!\left[z^{(i)}_t = k\right]$$

   i.e. **the fraction of trials that are in state $k$ at bin $t$.**
3. **Take the occupancy centre of mass** of each state:

$$\mathrm{com}_k \;=\; \frac{\sum_t t \cdot \mathrm{occ}_k(t)}{\sum_t \mathrm{occ}_k(t)}$$

   The denominator $\sum_t \mathrm{occ}_k(t)$ is the expected number of bins per trial spent in
   state $k$, so $\mathrm{com}_k$ is the **occupancy-weighted mean time at which the population
   is in state $k$**, pooled over all trials.
4. **`argsort(com)`** → relabel so state 0 is the earliest. Everything (λ, A, π, the Viterbi
   paths) is permuted consistently.

So: **it is a mean, over trials, of when each state is occupied.** Not per trial. It is computed
once, after fitting, from the decoded paths.

⚠️ **Known fragility:** when two states have similar mean times the sort **flips** between fits.
That is why the stability analysis must use a permutation-invariant metric (§4) — a metric that
cared about label identity would score a mere relabelling as a disagreement.

---

## 3. Directedness — and two things it is **not** (your questions 6 and 11)

### 3.1 The statistic

From the Viterbi paths, count within-trial transitions $N_{ij}$, zero the diagonal, and take

$$f \;=\; \frac{\sum_{i<j} N_{ij}}{\sum_{i<j} N_{ij} \;+\; \sum_{i>j} N_{ij}}$$

Because states are time-ordered, the upper triangle *is* "forward". $f = 0.5$ → symmetric,
$f \to 1$ → a strict forward chain.

### 3.2 ⚠️ It is NOT a transition *rate* — and this correction matters for your reading

> You wrote: *"VISp is less directed means it is transitioning to the other areas less
> frequently on average?"*

**Two things there need correcting, and the second is the important one.**

**(a) There are no cross-area transitions in this model. At all.** The HMM is fitted
**independently, per region**. VISp's states are rate vectors over VISp's 19 neurons; MRN's are
over MRN's 54. Nothing in the model connects them, and `forward_fraction` is entirely about a
region's **own internal state sequence**. So "transitioning *to the other areas*" is not a
quantity this model represents — the phrase has no referent here. (Testing inter-areal exchange
needs a **joint multi-region model**, which is the open Phase-3 item, or a cross-area
state-occupancy correlation.)

**(b) Directedness is an *asymmetry*, not a *frequency*.** VISp at $f = 0.53$ means: **when VISp
changes state, it goes forward about as often as it goes backward.** It says **nothing** about
how *often* VISp changes state. A region can:

| | switches often | switches rarely |
|---|---|---|
| **directed** | high switch_rate, $f \approx 1$ | low switch_rate, $f \approx 1$ |
| **undirected** | high switch_rate, $f \approx 0.5$ ← *VISp could be here* | low switch_rate, $f \approx 0.5$ |

Your parenthetical — *"there is only one main probable transition and the rest is pretty
static"* — describes the **bottom-left** cell, which would give a **high** $f$, not 0.53. So
that reading is not what the number says.

**Fix applied:** `hmm.py::switch_rate` now reports **state changes per trial** as a separate
statistic, and the bin-size check prints both side by side. They must never be quoted alone
again.

### 3.3 Your question 6: can we mitigate the ordering circularity by normalising by dwell time?

**Normalising by mean permanence time would change nothing** — because dwell time **already does
not enter** the statistic. `_transition_counts` zeroes the diagonal, so self-transitions are
discarded: a state entered once and held for 10 bins contributes exactly **one** off-diagonal
exit, the same as a state held for 2 bins. The statistic is already dwell-invariant.

But the concern *underneath* your question is real, so here is what actually addresses it. The
worry is that the ordering is derived from **trial-averaged** occupancy while the transitions
scored are **single-trial** — so we order the states to be time-ordered, then test whether
transitions respect that ordering. It is not vacuous (a population *can* have states with
distinct mean times and still hop back and forth on single trials — that is precisely how VISp
gets 0.53), but it is close enough to circular to deserve a clean test. Three fixes, all now
implemented:

| fix | function | what it closes |
|---|---|---|
| **Split-half ordering** | `forward_fraction_split_half` | Fit on half the trials, **fix the ordering from that half**, decode the *other* half and score its transitions. The ordering never sees the transitions it is judged on. **This is the real answer to your question.** |
| **Per-trial distribution** | `forward_fraction_per_trial` | Is the sequence there on *single trials*, or only in the pooled average? |
| **Direction-agnostic check** | `rslds.order_consistency` applied to the HMM's Viterbi paths | Asks *"is there a reliable order at all"* without imposing one |

If the split-half value tracks the pooled one, the pooled number was honest. If it collapses to
~0.5, the directedness was an artefact of ordering-then-testing-the-ordering.

---

## 4. The Rand Index and the ARI (your question 9)

### 4.1 ⚠️ First, the misconception — because it is the crux

> You asked: *"Given that it ignores the label, it just decodes which is the most probable state
> and thus tells which states are the same regardless of the label?"*

**No — and this is the whole elegance of it. The ARI never matches labels at all.** It does not
work out that model A's state 2 "is" model B's state 3. It never tries.

Instead it compares the two **partitions of time** by looking at **pairs of time bins**, a
description in which labels simply do not appear.

*(The thing you described — figuring out which state corresponds to which — **is** a real metric
in our code, but it is a **different** one: `matched_acc`, which Hungarian-matches the labels to
maximise the confusion-matrix diagonal. It is reported because it is interpretable ("the two
models agree on 91 % of bins"), but it is **not chance-corrected** and rises trivially as K
falls, so it is a read-out, never the criterion. See §4.4.)*

### 4.2 The Rand Index

Take any two time bins $i$ and $j$. Each decoding makes exactly one **label-free** statement
about that pair: *same state* or *different states*. Two decodings **agree** about the pair if
they both say "same", or both say "different". Count over all $\binom{n}{2}$ pairs:

| | decoding 2: same | decoding 2: different |
|---|---|---|
| **decoding 1: same** | $a$ (agree) | $c$ |
| **decoding 1: different** | $d$ | $b$ (agree) |

$$\mathrm{RI} \;=\; \frac{a+b}{a+b+c+d} \;=\; \frac{a+b}{\binom{n}{2}}$$

**No state name ever enters.** Relabel either decoding however you like — $a, b, c, d$ are
untouched. That is the permutation-invariance, and it is *structural*, not something we had to
engineer.

### 4.3 Why raw RI is unusable, and the chance correction

RI is badly inflated by chance, **and inflated differently at different K** — fatal, since
comparing across K is the entire point of the sweep. The reason: as K grows, two *random*
decodings put most pairs in "different states" simply because there are more states to fall
into, and those accidental different/different pairs count as agreement in $b$. So RI drifts
toward 1 with K for purely combinatorial reasons.

The ARI subtracts the agreement expected **by chance** and rescales:

$$\mathrm{ARI} \;=\; \frac{\mathrm{RI} - \mathbb{E}[\mathrm{RI}]}{\max(\mathrm{RI}) - \mathbb{E}[\mathrm{RI}]}$$

Concretely, build the **contingency table** $n_{ij}$ = number of bins that decoding 1 puts in
state $i$ and decoding 2 puts in state $j$; let $a_i = \sum_j n_{ij}$ and $b_j = \sum_i n_{ij}$
be the row/column sums. Then

$$\mathrm{ARI}=\frac{\displaystyle\sum_{ij}\binom{n_{ij}}{2}-\left[\sum_i\binom{a_i}{2}\sum_j\binom{b_j}{2}\right]\Big/\binom{n}{2}}
{\displaystyle\frac12\left[\sum_i\binom{a_i}{2}+\sum_j\binom{b_j}{2}\right]-\left[\sum_i\binom{a_i}{2}\sum_j\binom{b_j}{2}\right]\Big/\binom{n}{2}}$$

The expectation is taken under the **hypergeometric / permutation null**: random labellings
*with the same cluster sizes*. That null is why the correction is not merely "subtract 1/K" —
it accounts for the actual, possibly very unequal, occupancy of each state.

Properties, and why each is load-bearing:

| property | value | why we need it |
|---|---|---|
| permutation-invariant | — | state labels are arbitrary |
| chance level | **0** | so "no reproducible structure" has a *name* — the criterion can say **no** |
| identical partitions | **1** | the upper bound means something |
| **comparable across K** | — | the chance correction removes the K-dependent inflation. **This is the only reason the ARI-vs-K curve can be read at all.** |
| can go negative | yes | worse than chance; worth flagging if it ever happens |

`sklearn.metrics.adjusted_rand_score` implements this; we use it directly.

### 4.4 The second metric, and why it is not the criterion

`matched_acc`: Hungarian-match the two label sets (`scipy.optimize.linear_sum_assignment` on the
confusion matrix) to maximise the diagonal, then report the fraction of bins agreeing. **This is
the "which state is which" computation you had in mind.** It is interpretable — but it is *not*
chance-corrected, and under the null it *falls* with K for free, so read alone it would always
argue for the smallest K. **Read it next to the ARI, never instead of it.**

---

## 5. Correlated Poisson emissions — is there such a model? (your question 8)

**Yes, several — and you already have the best one for this project.** Not a detour; just worth
writing down. The Poisson HMM's fatal assumption is
$P(x_t \mid z_t=k) = \prod_n \mathrm{Poisson}(x_{nt};\lambda_{kn})$: **conditionally independent
neurons**. Four families relax it, in increasing order of relevance to us:

| model | how it adds neuron–neuron dependence | relevance here |
|---|---|---|
| **Negative-binomial HMM** | replaces Poisson with NB, so Var > Mean | fixes **over-dispersion**, *not* cross-neuron correlation. Half a fix. |
| **HMM with GLM emissions + coupling filters** (Escola & Paninski, *Neural Computation* 2011) | each state's emission is a GLM per neuron, with explicit **cross-neuron coupling terms** | the direct, literal answer to your question |
| **PLDS** — Poisson Linear Dynamical System (Macke et al. 2011; Buesing et al. 2012) | a **shared continuous latent** $x_t$ drives all neurons: $y_t \mid x_t \sim \mathrm{Poisson}(\exp(Cx_t + d))$. Marginalising $x_t$ induces correlations across neurons | the standard fix in this literature |
| **Poisson rSLDS** (Linderman et al. 2017) | PLDS **plus** discrete switching regimes | **this is exactly `ssm`'s Poisson backend — which is already in the project** |

> **So the answer is: the model you are asking about is the one we already planned to run.**
> `ssm`'s Poisson rSLDS has a continuous latent that absorbs shared variability *directly*,
> leaving the discrete variable free to describe **dynamics** instead of **covariance**. That is
> not a refinement of the HMM — it is the repair of its central flaw, and it is why
> [`choosing_K_for_the_HMM.md`](choosing_K_for_the_HMM.md) §4 calls the rSLDS a *necessity*.
>
> **Recommendation: do not implement anything new.** Note it, and let the existing `ssm` path
> carry the argument. If a reviewer asks "your HMM assumes conditional independence", the answer
> is *"yes — and that is precisely why the state-space model, which does not, is the fit of
> record."*
>
> *(Citations above are from memory and should be verified against the actual papers before any
> of them goes into a manuscript.)*

---

## 6. On your interpretation of the 3 states (your question 10)

Your reading — VISp: baseline / new-input / sustained-grating; SCm & MRN: richer back-and-forth
exchange — is **plausible and, importantly, testable**. It is not what the data currently says,
and here is the gap between the two.

**What would test it, cheaply:** the fitted $\lambda_k$ (each state's rate vector) and the
occupancy time-courses $\mathrm{occ}_k(t)$. If VISp's three states really are baseline →
transient → sustained, the occupancy plot shows it directly: state 0 dominant pre-stimulus,
state 1 a brief post-onset bump, state 2 rising and holding. That is a figure we already produce
and should now read carefully rather than glance at.

**Three cautions before you commit to the story:**

1. **VISp's three states are weakly supported.** Its ARI curve **declines** monotonically from
   K=2 (0.976 → 0.883 → 0.847 → …). There is no interior peak; "K=3" is just the best point on a
   falling curve (see §7.2). The three-state description may not be reproducible enough to hang a
   mechanism on.
2. **A "sustained" state may be a movement state.** Median RT is **171 ms**. By the time your
   "sustained grating representation" would appear, the mouse is already turning the wheel on
   most trials.
3. **Your SCm/MRN story ("switch more often") is the conflation corrected in §3.2.** Directedness
   is *not* switch frequency. If your hypothesis is genuinely about back-and-forth **exchange**,
   the statistic you want is a **high switch rate with a LOW forward_fraction** — and SCm/MRN
   show the *opposite* ($f$ = 0.77–0.80, i.e. strongly **one-directional**). As measured, the
   motor areas look *less* like back-and-forth exchange than VISp does, not more.

---

## 7. The limitations, addressed one by one (your question 12)

### 7.1 Recompute `forward_fraction` at the final K (3 / 4 / 3)

**Done** — `scripts/run_binsize_check.py` fits at K = 3/4/3 and reports `forward_fraction`,
`forward_fraction_split_half`, `switch_rate` and the robust baseline-exit latency, at four bin
sizes. The previously quoted values came from fits at K=2 and K=10, neither of which is the
chosen K.

### 7.2 Why VISp "does not peak", and where the 3 came from

The rule used is *"take the best **non-trivial** (K ≥ 3) peak"*, because K=2 is a trap (§7.4).
For VISp the ARI curve is **monotonically declining** from K=2, so the argmax over K ≥ 3 lands on
**the first candidate, K = 3, by construction**. It is not a discovered structure; it is *the
smallest non-trivial K*, and it would have been "3" whatever the curve did after that.

By the method note's own reading guide, a monotone decline from K=2 means: **the only structure
two independent fits agree on is the coarsest split** (baseline vs evoked). So the honest
statement about VISp is *"~2 reproducible states, and no evidence of a richer sequence"* — with
the caveat that ARI 0.883 at K=3 is still a *high* absolute value, so the three-state description
is not refuted either. It is **unresolved**, and your suspicion that the session is simply poor
for VISp (19 units at ≥1 Hz, median 1.1 Hz) is entirely consistent with the evidence.

### 7.3 How to strengthen the claim about MRN's K

First, the honest diagnosis: MRN's K=3 (ARI 0.8819) versus K=2 (0.8802) differ by **0.0017**,
against fold SDs of **0.04–0.05**. **This is not a precision problem.** More repeats would shrink
the error bars on each point, but the two points are *genuinely equally reproducible* — the
models are tied, not noisily measured. Grinding more repeats through the same session will not
break the tie.

What would actually strengthen it, in order of value:

1. **Replication across the 5 triplet sessions.** *"K = 3 comes out in 5 of 5 sessions"* is a far
   stronger claim than any tightening of the error bar in one. This is the real answer.
2. **Ask whether the third state is a state at all.** Independent of ARI: does it occupy a
   non-trivial fraction of bins? Is its $\lambda_3$ actually distinct from $\lambda_1, \lambda_2$
   (e.g. cosine distance between rate vectors)? Is its *entry time* reproducible across trials?
   A "state" that is reproducible but occupies 2 % of bins and has a rate vector nearly identical
   to another one is a numerical artefact wearing a state's clothes.
3. **A paired test.** Use the *same* A/B/C splits for K=2 and K=3 and test the paired difference,
   rather than comparing two independently-noisy means. Cheap, and it is the statistically
   correct comparison. (It will very likely confirm "no difference".)

### 7.4 Stability vs correctness (your question 12.4)

**Stability asks "do two independent fits agree?" Correctness asks "is the description true?"
A model can be *stably wrong*.**

The concrete failure: this data contains one huge, simple event — a **rate step at stimulus
onset**. A K=2 model only has to find *that*. Both independent fits will find it, every time, on
any split of the trials. So ARI at K=2 is ≈ 0.98 — **near-perfect reproducibility, and
scientifically empty.** It validates nothing whatsoever about metastable attractors; it says the
stimulus drives the neurons, which we knew.

Hence the rule: **"Is the ARI high?" is the wrong question. "How far up in K does it *stay* high?"
is the right one** — that asks *how many* states reproduce, which is the actual claim.

Stability is a **necessary, not sufficient** condition for a real state. It is a filter that
removes fitting artefacts; it cannot tell you the surviving states mean anything.

### 7.5 ⚠️ Pooling the sessions — your pseudo-population idea needs a correction

You wrote: *"the only way we can have more trials maintaining the A/B/C split is to do a
pseudo-population with the other sessions."*

**A pseudo-population will not work for the HMM or the rSLDS**, and the reason is the same one
that made us pick these regions in the first place.

A pseudo-population stitches neurons from *different sessions* into one population vector. But
those neurons **were never recorded at the same time**: session 1's trial 7 and session 2's trial
7 are unrelated events in different mice. A pseudo-population therefore has **no single-trial
joint activity** — and single-trial joint activity is *exactly* what a state-space model
consumes. The whole reason the project restricted itself to simultaneously-recorded triplets
(Phase 1) was to have it. Building a pseudo-population would throw away the property we selected
the sessions for. It does not add trials to the HMM; it destroys what a trial *is*.

**What is actually valid, per method:**

| method | can you pool across sessions? | how |
|---|---|---|
| **HMM, rSLDS** | ❌ not as a pseudo-population | ✅ **fit each session separately, pool the *statistics*.** "K = 3 in 5/5 sessions; forward_fraction 0.77 ± 0.04 across 5 sessions" — a **replication / meta-analysis**, and a much stronger claim than one session with tighter error bars. |
| **dPCA** | ✅ **yes** | dPCA works on **trial-averaged** activity per condition, so neurons from different sessions can be stacked. This is standard, and it **directly fixes the K = 25 trials/cell problem**. |
| **Decoding / geometry** | ⚠️ partly | fine for condition-averaged geometry; not for single-trial trajectories. |

Your instinct that *"the claim should depend on properties of the areas, not be animal-specific"*
is exactly right — and **replication across the 5 sessions is how you demonstrate that**, not
pseudo-population pooling. If the same K and the same directedness ordering appear in 5 mice,
that *is* the animal-independence claim, made properly.

⚠️ **This also corrects the project's own docs.** Several worklogs and `CLAUDE.md` say *"pool the
5 replicate sessions — add trials, not filters"*, which is too loose. It is right for dPCA and
wrong for the HMM/rSLDS. Corrected here; `CLAUDE.md` should be updated.

### 7.6 Keep the HMM cheap; let the rSLDS carry the refinement

Agreed, and it is now the working rule. The HMM's job is a **fast, honest proxy**: how many
reproducible states, and are they traversed directedly. The dynamical claims — attractor vs
saddle, sequence vs stopwatch, position-dependent switching — are the **rSLDS's** job and cannot
be answered by an HMM at any cost. Everything added today (`switch_rate`,
`forward_fraction_split_half`, `baseline_exit_latency_robust`) is seconds-to-minutes, not hours.

---

## 8. What was fixed in the code today

| function | status | why |
|---|---|---|
| `baseline_exit_latency` | **deprecated** | returned `NaN` in **2 of 3 regions**: it hard-coded baseline = state 0, and demanded a *single* state hold > 50 % of pre-stimulus bins |
| `baseline_states` | **new** | finds the baseline **set** empirically (smallest set covering 50 % of pre-stim occupancy) |
| `baseline_exit_latency_robust` | **new** | relative threshold — baseline falls to **half of its own pre-stimulus level**. Meaningful whether that level was 0.95 or 0.45, which is why the old absolute-0.5 rule failed |
| `switch_rate` | **new** | state changes per trial. **Directedness is not switch frequency** (§3.2) |
| `forward_fraction_per_trial` | **new** | is the sequence there on single trials, or only pooled? |
| `forward_fraction_split_half` | **new** | ordering derived on **held-out** trials — closes the ordering circularity (§3.3) |
