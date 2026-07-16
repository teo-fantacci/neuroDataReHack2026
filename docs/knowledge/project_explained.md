# The project, explained — answers to your questions

A pedagogical companion, organised around **your own questions**, across the three parts of the
project. Written to be *read*, not just referenced: each answer explains the idea, the maths where
it matters, what we actually did, and — honestly — what is verified versus assumed.

- **Part 1 — Preprocessing & the HMM** (this prompt) — filled.
- **Part 2 — rSLDS** — *to be written when you ask about it.*
- **Part 3 — Dimensionality reduction** — *to be written when you ask about it.*

Related docs (deeper or complementary):
[`where_we_are_HMM_rSLDS_dimred.md`](where_we_are_HMM_rSLDS_dimred.md) (synthesis by *topic*),
[`HMM_maths_and_readouts.md`](HMM_maths_and_readouts.md) (the HMM maths in full),
[`choosing_K_for_the_HMM.md`](choosing_K_for_the_HMM.md),
[`ARI_and_state_stability.md`](ARI_and_state_stability.md),
[`best_K_per_region.md`](best_K_per_region.md).
Code + figures: `src/data/preprocess.py`, `notebooks/00_preprocessing.ipynb`,
`results/figures/phase0_preproc/`.

> **A note on honesty.** Several of my first answers in this thread were *wrong* and got corrected
> by actually running the check. Where that happened, this document gives the corrected answer and
> says so — because the correction is usually the more interesting result.

---

# Part 1 — Preprocessing & the HMM

The overarching thing you asked for: **one preprocessing function, one frozen dataset, so every
model eats the same data.** That is now `src/data/preprocess.py` + `notebooks/00_preprocessing.ipynb`,
which write frozen tensors to `data/processed/*.npz` with the config serialised *inside* the file
(so a tensor can never be mis-described later). Everything below is a decision that function bakes
in, plus the HMM questions.

---

## 1. Input to the models: the Anscombe transform

**Your question:** what is Anscombe in detail, and can you show raw vs. smoothed so it's clear what
we feed the models?

### 1.1 What it is, and the problem it is meant to solve

The Anscombe transform is

$$y \;=\; 2\sqrt{x + \tfrac{3}{8}}.$$

The problem it targets: for a Poisson variable, **the variance equals the mean**. So in raw spike
counts a 20 Hz neuron is not merely *louder* than a 2 Hz neuron — it is intrinsically **noisier**.
Any method built on squared error (PCA, least squares, a Gaussian likelihood) will then spend its
variance budget chasing the shot noise of the loud units.

Anscombe (1948) is the classical *variance-stabilising* transform for the Poisson family. Applying
the delta method to $y = 2\sqrt{x+c}$ with $x\sim\mathrm{Poisson}(\lambda)$ gives

$$\mathrm{Var}[y] \;=\; 1 + O(1/\lambda),$$

and $c = 3/8$ is the offset that **cancels the leading $1/\lambda$ correction term**. So
*asymptotically* every neuron's noise becomes about one unit, whatever its firing rate. Note this is
different from z-scoring: z-scoring rescales each neuron by its own SD (fixes the *scale*), but the
count distribution stays skewed and heteroscedastic (Anscombe fixes the *shape*). In the geometry
pipeline we do both, in that order.

### 1.2 ⚠️ The correction: at *our* firing rates, Anscombe does not stabilise anything

This is the important part, and it is the opposite of what I first told you. `Var → 1` is an
**asymptotic** (large-$\lambda$) result. Before trusting it, you have to ask where "large" begins —
and where IBL sits. I put that check in the notebook (Part 2), on **simulated Poisson where the
truth is known**:

| $\lambda$ (spikes/bin) | 0.05 | 0.1 | 0.3 | 0.8 | 2.0 | **4.0** | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| $\mathrm{Var}[\text{Anscombe}]$ | 0.06 | 0.12 | 0.31 | 0.63 | 0.93 | **1.00** | 1.00 |

`Var → 1` only arrives at **$\lambda \gtrsim 4$ spikes/bin**. And IBL runs at:

| | VISp | SCm | MRN |
|---|---:|---:|---:|
| $\lambda$ @ 20 ms | 0.046 | 0.331 | 0.276 |
| $\lambda$ @ 50 ms | 0.116 | 0.827 | 0.689 |

We are **one to two orders of magnitude below** the regime where the transform works. On the real
tensor it produces median $\mathrm{Var}\approx 0.09$, not 1, and the mean–variance dependence is
untouched (Spearman $+0.99$).

This is **structural, not a tuning failure**: at $\lambda \ll 1$ a spike count is essentially
**Bernoulli** (0 or 1), and a Bernoulli's variance $p(1-p)\Delta^2$ depends on $p$ irreducibly — *no*
monotone transform can stabilise it. Variance stabilisation is simply not available in this data
regime.

### 1.3 So what do the models actually receive? (the raw-vs-smoothed figure)

The figure `input_chain_one_unit.png` shows one real unit, one real trial, through the whole chain:

1. **Raw integer counts** — mostly 0 and 1 (e.g. `0 2 0 0 0 1 1 0 …`). **This is what the Poisson
   HMM gets**, unaltered. The HMM's emission model *already* knows Var = Mean, so you must never
   pre-Anscombe it (you'd correct for the same thing twice).
2. **After Anscombe** — a *monotone relabelling* of the same counts: 0 → 1.225, 1 → 2.345,
   2 → 3.082. The data are still essentially binary. Anscombe did **not** turn the spike train into a
   smooth signal.
3. **After Gaussian smoothing (σ = 40 ms)** — *now* it looks like a continuous latent. **All of the
   continuity comes from the smoother**, not from Anscombe. This is what the rSLDS / PCA / geometry
   receive.

### 1.4 Bottom line, and why we keep Anscombe anyway

The table in your prompt (`counts → Anscombe → PCA`, `counts → √/Anscombe`) is what the geometry and
rSLDS pipelines do. We keep Anscombe — but for two *smaller*, honest reasons, not the advertised one:
it **compresses the high-count tail** (one occasional 7-spike bin stops dominating a squared-error
objective), and the rSLDS ground-truth calibration was done with it (changing it now would
invalidate that for no gain).

The real consequence is a scientific one: our fast rSLDS backend's Gaussian/PCA latent rests on a
variance-stabilising step that **stabilises nothing here**, which is a *second, independent* argument
(on top of "PCA finds variance, not dynamics") that the **Poisson `ssm` backend is the principled
model and the PCA latent is a concession**. It also explains, mechanistically, why the 20 ms latent
came out shot-noise dominated. (More on that in Part 2.)

---

## 2. Unit selection: a rate threshold, and why not MSR

**Your question:** a threshold to raise SNR — 1 or 2 Hz depending on how many units we lose, or drop
the lowest 10–20 % by MSR. Show the numbers.

### 2.1 Why any threshold at all

IBL "good" units are **well-isolated, not active** — `qc == 1.0` is a spike-*sorting* criterion.
VISp here contains units firing well under 1 Hz. Keeping them is not neutral for a Poisson HMM: each
adds $K$ rate parameters fitted to a handful of spikes, and dozens of near-silent units spend the
held-out likelihood on trivially predictable zeros, masking the units that carry the dynamics.

### 2.2 The decision: **1 Hz, on firing rate. Not MSR.**

The yield table (`unit_yield.png`) reports units kept **and spikes kept** — because "we kept 44 of 59
units" hides whether the dropped ones carried any data:

| threshold | VISp units | SCm units | MRN units | VISp spikes kept |
|---|---:|---:|---:|---:|
| none | 36 | 59 | 63 | 100 % |
| **1 Hz** | **19** | **44** | **54** | **96.7 %** |
| 2 Hz | **10** | 41 | 48 | 81.0 % |

- **1 Hz is nearly free**: it keeps 96.7 / 99.5 / 99.4 % of the spikes. The dropped units carry
  almost no data — which is exactly the case for dropping them.
- **2 Hz would halve VISp** (19 → 10 units, and lose 19 % of its spikes), gutting the region that is
  already the project's weakest and the anchor of every cross-area comparison, while SCm/MRN barely
  move. So 2 Hz buys nothing where it is safe and costs everything where it is not.

### 2.3 Why not MSR (three reasons, in order of force)

MSR (Multiscale Relevance) is attractive because it is unsupervised. But:

1. **It is a rate cut in disguise, exactly where it would matter.** Spearman(rate, MSR) is **0.64 in
   VISp and 0.55 in SCm**, so the bottom-MSR units *are* largely the quiet units. It buys the
   confound without buying independence. (`msr_vs_rate.png` shows the overlap.)
2. **It was already tested and rejected** (2026-07-13): in MRN — the one region with a valid
   rate-matched control — the top-MSR half decoded *no better than a random half*.
3. **It penalises a genuine stereotyped burster** (a delta multiplicity distribution has relevance
   ≈ 0), and SC/MRN contain bursting cells.

A rate floor is arbitrary, but *transparently* arbitrary — its failure mode fits in one sentence.
It is a **modelling choice, not a truth**: recorded in the frozen config, and any headline result
should be re-checked at 0.5 / 1 / 2 Hz.

---

## 3. Bin size: can the HMM and the rSLDS share one?

**Your question:** 50 ms is good for the rSLDS, but would it hurt the HMM? Can we use one parameter —
30 or 40 ms if 50 is too much for the HMM and 20 too little for the rSLDS? Should we check?

**Yes, we should check, and we did.** The answer is **40 ms for the whole project.**

### 3.1 The tension

- The **rSLDS needs ≥ ~40 ms.** Its ground truth (Part 2) shows that at 20 ms the PCA latent is
  shot-noise dominated, $|\lambda|\to 1$, and *every* regime gets mislabelled "line attractor /
  integrator" — a spurious result that *looks like a finding*.
- The **HMM had run at 20 ms**, and all its published numbers came from there.

### 3.2 What the check found (`scripts/run_binsize_check.py`, at the final K = 3/4/3)

`forward_fraction`, `switch_rate` and the baseline-exit latency are **essentially flat across
20–50 ms** — so none of the HMM's results are a bin-size artefact. But there is **one cost**:

| SCm state stability (ARI) | 20 ms | 40 ms | 50 ms |
|---|---:|---:|---:|
| | 0.943 ± 0.013 | **0.938 ± 0.025** | **0.826 ± 0.129** |

At 50 ms there are only 24 bins, and fitting K = 4 reproducibly into 24 bins is tight, so SCm's
stability degrades. **40 ms keeps it intact** (0.938), leaves every other read-out unchanged, and is
still well below the fastest GPFA timescale (107 ms, Part 3). Independently, the rSLDS ground truth
(Part 2) passes cleanly at 40 ms in every region.

**So 40 ms is the sweet spot you intuited** — big enough for the rSLDS, small enough for the HMM.
The frozen tensors are built at 40 ms (30 bins over the window).

---

## 4. The baseline window, and the length after onset

**Your question:** is 0.5 s pre-stimulus a good baseline, or shorter like 0.3? (And the plots in `01`
disagreed — some showed 0.5, some 0.3; make it consistent.) Sub-question: why 0.8 s after onset — is
that the average trial length?

### 4.1 The baseline: **0.4 s**, and it is answerable *exactly*

IBL *enforces* a **quiescence period** before the stimulus — the mouse must hold the wheel still for
a randomly drawn interval. In this session that interval is drawn from **[0.400, 0.700] s** (median
0.515). So (`trial_timing.png`, panel c):

- a **0.4 s** baseline window lies inside the enforced-still period on **100 %** of trials;
- the old **0.5 s** window steps *outside* it on **44 %** of trials — for nearly half the trials,
  part of the "baseline" is a period the mouse was free to move.

And there is no reason to go *shorter* than 0.4 s: contamination from the previous trial is
impossible (the minimum gap from the previous feedback to this stimulus onset is **2.7 s**), so 0.3 s
would only throw away clean baseline. **0.4 s** it is, and it is now consistent everywhere (the
0.5/0.3 mismatch you spotted was exactly the kind of drift the unified pipeline removes).

### 4.2 ⚠️ The 0.8 s after onset is *not* the trial length — and this reframes several results

The 0.8 s had **no recorded justification** — it was a round number, and no, it is not the average
trial length (trials vary). When I actually measured the timing (`trial_timing.png`), the finding was
important enough that it changes how earlier results must be read:

| quantity | median | consequence over `[0, +0.8] s` |
|---|---:|---|
| **reaction time** (first movement − stimOn) | **171 ms** | movement has begun on **92 %** of trials |
| **feedback** (feedback − stimOn) | **427 ms** | the outcome has arrived on **83 %** of trials |

**The mice are fast.** The post-stimulus window is *dominated by movement execution and reward*, not
by decision formation. And in IBL **the choice *is* the wheel-turn direction**, so "decision" and
"movement direction" cannot be separated by wishing. Consequences, all adopted:

- We keep 0.8 s (for continuity) but **relabel it**: no state after ≈ +0.2 s may be called a
  "decision" state without further argument.
- We now build a **second, movement-aligned tensor** (`firstMovement_times`) alongside the
  stimulus-aligned one, as a standing control. Disagreement between the two alignments is
  information, not noise.
- Shortening the window would *not* rescue this (feedback has already arrived on 45 % of trials by
  +0.4 s) — the confound is in the task, not the window. The honest fix is the movement alignment.

*(A parallel analysis thread independently confirmed this: the HMM's first "evoked" transition is
largely movement-locked, not stimulus-locked.)*

---

## 5. State ordering: how "forward" is computed

**Your question:** how is the time-order of states computed? After initialisation the discrete latent
evolves and differs every trial — do you take a mean?

Your intuition is exactly right about the **paths** (they differ every trial), and the answer is that
the ordering is a property of the **ensemble**, computed *once after fitting*
(`hmm.py::_order_by_time`):

1. **Decode every trial** with Viterbi → `states`, shape (trials, bins). Each trial has its own path.
2. **Average over trials** to get the occupancy $\mathrm{occ}_k(t)$ = fraction of trials in state $k$
   at bin $t$.
3. **Occupancy centre of mass** of each state:

$$\mathrm{com}_k \;=\; \frac{\sum_t t\,\mathrm{occ}_k(t)}{\sum_t \mathrm{occ}_k(t)}$$

   — the occupancy-weighted mean time at which the population is in state $k$, pooled over all trials.
4. **`argsort(com)`** → relabel so state 0 is earliest; permute $\lambda, A, \pi$ and the paths
   consistently.

So: it is a **mean over trials of when each state is occupied**, computed once, not per trial. One
fragility worth knowing: when two states have similar mean times, the sort can *flip* between fits —
which is exactly why the stability metric (ARI, §9) must be permutation-invariant.

---

## 6. Can we mitigate "back-and-forth on single trials" by normalising by dwell time?

**Your question:** can we reduce the "states have distinct mean times but still hop back and forth on
single trials" problem by normalising by the mean permanence (dwell) time?

**Normalising by dwell time would change nothing — because dwell time already does not enter the
statistic.** `forward_fraction` counts *off-diagonal* Viterbi transitions with the diagonal zeroed
(`_transition_counts`), so a state entered once and held for 10 bins contributes exactly **one**
off-diagonal exit, same as one held for 2 bins. The statistic is already dwell-invariant.

But the concern *underneath* your question is real and I took it seriously. The worry is that the
ordering is derived from **trial-averaged** occupancy while the transitions scored are
**single-trial** — so we order states to be time-ordered, then test whether transitions respect that
ordering. It is not vacuous (a population *can* have states with distinct mean times and still hop
back and forth — that is exactly how VISp gets ≈ 0.53), but it is close enough to circular to deserve
a clean test. Three fixes, all now implemented in `hmm.py`:

| fix | function | what it closes |
|---|---|---|
| **Split-half ordering** | `forward_fraction_split_half` | fit on half the trials, **fix the ordering from that half**, score transitions on the *other* half. The ordering never sees the transitions it is judged on. **This is the real answer to your question.** |
| **Per-trial distribution** | `forward_fraction_per_trial` | is the sequence there on *single trials*, or only in the pooled average? |
| **Switch rate, separated** | `switch_rate` | directedness is an *asymmetry*, not a *frequency* — see §11 |

And it earned its keep: the split-half test showed **VISp and SCm are unchanged** (their directedness
is real), but **MRN drops from 0.79 to 0.67 ± 0.10** — part of MRN's apparent directedness *was* the
ordering-then-testing-the-ordering effect. So we now quote the split-half number for MRN.

---

## 7. Fixing `baseline_exit_latency` (the NaN)

**Your question:** fix the read-out that returned `NaN` for VISp and MRN (only SCm gave a number).

**Fixed.** The old function had two bugs, each a way to return `NaN`:

1. It **hard-coded the baseline as state 0** (the earliest-centre-of-mass state), but the
   earliest-on-average state is *not necessarily* the pre-stimulus one — for MRN the baseline state
   was state 1.
2. It demanded that **a single state** hold > 50 % of the pre-stimulus bins; when baseline is shared
   across two states (an ordinary fit, not a pathology), the guard fired and the answer was thrown
   away.

The replacement (`baseline_exit_latency_robust` + `baseline_states`) identifies the baseline **set**
from the data, and uses a threshold **relative to baseline's own pre-stimulus level** (fall to half
of it) — meaningful whether that level started at 0.95 or 0.45, which is exactly why the absolute-0.5
rule failed. It now runs in all three regions and gives the predicted cascade:

$$\text{VISp} \approx 90\ \text{ms} \;\rightarrow\; \text{SCm} \approx 170\ \text{ms} \;\rightarrow\; \text{MRN} \approx 450\ \text{ms}$$

⚠️ **But read it against §4.2 before celebrating:** SCm's exit (~170 ms) sits on the median *reaction
time* (171 ms), and MRN's (~450 ms) on the median *feedback time* (427 ms). So the cascade may be
each area locking to the task event it already locks to (vision → movement → reward), not an internal
decision sequence. The movement-aligned tensor is the control that separates these.

---

## 8. Is there a Poisson emission model *with* neuron dependence?

**Your question (not to implement, just answer):** is there a model that introduces dependency across
neurons in the Poisson emission?

**Yes — and you already have the best one for this project.** The Poisson HMM's fatal assumption is
that neurons are *conditionally independent given the state*. Four families relax it:

| model | how it adds neuron–neuron dependence | relevance |
|---|---|---|
| Negative-binomial HMM | NB instead of Poisson, so Var > Mean | fixes over-dispersion, *not* cross-neuron correlation — half a fix |
| GLM-HMM with coupling filters (Escola & Paninski 2011) | explicit cross-neuron coupling terms in each state's emission | the literal answer |
| **PLDS** (Macke et al. 2011) | a **shared continuous latent** drives all neurons; marginalising it induces correlations | the standard fix in this literature |
| **Poisson rSLDS** (Linderman 2017) | PLDS **+ discrete switching regimes** | **this is exactly `ssm`'s Poisson backend, already in the project** |

So the model you're asking about *is* the state-space model we already planned. Its continuous latent
absorbs shared variability directly, freeing the discrete variable to describe *dynamics* rather than
*covariance* — which is why the rSLDS is a **necessity, not a refinement**. Recommendation: implement
nothing new; note it, and let the `ssm` path carry the argument. *(Citations from memory — verify
before a manuscript.)*

---

## 9. The Rand Index and the ARI

**Your question:** what is the Rand Index and the maths behind it? Since it ignores labels, does it
decode which state is which (regardless of label)? And how is the decoding done, mathematically?

### 9.1 First, the decoding (Viterbi)

Two different things you can ask a fitted HMM, and they disagree:

- **Per-bin posterior (forward–backward):** $\gamma_t(k) = P(z_t=k \mid x_{1:T})$, computed from the
  forward $\alpha_t$ and backward $\beta_t$ variables as $\gamma_t \propto \alpha_t\beta_t$. This is
  what `occupancy` uses.
- **Most-likely *path* (Viterbi):** the single sequence maximising the *joint* posterior,

$$z^*_{1:T} = \arg\max_{z_{1:T}} P(z_{1:T}\mid x_{1:T}),$$

  by dynamic programming: $\delta_t(k) = \max_j[\delta_{t-1}(j) + \log A_{jk}] + \log P(x_t\mid z_t=k)$,
  then backtrack. We use Viterbi (`model.predict`), because taking the per-bin $\arg\max$
  independently can produce a path the model considers *impossible*; Viterbi is a coherent
  trajectory, which is what a claim about a *sequence* needs. (`hmmlearn`'s Viterbi was validated
  against dynamax — 100 % bin agreement.)

### 9.2 ⚠️ The misconception — the ARI never matches labels

You asked whether the ARI "decodes which state is which regardless of label." **No — and this is its
whole elegance. It never matches labels at all.** It does not work out that model A's state 2 "is"
model B's state 3; it never tries. (The thing you described — figuring out which state corresponds to
which — *is* in our code, but it's a *different* metric, `matched_acc`, via the Hungarian algorithm,
and it is a read-out, not the criterion — see §9.5.)

Instead the ARI compares the two **partitions of time** by looking at **pairs of time bins**, a
description in which labels do not appear.

### 9.3 The Rand Index

Take any two time bins $i, j$. Each decoding makes one **label-free** statement: *same state* or
*different states*. Two decodings **agree** about the pair if both say "same" or both say
"different". Across all $\binom{n}{2}$ pairs:

| | dec 2: same | dec 2: different |
|---|---|---|
| **dec 1: same** | $a$ (agree) | $c$ |
| **dec 1: different** | $d$ | $b$ (agree) |

$$\mathrm{RI} = \frac{a+b}{a+b+c+d} = \frac{a+b}{\binom{n}{2}}.$$

**No state name ever enters** — relabel either decoding and $a,b,c,d$ are untouched. That is the
permutation-invariance we need (state labels are arbitrary; §5).

### 9.4 Why raw RI is unusable, and the chance correction

RI is inflated by chance, **and inflated differently at different K** — fatal, since comparing across
K is the whole point. As K grows, two *random* decodings put most pairs in "different states" simply
because there are more states, and those accidental agreements swell $b$. So RI drifts toward 1 with
K for combinatorial reasons.

The **Adjusted** Rand Index subtracts the chance agreement and rescales:

$$\mathrm{ARI} = \frac{\mathrm{RI} - \mathbb{E}[\mathrm{RI}]}{\max(\mathrm{RI}) - \mathbb{E}[\mathrm{RI}]},$$

with the expectation under the **hypergeometric / permutation null** (random labellings *with the
same cluster sizes*). In contingency-table form, with $n_{ij}$ bins that decoding 1 puts in state $i$
and decoding 2 in state $j$, and row/column sums $a_i, b_j$:

$$\mathrm{ARI}=\frac{\sum_{ij}\binom{n_{ij}}{2}-\big[\sum_i\binom{a_i}{2}\sum_j\binom{b_j}{2}\big]/\binom{n}{2}}
{\tfrac12\big[\sum_i\binom{a_i}{2}+\sum_j\binom{b_j}{2}\big]-\big[\sum_i\binom{a_i}{2}\sum_j\binom{b_j}{2}\big]/\binom{n}{2}}.$$

Properties, each load-bearing: **chance = 0** (so "no reproducible structure" has a name — the
criterion can say *no*), **identical = 1**, **negative possible** (worse than chance), and — the one
that matters — **comparable across K** (the chance correction removes the K-dependent inflation, the
only reason the ARI-vs-K curve can be read). We use `sklearn.metrics.adjusted_rand_score`.

### 9.5 Why the ARI, and not likelihood, for choosing K

The claim is *reproducible metastable states*, so we measure reproducibility: fit two **independent**
HMMs on disjoint trial-thirds A and B, decode **both** on an unseen third C, and score their
agreement with ARI. This has a **peak** (too few states → trivial agreement; too many → surplus
states split arbitrarily by each fit → agreement collapses), whereas held-out likelihood climbs
monotonically with K and cannot say "there are no states". (Full argument:
[`ARI_and_state_stability.md`](ARI_and_state_stability.md).) `matched_acc` (the Hungarian
label-matching you had in mind) is reported alongside because it is interpretable, but it is *not*
chance-corrected, so it is a read-out, never the criterion.

---

## 10. Your interpretation of the 3 states

**Your reading:** VISp = baseline / new-input-to-relay / sustained-grating-already-transmitted; the
two motor areas switch more (back-and-forth exchange to process the information into the output.

It is **plausible and testable**, and it is not what the data currently says. The gap:

- **What would test it, cheaply:** the fitted $\lambda_k$ (each state's rate vector) and the occupancy
  time-courses. If VISp's three states really are baseline → transient → sustained, the occupancy plot
  shows it: state 0 dominant pre-stimulus, state 1 a brief post-onset bump, state 2 rising and
  holding. That figure exists — read it carefully rather than glance.
- **Three cautions:**
  1. **VISp's three states are weakly supported** — its ARI curve *declines* from K=2, so "K=3" is
     the best point on a falling curve, not an interior peak (§12.2).
  2. **A "sustained" state may be a movement state** — median RT is 171 ms (§4.2).
  3. ⚠️ **Your SCm/MRN story ("switch more") is the conflation in §11.** Directedness is *not* switch
     frequency. If your hypothesis is genuinely about back-and-forth *exchange*, you want **high
     switch rate with LOW forward_fraction** — but SCm/MRN show the *opposite* (high forward_fraction
     ≈ 0.8, strongly one-directional). As measured, the motor areas look *less* like exchange than
     VISp, not more.

---

## 11. What "VISp is less directed" means

**Your question:** does "VISp less directed" mean it transitions to the other areas less often on
average? And you'd expect the motor areas to transition to visual less often (unless those are
"checks" that nothing changed).

**Two corrections, the second important:**

1. **There are no cross-area transitions in this model — at all.** The HMM is fitted **independently,
   per region**. VISp's states are rate vectors over VISp's neurons; MRN's over MRN's. `forward_fraction`
   is entirely about a region's **own internal** state sequence. "Transitioning *to the other areas*"
   is not a quantity this model represents. (Testing inter-areal exchange needs a **joint
   multi-region** model — the open item — or a cross-area occupancy correlation.)

2. **Directedness is an *asymmetry*, not a *frequency*.** VISp at ≈ 0.53 means: *when VISp changes
   state, it goes forward about as often as backward.* It says **nothing** about how *often* VISp
   changes. Your reading ("one main transition, the rest static") would give a **high**
   forward_fraction, not 0.53.

And I checked this directly, because it is the exact confusion `switch_rate` was added to prevent.
Measured (`run_binsize_check.py`): **VISp switches state as often as MRN — 1.59 vs 1.46 changes per
trial — it simply switches with no preferred direction (0.59 vs 0.79).** So VISp is **undirected, not
static.** The two quantities dissociate cleanly, and the data settles the reading.

---

## 12. The limitations, one by one

**12.1 — Recompute `forward_fraction` at the final K (3/4/3).** *Done.* The old 0.53/0.77/0.69 came
from fits at K=2 and K=10; at the final K the values are essentially flat across 20–50 ms bins. And
the split-half version (§6) revealed MRN's directedness is partly circular (0.79 → **0.67 ± 0.10**),
while VISp (0.59) and SCm (0.83) are unchanged. Quote the split-half number for MRN.

**12.2 — Why VISp "does not peak", and where the 3 came from.** The rule is "best *non-trivial*
(K ≥ 3) peak", because K=2 is a trap (§12.4). VISp's ARI curve is **monotonically declining** from K=2
(0.976 → 0.883 → 0.847 → …), so the K≥3 argmax lands on the *first* candidate, K=3, **by
construction** — it is the smallest non-trivial K, not a discovered structure. The honest statement:
VISp has ~2 reproducible states and no evidence of a richer sequence, though ARI 0.883 is still high,
so the three-state description is unresolved rather than refuted. Your suspicion that the *session* is
simply poor for VISp (19 units at ≥1 Hz, median 1.1 Hz) is entirely consistent with this.

**12.3 — How to strengthen the claim about MRN's K.** First the honest diagnosis: MRN's K=3 (ARI
0.8819) vs K=2 (0.8802) differ by **0.0017** against fold SDs of **0.04–0.05**. This is **not a
precision problem** — the two models are *genuinely tied*, not noisily measured, so more repeats will
not break the tie. What would: (1) **replication across the 5 sessions** ("K=3 in 5/5" is far stronger
than any error bar in one) — the real answer; (2) ask whether the third state is a state at all (does
it occupy a non-trivial fraction of bins? is its $\lambda_3$ actually distinct? is its entry time
reproducible?); (3) a **paired** test using the *same* A/B/C splits for K=2 and K=3.

**12.4 — Stability vs correctness.** Stability asks "do two independent fits agree?"; correctness asks
"is the description true?" **A model can be *stably wrong*.** The data contain one huge, simple event
— a rate step at stimulus onset — and a K=2 model only has to find *that*; both fits will, on any
split, so ARI at K=2 ≈ 0.98 is near-guaranteed and scientifically empty. Hence the rule: not "is the
ARI high?" but "how far up in K does it *stay* high?" — that asks *how many* states reproduce, which
is the actual claim. Stability is **necessary, not sufficient**: it removes fitting artefacts; it
cannot tell you the survivors mean anything.

**12.5 — ⚠️ Your pseudo-population idea needs a correction.** You proposed a pseudo-population with the
other sessions to gain trials while keeping the A/B/C split. **This does not work for the HMM or
rSLDS.** A pseudo-population stitches neurons from *different sessions* into one population vector —
but those neurons were **never recorded simultaneously**, so there is **no single-trial joint
activity**, which is exactly what a state-space model consumes (and is the property Phase 1 selected
these sessions *for*). It doesn't add trials to the HMM; **it destroys what a trial is.** What is
valid:

| method | pool across sessions? | how |
|---|---|---|
| **HMM, rSLDS** | ❌ not as a pseudo-population | ✅ fit each session separately, **pool the *statistics*** ("K=3 in 5/5, forward_fraction 0.8 ± 0.04") — a *replication*, and it *is* your animal-independence argument, made properly |
| **dPCA** | ✅ yes | it works on trial-averaged activity, so a pseudo-population is standard and directly fixes its 25-trials/cell problem |

Your instinct — *"the claim should depend on properties of the areas, not be animal-specific"* — is
right, and **replication across the 5 sessions is how you demonstrate it**, not pooling. *(This also
corrected the project's own docs, which had said "pool trials, not neurons" too loosely.)*

**12.6 — Keep the HMM as a fast proxy; refine with the rSLDS.** Agreed, and it is the working rule.
The HMM's job is a fast, honest proxy (how many reproducible states, traversed how directedly). The
dynamical claims — attractor vs saddle, sequence vs stopwatch, position-dependent switching — are the
**rSLDS's** job and cannot be answered by an HMM at any cost. Everything added here (`switch_rate`,
`forward_fraction_split_half`, `baseline_exit_latency_robust`) is seconds-to-minutes, not hours.

---

## Part 1 — the frozen configuration, in one table

| knob | value | chosen because |
|---|---|---|
| alignment | `stimOn_times` **+ `firstMovement_times`** | the movement-window confound (§4.2) |
| `pre_s` | **0.4 s** | inside the enforced quiescence period on 100 % of trials (§4.1) |
| `post_s` | **0.8 s** | kept for continuity — but a *movement + outcome* window, not a decision window |
| `bin_size_s` | **0.04 s (40 ms)** | rSLDS ground truth + GPFA timescales, *and* preserves SCm's HMM stability (§3) |
| `min_rate_hz` | **1.0** | nearly free (96.7/99.5/99.4 % of spikes); 2 Hz halves VISp (§2) |
| transform on disk | **none** | HMM needs raw counts; Anscombe is applied downstream — and doesn't stabilise anything anyway (§1.2) |

Frozen tensors: **VISp 19 / SCm 44 / MRN 54** units, 843 trials, 30 bins.

---

# Part 2 — rSLDS

*To be written when you ask about it. It will cover: the SLDS/rSLDS model and what the recurrence
term means; why the flow geometry (`|λ|`, line-attractor labels) turned out to be a smoothing
artefact and how the shuffle-null test proved it; the recurrence test (`delta_recurrent`) and its
position-shuffle null; `order_consistency` vs `forward_fraction`; the two backends and why `ssm` is
the principled one; and the ground-truth calibration that fixed the bin size and the `smooth_sigma_ms`
API.*

# Part 3 — Dimensionality reduction

*To be written when you ask about it. It will cover: dPCA (variance by cause — the project's core
prediction); GPFA (inferred vs imposed timescales — the honest read on "how slow"); RRR /
communication subspace (the SCm↔MRN common-drive finding); the joint manifold + per-area partial
projections; and why t-SNE/UMAP clusters are not states.*
