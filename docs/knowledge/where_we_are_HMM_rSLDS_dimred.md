# Where we are — HMM, rSLDS, dimensionality reduction

Written 2026-07-14. A synthesis document, not a new result: it collects what the three
analysis strands **are** (the model, the maths, what each symbol represents), what they have
**actually produced so far**, how they **feed each other**, and where they are **weak**.

Everything below is traceable to code or to an executed run. Where a number is *not* yet
earned, it says so. Sources are linked at each claim.

**How to use the checklists.** Each section ends with a numbered list of concepts (`H1`,
`R4`, `D6`, `X2`…). Reply with the IDs you want expanded — e.g. *"expand H4 and R5"* — and I
will write those out properly rather than guessing what was unclear.

---

## 0. The substrate everything runs on

One session, `5c0c560e-9e1f-45e9-b66e-e4ee7855be84`, chosen in Phase 1 because IBL
co-records our visual→motor axis triplet `{VISp, SCm, MRN}` in only 5 sessions and this is
the working one. **843 usable trials**, aligned to `stimOn_times`.

| | units used | bin | window | input | smoothing |
|---|---:|---:|---|---|---|
| **HMM** (nb 01) | 20 / 44 / 55 (≥ 1 Hz) | 20 ms | −0.5 → +0.8 s | **raw integer counts** | none (forbidden) |
| **rSLDS** (nb 03) | 36 / 59 / 63 (all) | 20 + **50** ms | **−0.2 → +0.8 s** | counts → Anscombe → PCA | σ = 1.5 bins |
| **Geometry** (nb 02, 04) | 36 / 59 / 63 (all) | **50 ms** | −0.5 → +0.8 s | counts → √ / Anscombe | **σ = 40 ms** |

Units are VISp / SCm / MRN throughout.

⚠️ **Those three rows are not the same preprocessing, and two of the differences are
accidents rather than decisions.** The rSLDS notebook reads `pre_s` from
`config/params.yaml` (**0.2 s**) while notebooks 02/04/05 hard-code **0.5 s**; and the HMM
filters units at ≥ 1 Hz while the other two do not. See §5 — this matters most for the K
hand-off, which is the main bridge between topics 1 and 2.

---

## 1. The HMM — *where* the population is

Code: `src/dynamics/hmm.py` · Notebook: `02_hmm_triplet.ipynb` · Method notes:
[`choosing_K_for_the_HMM.md`](choosing_K_for_the_HMM.md),
[`ARI_and_state_stability.md`](ARI_and_state_stability.md),
[`best_K_per_region.md`](best_K_per_region.md)

### 1.1 The model

A hidden Markov model with **Poisson emissions**, per region, per time bin.

- Hidden state $z_t \in \{1,\dots,K\}$, one per 20 ms bin, evolving as a first-order Markov
  chain: $P(z_t = j \mid z_{t-1} = i) = A_{ij}$.
- Observations $x_t \in \mathbb{N}^N$ — the spike count of each of the $N$ neurons in bin $t$.
- Emission: **given the state, the neurons are independent Poissons**

$$P(x_t \mid z_t = k) \;=\; \prod_{n=1}^{N} \frac{\lambda_{kn}^{\,x_{nt}}\, e^{-\lambda_{kn}}}{x_{nt}!}$$

Parameters: $\lambda \in \mathbb{R}^{K \times N}$ (rate of each neuron in each state),
$A \in \mathbb{R}^{K\times K}$ (transitions), $\pi \in \mathbb{R}^{K}$ (initial state).
Fitted by EM (Baum–Welch), decoded by Viterbi (the single most likely state *path*, not the
per-bin marginal).

### 1.2 What each object *represents*

- **$\lambda_k$ is a state.** It is a vector of firing rates over the whole population — a
  *point in $N$-dimensional rate space*, a stereotyped activity pattern.
- **$z_t$ says which pattern is active now.** The model's entire claim about the data is:
  *at every instant the population sits in one of $K$ stereotyped rate patterns, and it hops
  between them.*
- **$A$ is the hopping rule** — memoryless, and (crucially) **not a function of anything**.
  It does not depend on time, on the stimulus, or on where the population actually is. This
  is the assumption the rSLDS exists to relax.

So the HMM answers ***where*** the population is, and *when it moves*. It says nothing about
***why*** it moves or ***what kind of dynamical object*** each state is.

### 1.3 Our two additions

**(a) Baseline-biased initial state.** Every trial's window starts 0.5 s before the stimulus,
so a resting regime genuinely exists in the data. After a first fit we measure the state
distribution over the pre-stimulus bins and *install it* as $\pi$, then re-decode. The
post-stimulus trajectory is then read as a **departure from baseline** rather than from an
arbitrary EM initialisation. It is a modelling choice, switchable (`bias_startprob=False`).

**(b) Time-ordering, which is what makes "forward" mean anything.** EM's state labels are
arbitrary. We order states by **occupancy centre-of-mass**,

$$\mathrm{com}_k = \frac{\sum_t t \cdot \mathrm{occ}_k(t)}{\sum_t \mathrm{occ}_k(t)}, \qquad
\mathrm{occ}_k(t) = P(z_t = k)\ \text{across trials},$$

and relabel so state 0 is the earliest. Now "later state" is well defined.

### 1.4 The read-outs

**`forward_fraction` — directedness.** From the *Viterbi* paths (not the fitted $A$ — the
empirically realised transitions are the more honest object), count within-trial transitions
$N_{ij}$, zero the diagonal, and take

$$f \;=\; \frac{\sum_{i<j} N_{ij}}{\sum_{i<j} N_{ij} + \sum_{i>j} N_{ij}}$$

Because states are time-ordered, the upper triangle *is* "forward". Equivalently, with net
flux $\Phi = N - N^{\top}$, $f = \tfrac12\left(1 + \Phi_{\text{fwd}}/\text{total}\right)$ —
the two descriptions are the same statistic.

- $f = 0.5$: transitions are symmetric. The population **wanders** among the states.
- $f \to 1$: a strict forward chain. The population **runs a sequence**.

Validated against synthetic forward / reversed / shuffled data, and against a **within-trial
bin-shuffle null**.

*The subtlety worth understanding (this is not circular, but it is close):* the ordering is
derived from **trial-averaged** occupancy, while $f$ is computed from **single-trial**
transitions. A population whose states have distinct mean times can still hop back and forth
on individual trials — and that is exactly what VISp does ($f \approx 0.53$). So the statistic
can, and does, return "no sequence".

**`baseline_exit_latency`** — the first post-stimulus time at which $P(\text{baseline state})$
stays below 0.5 for ≥ 3 consecutive bins. Intended as the inter-areal timing read-out (does
VISp leave baseline before SCm before MRN?). ⚠️ **It currently returns `NaN` for VISp and MRN**
(`results/hmm/phase2_final.csv`) because the guard `require_dominant_baseline` fires — the
baseline state was not dominant pre-stimulus in those regions. **Only SCm produces a number
(30 ms).** This read-out is, as of now, not working.

### 1.5 Choosing K — the part that took the longest and taught the most

**Held-out log-likelihood does not work here, and the reason is structural.** A Poisson HMM
assumes neurons are *conditionally independent given the state*. Real populations have
**noise correlations** and **over-dispersion** (variance > mean; Poisson forces variance =
mean). The model has exactly one mechanism to express any of that: **add another state**.
Mixing over more rate vectors is how a conditionally-independent model imitates correlation.

So held-out LL climbs with K *whether or not there are more real states*. Observed twice, in
opposite directions, on the same data:

- argmax → **K = 2** for every region (an artifact: every fold scored `-inf`, because
  `hmmlearn` fits $\lambda = 0$ for a neuron silent in training, and $\text{argmax}$ over a
  column of $-\infty$ returns index 0). Fixed with `LAMBDA_FLOOR = 1e-6` + a hard `raise`.
- argmax → **K = 10** for every region, once finite. The top of the search range.

Both are **boundaries, not answers**. The implementation is not at fault: `hmmlearn` was
validated against **dynamax** — held-out LL agrees to $3.6\times10^{-16}$ relative, Viterbi
paths agree on **100 %** of bins (`scripts/validate_hmm_vs_dynamax.py`).

**What we use instead: state stability (ARI).** The claim is *reproducible metastable states*,
so measure reproducibility:

1. Split trials into three disjoint thirds **A**, **B**, **C**.
2. Fit one HMM on **A**, a completely independent one on **B** (no shared trials, no shared
   parameters; each refitted `n_restarts` times, keeping the best-LL solution).
3. Decode **both** on the same unseen **C**. Two state sequences over the same bins.
4. Score their agreement.

The metric must ignore labels (model A's "state 3" has nothing to do with model B's). The
**Rand Index** does this by scoring *pairs of bins*: for bins $i,j$, each decoding says only
"same state" or "different state". But raw RI inflates with K for combinatorial reasons, so
we use the **Adjusted Rand Index**:

$$\mathrm{ARI} = \frac{\mathrm{RI} - \mathbb{E}[\mathrm{RI}]}{\max(\mathrm{RI}) - \mathbb{E}[\mathrm{RI}]}$$

with the expectation under random labellings with the same cluster sizes. Chance = **0**,
identical = **1**, and — the property that matters — **comparable across K**.

Why this works where likelihood fails: **it has a peak.** Too few states → both models agree
trivially but describe little. Too many → the surplus states are split arbitrarily and
*differently* by each fit, so agreement collapses. The extra "correlation-absorbing" states
cannot game it, because they are fitted to noise, and noise does not reproduce.

### 1.6 Where the HMM actually is — the numbers

All 843 trials, 20 ms, units ≥ 1 Hz, 5 repeats × 2 EM restarts
(`results/hmm/state_stability_maxtrials.json`):

| region | units | **K** | ARI at K | `forward_fraction` |
|---|---:|---:|---|---|
| VISp | 20 | **3** | 0.883 ± 0.019 | **0.53 – 0.55** (≈ undirected) |
| SCm | 44 | **4** | **0.920 ± 0.031** | **0.77 – 0.80** |
| MRN | 55 | **3** | 0.882 ± 0.052 | **0.69 – 0.80** |

**The result that survives everything: motor midbrain traverses its states far more
directedly than visual cortex.** VISp is at chance; SCm and MRN are not. The ordering is
stable across every K we have fitted, which is why it is quoted as a range.

**The result that was retracted: there is *no* state-count gradient.** An earlier pass claimed
VISp 2 < MRN 3 < SCm 4. With EM restarts the counts are **3 / 4 / 3** — VISp and MRN are
equal. The honest statement is: *the areas hold a similar number of states, but the motor
areas traverse them far more directedly.* Which arguably fits the project's framing better
(*shared alphabet, area-specific sequence*) — but it is a **different claim** from the one
that was briefly on the table.

### 1.7 Limitations — stated plainly

1. **`forward_fraction` has never been computed at the final K.** The values above come from
   fits at K = 2 (first pass) and K = 10 (`phase2_final.csv`, the elbow rule). Not at
   3 / 4 / 3. The *ordering* is robust across those, which is reassuring, but the definitive
   number is not yet on the table.
2. **VISp's ARI curve does not peak — it declines.** 0.976 (K=2) → 0.883 → 0.847 → 0.776 →
   0.672 … "K = 3" is *the best non-trivial point on a falling curve*, not an interior peak.
   Per the method note's own reading guide, a monotone decline from K=2 means *"the only
   structure two strangers agree on is the coarsest split"*. VISp should be treated as a
   region with weak reproducible state structure.
3. **MRN's K=3 is statistically indistinguishable from K=2.** ARI 0.8819 vs 0.8802 — a
   difference of **0.0017** against fold SDs of **0.04–0.05**. `best_K_per_region.md` says
   MRN's "non-trivial peak *beats* the trivial split"; that is numerically true and
   statistically empty. **Only SCm has a genuine interior peak** (a dip to 0.721 ± 0.284 at
   K=3, recovering to 0.920 at K=4 — the classic signature of EM arbitrarily splitting one
   real state at K=3). SCm K=4 is the one solid state-structure result in the project.
4. **Stability is not correctness.** A model can be *stably wrong*: a single rate step at
   stimulus onset reproduces perfectly, which is exactly why the K=2 ARI ≈ 0.98 is a trap and
   not a result.
5. **Stability confounds "no such state" with "not enough trials to find it twice."** Each fit
   sees ~281 trials. Diagnostic (not yet run): re-run at several trial counts — if the peak K
   marches right as trials grow, we are data-limited.
6. **The conditional-independence violation is not only a K problem.** If states are partly
   encoding noise correlations, then state *identity*, *count* and *transition structure* are
   all contaminated. This is the formal argument that the rSLDS is a **necessity, not a
   refinement** — see §4.3.
7. **One session.**

### ✅ HMM — tell me which of these to expand

- **H1** — Why Poisson emissions, and what "conditionally independent given the state" buys
  and costs.
- **H2** — EM / Baum–Welch itself: forward–backward, why Viterbi ≠ per-bin argmax.
- **H3** — The baseline-biased `startprob_`: what it changes, and why it is defensible rather
  than cheating.
- **H4** — `forward_fraction`: the net-flux algebra, the shuffle null, and the near-circularity
  of ordering-then-testing-the-ordering.
- **H5** — Why held-out likelihood *structurally* cannot select K on spike counts (the
  correlation-absorbing-states argument).
- **H6** — The ARI: the pair-counting construction, the chance correction, and why it is the
  only thing comparable across K.
- **H7** — The A/B/C stability protocol: why a third held-out set, and why compare *decodings*
  rather than fitted rate vectors.
- **H8** — How to read the ARI-vs-K curve, and why VISp's declining curve and MRN's flat one
  mean what §1.7 says they mean.

---

## 2. The rSLDS — *what kind of dynamics* the population is running

Code: `src/dynamics/rslds.py` (ours, exact EM) + `src/dynamics/rslds_ssm.py` (bridge to
Linderman's `ssm`) · Notebook: `03_rslds_triplet.ipynb` · Worklog:
[`2026-07-13_rslds.md`](../worklog/2026-07-13_rslds.md)

### 2.1 The model

A **recurrent switching linear dynamical system** (Linderman et al. 2017, AISTATS). Per trial:

$$z_t \mid z_{t-1}, x_{t-1} \;\sim\; \mathrm{softmax}_k\!\big(\log\Pi[z_{t-1}, k] \;+\; R_k^{\top} x_{t-1}\big)$$

$$x_t \mid z_t, x_{t-1} \;\sim\; \mathcal{N}\!\big(A_{z_t} x_{t-1} + b_{z_t},\; Q_{z_t}\big)$$

with observations $y_t$ (spike counts) emitted from $x_t$ — Poisson in `ssm`, Gaussian on a
PCA latent in ours.

### 2.2 What each object represents — and the one idea that separates it from the HMM

- **$x_t \in \mathbb{R}^D$ ($D = 3$): the continuous latent.** The population's *position* in a
  low-dimensional space.
- **$z_t$: which *linear flow* is currently governing motion.** Not "where you are" — **"what
  rule is moving you"**.
- **$(A_k, b_k)$: the flow itself.** Its increment is $\Delta x = (A_k - I)x + b_k$, and its
  **fixed point** solves $x^* = A_k x^* + b_k$, i.e. $x_k^* = (I - A_k)^{-1} b_k$.
- **The eigenvalues of $A_k$ name the dynamical object** (`classify()`):
  - all $|\lambda| < 1$ → **point attractor** (the trajectory is pulled in and held);
  - complex $\lambda$ → **rotation**;
  - some $|\lambda| \approx 1$ → **slow mode / line attractor / integrator** — the object a
    decision variable is supposed to live on;
  - mixed → **saddle**: a waypoint the trajectory passes *through*.
  - Timescale of the slowest mode: $\tau_k = -\Delta t / \ln|\lambda|_{\max}$.
- **$R_k$: the recurrence.** This is the whole point of the "r" in rSLDS. If $R = 0$, the
  switching probability depends only on *which regime you were in* — a plain ARHMM, i.e. **a
  stopwatch**: the brain runs a fixed timed script. If $R$ is free, you switch **when the
  trajectory reaches a region of state space** — i.e. an **attractor sequence**.

> **This is the single most important comparison in the project, and it has not been run
> yet.** `delta_recurrent` = held-out LL with $R$ free minus held-out LL with $R = 0$. It is
> the *only* statistic that distinguishes "a sequence of attractors" from "a stopwatch". It
> lives in Part 9 of the notebook, which is **`RUN_CV = False` by default**.

**HMM vs rSLDS in one line:** the HMM's state is a **place** (a rate vector); the rSLDS's
regime is a **rule of motion** (a flow). The HMM can tell you the population visited three
regimes in order; only the rSLDS can tell you whether it was *drawn* there.

### 2.3 The sequence statistic: `order_consistency`

$P[k,l]$ = fraction of trials in which regime $k$ is **first entered** before regime $l$
(over trials visiting both). Then

$$\mathrm{OC} \;=\; \underset{k<l}{\mathrm{mean}}\;\max\big(P[k,l],\, 1 - P[k,l]\big)$$

- **0.5** → the regimes occur in random order: a *set* of states the population wanders among.
  **No sequence.**
- **1.0** → every trial visits them in the same order. **A sequence.**

Deliberately **blind to which direction**, because the claim under test is *"decision-making
is a temporal sequence of metastable states"* — not *"the sequence runs in the direction we
labelled forward"*. It is the **direction-agnostic complement to the HMM's
`forward_fraction`** (which measures directedness *given* an imposed ordering). See §4.2.

⚠️ **Its null is not 0.5.** Within-trial bin shuffling lands at **0.53–0.62**, because
shuffling still changes *which* regime happens to be entered first. Compare to the null, never
to 0.5 — and report the **effect size**: on hundreds of trials a trivial order bias still
reaches p < 0.001 (the random-order control hit p < 0.001 at an effect of **+0.06**, versus
**+0.44** for a real sequence).

### 2.4 Two backends, on purpose

| | ours (`rslds.py`) | `ssm` (Linderman) |
|---|---|---|
| latent | PCA on Anscombe counts, **fixed** | **inferred jointly** with the dynamics |
| emissions | Gaussian | **Poisson** |
| inference | exact forward–backward + weighted LS | Laplace-EM (**ELBO — a lower bound**) |
| cost | seconds | **~15–20 min / region** |
| role | cross-validate K, test recurrence, screen | **the fit of record** |

`ssm` is the published, tested method and is what we would report — but at ~15–20 min a fit,
cross-validating K would take **≈ 13 h**, and **its ELBO is a bound, not a likelihood you may
argmax** (its tightness varies with K and D). So the fast exact-EM model does the selection and
`ssm` is fitted at the chosen K. Both return the same `RSLDSResult`, so every figure and
statistic is backend-agnostic. Agreement between two very different inference schemes would be
a much stronger claim than either alone.

`ssm` lives in a **separate conda env (`ndrh_ssm`)** because it is a numpy-1.x package that
would drag `neurodatarehack` back from numpy 2.3.5 and take pynapple/numba/pandas with it. The
notebook calls it **out of process**.

### 2.5 What ground truth taught us (this is the most valuable part of this strand)

Synthetic population, 3 known attractors visited 0→1→2 on **every** trial, Poisson spikes at an
IBL-like 4 Hz:

| bin | sticky | regime acc | switches/trial (**true 2.0**) | $\|\lambda\|_{\max}$ | OC |
|---|---|---|---|---|---|
| 20 ms | 0 | 0.55 | **8.3** | 1.07 | 0.84 |
| 20 ms | 4 | 0.84 | 2.5 | 0.98 | 1.00 |
| **50 ms** | **4** | **0.90** | **2.0** | **0.875** | **1.00** |

**(a) A prior applied at initialisation is not a prior.** The `sticky` (self-transition)
pseudo-count was being set on the initial $\log\Pi$ and then silently **overwritten by the
first M-step**. Regimes flickered 8× per trial where the truth was 2, and every sequence
statistic would have been computed on that noise. It must be applied in **every** M-step. It
adds mass to $k\!\to\!k$ only, never to $k\!\to\!l$ — so it is a prior for *persistence*, not
for *order*, and cannot manufacture the sequence we are testing for.

**(b) At 20 ms bins the latent is noise, and it fails as a *plausible result*.** With IBL's
sparse units the PCA latent is shot-noise dominated, so $|\lambda| \to 1$ and **every** regime
is classified *"line attractor / integrator"*. That is a near-unit-root random walk fitted to
shot noise — and it reads exactly like the paper-ready sentence *"we found the integrator a
decision variable lives on"*. At 50 ms the same pipeline recovers genuinely contracting
attractors ($|\lambda| = 0.875$).

> **Any integrator / line-attractor claim from a 20 ms fit is inadmissible.**

**(c) A 96 %-of-runtime bug that changed no digit.** The transition M-step was running L-BFGS
*to convergence* inside every EM iteration. EM only requires the M-step to **increase** the
expected complete-data log-likelihood, not maximise it (textbook **generalised EM**). Cutting
the budget from 100 to 10 L-BFGS steps left the final log-likelihood **bit-identical** and took
the K-sweep from **45 min → 7.4 min**.

### 2.6 Where the rSLDS actually is

> **It runs end-to-end on the real session — both backends — and *no scientific numbers are
> claimed*.** The executed pass was a **smoke test** (120 trials, 8 Laplace-EM iterations);
> `ssm` correctly reported `converged=False` at that setting.

A real pass needs `MAX_TRIALS=None`, `SSM_ITERS ≈ 50`, then the 5 replicate sessions. The
notebook is configured for that (`RUN_SSM = True`, `SSM_BIN = "50ms"`) but has not been run
that way.

**K is not selected here.** It is taken **per region from the HMM stability analysis**, and the
notebook runs no model selection at all. That hand-off is the main bridge between §1 and §2 —
and it is currently **stale**: see §5.

### 2.7 Limitations — stated plainly

1. **The recurrence test has not been run.** Until `delta_recurrent` is computed we have **not**
   shown that switching depends on latent *position* rather than elapsed *time* — and that
   distinction *is* the entire difference between an attractor sequence and a stopwatch. This
   is the biggest single gap in the project.
2. **Our backend fixes the latent before fitting the dynamics.** PCA picks directions of
   **variance**, not directions the **dynamics** act on. A real rSLDS could find a low-variance
   but dynamically decisive direction that PCA discards. Consequence: **$D$ must never be
   selected by held-out likelihood** (different $D$ = different data = incomparable numbers).
   Only K, and $R$-free-vs-$R{=}0$, may be.
3. **`ssm`'s ELBO is a lower bound** and must not be argmaxed over K or D.
4. **K is a transfer, not a measurement.** It comes from a *different model* (Poisson HMM) at a
   *different timebase* (20 ms) on a *different unit set* (≥ 1 Hz) over a *different window*
   (1.3 s vs 1.0 s). HMM states and rSLDS regimes **are not the same object** (§2.2). This is a
   principled prior; it is not evidence.
5. **`sticky = 2.0` is fixed by fiat** (chosen on synthetic ground truth). It should be a
   hyper-parameter chosen by held-out LL.
6. **Trials, not neurons, limit this model** — see §4.4. Never prune units to "afford" it.
7. **One session, and only a smoke run.**

### ✅ rSLDS — tell me which of these to expand

- **R1** — SLDS vs HMM vs LDS: what "switching" buys, and why a *flow* is a different object
  from a *state*.
- **R2** — The recurrence term $R_k^\top x_{t-1}$: why it is the attractor-vs-stopwatch test,
  and what `delta_recurrent` would actually prove.
- **R3** — Reading $A_k$: fixed points, eigenvalues, $|\lambda|$, rotation, saddles, and the
  $\tau = -\Delta t/\ln|\lambda|$ timescale.
- **R4** — Why $|\lambda| \to 1$ under shot noise, i.e. *how noise manufactures an integrator*.
- **R5** — Exact EM here vs Laplace-EM in `ssm`; why an ELBO cannot be argmaxed.
- **R6** — `order_consistency` vs `forward_fraction`: sequence vs direction, and why the null
  is not 0.5.
- **R7** — The `sticky` prior: what a Dirichlet self-transition pseudo-count does, and why
  M-step vs initialisation is not a detail.
- **R8** — Generalised EM: why a partial M-step is legal and lost nothing.
- **R9** — Why the parameters that carry the science are estimated from *regime transitions*,
  and hence why trials are the currency (§4.4).

---

## 3. Dimensionality reduction — *what shape* the activity has, and *what causes* it

Code: `src/geometry/{embed,joint}.py`, `src/dynamics/{dpca,gpfa,rrr}.py`, `src/selection/msr.py`
· Notebooks: `04_dimred_msr.ipynb`, `05_manifold_dpca_gpfa_rrr.ipynb` · Worklogs:
[MSR + dim-red](../worklog/2026-07-13_msr-and-dimred.md),
[dPCA/GPFA/RRR/manifold](../worklog/2026-07-13_dpca-gpfa-rrr-manifold.md)

This "topic" is really **five different questions**. Keeping them separate is the point.

> **Notebook 04 was cleaned and then made the consolidated dim-red notebook (2026-07-15).** It
> runs on the **unified 40 ms pipeline** (`data/processed/…bin40…npz`) and shows **PCA, dPCA,
> MDS, Laplacian eigenmaps in 3-D, coloured by time** (all units vs MSR-top; post-stimulus +
> full-window diverging view, blue = baseline, red = evoked). **04's scientific results are now
> merged into 02 §6–9** (dPCA variance split, GPFA, RRR, joint manifold), re-run on the same
> 40 ms tensor — every one reproduced the 50 ms numbers below, so none is a bin-size artefact.
> The MSR validation battery / decoding / choice-coloured maps were dropped (retained in §3.6 +
> worklog). Notebook 05 now only uniquely holds the bin×smoothing sweep and the 7-method
> quality zoo; it is a retirement candidate.

### 3.1 The embedding zoo — *illustration*, mostly not evidence

Points = (trial, bin) population vectors; √-transformed (variance-stabilises Poisson counts),
smoothed, z-scored per unit. Same subsample for **every** method, so differences are the
method, never the points. Run: **PCA, MDS, t-SNE, UMAP, Isomap, Laplacian eigenmaps, PHATE**.

The useful output is not the pictures, it is the **quality metrics** — because *every* method
produces a convincing picture:

- **PCA** → variance explained. The honest one: **PC1+2 = 23–31 %**, i.e. ~70 % of the geometry
  is off-screen.
- **MDS** → Kruskal **stress-1** (0.23–0.29).
- **t-SNE** → KL divergence. **Isomap** → reconstruction error.
- **UMAP, Laplacian, PHATE** → **no goodness-of-fit scalar exists.** Anyone quoting "variance
  explained" for UMAP is inventing it.

So all seven also get two *common* metrics: **trustworthiness** (are the low-D neighbours real
high-D neighbours? — penalises *invented* structure) and **stress-1** (global distortion). The
result is a clean signature:

> **t-SNE and UMAP have the highest trustworthiness (0.88–0.93) and the worst stress-1.** They
> preserve neighbourhoods and destroy global geometry. MDS is the reverse.

⚠️ **Therefore t-SNE/UMAP cannot support any claim about trajectories or distances — and their
apparent "clusters" are NOT evidence of discrete metastable states.** They manufacture clusters
from continuous data. Testing for discrete states is the **HMM's** job.

### 3.2 dPCA — the method that speaks to the actual hypothesis

> **Full maths + the link to the attractor sequence:**
> [`dPCA_maths_and_the_attractor_sequence.md`](dPCA_maths_and_the_attractor_sequence.md). Its
> one-line thesis: **dPCA gives the *existence and geometry* of the shared sequence in a
> coordinate system that is blind to *direction*; the condition-independent (time) component
> IS the trial-averaged shadow of the metastable sequence.** Direction and attractor-vs-stopwatch
> are the dynamics' job (HMM `forward_fraction`, rSLDS `order_consistency` / `delta_recurrent`).

Kobak et al. 2016. PCA finds the dominant directions of activity but **cannot say why they
vary**. dPCA decomposes the trial-averaged activity into **marginalisations** *before* reducing
dimension — by averaging over the other factors:

$$X \;\approx\; X_{t} \;+\; X_{s} \;+\; X_{d} \;+\; X_{sd}$$

(condition-independent *time*; *stimulus* = grating side; *decision* = the animal's choice;
their interaction). For each marginalisation $\phi$ it then finds an encoder $F_\phi$ and
decoder $D_\phi$ minimising $\|X_\phi - F_\phi D_\phi X\|^2$ — so each component is
**demixed**: it captures one cause, while still being a readable projection of the *full* data.

**This is the project's core prediction, tested with a number for the first time:**

| region | stimulus % | decision % | time % | **decision − stimulus** |
|---|---:|---:|---:|---:|
| VISp | 11.8 | 5.0 | 74.1 | **−6.8** |
| SCm | 4.5 | 6.3 | 63.5 | **+1.8** |
| MRN | 8.3 | 14.3 | 49.2 | **+6.0** |

**The ordering comes out exactly as predicted** — VISp stimulus-dominated, MRN
decision-dominated, SCm between — and the condition-independent (time) share *falls*
monotonically along the visual→motor axis. This is the strongest evidence the project has, and
the first result that speaks to the **claim** rather than to the machinery.

**Two things stop it being a claim yet:**

1. **The time component dominates everywhere (74/64/49 %).** Most of what *any* raw PCA
   trajectory plot shows is this, in every area. That is precisely the trap dPCA exists to
   avoid — and a reason to distrust any story told from a plain PCA trajectory, ours included.
2. **The balanced design gives only K = 25 trials per (stimulus × decision) cell.** The
   off-diagonal cells are **error trials** and mice are ~85 % correct, so the *decision*
   components rest on 25 trials. They are noisy. **Pool the 5 replicate sessions before
   believing the ordering.**

### 3.3 GPFA — the dynamics are slower than any bin we argued about

Yu et al. 2009, via `elephant`. Every other method here treats each (trial, bin) vector as an
**unordered point** — it does not know bin 12 follows bin 11. Any smoothness you see in their
trajectories **was put there by us**, in the pre-smoothing step.

GPFA does not need that. Each latent dimension is a **Gaussian process across time**, with a
squared-exponential kernel whose timescale $\tau_i$ is **fitted**, and the spikes are noisy
observations of it. So the smoothing is *inferred*, not imposed — which makes it the honest
check on whether our smoothed trajectories were our own doing.

Fitted $\tau$ per latent (raw counts):

| region | fitted τ (ms), slowest → fastest |
|---|---|
| VISp | 176, 152, 125, 111, 107 |
| SCm | 226, 174, 153, 145, 107 |
| MRN | **283**, 232, 228, 198, 117 |

- **Every timescale is far above 50 ms** → the bin-size debate is settled from the data's side:
  50 ms bins lose nothing, and our trajectories are not a smoothing artefact.
- **The motor areas are slower than the visual one** (MRN 283 ms vs VISp 176 ms) — an
  independent echo of the HMM's directedness result.

### 3.4 RRR / communication subspace — the result that changed the project's mind

Semedo et al. 2019. Regress area B's population on area A's, **restricted to rank $r$**:

$$\hat{B}_r = \arg\min_{\mathrm{rank}(B) \le r} \| Y - XB \|_F^2$$

(closed form: ridge, then truncate the SVD of the *fitted values*). The rank at which held-out
$R^2$ **saturates** is the **dimensionality of the channel through which A influences B**. A
small rank ⇒ B sees only a narrow, low-dimensional projection of A.

**The control is everything.** "SCm predicts MRN at rank 7" means nothing alone — *any* two
populations correlate. The reference is the **within-area control**: split the *target's* own
units in half and predict one half from the other. CV is **grouped by trial** (bins within a
trial are correlated; an ungrouped split leaks and inflates everything).

| source → target | held-out R² | sat. rank | | within-area control | R² | rank |
|---|---:|---:|---|---|---:|---:|
| SCm → MRN | 0.414 | 7 | | MRN half → half | 0.470 | 8 |
| **MRN → SCm** | **0.366** | **7** | | **SCm half → half** | **0.383** | **7** |
| VISp → SCm | 0.116 | 4 | | SCm half → half | 0.383 | 7 |
| VISp → MRN | 0.165 | 5 | | MRN half → half | 0.470 | 8 |

> **MRN predicts SCm (0.366, rank 7) almost as well as SCm predicts *itself* (0.383, rank 7).**
> There is **no restricted communication subspace** between SCm and MRN. The two areas share
> essentially everything they have, at the same dimensionality.

This is the **third independent method** pointing the same way: the pynapple cross-correlogram
showed a sharp **zero-lag** peak (common drive, not a directed exchange), and the HMM found both
areas similarly directed. **The SCm↔MRN "feedback loop" hypothesis now looks less like a
directed loop and more like two readouts of a shared signal.** VISp, by contrast, shares little
with either (0.12–0.17).

### 3.5 The joint manifold — and the number that may be the most interesting in the project

**You cannot simply embed three areas together.** A VISp state vector lives over 36 VISp
neurons, an MRN one over 63 MRN neurons — **different spaces**. Stacked and handed to t-SNE the
areas would separate **trivially**, because their axes differ, and the plot would be pure
artefact. (It is an easy plot to make and it looks convincing.)

What licenses a common space here is that the areas are **simultaneously recorded**: they share
the same (trial, bin) samples. So: concatenate all 158 units, take the **joint PCA**, and —
because PCA is **linear** — decompose the joint trajectory **exactly** into per-area
contributions using each area's own loadings:

$$Y(t) \;=\; (X(t) - \mu)V^{\top} \;=\; \sum_a (X_a(t) - \mu_a)V_a^{\top} \;=\; \sum_a Y_a(t)$$

`src/geometry/joint.py` **asserts** this identity rather than assuming it — it is the entire
justification for plotting three areas on one pair of axes.

| region | share of joint latent variance | n units |
|---|---:|---:|
| VISp | 0.056 | 36 |
| SCm | 0.222 | 59 |
| MRN | 0.307 | 63 |

**The shares sum to 0.585.** The areas' contributions are **not orthogonal**, and the missing
**~0.42 is cross-area covariance** — i.e. *the shared signal the whole project is arguing
about*, quantified.

⚠️ *Caveat:* units are z-scored so **loudness** is controlled, but **unit count is not**. VISp's
small share is partly its 36 units. Compare at matched unit count before claiming MRN "drives"
the shared state.

### 3.6 MSR — tested, and **not adopted**

Multiscale Relevance (Cubero et al. 2020): sweep the bin width $dt$, and at each scale compute
the **resolution** $H[s]$ (entropy over bins) and the **relevance** $H[k]$ — the entropy of the
*frequency of frequencies* ($m_k$ = number of bins holding exactly $k$ spikes). MSR is the area
under the $H[k]$-vs-$H[s]$ curve. Attractive because it is **model-free and unsupervised** — it
never sees the stimulus, so selecting units with it cannot double-dip.

Three findings, all negative or cautionary:

1. **MSR correctly gives a LOW score to a perfectly stereotyped burster.** If every burst has
   exactly 4 spikes, $m_k$ is a delta and $H[k] \to 0$. The train has *structure* but no
   *variety of multiplicities* — and variety of multiplicities is exactly what relevance
   measures. **Consequence for real data: a low MSR is not automatically an uninformative
   unit** (SC and MRN contain bursting cells).
2. **The rate-matched control silently degenerated into "the slow half."** MSR correlates with
   firing rate (Spearman ρ: VISp **0.64**, SCm **0.55**, MRN 0.08), so the top-MSR half *is* the
   loud half and the leftover pool contains nothing loud enough to match. In VISp and SCm the
   "control" was 3.5× and 4.6× quieter. **Any selection-beats-control claim in those rows is a
   firing-rate difference wearing a costume.**
3. **In MRN — the only region where the control is real — the top-MSR half is indistinguishable
   from a RANDOM half** (choice AUC 0.934 vs **0.939**). MSR bought nothing.

> **Standing decision: MSR is not adopted.** Defensible as cheap unsupervised compression, but
> there is no evidence here that it beats firing rate, and where it could be tested fairly it
> did not beat *chance selection*.

**But it produced one result that matters everywhere else:** halving the population is nearly
free for **choice** (SCm AUC 0.946 → 0.943) and **costly for time** (MRN time R² 0.550 → 0.483).
**Choice information is redundant across the population; the temporal trajectory is not.**
Hence: **selection is fine for decoding, and forbidden for dynamics.** Do not fit the HMM or the
rSLDS on a pruned population.

### 3.7 Limitations — stated plainly

1. **One session** — and dPCA's decision components rest on **25 trials per cell**.
2. **Unit count is uncontrolled** in the joint manifold; VISp's small share is partly just 36
   units.
3. **Everything except the decoding table is fitted and displayed on the same trials.** Only the
   held-out decoding is evidence.
4. **UMAP/Laplacian/PHATE have no fit metric at all** — they can illustrate, never establish.
5. **GPFA's observation model is Gaussian on √-counts, not Poisson** — fine at these rates,
   wrong for the very sparse units that already strain the Poisson HMM.
6. **The bin/smoothing configuration could not be chosen from within.** `time_R2` rises
   *monotonically* with σ and its argmax lands on whatever boundary you sweep to (0.32 → 0.48 as
   σ goes 0 → 160 ms) — because **a Gaussian kernel imposes exactly the temporal
   autocorrelation that `time_R2` rewards**. It is a hair from circular. **50 ms / σ = 40 ms was
   chosen on external grounds** (the rSLDS ground-truth bin + the fitted GPFA timescales). The
   saving grace: **choice AUC is flat across the entire sweep (0.854–0.860)** — *the choice
   signal does not depend on bin size or smoothing at all*, so nothing downstream rests on this
   knob.

### ✅ Dimensionality reduction — tell me which of these to expand

- **D1** — Why PCA on spike counts needs a variance-stabilising transform (√ / Anscombe) at all.
- **D2** — dPCA's marginalisation algebra: how $X$ is split, what the encoder/decoder pair does,
  and why it is *not* just "PCA on the condition means".
- **D3** — Why the condition-independent (time) component dominates everywhere, and what that
  implies about every trajectory plot in systems neuroscience.
- **D4** — GPFA: the GP prior, the fitted $\tau$, and why "inferred smoothness" is a stronger
  claim than "imposed smoothness".
- **D5** — Reduced-rank regression: the rank constraint, why the **within-area control** is the
  only meaningful reference, and what "no restricted communication subspace" does and does not
  license us to say.
- **D6** — The joint-PCA partial projection: why $Y = \sum_a Y_a$ is exact, and what the missing
  0.42 (cross-area covariance) actually *is*.
- **D7** — Trustworthiness vs stress-1, and why t-SNE/UMAP clusters are not states.
- **D8** — MSR: resolution vs relevance, the frequency-of-frequencies entropy, and the
  stereotyped-burster degeneracy.
- **D9** — Why the smoothing sweep is circular, and what a non-circular criterion for σ would
  even look like.

---

## 4. How the three interact

They are **not three parallel analyses**. They are a dependency graph, and in two places one
strand's *failure* is another strand's *justification*.

### 4.1 The HMM hands K to the rSLDS (a parameter transfer — and an assumption)

The rSLDS runs **no model selection**. K comes per region from the HMM's ARI stability, because
that criterion works on this data and likelihood does not. This is the main structural bridge —
and it is an **assumption**, because an HMM state (a *rate vector*, at 20 ms, on ≥1 Hz units,
over a 1.3 s window) is **not the same object** as an rSLDS regime (a *flow*, at 50 ms, on all
units, over a 1.0 s window). It is a principled prior. It is not a measurement. And right now it
is also **stale** (§5.1).

### 4.2 The HMM and the rSLDS measure two *different halves* of the same claim

The claim is *"decision-making is a temporal **sequence** of metastable states."* It has two
separable parts, and each model can only see one:

| | statistic | asks | chance |
|---|---|---|---|
| **HMM** | `forward_fraction` | *Given* a time-ordering, does the traffic run **forward**? | 0.5 (validated by shuffle) |
| **rSLDS** | `order_consistency` | Is there **any reliable order at all** — in either direction? | **0.53–0.62**, *not* 0.5 |

`forward_fraction` presupposes an ordering (it takes it from occupancy centre-of-mass);
`order_consistency` is deliberately blind to direction. **Neither alone establishes the claim.**
Together they separate *"there is a sequence"* from *"the sequence runs the way we say it does"*.

### 4.3 The HMM's structural failure is the rSLDS's justification

This is the deepest link, and it is causal, not rhetorical:

> The Poisson HMM assumes neurons are **conditionally independent given the state**. Real
> populations are **correlated** and **over-dispersed**. The model's *only* mechanism for
> expressing that is **to add a state**.

Two consequences, one shallow and one deep:

- *Shallow:* held-out likelihood cannot select K (it climbs forever) — hence the ARI.
- *Deep:* **if states are partly absorbing noise correlations, then state identity, state count
  and transition structure are all contaminated.** The fix is a model with a **continuous
  latent** that soaks up shared variability *directly* — leaving the discrete variable free to
  describe *dynamics* rather than *covariance*. That is exactly the rSLDS.

So the rSLDS is not a fancier HMM we run because we can. **The HMM's own failure mode is the
formal argument that it is necessary.**

### 4.4 Dimensionality reduction is *inside* the rSLDS, not beside it

Our rSLDS backend literally *is* a dim-red pipeline followed by a dynamical model: Anscombe →
smooth → z-score → **PCA (D = 3)** → fit switching dynamics on the latent. So §3's central
weakness is inherited wholesale by §2: **PCA picks directions of variance, not directions the
dynamics act on.** (`ssm`'s Poisson backend, which infers the latent *jointly with* the
dynamics, is the principled answer — which is why it is worth the 15–20 min a fit.)

### 4.5 Dim-red chose the timebase that the rSLDS runs on — and vice versa

Neither strand could pick its own hyper-parameters honestly, so they picked each other's:

- **GPFA (dim-red) → the geometry config.** Fitted τ = 107–283 ms ⇒ 50 ms bins lose nothing and
  σ = 40 ms sits safely below the *fastest* real timescale.
- **rSLDS ground truth → the geometry config.** 50 ms recovers real attractors; 20 ms
  manufactures integrators.

And this was necessary because the *internal* criterion was circular: **`time_R2` is inflated by
the very smoothing it was being used to select** (§3.7.6).

### 4.6 Three methods, three currencies, one axis — and one retraction

The project's central prediction is a **visual → motor gradient**. Three *independent* methods,
measuring in three different units, agree on the ordering:

| method | currency | VISp | SCm | MRN |
|---|---|---|---|---|
| **dPCA** | decision − stimulus variance | **−6.8** | +1.8 | **+6.0** |
| **HMM** | `forward_fraction` (directedness) | **0.53** ≈ undirected | 0.77–0.80 | 0.69–0.80 |
| **GPFA** | slowest fitted τ | **176 ms** | 226 ms | **283 ms** |
| ~~stability~~ | ~~number of reproducible states~~ | 3 | 4 | 3 | ← **no gradient** |

**The state-count row is retracted** (it was 2/3/4 before EM restarts; it is 3/4/3 after). The
honest synthesis is therefore:

> **The areas hold a similar number of states, but the motor areas traverse them far more
> directedly, carry more decision variance, and run slower.**

Which is arguably a *better* fit to the project's framing — *shared alphabet, area-specific
sequence* — than a state-count gradient would have been. But it is a **different claim**, and it
must be stated as one.

### 4.7 Dim-red has already constrained what a future rSLDS is allowed to conclude

The open scientific question was an **SCm↔MRN feedback loop** (state exchange before a shared
terminal attractor), to be tested with a **joint multi-region rSLDS**. Three independent methods
now point at **common drive** instead:

1. **RRR** — no restricted communication subspace; MRN predicts SCm about as well as SCm
   predicts itself.
2. **Cross-correlogram** (pynapple) — a sharp **zero-lag** peak, the signature of shared input,
   not a lead–lag exchange.
3. **HMM** — both areas similarly directed, neither leading.

> **The joint rSLDS must now *overturn* three converging pieces of evidence, not merely confirm
> a feedback loop.** That is a much harder — and much more interesting — thing to ask of it. It
> is also a case of the geometry strand doing real scientific work *before* the dynamics strand
> gets there.

### 4.8 The shared limiting resource: **trials, not neurons**

Four independent lines of reasoning converged on this, which is why it is now a standing rule:

- **rSLDS parameter counting.** The scientific payload — attractors and sequence — lives in the
  dynamics + transition blocks (≈100 parameters at D=3, K=4), and those are estimated from the
  number of **regime transitions observed**. A trial supplies only **2–4**. Neurons, by
  contrast, enter through the emission matrix and bring a whole new time series *with* their
  parameters — they are nearly free, and they sharpen the per-bin latent.
- **dPCA.** The balanced design caps at **25 trials per cell**.
- **ARI stability.** Each fit sees only ~1/3 of the trials.
- **MSR/decoding.** Halving the population barely costs choice info, but *does* cost the
  temporal trajectory.

> **Never prune units to "afford" a state-space model — it buys no statistical power and costs
> latent SNR. To strengthen a claim, pool the 5 replicate triplet sessions.** (The original
> plan's framing of MSR as a way to *shrink* the population is right for ELA and **wrong** for
> the rSLDS.)

### 4.9 The shared failure mode: an argmax on a boundary, or a criterion the knob inflates

The same error, in all three strands, three times:

| strand | the argmax | why it was wrong |
|---|---|---|
| **HMM** | K = 2, then K = 10 | both **boundaries**; first off a column of `-inf`, then off the top of the range |
| **rSLDS** | 20 ms bins ⇒ every regime an "integrator" | the criterion was **satisfied by noise**, and it *looked like a finding* |
| **dim-red** | σ = 160 ms (the sweep's ceiling) | the metric (`time_R2`) is **inflated by the very knob it selects** |

> **The rule, extracted: never take an argmax without checking that (a) the winner is
> *interior*, and (b) the criterion is *independent of the parameter being chosen*.**

---

## 5. Known inconsistencies — concrete and actionable

These are live defects, not philosophy. Ordered by how much they could bite.

**5.1 ⚠️ `03_rslds_triplet.ipynb` is running a STALE K.** It sets
`K_PER_REGION = {VISp: 2, SCm: 4, MRN: 3}` and its markdown still carries the retracted claim
*"VISp is a documented negative … ~2 reproducible states"*. But
[`best_K_per_region.md`](best_K_per_region.md) is **FINAL as of 2026-07-14** with **VISp 3**,
SCm 4, MRN 3 — and explicitly retracts the VISp=2 reading (it was an artefact of running the
stability scan with a single EM fit; with restarts, VISp K=3 gives ARI 0.883 ± 0.019). **The
notebook must be updated to VISp = 3 before any rSLDS pass is treated as real.** At VISp K=2,
`order_consistency` has exactly *one* regime pair and cannot express a sequence at all.

**5.2 ⚠️ The rSLDS uses a different trial window from everything else.** Notebook 03 reads
`pre_s` from `config/params.yaml` (**0.2 s**); notebooks 02/04/05 hard-code **0.5 s**. So the
rSLDS sees a 1.0 s window with a 0.2 s baseline, while the HMM that *chose its K* saw 1.3 s with
a 0.5 s baseline. This should be made deliberate (pick one, write it in the config, and have all
four notebooks read it) rather than left as an accident of which notebook reads which source.

**5.3 `forward_fraction` has never been computed at the final K** (3 / 4 / 3). The quoted values
come from K=2 and K=10 fits. The ordering is stable across both, so the *conclusion* is safe —
but the *number* is not yet the one we should publish.

**5.4 `baseline_exit_latency` returns `NaN` for VISp and MRN.** The inter-areal timing read-out
— *does VISp leave baseline before SCm before MRN?* — is currently **not working** for two of
three regions.

**5.5 The recurrence test (`delta_recurrent`) has not been run.** It is the only thing that
distinguishes an attractor sequence from a stopwatch, and it is off by default.

**5.6 `config/areas.yaml::selected` is still empty.** Four worklogs in a row have said so. It is
clearly `{VISp, SCm, MRN}`.

**5.7 Minor:** `results/manifold/summary.md` reports joint PC1 = 16.7 % / PC2 = 8.0 % /
PC1–10 = 44.7 %, while the worklog of 2026-07-13 says 19.1 / 9.4 / 50.0. The per-area
`var_share` numbers agree in both. The results file is from the later executed run; the worklog
line is probably from an earlier pass and should be corrected.

**5.8 `config/params.yaml::hmm.n_states = 4`** is a leftover Phase-0 smoke-test value and no
longer reflects anything (K is now per-region, from stability).

---

## 6. The single highest-value next step

Three separate analyses arrived at it independently, which is about as strong a signal as
methods work gives you:

> ### Pool the 5 replicate `{VISp, SCm, MRN}` sessions. Add trials; do not filter units.

It is what dPCA needs (25 trials/cell), what the rSLDS needs (transitions are the currency),
what the stability analysis needs (each fit sees ~281 trials), and what a real rate-matched
control for VISp/SCm would need. Almost every caveat in this document ends in the same place.

After that, in order: **(1)** fix the stale K and the window (§5.1, §5.2), **(2)** run the
**recurrence test** (§5.5) — it is the project's biggest unanswered question, **(3)** run the
**joint SCm+MRN rSLDS**, remembering it now has to *overturn* the common-drive evidence, not
confirm a loop.

---

## ✅ Cross-cutting — tell me which of these to expand

- **X1** — Why "metastable state" (HMM), "dynamical regime" (rSLDS) and "cluster in an
  embedding" (t-SNE) are three different things that all look like blobs.
- **X2** — The K hand-off from HMM to rSLDS: what would have to be true for it to be valid, and
  how we could actually test it.
- **X3** — What a *joint multi-region* rSLDS would look like, what $R$ carrying cross-area
  structure would mean, and what would count as overturning the common-drive evidence.
- **X4** — Common drive vs feedback loop: what evidence *could* still distinguish them, given
  RRR, the zero-lag correlogram and the HMM all point one way.
- **X5** — Why trials and not neurons, in full — the parameter-counting argument.
- **X6** — Model selection generally: likelihood vs stability vs ELBO vs held-out decoding —
  which criterion answers which question, and which of ours are circular.
