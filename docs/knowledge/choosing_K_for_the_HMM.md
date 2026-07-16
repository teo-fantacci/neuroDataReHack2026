# Choosing K for the HMM — why held-out likelihood fails, and what to use instead

Written 2026-07-12 after the Phase-2 CV curve refused to plateau. This is a methods note,
not a result: it records *why* we select K the way we do, so the decision is not
re-litigated.

---

## 1. The symptom

Cross-validated held-out log-likelihood, per region, `K = 2 … 10`, units ≥ 1 Hz:

* it **rises monotonically and never turns over**;
* `argmax` therefore returns **K = 10 — the top of the search range**, for every region;
* and before the `-inf` bug was fixed, the same `argmax` returned **K = 2 — the bottom of
  the range**, for every region.

Two opposite "results", both produced by an argmax pinned to a boundary. That is the
signature of a **criterion that does not discriminate**, not of a dataset that wants ten
states.

The implementation is not at fault. `hmmlearn`'s likelihood and Viterbi path were validated
against **dynamax** (Linderman lab) with identical injected parameters: held-out LL agrees
to machine precision (relative difference 3.6e-16 on VISp, exactly 0.0 on SCm) and the
Viterbi paths agree on **100%** of bins. See `scripts/validate_hmm_vs_dynamax.py`.

## 2. Why held-out likelihood keeps climbing (the real reason)

A Poisson HMM assumes the neurons are **conditionally independent given the state**:

> P(spikes at time t | state k) = ∏_n Poisson(x_nt ; λ_kn)

Real cortical and midbrain populations violate this. They have **noise correlations** —
co-fluctuations shared across neurons that are not explained by the stimulus or by the
state — and their spike counts are **over-dispersed** (variance > mean), whereas a Poisson
sets variance = mean by definition.

The model has exactly **one mechanism** available to express any of that: **add another
state**. Each new state is a new rate vector, and mixing over more rate vectors is how a
conditionally-independent model imitates correlation and extra variance.

So held-out likelihood improves with K *whether or not there are more real states*, because
the extra states are being spent on correlation structure rather than on temporal dynamics.
The likelihood is buying the right answer to the wrong question.

This is a known property of mixture-like models on spike counts, not a quirk of our data.
**It means CV-log-likelihood argmax is simply not a valid K-selection rule here**, and no
amount of extending the K range will fix it.

## 3. What we do instead

### 3a. The elbow (`hmm.select_k_by_elbow`) — the pragmatic rule

Ask not "which K is best" but **"is the next state buying more than the noise?"** Compare
the increment `LL(K) − LL(K−1)` against the **standard deviation across CV folds** at K. The
elbow is the smallest K after which *no* later increment clears its own noise floor.

On session `5c0c560e` (units ≥ 1 Hz):

| region | argmax | **elbow** | what the increments do |
|---|---:|---:|---|
| VISp | 10 | **4** | every gain after K=4 is ~0.02 against a fold SD of ~0.06 |
| SCm | 10 | ~6–8 | gains decay 0.85 → 0.11, hugging the noise line |
| MRN | 10 | ~6–9 | gain at K=8 is **negative** (−0.15), then positive again — pure noise |

MRN's sign flip is the tell: past K≈6 the increments are fold-to-fold variability, not
model improvement.

This is a **judgement call encoded as a rule**. It must be reported, and results must be
checked for sensitivity to it.

### 3b. State stability (`hmm.state_stability`) — the criterion that matches the claim

The deeper fix. Our scientific claim is that decision-making passes through **metastable
states** — that identifiable, *reproducible* regimes exist. **Held-out likelihood does not
test that claim.** It asks "does this model predict spikes well," which rewards flexibility.
Reproducibility is what we are asserting, so reproducibility is what we should measure.

A state that appears when you fit trials 1–100 but vanishes when you fit trials 101–200 is
not an attractor. It is a fitting artifact. So:

1. Split trials three ways: **A**, **B**, **C**.
2. Fit one HMM on **A**, and a **completely independent** HMM on **B**.
3. Decode **both** models on the **same** held-out trials **C**.
4. Measure how much the two decoded state sequences agree.

Both models meet C for the first time, so neither is advantaged, and the two fits never
shared a trial. If the K states are real, two strangers should carve C's timeline the same
way.

**Metrics.**

* **Adjusted Rand Index (ARI)** — the headline. It compares the two *partitions of time*,
  and is **permutation-invariant** (state labels are arbitrary, so any metric that cares
  about label identity is meaningless) and **chance-corrected** (0 = chance, 1 = identical).
* **Matched accuracy** — fraction of bins agreeing after optimally matching labels
  (Hungarian algorithm on the confusion matrix). Interpretable, but **not chance-corrected**:
  it rises trivially as K falls, so it must be read alongside the ARI, never alone.

**Why this succeeds where likelihood fails: it has a peak.** With too few states, both
models agree easily but describe little. With too many, the surplus states are split
arbitrarily and differently by each fit, so agreement *collapses*. The turnover is the
signal — and unlike the likelihood, it is a criterion the extra "correlation-absorbing"
states cannot game, because those states are fit to noise and noise does not reproduce.

## 4. Consequences beyond K

The mechanism in §2 is not only a model-selection nuisance — it is a **substantive limit on
the Poisson HMM for this project**. If states are partly encoding noise correlations, then
state identity, state count, and transition structure are all contaminated.

That is an argument for a model with a **continuous latent** that can absorb shared
variability directly — an rSLDS, or a Poisson linear dynamical system — leaving the discrete
states to describe dynamics rather than covariance. Phase 3 was already going there; this
finding says it is a necessity, not a refinement.

## 5. Rules of thumb, extracted

* **Never take an argmax without checking the winner is (a) finite and (b) not on a
  boundary.** Both failures happened here, in the same cell, within one week.
* `-inf` is **not** NaN. Guard model selection against **non-finite**, not just NA.
* Prefer a criterion that **can fail**. Likelihood cannot say "there are no states";
  stability can.
* Match the criterion to the claim. We are claiming reproducible states, so we should be
  measuring reproducibility.
