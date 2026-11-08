# rSLDS backends & the Anscombe transform — why we keep the exact-EM

**Written 2026-07-16.** Resolves the standing "should we drop the exact-EM rSLDS?" question
(decision: **keep it**), and records the theory the decision rests on: what the Anscombe
transform is, and how our fast exact-EM backend differs from the Poisson `ssm` backend.

Companion to [`where_we_are_HMM_rSLDS_dimred.md`](where_we_are_HMM_rSLDS_dimred.md) §2 (the rSLDS
model in full) and [`project_explained.md`](project_explained.md) §1 (the measured Anscombe
failure). Code: [`src/dynamics/rslds.py`](../../src/dynamics/rslds.py) (exact-EM),
[`src/dynamics/rslds_ssm.py`](../../src/dynamics/rslds_ssm.py) (`ssm` bridge).

---

## 1. Decision: keep the exact-EM backend — and *why*

The exact-EM rSLDS produces one thing we **retracted** and one thing we **certified**, and the
two are separable, which is the whole reason it stays.

**What we retired (and no longer trust):** its **flow geometry** — the eigenvalue magnitudes
`|λ|`, the "line attractor / integrator / point attractor" labels. At IBL firing rates its
Gaussian-on-a-smoothed-PCA-latent pushes `|λ| → 1` as a *smoother* artefact (>97 % reproduced on
temporally-shuffled data). Nothing in the project reads these numbers any more.

**What it is the engine for (and cannot be replaced on):**

- The **recurrence test** `delta_recurrent` = held-out `LL(R free) − LL(R = 0)` — the **only**
  statistic that separates a *position-dependent attractor sequence* from a *timed stopwatch*.
  This is a **certified** result (positive in every region, both alignments, beats a
  position-shuffle null in 100 % of folds; [`results/rslds/recurrence_test.json`](../../results/rslds/recurrence_test.json)).
- The **`order_consistency`** sequence screen and the **K** screen.

**Why `ssm` cannot take over these:** the recurrence test must be **cross-validated** (it is a
held-out likelihood *comparison*), and a single `ssm` Poisson fit takes ~18–22 min/region, so
cross-validating it across folds/regions is ≈ 13 h. Worse, `ssm`'s objective is an **ELBO — a
lower bound**, not a likelihood you may arg-max or difference cleanly across two models. So the
fast, exact-likelihood backend is the *only* practical way to run the recurrence test.

**Why the retracted geometry does not contaminate the certified result:** the recurrence test is
a **relative** likelihood comparison (same latent, same data, `R` free vs `R = 0`) with its own
shuffle null. It never reads `|λ|` or a flow label. The thing that is unreliable (absolute flow
geometry from a smoothed latent) is simply not an input to the thing that is reliable (does
switching depend on latent *position*).

> **So: keep the exact-EM as the recurrence-test / `order_consistency` / K-screen engine. Keep
> `ssm` (Poisson) as the fit of record. Trust neither backend's `|λ|` geometry.** Dropping the
> exact-EM would delete the engine behind a certified result to remove a geometry we already
> ignore — a bad trade.

---

## 2. The Anscombe transform — the theory, briefly

**What it is.** `y = 2·√(x + 3/8)`. The classical **variance-stabilising** transform for Poisson
counts (Anscombe 1948).

**The problem it targets.** For a Poisson variable **Var = Mean**. So in raw counts a 20 Hz neuron
is not merely *louder* than a 2 Hz neuron — it is intrinsically **noisier**, and any squared-error
method (PCA, least squares, a Gaussian likelihood) spends its variance budget chasing the loud
units' shot noise.

**Why √, and why the 3/8.** A delta-method (first-order Taylor) expansion of `y = 2√(x + c)` for
`x ~ Poisson(λ)` gives

```
Var[y] = 1 + O(1/λ),
```

i.e. asymptotically every neuron's noise becomes ≈ 1 unit regardless of its rate; `c = 3/8` is the
offset that **cancels the leading 1/λ correction term**, making the approach to Var = 1 fastest.
Note it is *not* z-scoring: z-scoring rescales each neuron by its SD (fixes the *scale*, leaves the
distribution skewed); Anscombe reshapes the count *distribution* toward homoscedastic-Gaussian.

**⚠️ Why it does nothing here.** `Var → 1` is a **large-λ** result — it needs **λ ≳ 4 spikes/bin**.
IBL runs at **λ = 0.05–0.83** (VISp–SCm, 20–50 ms), one to two orders of magnitude below. On the
real tensor the transform yields median Var ≈ 0.09, not 1, and the mean–variance dependence
survives intact (Spearman ≈ +1). This is **structural, not tunable**: at λ ≪ 1 a count is
essentially **Bernoulli** (0/1), and a Bernoulli's variance `p(1−p)Δ²` depends on `p`
irreducibly — no monotone transform can stabilise it.

**Where it is used, and why we keep it anyway.** Anscombe is applied in **exactly one place** —
the exact-EM latent ([`counts_to_latents`](../../src/dynamics/rslds.py), the inline
`2·√(counts + 3/8)`). It is **not** in the Poisson `ssm` backend (which keeps integer counts),
**not** in the geometry embeds (`src/geometry/embed.py` uses a plain `√`), and **never** in the
Poisson HMM (raw counts by construction — pre-stabilising would correct for Var = Mean twice). We
keep it for two small, honest reasons: it **compresses the high-count tail** (one 7-spike bin stops
dominating a squared-error objective), and the rSLDS **ground truth was calibrated with it**, so
removing it would invalidate that calibration for no gain. It is a **monotone relabelling**
(0 → 1.225, 1 → 2.345, 2 → 3.082), *not* smoothing — all continuity downstream comes from the
Gaussian smoother.

**The scientific consequence** is the real point: our fast backend's Gaussian/PCA latent rests on a
variance-stabilising step that stabilises **nothing** at these rates. That is a *second*,
independent argument (on top of "PCA finds variance, not dynamics") that the **Poisson `ssm`
backend is the principled model and the exact-EM's Gaussian latent is a concession for speed.**

---

## 3. Exact-EM vs `ssm` Poisson — the difference

| | **exact-EM** (`rslds.py`, ours) | **`ssm` Poisson** (Linderman) |
|---|---|---|
| emission model | **Gaussian** on an Anscombe→smooth→z-score→**PCA** latent | **Poisson** directly on integer counts |
| the latent | **fixed before fitting** (PCA = variance directions) | **inferred jointly** with the dynamics |
| inference | exact forward–backward + weighted least squares | Laplace-EM (variational) |
| objective | an **exact likelihood** (can be differenced/arg-maxed) | an **ELBO — a lower bound** (cannot be arg-maxed over K/D) |
| cost | **seconds** | **~18–22 min / region** |
| role | model selection (K), `order_consistency`, **the recurrence test** | **the fit of record** |
| environment | `neurodatarehack` (numpy 2.x) | isolated `ndrh_ssm` (numpy 1.x), called out-of-process |

**The conceptual difference, in one line.** The exact-EM first squashes the counts into a Gaussian
shape (Anscombe), smooths them, and reduces to the **directions of most variance** (PCA) — *then*
fits dynamics on that latent. Two things are wrong-in-principle there: **Anscombe stabilises
nothing** at these rates (§2), and **PCA picks variance directions, not the directions the dynamics
act on** — a low-variance but dynamically decisive direction can be discarded. The `ssm` backend
avoids both: it keeps the counts as **Poisson** (the correct noise model for spikes) and **infers
the latent jointly with the dynamics**, so the latent is chosen to be dynamically informative, not
merely high-variance.

**Why we still run both.** The exact-EM is **fast and exact-likelihood**, so it can cross-validate
K and — crucially — run the recurrence test, which `ssm` cannot afford (§1). The `ssm` fit is the
**principled** one we would report. And when two very different inference schemes **agree** — as
they do on `order_consistency` (SCm/MRN ~0.93–0.99, ≫ null on *both* backends) — that agreement is
a stronger claim than either backend alone. Neither backend's **`|λ|` geometry** is trusted.

---

## 4. Note on the frozen `.npz` files (raw counts only)

For completeness, because it connects: the frozen tensors
(`data/processed/*.npz`) store **raw `int32` spike counts only** — there is no smoothed or
transformed series in the file. Anscombe and smoothing are *both* downstream, model-specific steps
applied at load time, never persisted. This is deliberate: the two Poisson models (HMM, `ssm`) need
integers, and freezing a smoothed copy would invite feeding it to a model that must not see it.

---

*Provenance: worklogs 2026-07-13 (rSLDS), 2026-07-15 (rSLDS first real test); knowledge docs
where_we_are_HMM_rSLDS_dimred §2, project_explained §1; results/rslds/recurrence_test.json;
src/dynamics/rslds.py, rslds_ssm.py, src/data/preprocess.py::anscombe.*
</content>
