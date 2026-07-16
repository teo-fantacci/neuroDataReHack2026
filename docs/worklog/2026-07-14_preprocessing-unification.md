# 2026-07-14 — One preprocessing pipeline, and two findings that reframe the project

New: `notebooks/00_preprocessing.ipynb` (executed end-to-end), `src/data/preprocess.py`,
`scripts/run_binsize_check.py`, `docs/knowledge/HMM_maths_and_readouts.md`. Fixes to
`src/dynamics/hmm.py`.

The goal was housekeeping: stop each notebook building its own tensors. It turned into two
substantive findings, both of which change how existing results must be read.

---

## 0. The problem that started it

The four notebooks silently disagreed about what "the data" is:

| notebook | window | bins | units |
|---|---|---|---|
| `01_hmm` | −0.5 → +0.8 s | 20 ms | ≥ 1 Hz |
| `03_rslds` | **−0.2** → +0.8 s (it read `params.yaml`; the others hard-code) | 20 + 50 ms | **all** |
| `02`, `04` | −0.5 → +0.8 s | 50 ms | **all** |

So **"K from the HMM" was handed to an rSLDS fitted on a different window and a different unit
set**, and nobody had written that down. `src/data/preprocess.py` is now the single place the
choice is made; the config is serialised *inside* the `.npz` with the tensors, so a tensor can
no longer be mis-described later.

---

## 1. ⚠️ The analysis window is mostly MOVEMENT and REWARD, not decision

Nobody had ever measured the trial timing. Measured now (843 trials, `5c0c560e`):

| quantity | median | consequence over `[0, +0.8] s` |
|---|---:|---|
| **reaction time** (first movement − stimOn) | **171 ms** | movement has begun on **92 %** of trials |
| **feedback** (feedback − stimOn) | **427 ms** | the outcome has arrived on **83 %** of trials |

**The mice are fast.** The `post_s = 0.8 s` window — chosen at some point for no recorded reason
— is *dominated* by movement execution and reward consumption. And in IBL **the choice IS the
wheel-turn direction**, so "decision" and "movement direction" are not separable by wishing.

**This reframes several results without invalidating them:**

- The HMM's **late states** in SCm/MRN are at least as likely to be *movement* and *outcome*
  states as decision states.
- **dPCA's "decision" variance** (VISp −6.8 → SCm +1.8 → MRN +6.0) is entangled with movement
  direction. The visual→motor ordering is still a real gradient — but the axis may be
  *sensory → motor execution*, which is a weaker and more obvious claim than the one we want.
- **"Motor areas are more directed than visual cortex"** now has an unglamorous null that must be
  excluded: *motor areas look directed because the window is full of movement.*

**Adopted:** a **`firstMovement_times`-aligned tensor is now built alongside the stimulus-aligned
one**, as a standing control. Disagreement between the two alignments is information. And no
state after ≈ +0.2 s gets called a "decision" state without an argument that survives it.

*(Shortening the window does not rescue this: feedback has already arrived on 45 % of trials by
+0.4 s. The confound is in the task, not the window.)*

---

## 2. ⚠️ The Anscombe transform does not stabilise variance at IBL's firing rates

`counts → Anscombe → PCA` is the input chain for the rSLDS and all the geometry. The transform's
whole purpose is to make Poisson noise rate-independent (Var = 1 regardless of λ). **It does not
do that here, and the gate that proves it is now in the notebook.**

On **simulated Poisson**, where the truth is known:

| λ (spikes/bin) | 0.05 | 0.1 | 0.3 | 0.8 | 2.0 | **4.0** | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Var[Anscombe] | 0.06 | 0.12 | 0.31 | 0.63 | 0.93 | **1.00** | 1.00 |

Var → 1 is an **asymptotic** result and only arrives at **λ ≳ 4**. And IBL sits at:

| | VISp | SCm | MRN |
|---|---:|---:|---:|
| λ @ 20 ms | **0.046** | 0.331 | 0.276 |
| λ @ 50 ms | 0.116 | **0.827** | 0.689 |

**One to two orders of magnitude below the regime where the transform works.** On the real tensor
it produces median Var ≈ **0.09**, not 1, and the mean–variance dependence is untouched
(Spearman **+0.99**).

**This is structural, not a tuning failure.** At λ ≪ 1 a spike count is essentially **Bernoulli**,
and Var = p(1−p)·Δ² depends on p irreducibly. **No monotone transform can stabilise a Bernoulli.**
Variance stabilisation is simply *unavailable* in this data regime. Looking at the per-trial input
figure makes it obvious: Anscombe relabels 0 → 1.22 and 1 → 2.35. The signal is still binary. All
the continuity in the "latent" comes from the **smoother**.

**Kept anyway** (it compresses the high-count tail, and the rSLDS ground-truth calibration was
done with it — changing it now would invalidate that for no gain). But the consequence is the
real result:

> **Our fast rSLDS backend's Gaussian/PCA latent rests on a variance-stabilising step that
> stabilises nothing.** That is a *second, independent* argument — on top of "PCA picks variance,
> not dynamics" — that **`ssm`'s Poisson emissions are the principled model and our PCA latent is
> a concession.** It also explains *mechanistically* why the 20 ms latent came out shot-noise
> dominated and manufactured fake line attractors.

> **Lesson.** The transform was used for a year of project-time on the strength of what it says on
> the tin. One 10-line simulation at our actual firing rates showed it does not apply. **Check that
> an asymptotic result has reached its asymptote before relying on it.**

---

## 3. The frozen configuration

| knob | value | evidence |
|---|---|---|
| `align_event` | `stimOn_times` **+ `firstMovement_times`** | §1 |
| `pre_s` | **0.4 s** | IBL's enforced **quiescence period** is drawn from **[0.400, 0.700] s**. A 0.4 s baseline is inside the still-period on **100 %** of trials; the old 0.5 s steps outside it on **44 %**. Previous-trial contamination is impossible (min gap 2.7 s), so going *shorter* than 0.4 s would only discard clean baseline. |
| `post_s` | **0.8 s** | kept for continuity, **relabelled** (§1) |
| `bin_size_s` | **0.05 s** | rSLDS ground truth + GPFA timescales (107–283 ms). HMM comparability checked by `scripts/run_binsize_check.py`. |
| `min_rate_hz` | **1.0** | **nearly free**: keeps **96.7 / 99.5 / 99.4 %** of spikes. At 2 Hz **VISp collapses 19 → 10 units** and loses 19 % of its spikes — gutting the weakest region — while SCm/MRN barely move. |
| transform | **none on disk** | HMM needs raw counts; Anscombe applied downstream |

Frozen tensors: **VISp 19 / SCm 44 / MRN 54** units, 843 trials, 24 bins.

**MSR was tested as a unit-selection criterion and rejected again**, now on three grounds:
Spearman(rate, MSR) is high in VISp/SCm so an MSR cut *is* a rate cut in the regions where it
would matter; it was already a standing decision (it did not beat a random half in the one region
with a valid control); and it penalises stereotyped bursters, which SC/MRN contain. A rate floor
is arbitrary but *transparently* so.

---

## 4. `hmm.py` fixes

| function | status | why |
|---|---|---|
| `baseline_exit_latency` | **deprecated** | It returned `NaN` in **2 of 3 regions**, i.e. the inter-areal timing read-out never worked. Two bugs: it hard-coded baseline = *state 0* (the earliest-centre-of-mass state is not necessarily the pre-stimulus one), and it demanded a **single** state hold > 50 % of pre-stimulus bins — so a perfectly ordinary fit that splits baseline across two states was thrown away as a failure. |
| `baseline_states` | new | finds the baseline **set** from the data |
| `baseline_exit_latency_robust` | new | threshold **relative to baseline's own pre-stimulus level** (fall to half of it). Meaningful whether that level was 0.95 or 0.45 — which is exactly why the absolute-0.5 rule failed. |
| `switch_rate` | new | state changes per trial. **`forward_fraction` is an ASYMMETRY, not a FREQUENCY**, and the two were being conflated. VISp's 0.53 means its transitions are *symmetric*, **not** that it transitions rarely. |
| `forward_fraction_per_trial` | new | is the sequence there on single trials, or only in the pooled average? |
| `forward_fraction_split_half` | new | ordering derived on **held-out** trials, closing the ordering-then-testing-the-ordering circularity |

---

## 5. Corrections to the project's own doctrine

**"Pool the 5 replicate sessions — add trials, not filters"** (in `CLAUDE.md` and three worklogs)
is **too loose, and wrong for the models it was aimed at.**

A **pseudo-population** stitches neurons from different sessions into one population vector — but
those neurons were **never recorded simultaneously**, so there is no single-trial joint activity,
which is precisely what a state-space model consumes. It is the property we chose these sessions
*for* (Phase 1). Building a pseudo-population does not add trials to the HMM; **it destroys what a
trial is.**

| method | pooling across sessions |
|---|---|
| **HMM, rSLDS** | ❌ **not** as a pseudo-population. ✅ Fit each session separately and **pool the statistics** — "K=3 in 5/5 sessions, forward_fraction 0.77 ± 0.04" is a **replication**, and a stronger claim than one session with tighter error bars. |
| **dPCA** | ✅ **yes** — it works on trial-averaged activity, so a pseudo-population is standard and **directly fixes the K = 25 trials/cell problem**. |

Also corrected: **MRN's "K=3 beats the trivial split"** is numerically true (ARI 0.8819 vs 0.8802)
and **statistically empty** — the difference is 0.0017 against fold SDs of 0.04–0.05. More repeats
will *not* fix this; the two models are genuinely tied, not noisily measured. Only **replication
across sessions** can settle it. And **VISp's K=3 is not a peak** — its ARI curve declines
monotonically from K=2, so "K=3" is the smallest non-trivial K by construction, not a discovery.

---

## 6. The bin-size check — the HMM survives at 50 ms, and produces two new results

`scripts/run_binsize_check.py`, at the **final K (3 / 4 / 3)**, `pre_s=0.4`, units ≥ 1 Hz,
4 bin sizes × 3 regions, 3 stability repeats × 2 EM restarts.

### 6.1 One timebase is defensible — but 40 ms, not 50 ms

`forward_fraction`, `switch_rate` and the baseline-exit latency are **essentially flat across
20–50 ms**. The directedness result is **not** a bin-size artefact.

| | VISp | SCm | MRN |
|---|---|---|---|
| `forward_fraction` 20 → 50 ms | 0.592 → 0.594 | 0.785 → 0.833 | 0.779 → 0.790 |
| `switch_rate` (changes/trial) | 1.58 → 1.59 | 2.58 → 2.22 | 1.52 → 1.46 |

**The one cost: SCm's state stability degrades at 50 ms** — ARI **0.943 ± 0.013** at 20 ms and
**0.938 ± 0.025** at 40 ms, but **0.826 ± 0.129** at 50 ms. At 50 ms there are only 24 bins, and
fitting K=4 reproducibly into 24 bins is tight.

> **Recommendation: 40 ms as the common bin.** It preserves SCm's ARI (0.938), keeps every other
> read-out unchanged, and sits comfortably below the fastest GPFA timescale (107 ms). Caveat, and
> it is a real one: the **rSLDS synthetic ground truth is only validated at 20 ms (fails) and
> 50 ms (works)** — 40 ms is *untested*. **Re-run the rSLDS ground truth at 40 ms before
> adopting it**; that is cheap (synthetic data) and it is the only thing standing in the way of a
> single project-wide timebase. If 40 ms fails the ground truth, use 50 ms and accept SCm's
> looser ARI.

### 6.2 ⚠️ The split-half test caught something — MRN's directedness is partly circular

`forward_fraction` needs a state ordering, and we take it from occupancy centre-of-mass measured
on **the same trials** whose transitions we then score. `forward_fraction_split_half` derives the
ordering on **held-out** trials instead:

| region | pooled (50 ms) | **split-half** | verdict |
|---|---:|---:|---|
| VISp | 0.594 | **0.590 ± 0.003** | unchanged — real |
| SCm | 0.833 | **0.833 ± 0.013** | unchanged — real |
| **MRN** | 0.790 | **0.666 ± 0.102** | ⚠️ **drops** |

**Part of MRN's apparent directedness came from ordering-the-states-then-testing-the-ordering.**
It is still above 0.5, but weaker than advertised, and the variance is large. **Quote the
split-half number for MRN from now on.** VISp and SCm are untouched by the test, so their numbers
were honest.

### 6.3 VISp is UNDIRECTED, not static — the dissociation, measured

**VISp switches state as often as MRN (1.58 vs 1.52 changes/trial) — it simply switches with no
preferred direction (0.59 vs 0.79).** This is the clean dissociation between `forward_fraction`
(asymmetry) and `switch_rate` (frequency), and it settles a reading that was on the table: VISp
is *not* "mostly static with one main transition" — that description would produce a **high**
forward_fraction. It is genuinely **undirected**.

### 6.4 NEW: the baseline-exit latency now works, and gives the visual→motor cascade

The old function returned `NaN` in 2 of 3 regions. The fixed one runs everywhere, and:

| | VISp | SCm | MRN |
|---|---:|---:|---:|
| baseline exit (ms) | **~90** | **~170** | **~450** |

**VISp → SCm → MRN, exactly the predicted ordering**, and stable across all four bin sizes. The
old function failed on MRN because **MRN's baseline state is state 1, not state 0** — precisely
the hard-coded assumption that was wrong.

> ⚠️ **But read it against §1 before celebrating.** SCm's exit (~170 ms) sits on the **median
> reaction time (171 ms)**. MRN's (~450 ms) sits on the **median feedback time (427 ms)**. So the
> "cascade" may be nothing more than each area locking to the task event it is already known to
> lock to — vision, then movement, then reward — rather than an internal decision sequence. The
> movement-aligned tensor is the control that can separate these, and it has not been run.

---

## 7. rSLDS ground truth — rebuilt, run at 40 ms, and it changed the story

`scripts/validate_rslds_ground_truth.py` (**new, committed**) →
`results/rslds/ground_truth_binsize.json`.

### 7.1 ⚠️ The old validation had no code and was not reproducible

The bin-size table in `rslds.py`'s docstring — the basis for *"20 ms fits are inadmissible"*,
the single most load-bearing methods decision in the project — **was run ad hoc and only the
numbers were saved.** The rebuild (OU attractors simulated in continuous time at 1 ms, then
binned, so the truth is identical across bin sizes) matches its **20 ms** row closely but **not**
its 50 ms row. The old "|λ| = 0.875 at 50 ms" **is retracted**; do not quote it.

### 7.2 40 ms is safe everywhere — the answer to the open question

Spurious "integrator" regimes (truth = 0 of 3), sigma held **fixed at 40 ms**:

| bin | benchmark | VISp-like | MRN-like | SCm-like | regime acc (SCm-like) |
|---|---|---|---|---|---|
| 20 ms | 3/3 | 1/3 | 3/3 | 3/3 | 0.73 ← **fails** |
| 30 ms | 1/3 | 0/3 | 0/3 | 0/3 | 0.92 marginal |
| **40 ms** | **0/3** | **0/3** | **0/3** | **0/3** | **0.94 ← safe** |
| 50 ms | 0/3 | 0/3 | 0/3 | 0/3 | 0.94 safe |

**40 ms passes the rSLDS ground truth AND preserves SCm's HMM stability** (which 50 ms degraded,
§6.1). **One timebase for the whole project: 40 ms.**

### 7.3 ⚠️ The reason the project believed in "50 ms" was partly an artefact

The old validation held `smooth_sigma_bins = 1.5` — sigma in **bins** — so it smoothed with 30 ms
at 20 ms bins but **75 ms** at 50 ms bins. **The bin-size sweep was secretly also a smoothing
sweep.** Reproducing that convention, `|λ|` climbs *monotonically* toward 1 as the bin grows
(0.951 → 0.974 → 0.990 → 0.996) and **50 ms comes out WORSE than 20 ms** — because a Gaussian
kernel imposes the very autocorrelation an AR fit reads as a slow mode. So `|λ| → 1` has **two**
causes — shot noise (small bins) *and* over-smoothing (large sigma) — and the original comparison
confounded them. Same family as the geometry σ-sweep and the K-on-a-boundary traps.

### 7.4 VISp's rSLDS fit will be the least trustworthy — quantified

At VISp's real size (19 units, 3.5 Hz), regime recovery is only **0.71** even at 40 ms, and
switches **undershoot** (1.6 vs true 2.0) — while the *order* is recovered perfectly (OC 1.00).
So: **trust VISp's sequence, distrust its regime boundaries and its regime count.** Consistent
with everything else about VISp.

### 7.5 What was cleaned in the code

- `counts_to_latents` now takes **`smooth_sigma_ms` + `bin_size_s`** (sigma-in-bins deprecated
  with a warning). Sigma-in-ms is bin-invariant, so a bin sweep can no longer confound smoothing.
- `cross_validate_K` signature updated to match; defaults 40 ms / 40 ms.
- **New `RSLDSResult.admissibility()`** — flags the shot-noise failure mode mechanically
  (`|λ| ≥ 1.02`, all regimes integrators, or flickering) so the "we found an integrator" trap is
  caught by code, not by memory. Returns `admissible=False` + a reason.
- Docstring rewritten with the reproducible table; the stale *"ssm does not install"* section
  corrected (it installs, in `ndrh_ssm`, since 2026-07-13).

---

## Open

1. **Run the movement-aligned control.** It is built and unused, and §6.4 now *needs* it.
2. **Point `01`–`04` at `load_processed(...)`** and delete their private tensor-building cells.
   Notebook 03 additionally needs the stale K fixed (VISp 2 → 3) and the window corrected.
3. **The recurrence test (`delta_recurrent`)** — still the project's biggest unanswered question,
   and the thing the "test it" pass should target (see §8).
3. **Notebook 03 still holds a stale K** (`VISp 2`) and the retracted "VISp is a negative result"
   text. `best_K_per_region.md` says **VISp 3**.
4. **The movement-aligned control has not been run** on anything yet — it is built, not used.
5. **The recurrence test (`delta_recurrent`)** remains the project's biggest unanswered question.
6. `config/areas.yaml::selected` **still empty** (fifth worklog in a row).
