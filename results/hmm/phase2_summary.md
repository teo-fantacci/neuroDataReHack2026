# Phase 2 — HMM cross-validation & directedness

> ## ⚠️ RETRACTED (2026-07-12): the "best K = 2" result was an artifact
>
> The previous version of this file reported **"CV picks K = 2 for every region"** and read a
> story into it ("a parsimonious baseline + evoked description is what generalises"). **That
> was not a result.** Every candidate K scored a held-out log-likelihood of **`-inf`**, and
> `argmax` over a column of `-inf` returns index 0 — i.e. the *smallest* K in the list. The
> number 2 was the first entry of `CV_KS`, nothing more.
>
> **Root cause.** `hmmlearn` fits `lambdas_ == 0` for a (state, neuron) pair that never spiked
> in training. A Poisson with rate exactly 0 assigns probability 0 — log-likelihood `-inf` —
> to any count > 0. VISp has units with ~2 spikes in the entire analysis window; under a
> 250-trial CV subsample such a unit can be **completely silent in a training fold**, so every
> state learns λ = 0 for it, and a single held-out spike sends the fold score to `-inf`.
>
> **The tell we ignored.** The CV *plot* was blank — a title, axes and a full legend over no
> data. matplotlib draws exactly that when handed non-finite arrays: `errorbar` still creates
> the legend handle, but with no finite points the axes never autoscale. The blank figure was
> the bug announcing itself, and it was read as a rendering glitch.
>
> **Fixed** in `src/dynamics/hmm.py`: drop units that are all-zero in the CV tensor, floor
> `lambdas_` at 1e-6, and **raise** if every K is non-finite rather than taking an argmax.

---

## What survives

Session `5c0c560e`, 400 trials (250 for CV), 20 ms bins, 0.5 s pre-stimulus baseline.
Regions **{VISp, SCm, MRN}** — VISpm dropped, see the worklog.

### 1. The binning is correct — verified independently

`build_matrices.region_tensor_from_probes` was cross-checked against **pynapple**'s
`TsGroup.count()` (an independent implementation) on the real session:

| region | our spikes | pynapple | per-(trial,bin) identical |
|---|---:|---:|---|
| VISp | 42,085 | 42,085 | **yes** |
| SCm | 500,404 | 500,404 | **yes** |
| MRN | 445,569 | 445,569 | **yes** |

Nearly a million spikes, agreeing on every bin. **The tensor feeding the HMM was never the
problem** — the bug was confined to CV scoring. This is what adopting pynapple bought.

### 2. Directedness (`forward_fraction`) — unaffected by the CV bug

Computed from Viterbi paths on a single fit, so it never touched the held-out score. Motor
midbrain remains clearly more directed than visual cortex, and the earlier shuffle null had
these significant at p<0.001. **But every one of these was measured at a K chosen by the
broken CV**, so they must be recomputed once K is settled honestly.

### 3. Cross-validated K — NOT YET DETERMINED

With the `-inf` fixed, held-out LL is finite and **rises monotonically with K** across
2–7 for all three regions — the *opposite* of the retracted "parsimony" story. But the
argmax landed on **7, the top of the search range**, which is a boundary, not a choice: the
curve had not plateaued.

The sweep is now `CV_KS = [2 … 10]` with an explicit **boundary warning**. If it still has
not plateaued by K=10, that is not a licence to keep raising K — it is evidence that
something is wrong with the model or the preprocessing (e.g. sparse units contributing
likelihood without contributing dynamics), and the approach should be reconsidered.

## Caveats

- One session, subsampled. Repeat on the 5 replicate `{VISp, SCm, MRN}` sessions.
- **Read the CV curve, not the argmax.** That habit is what would have caught this on day one.
- The SCm↔MRN cross-correlogram (see `results/figures/phase2_nap/`) shows a **sharp peak at
  zero lag on a broad elevated baseline** — the signature of **common drive**, not a directed
  SCm→MRN exchange, at the millisecond timescale. This does not kill the feedback hypothesis
  (HMM states live at 20 ms / hundreds of ms, a different timescale), but it means the
  spike-timing evidence does **not** support a lead–lag, and the joint model must carry the
  argument.
