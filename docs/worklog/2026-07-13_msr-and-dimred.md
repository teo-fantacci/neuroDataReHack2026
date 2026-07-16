# 2026-07-13 — MSR unit selection + first dimensionality reduction (`02_dimred_msr.ipynb`)

New notebook `notebooks/02_dimred_msr.ipynb`, run end-to-end on the working session
`5c0c560e` (VISp 36 / SCm 59 / MRN 63 units, 843 trials, 20 ms bins). Everything below
comes from that executed run — figures in `results/figures/phase3_dimred/`, tables in
`results/dimred/`.

---

## 0. `src/selection/msr.py` had never been executed

It was written, docstring'd (*"validated on synthetic trains ... see the notebook"*) and
imported by **nothing**. The notebook that was supposed to validate it did not exist. First
job was therefore to build that gate, and it immediately earned its keep.

Also added `build_matrices.region_spike_trains()`: each unit's **continuous** spike train
keyed by **its column index in `region_tensor_from_probes`**. MSR needs continuous time
(it sweeps dt from 2 ms to 1 s; a trial window is only 1.3 s), but the ranking it produces
must index the tensor. The notebook **asserts** the alignment (re-bins every train inside
the trial windows and requires the totals to equal the tensor's) rather than assuming it —
a mismatch here would have silently kept *the wrong half of the population* with nothing
looking broken. Verified on all 36/59/63 units.

---

## 1. ⚠️ MSR gives a *low* score to a perfectly stereotyped burster — and that is correct

The first gate **failed**: at matched rate (5 Hz, 600 s), fixed-size bursts scored
**below** homogeneous Poisson (0.031 vs 0.040), and my "multiscale" train was dragged down
with it. That looked like a bug in `msr.py`. It is not.

**Relevance `H[k]` is the entropy of the frequency of frequencies** — of `m_k`, the number
of bins holding exactly `k` spikes. If every burst contains *exactly 4* spikes, then every
occupied bin holds the same count, `m_k` is a **delta**, and `H[k] -> 0`. The train has
structure but no **variety of multiplicities**, and variety of multiplicities is precisely
what relevance measures.

Draw the burst sizes from a geometric distribution instead and the gate passes cleanly:

| train | MSR | rate |
|---|---:|---:|
| multiscale (variable bursts + slow modulation) | 0.0608 | 5.03 |
| bursty (variable size) | 0.0564 | 4.92 |
| slow_mod | 0.0500 | 4.95 |
| **poisson (the null)** | **0.0395** | 5.04 |
| bursty_fixed (exactly 4 spikes/burst) | 0.0219 | 5.08 |

`bursty_fixed` is **kept in the notebook permanently** as a documented degeneracy, because
it has a real consequence for our data: **MSR can penalise a genuine neuron with highly
stereotyped bursts.** SC and MRN contain bursting cells. Do not read a low MSR as
"uninformative unit" without checking whether it is a stereotyped burster.

> **Lesson.** The gate failed, and the *implementation was fine* — my expectation was
> wrong. Which is the point of a gate: it fired before the statistic was used to throw
> away half the units. Had I skipped it and trusted the docstring, I would have shipped a
> selection built on a misunderstanding of what the measure rewards.

---

## 2. ⚠️ The rate-matched control silently degenerated into "the slow half"

`msr.py`'s own docstring warns that MSR rises with firing rate and demands a rate-matched
control. It has one — `rate_matched_control()` — and **it failed quietly in 2 of 3
regions**, because of a structural problem the function cannot fix:

MSR correlates with rate (Spearman rho: **VISp 0.64, SCm 0.55**, MRN 0.08). So the top-MSR
half *is* the loud half, and the leftover pool **contains no units loud enough to match**.
`rate_matched_control` then falls back to "nearest available" (as documented) and returns
what is simply the quiet half:

| region | msr_top median rate | "rate-matched" control | ratio | matched? |
|---|---:|---:|---:|:--|
| VISp | 3.17 Hz | 0.92 Hz | 3.5x | **no** |
| SCm | 20.92 Hz | 4.52 Hz | 4.6x | **no** |
| MRN | 6.97 Hz | 7.89 Hz | 1.1x | yes |

So the eye-catching result — **SCm choice-decoding AUC 0.943 (msr_top) vs 0.787
(rate_ctrl)** — is **not usable**. It is a 4.6x firing-rate difference wearing a costume.
The notebook now **hatches those bars in the figure** and marks the rows `matched=False`.

**Fix added:** `msr_resid` — rank by MSR **residualised on log firing rate** ("dynamically
rich *for its rate*"). Uncorrelated with rate by construction, so a real control exists.
It helps but does not fully rescue VISp/SCm (ratios still 1.6x / 2.7x): with ~30–60 units
the pool is just too small to match a heavy-tailed rate distribution. **MRN is the only
region where the comparison is trustworthy.**

---

## 3. What the honest comparison actually says

Held-out decoding, `GroupKFold` **by trial** (bins of one trial are correlated — mixing
them across folds leaks and inflates everything), fixed 10-PC space so no set is rewarded
for having more dimensions.

| region | metric | all | msr_top | msr_resid | rate_ctrl | random |
|---|---|---:|---:|---:|---:|---:|
| MRN (**the trustworthy row**) | choice AUC | 0.935 | 0.934 | 0.934 | 0.872 | **0.939** |
| MRN | time R² | 0.550 | 0.483 | 0.483 | 0.502 | 0.482 |
| SCm | choice AUC | 0.946 | 0.943 | 0.942 | *0.787* | 0.913 |
| SCm | time R² | 0.608 | 0.578 | 0.605 | *0.539* | 0.530 |
| VISp | choice AUC | 0.592 | 0.612 | 0.605 | *0.551* | 0.605 |

*(italics = control failed its rate match; not evidence)*

1. **In MRN — the one region with a real control — the top-MSR half is indistinguishable
   from a RANDOM half** (0.934 vs 0.939 AUC). MSR bought nothing there.
2. **Halving is nearly free for choice, but not for time.** Choice AUC barely moves
   (SCm 0.946→0.943, MRN 0.935→0.934) while time R² drops (MRN 0.550→0.483). Choice
   information is redundant across the population; the *temporal trajectory* is not — it
   needs the whole population. **Relevant to Phase 2/3: do not fit the HMM on half the
   units.** Selection is fine for decoding, not for dynamics.
3. `msr_resid` preserves time R² better than raw `msr_top` in SCm (0.605 vs 0.578, against
   0.608 for all units) — mildly interesting, on an unmatched control, so: a lead, not a
   result.

**Bottom line: MSR is not (yet) earning its place on this data.** It is defensible as a
cheap unsupervised compression (half the units, ~99% of the choice signal), but there is
no evidence here that it beats firing rate — and in the only region where that could be
tested fairly, it did not beat *chance selection*.

---

## 4. Non-circular cross-check: are MSR units task-selective?

MSR never sees the stimulus, so this is a real question (project-notes item 1). Mann-Whitney
on 50–250 ms counts, stimulus-right vs -left, 0%-contrast trials dropped:

| region | msr_top frac p<0.05 | msr_bottom |
|---|---:|---:|
| SCm | **0.679** | 0.393 |
| MRN | 0.548 | 0.613 |
| VISp | 0.308 | 0.308 |

**Only SCm** shows the hoped-for enrichment. MRN goes slightly the *wrong* way and VISp is
flat. So "unsupervised dynamical richness lands on task-relevant cells" is **not a general
truth here** — it holds in one of three areas. (Uncorrected p; the top-vs-bottom contrast
is relative, so the correction cancels.)

---

## 5. Dimensionality reduction — what was run

Points = **(trial, 20 ms bin) population vectors** (54,795 per region), sqrt-transformed
(variance-stabilises Poisson counts), smoothed (sigma = 2 bins = 40 ms), z-scored per unit.
Same random subsample (2,000 points; 700 for MDS) for **every** method and unit set, so
differences are the method or the units, never the points. Geometry only — **the HMM still
gets raw counts**.

Run per region x {all units, top-half MSR}: **PCA, MDS, t-SNE, UMAP, Isomap, Laplacian
eigenmaps, PHATE** (~115 s for all 42 embeddings). Plus a neuron-space view (each point a
unit, features = z-scored PSTH shape, coloured by MSR).

New deps, dry-run-verified additive *before* installing (the `ssm` lesson) and now pinned
in `requirements.txt`: **`umap-learn`**, **`phate`**.

**Standing warning written into the notebook:** t-SNE and UMAP **manufacture clusters from
continuous data**. Apparent "discrete states" in those panels are *not* evidence of
metastable states — that is the HMM's job. PCA/MDS/PHATE are the ones whose global
geometry means something; PHATE is the one actually built for trajectories.

Everything except the §7 decoding table is fitted and displayed on the same trials. **Only
the decoding table is evidence.**

---

## Suggested next methods (full list + pros/cons in the notebook's §8)

- **dPCA** — the one I would run next. Splits population variance into time / stimulus /
  choice / interaction components, which is *the* project question ("how much of this
  region's geometry is decision-related?"). PCA cannot answer it.
- **GPFA** — dimensionality reduction *with a time prior*; the classic single-trial latent.
- **T-PHATE** — PHATE plus an explicit temporal-autocorrelation kernel (already named in
  the project notes).
- **CCA / reduced-rank regression / communication subspaces** — for the open SCm↔MRN
  question; the geometric counterpart of the HMM flow argument.
- **CEBRA** — powerful but **supervised**: it cannot be used to argue a manifold exists
  independently of the labels you fed it.

## Open

1. **`config/areas.yaml::selected` is still empty** (unchanged; still clearly `{VISp, SCm, MRN}`).
2. Does the MSR cut change the *HMM* (not just decoding)? Point 3.2 above predicts it hurts.
3. VISp/SCm rate-matching is unfixable within one session (too few units). Pool the 5
   replicate triplet sessions to build a real matched control?
4. Re-run Phase 2 CV at `MAX_TRIALS=None` and read the CV *curve* — still open from
   2026-07-12.
