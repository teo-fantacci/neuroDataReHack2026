# 2026-07-13 (2) — dPCA, GPFA, RRR, and one common manifold (`04_manifold_dpca_gpfa_rrr.ipynb`)

Executed end-to-end on `5c0c560e` (VISp 36 / SCm 59 / MRN 63 units, 843 trials). Figures in
`results/figures/phase3_manifold/`, tables in `results/manifold/`. New src modules:
`src/geometry/{embed,joint}.py`, `src/dynamics/{dpca,gpfa,rrr}.py`.

New deps, dry-run-verified additive **and then smoke-tested before use**: `dPCA`, `elephant`
(GPFA), `numexpr`. **`dPCA` has an undeclared runtime dependency on `numexpr`** — it installs
happily and then `ImportError`s. "It installed" is not "it works"; that is now pinned.

---

## 1. ⚠️ The smoothing sweep cannot choose the smoothing — and the metric that "chooses" it is circular

Swept 10 / 20 / 50 ms bins x 0 / 20 / 40 / 80 / **160** ms Gaussian sigma (sigma in
**milliseconds**, not bins — otherwise changing the bin silently changes the smoothing too).

| sigma (ms) | time R² | choice AUC | PC1+2 var |
|---:|---:|---:|---:|
| 0 | 0.319 | 0.858 | 0.157 |
| 20 | 0.389 | 0.859 | 0.225 |
| 40 | 0.422 | 0.860 | 0.269 |
| 80 | 0.451 | 0.860 | 0.301 |
| 160 | **0.479** | 0.854 | 0.319 |

`time_R2` rises **monotonically** and its argmax sits on whatever boundary you happen to
sweep to. Raising the ceiling from 80 to 160 ms simply moved the "winner". The reason is not
subtle once seen: **a Gaussian kernel imposes temporal autocorrelation, and `time_R2`
measures how predictable time is from the smoothed signal.** Maximising it maximises the
smoother. It is a hair away from circular, and it would have silently justified smoothing the
data until the trajectory we wanted appeared.

The notebook now **detects the boundary and refuses the argmax** (same discipline as the
Phase-2 `K=7`-on-the-boundary lesson).

**`choice_AUC` is flat across the entire sweep (0.854–0.860).** It cannot select the
configuration either — and *that flatness is the actual result*: **the choice signal does not
depend on bin size or smoothing at all**, so nothing downstream rests on this knob.

**Configuration chosen on external grounds and stated openly: 50 ms bins, sigma 40 ms.**
50 ms because the rSLDS ground-truth (`src/dynamics/rslds.py`) recovers real attractors there
and shot-noise at 20 ms; sigma 40 ms because it sits comfortably below the **fastest** GPFA
timescale fitted below (~107 ms), so it denoises without blurring the latent it is meant to
reveal.

---

## 2. dPCA — the project's central prediction, tested with a number for the first time

Variance decomposed by **cause** (Kobak et al. 2016). The prediction was: VISp carries
relatively more *stimulus* variance, SCm/MRN more *decision* variance.

| region | stimulus % | decision % | time % | **decision − stimulus** |
|---|---:|---:|---:|---:|
| VISp | 11.8 | 5.0 | 74.1 | **−6.8** |
| SCm | 4.5 | 6.3 | 63.5 | **+1.8** |
| MRN | 8.3 | 14.3 | 49.2 | **+6.0** |

**The ordering comes out exactly as predicted** — VISp stimulus-dominated, MRN
decision-dominated, SCm between — and the condition-independent (time) share *falls*
monotonically along the visual→motor axis. This is the strongest evidence the project has so
far, and it is the first result that speaks to the claim rather than to the machinery.

**Two things that stop it being a claim yet:**

1. **The time component dominates everywhere** (74 / 64 / 49 % of explained variance). Most
   of what any raw PCA trajectory shows is this, in *every* area. It is exactly the trap
   dPCA exists to avoid — and a reason to distrust any story told from a plain PCA
   trajectory plot, ours included.
2. **The design balances to K = 25 trials per (stimulus x decision) cell.** The off-diagonal
   cells are *error trials* and mice are ~85% correct, so the decision components rest on 25
   trials. They are noisy. **Pool the 5 replicate triplet sessions before believing the
   ordering** — the same conclusion the rSLDS work reached from a different direction
   (trials, not neurons, are the limiting resource).

---

## 3. GPFA — the dynamics are slower than any bin we have been arguing about

Fitted GP timescales (`tau`) per latent dimension, on **raw counts** (GPFA infers the
smoothness instead of having it imposed — so it is the check on whether our smoothed
trajectories were our own doing):

| region | fitted tau (ms), slowest → fastest |
|---|---|
| VISp | 176, 152, 125, 111, 107 |
| SCm | 226, 174, 153, 145, 107 |
| MRN | **283**, 232, 228, 198, 117 |

* Every timescale is **far above 50 ms**, so the bin-size debate is settled from the data's
  side: 50 ms bins lose nothing.
* The **motor areas are slower than the visual one** (MRN's slowest latent is 283 ms vs
  VISp's 176 ms) — an independent echo of the HMM's directedness result (SCm/MRN more
  directed than VISp).
* It also justifies sigma = 40 ms: below the fastest real timescale (107 ms).

---

## 4. ⚠️ RRR — SCm and MRN behave almost like ONE population, not two areas with a channel

Reduced-rank regression (Semedo et al. 2019), CV grouped **by trial**. Read against the
**within-area control** (predict half an area's units from its other half), never against zero.

| source → target | best held-out R² | saturation rank | | control for that target | R² | rank |
|---|---:|---:|---|---|---:|---:|
| SCm → MRN | 0.414 | 7 | | MRN half → half | 0.470 | 8 |
| MRN → SCm | 0.366 | 7 | | SCm half → half | 0.383 | 7 |
| VISp → SCm | 0.116 | 4 | | SCm half → half | 0.383 | 7 |
| VISp → MRN | 0.165 | 5 | | MRN half → half | 0.470 | 8 |

**MRN predicts SCm (R² 0.366, rank 7) almost as well as SCm predicts itself (0.383, rank
7).** There is **no restricted communication subspace** between SCm and MRN — the areas share
essentially everything they have, at the same dimensionality.

This is the *third* independent method now pointing the same way: the pynapple
cross-correlogram showed a sharp **zero-lag** peak (common drive, not a directed exchange),
and the HMM found both areas similarly directed. **The SCm↔MRN "feedback" hypothesis is
looking less like a directed loop and more like two readouts of a shared signal** — which is
a real finding, and one that a joint multi-region rSLDS should now be asked to confirm or
overturn rather than assume.

VISp, by contrast, shares little with either (R² 0.12–0.17) — consistent with the sparse,
weakly-decodable VISp population seen everywhere else in this project.

---

## 5. The common manifold — and why stacking the areas naively would have been an artefact

**You cannot simply embed three areas together.** A VISp state vector lives over 36 VISp
neurons, an MRN one over 63 MRN neurons — *different spaces*. Stacked and handed to t-SNE,
the areas would separate **trivially**, because their axes differ, and the picture would be
pure artefact. Worth stating plainly, because that plot is easy to make and looks convincing.

What licenses a common space here is that the three areas are **simultaneously recorded**:
they share the same (trial, bin) samples. So: concatenate all 158 units, take the **joint
PCA** (PC1 19.1%, PC2 9.4%, PC1–10 50.0%), and — because PCA is *linear* — decompose the
joint trajectory **exactly** into per-area contributions, `Y(t) = Σ_a Y_a(t)`, using each
area's own loadings. `src/geometry/joint.py` **asserts** that identity rather than assuming
it; it is the entire justification for plotting the three areas on one pair of axes.

| region | share of joint latent variance | n units |
|---|---:|---:|
| VISp | 0.056 | 36 |
| SCm | 0.222 | 59 |
| MRN | 0.307 | 63 |

The shares sum to **0.585** — the areas' contributions are not orthogonal, and **the missing
~0.42 is cross-area covariance**, i.e. the shared signal the whole project is arguing about.
That number is arguably the most interesting one in this notebook.

Figures: contributions coloured by area; then **shape = area, colour = time** (so neither
channel does double duty, and the area distinction survives red–green deficiency *and*
greyscale); then a nonlinear version (PHATE run on the shared linear basis, since a nonlinear
map cannot be partially projected).

*Caveat kept in view:* units are z-scored, so loudness is controlled — but **unit count is
not**. VISp's small share is partly its 36 units. Compare at matched unit count before
claiming MRN "drives" the shared state.

---

## 6. Embedding quality — and what does not have a quality metric

Variance explained is now printed **on the PCA axes** (asked for, and it is the honest one:
"PC1+2 = 27–34%" says most of the geometry is off-screen). Each method reports what it
actually has: MDS **stress-1** (computed explicitly — `sklearn`'s `stress_` normalisation has
changed across versions), t-SNE **KL**, Isomap **reconstruction error**.

**UMAP, Laplacian eigenmaps and PHATE define no goodness-of-fit scalar.** Anyone quoting a
"variance explained" for UMAP is inventing it. So all seven also get two *common* metrics:
**trustworthiness** and **stress-1**.

The result is a clean signature: t-SNE/UMAP have the **highest trustworthiness** (0.89–0.94)
and the **worst stress-1** — they preserve neighbourhoods and destroy global geometry. MDS is
the reverse (stress-1 0.24–0.27). **So t-SNE/UMAP cannot support a claim about trajectories
or distances, only about neighbourhoods** — and their apparent "clusters" are not evidence of
discrete states. That is the HMM's job.

---

## Open

1. **Pool the 5 replicate triplet sessions.** dPCA's decision components (K=25/cell) and any
   rate-matched control both need trials, not neurons. This is now the single highest-value
   next step, and three separate analyses have independently arrived at it.
2. The dPCA ordering (VISp → SCm → MRN) is the project's core claim and currently rests on
   **one session**. Replicate before it goes in a figure.
3. Joint multi-region rSLDS for SCm↔MRN — but note it now has to *overturn* three converging
   pieces of evidence for common drive, not merely confirm a feedback loop.
4. Compare areas at **matched unit count** in the joint manifold before attributing "drive".
5. `config/areas.yaml::selected` still empty.
