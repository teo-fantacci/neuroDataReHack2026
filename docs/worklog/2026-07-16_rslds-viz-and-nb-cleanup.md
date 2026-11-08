# 2026-07-16 — rSLDS visualization, notebook-03 cleanup, and the Q&A explainer

Companion to [`2026-07-15_rslds-first-real-test.md`](2026-07-15_rslds-first-real-test.md), which
holds the rSLDS **analysis** (screen, `ssm` fit of record, `|λ|`-stability, recurrence test). This
entry records the things that chat produced *after* the analysis and had **not** yet been written
down: the **visualizations**, the **notebook-03 restructuring**, the **Q&A explainer doc**, and a
**`pre_s` bookkeeping discrepancy** the next chat needs to know about.

---

## 0. ⚠️ READ FIRST — the `pre_s` discrepancy (pre400 vs pre300)

The whole rSLDS analysis and every visualization below were computed on the **`pre400`** 40 ms
tensors (`data/processed/5c0c560e__stimOn_pre400_post800_bin40_rate1.npz`, 30 bins, 0.4 s baseline).

Meanwhile the parallel preprocessing chat changed the config default to **`pre_s = 0.3`** (2026-07-16;
`PreprocessConfig`, `build_processed_40ms.py`). Both `pre300` and `pre400` tensors exist on disk for
all 5 sessions, so **nothing is broken right now** — but:

- All rSLDS result JSONs (`results/rslds/{screen,ssm_record,lambda_stability,recurrence}_40ms.json`)
  and all four GIFs are **pre400** numbers.
- The rSLDS scripts and `notebooks/06` **hard-code the `pre400` filename**
  (`..._pre400_post800_bin40_rate1.npz`).

**Decision for the next chat:** either (a) accept pre400 for the rSLDS strand (the baseline length
is a 0.1 s cosmetic difference and the conclusions — sequence certified, geometry retired — are
robust to it), or (b) re-point the four `run_rslds_*.py` scripts + `notebooks/06` at the `pre300`
files and re-run (~1.5 h for the `ssm` fit; the rest are minutes). If you unify, do it once and
note it. Do **not** leave half the project on pre300 and the rSLDS on pre400 silently.

---

## 1. rSLDS visualizations — new module + notebook `06`

*(This notebook was created as `05_results_visualization` and renamed to `06` by the repo-cleanup
chat on 2026-07-16 due to a number collision with `05_hmm_summary`.)*

**`src/viz/rslds_anim.py`** (new, reusable) and **`notebooks/06_results_visualization.ipynb`** (new,
self-contained, runs clean, ~3 min). Turns the fitted rSLDS into animations so the *certified*
result — a reliable, position-dependent **sequence** of regimes — is watchable. Two builders, one
figure of each **per alignment** (stimOn + firstMovement), written to `results/figures/phase5_viz/`:

- **`single_trial_gif`** — one representative trial: a marker walks each area's latent path,
  flashing on each regime switch, with a **3-lane regime tape** (▼ at each switch) building up
  underneath. The intuition version. `pick_representative_trial` chooses a trial matching each
  area's modal sequence, preferring one that shows a post-onset switch in as many areas as
  possible (VISp usually sits in one regime — honestly shown, its modal sequence *is* a single
  state, consistent with its weak directedness).
- **`occupancy_sweep_gif`** — the **trustworthy** version: a time cursor sweeps the trial-averaged
  regime occupancy of all three areas at once, with a **switch-time density** panel below (mean
  switches/trial entering each bin, per area). The *order of the density peaks is the cross-area
  cascade* (SCm ~+0.06 s, MRN ~+0.18 s under stimOn) — direction-agnostic, certified, and it never
  touches the flow geometry.

**The guardrail, enforced in every panel:** the latent-plane *shape* is captioned "illustrative (a
smoothing artefact) — watch the switches." The load-bearing content is the regime tape and the
switch-time density; the attractor shapes are not claimed (per the `|λ|`-stability finding).

**One fix inside the module (2026-07-16):** the single-trial latent panels first looked "extremely
compressed" — a single trial fills only ~15-19 % of the axis box when the box is set to the full
(all-trials) cloud, because most of that cloud's extent is *cross-trial* variability, not
within-trial motion (itself a symptom of the low single-trial SNR). `single_trial_gif` now zooms the
axes to **that trial's** extent + padding (`_latent_limits(..., trial_idx=)`), so the path fills the
panel.

To regenerate on the 5-session pool later: the builders take any `{region: RSLDSResult}` dict, so
it is a one-line change once those fits exist.

---

## 2. Notebook `03_rslds_triplet.ipynb` — restructured

### 2.1 Part 10 added — a self-contained certified-results dashboard

New **Part 10** (self-contained; does *not* depend on the stale Parts 1–9). It loads the frozen
40 ms tensors + the result JSONs and produces three figures:

- **Figure 1** (2×2): (a) `delta_recurrent` with per-fold error bars, both alignments; (b) the
  position-shuffle gap (the MRN ≈ SCm ≫ VISp gradient); (c) `order_consistency` vs its shuffle
  null; (d) **`|λ|` real vs time-shuffled** — the two bars sit on top of each other, so you can
  *see* the geometry is the smoother.
- **Figure 2**: regime occupancy over the trial, one row per area (the sequence made visible).
- **Figure 3**: latent trajectories, explicitly labelled illustrative.

Verified: all cells run headless, figures are real (checked — not the blank-plot trap). Ships with
outputs cleared; the PNGs are in `results/figures/phase3_rslds_results/`.

### 2.2 Part 6 (flow geometry) — REMOVED

The old **Part 6** ("what kind of dynamical object is each regime" — fixed points, eigenvalue
classification, the flow-field "money figure") was **deleted** and replaced with a short markdown
note recording *why*: the flow geometry is a **smoothing artefact** (`|λ|`-stability test), not
certifiable from this data, and both backends hit `|λ| ≈ 1` by the same mechanism. The certified
content (sequence, recurrence) uses regime *order* and *switching*, not `A`'s eigenvalues, and is
untouched. The geometry code still lives in `src/dynamics/rslds.py` (`fixed_points`, `eigenvalues`,
`classify`, `flow_field`, gated by `admissibility()`) and git history.

**Bug found on the way out, worth keeping:** `plot_flows` mis-scaled ("compressed subplots") for a
diagnosable reason — it plotted every regime's fixed point `x* = (I−A)⁻¹b` regardless of position,
and for a near-integrator (`|λ| → 1`) that point **diverges** (near-singular `I−A`), blowing up the
axes and squashing the data into a corner. Outlier scatter points did the same. The fix (pin the
axes to the 1–99 pct data cloud; mark off-view fixed points with an edge arrow) was applied and
verified before the section was removed — so it is in git history if the geometry is ever revived.

### 2.3 Other notebook-03 fixes

- **Stale `K_PER_REGION` fixed**: `VISp 2 → 3` (the old K=2 was a 1-restart artefact; final is
  3/4/3 per `best_K_per_region.md`). Intro prose updated to match.
- **Status banner** added to the intro: sequence/recurrence certified, geometry retired.
- ⚠️ Parts 1–9 still run the **stale config** (50 ms, `smooth_sigma_bins`, own tensor building).
  Only Part 10 reflects the unified 40 ms pipeline. Migrating 1–9 to `load_processed(...)` is still
  open (same as notebooks 02/04).

---

## 3. New doc: `docs/knowledge/project_explained.md`

A **pedagogical Q&A explainer**, organised around the user's own prompts, meant to be *read*:

- **Part 1 — Preprocessing & the HMM** — **written** (answers all 12 questions from the
  preprocessing prompt: Anscombe + the variance-stabilisation failure, the 1 Hz cut, 40 ms bins,
  the 0.4 s baseline + movement-window finding, state ordering, the ARI maths, the pseudo-population
  correction, etc.). Reflects the **final** answers, flagging where an earlier answer was corrected.
- **Part 2 — rSLDS** and **Part 3 — Dimensionality reduction** — **stubbed**, to be filled when the
  user asks about those parts. **Next chat: this is the doc to extend** with the rSLDS write-up
  (the sequence/recurrence certification, the `|λ|`-smoother finding, the two backends) and the
  dim-red write-up.

Distinct from `where_we_are_HMM_rSLDS_dimred.md` (synthesis by topic) and `HMM_maths_and_readouts.md`
(the HMM maths in full). `project_explained.md` is the reader-friendly narrative that points to
those for depth.

---

## 4. State of the rSLDS strand, for the next chat

**Certified (robust, both backends, both alignments, vs nulls):**
- **Attractor *sequence*, not a stopwatch** — `delta_recurrent > 0` every region; the
  position-shuffle null passes in **100 % of folds** (`results/rslds/recurrence_test.json`).
- **A reliable temporal sequence** — `order_consistency ≫` null.
- **Gradient MRN ≈ SCm ≫ VISp** in recurrence strength / position-dependence.

**Retired (not certifiable from this data):**
- The **flow geometry** (attractor vs line-attractor/integrator). `|λ| ≈ 1` is >97 % reproduced on
  time-shuffled data → it is the smoother. GPFA's finite τ (107–283 ms) is the honest timescale.

**Open / next steps:**
1. **Resolve pre400 vs pre300** (§0).
2. **Fill `project_explained.md` Parts 2 & 3.**
3. **Pool the 5 sessions** — as *replication* of the sequence/recurrence (does "position-dependent
   sequence, motor > visual" hold in 5/5 mice?), **not** to rescue the geometry (the smoothing
   artefact caps it regardless). `TRIPLET_EIDS` is populated; all 5 sessions built at 40 ms.
4. **The joint SCm+MRN rSLDS** — the direct exchange-vs-common-drive test; it must overturn three
   converging pieces of common-drive evidence (RRR, zero-lag correlogram, HMM).
5. **`ssm` `|λ|`-stability is by analogy, not directly tested** — the shuffle-null was run on the
   exact-EM backend; the `ssm` extension rests on "its Laplace posterior is also smooth" + the
   empirical agreement. Airtight would need a shuffle-null on the `ssm` fit (~1 h).
6. Migrate notebook 03 Parts 1–9 to `load_processed(...)` (still their own 50 ms tensors).

**Artifacts produced this chat:** `src/viz/rslds_anim.py`; `notebooks/06_results_visualization.ipynb`;
`notebooks/03` Part 10 + Part 6 removal; `scripts/{run_rslds_screen,run_rslds_ssm_record,run_rslds_lambda_stability,run_rslds_recurrence,validate_rslds_ground_truth,build_processed_40ms}.py`;
`results/rslds/*.json`; `results/figures/phase5_viz/*.gif`; `docs/knowledge/project_explained.md`;
`docs/knowledge/HMM_maths_and_readouts.md`.
