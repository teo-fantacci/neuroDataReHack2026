# CLAUDE.md — context for Claude Code sessions

Auto-loaded each session. If you are a fresh session (e.g. on a different machine after
`git pull`), read this first, then `docs/worklog/` newest-first for the running state.

## What this project is

NeuroDataReHack 2026 (HHMI Janelia, 11–17 July 2026). Reanalysis of the **IBL Brain-Wide
Map** (DANDI:000409) — and possibly **SHIELD** (DANDI:001051). Scientific claim:
decision-making is a **temporal sequence of metastable/attractor states** that is
**shared across areas** but realized on **geometrically distinct, area-specific
manifolds**. Framing guardrail: *distributed temporal sequences within populations, not a
clean anatomical relay.* Full scientific plan: `docs/project_notes/` (static,
pre-hackathon). Running decisions/findings: `docs/worklog/` (updated each session).

## Who the user is, and how to work with them

- PhD student (Italy), systems/computational neuroscience. Strong on the science; wants
  you to **explain what you do and why**, not just produce code. Answer conceptual
  questions in depth.
- **Use colour-vision-safe figures.** NEVER rely on red-vs-green to
  distinguish things, and do not use greyscale-only as a substitute. Use the helpers in
  `src/viz/style.py`: **Okabe-Ito** categorical palette, **cividis** sequential heatmaps,
  **red–blue** (`RdBu_r`) diverging. `set_style()` applies it.
- Wants **important steps written down in `docs/`** as `.md` (that is why `docs/worklog/`
  and `docs/knowledge/` exist). Keep doing this.
- Values scientific honesty: separate what is verified from what is assumed, flag fragile
  metrics, prefer real smoke-tests over claims. Do not fabricate APIs/citations.
- Prints of **timing** on long cells, and **structured** output/plot paths, are expected.

## Environment (must exist on each machine)

Conda env **`neurodatarehack`**, Python 3.11. Recreate with:

```bash
conda create -n neurodatarehack python=3.11 -y
conda activate neurodatarehack
pip install -r requirements.txt
```

Interpreter path (Windows): `C:\Users\<user>\miniconda3\envs\neurodatarehack\python.exe`.
Register the Jupyter kernel once: `python -m ipykernel install --user --name neurodatarehack`.
In VSCode: Ctrl+Shift+P → "Python: Select Interpreter" → `neurodatarehack`.

**On every fresh clone, also run once:** `nbstripout --install --attributes .gitattributes`
(2026-07-16). This activates the git **clean filter** that strips notebook **outputs** from
committed blobs while keeping the plots in your working `.ipynb` on disk — the committed notebooks
are ~30 KB, not the 0.5–4 MB they are on disk. If you skip it, your next notebook commit re-bloats
the repo with embedded images. Committed notebooks are already stripped, so *reading* the repo needs
nothing. See [worklog 2026-07-16](docs/worklog/2026-07-16_repo-cleanup-and-sharing.md).

**Second env `ndrh_ssm`** (2026-07-13) — holds Linderman's `ssm` (the real rSLDS) and *nothing
else touches it*. `ssm` is a **numpy-1.x** package: installing it into `neurodatarehack` would
downgrade numpy 2.3.5 and take pynapple/numba/pandas 3.0 with it. Build once with
`bash scripts/install_ssm_env.sh` (**no admin rights** — it uses conda's mingw toolchain instead
of a 6 GB MSVC install). The notebook stays in `neurodatarehack` and calls it **out of process**
(`src/dynamics/rslds_ssm.py`). Call the env's `python.exe` **directly** — on Windows `conda` is a
`.bat` a subprocess cannot see (`WinError 2`), so `conda run` fails from a kernel.

## Data — NOT in git

- IBL data is **downloaded & cached inside the repo at `data/ONE/`** (gitignored). It is
  NOT synced by GitHub, so a freshly-pulled machine re-downloads on first notebook run
  (~440 MB for the working session). `config/params.yaml::data.one_cache_dir` points there.
  - ⚠️ **ONE also caches to `notebooks/data/ONE/` when a notebook uses a relative cache dir**
    (~500 MB). That path is OUTSIDE the `/data/**` ignore rule (leading slash = repo root only),
    so it was NOT ignored and a `git add -A` nearly committed half a GB. Now covered by
    `**/data/ONE/` + `/notebooks/data/` in `.gitignore` (2026-07-16). Never relax the leading-slash
    on `/data/**` — without it, the pattern also swallows `src/data/`.
- **No account needed**: ONE-api ships public `openalyx` credentials; `src/data/access.py`
  uses them. A DANDI account is only needed if we pivot to SHIELD.
- Reproducibility pins live in `src/data/access.py` (`PAPER_FREEZE`, `AGGREGATE_TAG`).
  Two upstream `brainwidemap` bugs are worked around there — do not "fix" by passing
  `enforce_version=False` (that silently changes the unit set). See the Phase 0 worklog.

## Layout

```
config/      areas.yaml, params.yaml (bins, HMM, reproducibility pins, cache dir)
src/data/    access.py (ONE + brainwidemap), build_matrices.py (tensor/rasters/stats/
             region_spike_trains), select_areas.py (Phase 1 co-occurrence)
src/dynamics/ hmm.py    (Poisson HMM, baseline-biased startprob, CV of K, directedness)
             rslds.py    (OUR exact-EM rSLDS on a PCA latent + sequence statistics)
             rslds_ssm.py (out-of-process bridge to Linderman `ssm` in the ndrh_ssm env)
src/selection/ msr.py   (multiscale relevance, task-selectivity, rate-matched control)
src/viz/     style.py (colour-vision-safe palette, stroke(), structured savefig)
             nap_plots.py (pynapple: raster/PSTH/sequence/tuning/correlograms)
scripts/     run_phase1.py, run_hmm_triplet.py   (batch versions of the notebooks)
             install_ssm_env.sh (builds the ndrh_ssm env), run_rslds_ssm.py (runs INSIDE it)
             validate_hmm_vs_dynamax.py
notebooks/   00_preprocessing (pipeline entry), 00_data_smoketest (Phase-0 access check),
             01_hmm_triplet, 02_dimred_msr, 03_rslds_triplet, 04_manifold_dpca_gpfa_rrr (legacy,
             merged into 02), 05_hmm_summary (write-up), 06_results_visualization
docs/project_notes/  static scientific plan
docs/knowledge/      recorded_areas_IBL_SHIELD.md (areas, full names, functions)
docs/worklog/        dated session logs (READ THESE for current state)
results/     figures/<phase>/*.png and *.csv  (gitignored; regenerated)
```

## pynapple (adopted 2026-07-12) — scope it, don't let it spread

Exploration + basic analysis only: rasters, PSTHs, tuning curves, auto/cross-correlograms,
and an **independent check on our own binning** (`TsGroup.count()` vs
`build_matrices.region_tensor_from_probes`). It does **no state-space modelling** — HMM/rSLDS
stay in `src/dynamics/`. Do **not** rewrite the working pipeline around it.

Gotchas found by running it (v0.11.3):
- **`nap.Ts` silently re-sorts unsorted timestamps** (UserWarning only). Never hand it
  RT-ordered events — you get a quietly mislabelled raster. Pass events in time order and
  reorder the plotted *rows* instead.
- `compute_crosscorrelogram(..., ep=None)` raises `TypeError`; omit the kwarg.
- `nap.compute_perievent_continuous` does not exist.

## Conventions

- Tensors are `(trials, time, neurons)`. Poisson HMM needs **raw counts** — never smooth
  before fitting. Smoothing is for geometry/plots only.
- Single-trial methods need **simultaneously recorded** populations → work per session,
  per region; pool a session's probes, never cross-session (except final viz).
- Save figures via `src.viz.style.savefig(fig, "<phase>", "<name>")`.
- Verify package APIs against the installed package before using them; the plan docs
  predate install and some APIs differed.

## Current state (keep this section current; details in the latest worklog)

- **Preprocessing is UNIFIED (2026-07-14)** — `src/data/preprocess.py` + `notebooks/00_preprocessing.ipynb`.
  Every model now eats the same frozen tensors (`data/processed/*.npz`, config serialised inside).
  **stimOn + firstMovement alignment, `pre_s` 0.4 s, `post_s` 0.8 s, 40 ms bins (default since
  2026-07-15), units ≥ 1 Hz**. `pre_s=0.4` because IBL's enforced **quiescence period** is drawn
  from [0.400, 0.700] s — 0.5 s stepped outside it on 44% of trials. `01_hmm_triplet.ipynb` reads
  `load_processed(...)`; notebooks 02–04 still build their own tensors and must be pointed at it.
- **ALL 5 TRIPLET SESSIONS BUILT AT 40 ms (2026-07-15)** — `scripts/build_processed_all.py`
  downloaded + preprocessed every `{VISp,SCm,MRN}` session (5c0c560e 843 tr / 90c61c38 656 /
  111c1762 364 / 90d1e82c 353 / 8a3a0197 290), both alignments. `run_hmm_sessions.py` +
  `run_hmm_kselect.py` replicate the HMM across them and pool the statistics.
- **⚠️ THE WINDOW IS A MOVEMENT WINDOW (2026-07-14).** Median RT **171 ms**, median feedback
  **427 ms** → over `[0,+0.8]s` movement has begun on **92%** of trials and the outcome arrived on
  **83%**. In IBL the choice *is* the wheel-turn direction. So the HMM's late states, dPCA's
  "decision" variance, and the SCm/MRN directedness are all **entangled with movement and reward**.
  A `firstMovement`-aligned tensor is now built as the standing control. **No state after ~+0.2 s
  may be called a "decision" state without it.**
- **⚠️ ANSCOMBE DOES NOT STABILISE VARIANCE HERE (2026-07-14).** `2*sqrt(x+3/8)` reaches Var=1 only
  at **λ ≳ 4** spikes/bin; IBL runs at **λ = 0.05–0.83**. At λ≪1 a count is ~Bernoulli and
  Var=p(1-p)Δ² is irreducible — **no monotone transform can stabilise it**. Kept (it compresses the
  tail; the rSLDS ground truth used it), but this is a **second, independent** argument that the
  **Poisson `ssm` backend is principled and our PCA latent is a concession** — and it explains why
  the 20 ms latent manufactured fake line attractors.
- **Phase 0 done** — env, public data access, scaffold, smoke-test.
- **Phase 1 done** — area selection. IBL co-records ≤3 of our visual→motor axis regions
  per session; the one replicable axis triplet is `{MRN, SCm, VISp}` (5 sessions). A
  clean *visual→higher-order→motor* triplet essentially does not co-occur in IBL (that
  is a real argument for SHIELD). Working session: `5c0c560e`.
- **Regions frozen to `{VISp, SCm, MRN}`** (2026-07-12). **VISpm dropped** — it was a
  stand-in for a "higher-order" tier that did not actually supply one.
- **Phase 2 in progress** — per-region Poisson HMM with pre-stimulus-baseline-biased
  startprob; `forward_fraction` (directedness, validated on synthetic + shuffle null);
  cross-validation of K. Motor areas are more directed than visual (SCm .77, MRN .69 vs
  VISp .53). **Treat "CV prefers K=2" with suspicion** — read the CV *curve*, not the
  argmax; if held-out LL is flat across K the argmax is noise. Re-run at `MAX_TRIALS=None`.
- **`02_dimred_msr.ipynb` is now the CONSOLIDATED dim-red notebook (2026-07-15).** Runs on the
  **40 ms unified pipeline** (`load_processed`), MSR demoted to a comparison set (kept, not
  adopted). §1–5 = **PCA/dPCA/MDS/Laplacian in 3-D, coloured by time** (all vs MSR-top;
  post-stim + full-window diverging blue=baseline/red=evoked). §6–9 = **04's results merged in**
  (dPCA split, GPFA, RRR, joint manifold), re-run at 40 ms — all reproduce the 50 ms numbers.
  **Notebook 04 is now a retirement candidate** (only the bin×smoothing sweep + 7-method quality
  zoo are unique to it). dPCA maths + the sequence link:
  `docs/knowledge/dPCA_maths_and_the_attractor_sequence.md`.
- **Phase 3 started (2026-07-13)** — `notebooks/02_dimred_msr.ipynb`: MSR unit selection +
  PCA/MDS/t-SNE/UMAP/Isomap/Laplacian/PHATE on the triplet. Two results to carry forward:
  **(a) MSR is not adopted** — where a rate-matched control is actually valid (MRN only),
  the top-MSR half decodes no better than a *random* half; in VISp/SCm the control silently
  degenerates into "the slow half" (MSR–rate rho ≈ .55–.64), so those comparisons are
  confounded. **(b) Do not fit the HMM on a halved population**: choice info survives
  halving, the *temporal trajectory* does not. Also: MSR correctly gives a LOW score to a
  stereotyped burster (delta multiplicity distribution ⇒ relevance ≈ 0) — a low MSR is not
  automatically an uninformative unit.
- **⚠️ rSLDS TESTED FOR REAL (2026-07-15/16)** — supersedes the "smoke settings only" note below.
  Fitted on the frozen 40 ms tensors, both alignments, K=3/4/3, exact-EM screen + Poisson `ssm`
  fit of record. Full write-up: `docs/worklog/2026-07-15_rslds-first-real-test.md` (analysis) +
  `2026-07-16_rslds-viz-and-nb-cleanup.md` (viz + notebook cleanup). **Two verdicts:**
  - ✅ **CERTIFIED — attractor *sequence*, not a stopwatch.** `delta_recurrent > 0` every region,
    both alignments; a clean **position-shuffle null passes in 100 % of folds**
    (`results/rslds/recurrence_test.json`); `order_consistency ≫` its shuffle null on both
    backends. Gradient **MRN ≈ SCm ≫ VISp**. This is the project's core claim, now with an error
    bar and a clean null.
  - ❌ **RETIRED — the flow geometry** (attractor vs line-attractor/integrator). Both backends hit
    `|λ| ≈ 1`; the `|λ|`-stability test (`run_rslds_lambda_stability.py`) proved it is **>97 %
    reproduced on time-shuffled data** → the **smoother**, not the dynamics. GPFA's finite τ
    (107–283 ms) is the honest timescale (slow but *contracting*). **Notebook-03 Part 6 (flow
    geometry) was DELETED**; Part 10 (certified dashboard) added; `notebooks/06_results_visualization.ipynb`
    + `src/viz/rslds_anim.py` animate the certified sequence (single-trial GIF + occupancy sweep).
  - ⚠️ **`pre_s` BOOKKEEPING**: all rSLDS results/viz used **pre400** 40 ms tensors; the config
    default is now **pre300** (2026-07-16). Both exist on disk; the `run_rslds_*.py` scripts
    hard-code the pre400 filename. Re-point + re-run, or accept pre400 for this strand — but decide.
- **Phase 3 rSLDS (2026-07-13)** — `notebooks/03_rslds_triplet.ipynb`, runs end-to-end on the
  real session (**smoke settings only — no scientific numbers claimed yet**). Two backends by
  design: our **exact-EM** rSLDS (`src/dynamics/rslds.py`, seconds — does the CV of K and the
  recurrence test) and the **real `ssm`** (`rslds_ssm.py`, ~15–20 min/region — the fit of
  record). `ssm` is too slow to cross-validate (≈13 h) and its **ELBO is a lower bound, not a
  likelihood you may argmax**. New statistic `order_consistency`: does a reliable *sequence*
  exist, **in either direction** (0.5 = no order, 1.0 = same order every trial) — the
  direction-agnostic complement to the HMM's `forward_fraction`.
- **Phase 3 geometry (2026-07-13)** — `notebooks/04_manifold_dpca_gpfa_rrr.ipynb`, executed
  end-to-end. **dPCA reproduces the project's core prediction**: decision−stimulus variance
  runs **VISp −6.8 → SCm +1.8 → MRN +6.0**, and the condition-independent (time) share falls
  74→64→49% along the axis. Caveat: the balanced design gives only **K=25 trials/cell** (the
  off-diagonal cells are error trials), so **pool the 5 replicate sessions before claiming it**.
  **GPFA** (raw counts, inferred smoothness) fits **tau = 107–283 ms**, slowest in MRN → the
  dynamics are far slower than any bin we argued about, and our smoothed trajectories are not
  an artefact of the smoother. **RRR/communication subspace: SCm↔MRN show NO restricted
  channel** — MRN predicts SCm (R²=.366, rank 7) about as well as SCm predicts *itself*
  (.383, rank 7). That is now the **third** independent method pointing at common drive
  (with the zero-lag cross-correlogram and the HMM). A joint rSLDS must now *overturn* this,
  not merely confirm a feedback loop. **Joint manifold** = PCA of all 158 units; per-area
  *partial projections* (exact by linearity, asserted in `src/geometry/joint.py`); area shares
  VISp .06 / SCm .22 / MRN .31, summing to .58 — **the missing ~.42 is cross-area covariance**.
- **Geometry config frozen: 50 ms bins, sigma 40 ms** (2026-07-13). Do **not** pick smoothing
  by maximising time-decoding R²: a Gaussian kernel *imposes* the temporal autocorrelation
  that the metric rewards, so the argmax runs to whatever sigma you swept to (0.32→0.48 as
  sigma goes 0→160 ms; the argmax moved when the ceiling moved). Choice AUC is **flat** across
  the whole sweep (.854–.860) — which is the real news: **the choice signal is robust to bin
  size and smoothing.** The config is chosen on external grounds (rSLDS ground-truth bin +
  fitted GPFA timescales).
- **Open**: `config/areas.yaml::selected` still empty (now clearly `{VISp, SCm, MRN}`);
  IBL-vs-SHIELD pivot (user keeping both open); GPU at Janelia; the SCm↔MRN feedback
  hypothesis → needs a **joint multi-region rSLDS** (one model on the concatenated SCm+MRN
  population; ask whether the recurrence matrix `R` carries cross-area structure). Next method
  suggestion: **dPCA** (splits variance into time/stimulus/choice — the actual project question).
- **rSLDS is limited by TRIALS, not neurons** (2026-07-13). The attractors + sequence live in the
  dynamics/transition parameters (~100 of them), estimated from the number of *regime
  transitions* — and a trial supplies only 2–4. Neurons enter through the emission matrix and are
  nearly free information: they sharpen the per-bin latent. **So never prune neurons to "afford"
  a state-space model** — it buys no power and costs latent SNR. (The plan's framing of MSR as a
  way to shrink the population is right for ELA, wrong for rSLDS — and the parallel MSR notebook
  reached the same conclusion from decoding.)
- **⚠️ CORRECTION (2026-07-14) — "pool the 5 sessions" means REPLICATE, not pseudo-population.**
  Earlier worklogs said "pool trials, not neurons", which is too loose and **wrong for the models
  it was aimed at**. A pseudo-population stitches neurons from *different* sessions into one
  population vector — but they were **never recorded simultaneously**, so there is no single-trial
  joint activity, which is exactly what a state-space model consumes (and is the property Phase 1
  selected these sessions *for*). It does not add trials to the HMM; **it destroys what a trial is.**
  - **HMM / rSLDS**: fit each session separately, **pool the STATISTICS**. "K=3 in 5/5 sessions,
    forward_fraction 0.77 ± 0.04" is a *replication* — a stronger claim than one session with
    tighter error bars, and it *is* the animal-independence argument, made properly.
  - **dPCA**: a pseudo-population **is** valid (it works on trial-averaged activity) and **directly
    fixes the K = 25 trials/cell problem**.
- **Dataset default**: IBL primary, SHIELD as a de-risked second substrate (public,
  streamable — verified). pynapple is NWB-native, which helps if we pivot.

## Traps that already bit us

- **Do not trust committed notebook outputs.** The Phase-2 CV figure was a *blank* plot
  (legend over no data) left behind by a broken run, while the cell above it showed results
  from a different, good run. matplotlib draws a convincing empty figure when handed
  all-NaN/empty arrays. A figure is evidence only if its cell ran in the same pass.
- **pandas `attrs` does not survive most operations.** `cross_validate_states` returned
  `best_k` in `df.attrs`; anything downstream must recompute it from the table.
- **At 20 ms bins the rSLDS latent is noise, and it fails as a *plausible result*.** With IBL's
  sparse units the PCA latent is shot-noise dominated, so the fitted `|λ| → 1` and **every**
  regime is classified "line attractor / integrator" — a near-unit-root random walk that reads
  exactly like "we found the integrator a decision variable lives on". **Integrator claims from a
  20 ms fit are inadmissible.** Now caught mechanically by `RSLDSResult.admissibility()`.
  - **Ground truth is now REPRODUCIBLE (2026-07-14):** `scripts/validate_rslds_ground_truth.py`
    → `results/rslds/ground_truth_binsize.json`. The old docstring table had no code; its
    **"|λ| = 0.875 at 50 ms" is retracted** (not reproducible). 40 ms passes in every region
    (0/3 spurious integrators), 20 ms fails (1–3/3).
  - **⚠️ `|λ| → 1` has TWO causes.** Shot noise (small bins) AND over-smoothing (a Gaussian
    kernel imposes the autocorrelation an AR fit reads as a slow mode). The old validation used
    sigma **in bins**, so its bin sweep was also a smoothing sweep (75 ms of smoothing at 50 ms
    bins), which is why "50 ms looked best". `counts_to_latents` now takes **`smooth_sigma_ms`**,
    not bins.
- **Project-wide bin size is now 40 ms** (supersedes 50 ms): it passes the rSLDS ground truth
  *and* preserves SCm's HMM state-stability, which degrades at 50 ms (ARI 0.938 → 0.826). Still
  re-fit the rSLDS ground truth if you change it.
- **A shuffle null is not automatically 0.5.** `order_consistency`'s within-trial bin-shuffle null
  sits at **0.53–0.62**. Compare to the null, never to the "obvious" chance value — and report the
  **effect size**, because with hundreds of trials a trivial bias still gets p < 0.001.
- **"Package X won't install" is a symptom, not a diagnosis.** `ssm` was written off as
  "CONFIRMED UNINSTALLABLE, do not retry" on the strength of *one* error message. There were
  **two** stacked, unrelated errors (missing build-time numpy; then no C++ compiler) — both
  fixable, no admin needed. A tested, published method was nearly dropped for a day. Read past
  the first traceback before recording a permanent verdict.
- **A metric that your knob inflates cannot choose that knob.** Smoothing sigma was going to be
  picked by maximising held-out *time*-decoding R² — but a Gaussian kernel *imposes* temporal
  autocorrelation, which is exactly what that metric rewards. It rose monotonically (0.32→0.48)
  and its argmax followed the sweep's ceiling wherever we put it. Same family as the K=7
  boundary: **never argmax without checking the winner is interior and the criterion is
  independent of the parameter.**
- **A prior applied at initialisation is not a prior.** The rSLDS `sticky` (self-transition)
  prior was set on the initial transition matrix and then silently overwritten by the first
  M-step. Regimes flickered 8× per trial where the truth was 2, and every sequence statistic
  would have been computed on that noise. It must be applied in **every** M-step.

## Note on history/memory portability

Claude Code conversation transcripts and the `memory/` dir are **machine-local**
(`~/.claude/projects/<path-hash>/`), NOT synced by GitHub. Cross-machine continuity comes
from THIS file + `docs/worklog/`, not from chat history. Keep them current.
