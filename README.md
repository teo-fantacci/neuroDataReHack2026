# IBL Decision Dynamics

**NeuroDataReHack 2026 — HHMI Janelia, 11–17 July 2026**

Reanalysis of the IBL Brain-Wide Map ([DANDI:000409](https://dandiarchive.org/dandiset/000409);
IBL et al. 2025, *Nature* 645:177–189).

**The claim.** Decision-making is a **temporal sequence of metastable / attractor states**,
**shared across areas** but realised on **geometrically distinct, area-specific manifolds** —
*distributed temporal sequences within populations, not a clean anatomical relay.*

We test it on the one visual→motor axis IBL co-records simultaneously in ≥5 sessions:
**`{VISp, SCm, MRN}`** (primary visual cortex → superior colliculus motor → midbrain
reticular nucleus). Working session `5c0c560e`; all 5 triplet sessions are built for replication.

## Where the claim stands (honest status)

The full, sourced status is
[`docs/knowledge/project_synthesis_2026-07-15.md`](docs/knowledge/project_synthesis_2026-07-15.md).
In one paragraph:

- ✅ **The sequence half is certified** (one session; replication across 5 pending). There is a
  reliable, position-dependent temporal **sequence** of states in all three regions — not a timed
  stopwatch (rSLDS recurrence test beats a position-shuffle null in 100% of folds) and not a
  movement-timing artefact. Motor areas traverse their states more directedly than visual cortex
  (HMM `forward_fraction`: VISp ≈ 0.53 *undirected*, SCm ≈ 0.83, MRN ≈ 0.67).
- 🔶 **The area-specific-manifold half is a strong single-session lead.** The same visual→motor
  ordering falls out of **four** independent methods (dPCA decision-variance gradient, GPFA
  timescales, RRR common-drive, HMM directedness) — but every number is one session. **Replicate
  before it becomes a figure.**
- ❌ **Retracted:** "we found the line attractor / integrator a decision variable lives on." The
  `|λ| ≈ 1` behind it is >97% reproduced on temporally-shuffled data — it is manufactured by the
  smoother, not the dynamics. The honest replacement is *"slow but contracting dynamics, motor
  slower than visual."*

## Setup

One conda env runs **everything** below:

```bash
conda create -n neurodatarehack python=3.11 -y
conda activate neurodatarehack
pip install -r requirements.txt
python -m ipykernel install --user --name neurodatarehack   # register the Jupyter kernel once
nbstripout --install --attributes .gitattributes            # strip notebook OUTPUTS from commits (keeps them on disk)
```

### The `ssm` env is OPTIONAL — you can skip it

Linderman's `ssm` (the Poisson rSLDS "fit of record") is **numpy-1.x** and would downgrade the
main env, so it lives in a *separate* env `ndrh_ssm`. **You do not need it to run the notebooks.**
The code auto-detects it: `ssm_env_available()`
([`src/dynamics/rslds_ssm.py`](src/dynamics/rslds_ssm.py)) returns `False` when the env is absent,
and notebook `03` / the batch scripts fall back to the fast **exact-EM** rSLDS
([`src/dynamics/rslds.py`](src/dynamics/rslds.py)) — which is what produces the certified
recurrence result anyway. Skipping `ssm` only costs you the second-backend cross-check.

Build it only if you want that cross-check (one-off, no admin rights):

```bash
bash scripts/install_ssm_env.sh        # builds ndrh_ssm via conda's mingw toolchain
```

## Data — not in git

- **No account needed.** ONE-api ships public `openalyx` credentials; `src/data/access.py` uses
  them. (A DANDI account is only needed if we pivot to SHIELD, DANDI:001051.)
- IBL data is **downloaded and cached inside the repo** and is gitignored, so a freshly-cloned
  machine **re-downloads on first run** (~440 MB for the working session). The cache path is
  `config/params.yaml::data.one_cache_dir`.
- Reproducibility is pinned to the paper freeze `2023_12_bwm_release` (aggregate tag
  `2024_Q2_IBL_et_al_BWM`); `bwm_units` asserts a hash of the unit UUIDs so we analyse **exactly**
  the paper's 62,990 units. Two upstream `brainwidemap` bugs are worked around in
  [`src/data/access.py`](src/data/access.py) — see the
  [Phase-0 worklog](docs/worklog/2026-07-08_phase0-setup.md). Do **not** "fix" them with
  `enforce_version=False`; that silently changes the unit set.

## Run order

Every model eats the same frozen tensors, so preprocessing runs first:

1. **[`notebooks/00_preprocessing.ipynb`](notebooks/00_preprocessing.ipynb)** — builds the frozen
   `data/processed/*.npz` tensors (stimOn **and** firstMovement alignment, `pre` 0.4 s, `post`
   0.8 s, **40 ms bins**, units ≥ 1 Hz). Config is serialised inside each `.npz`.
2. **[`01_hmm_triplet`](notebooks/01_hmm_triplet.ipynb)** → state sequences and directedness.
3. **[`02_dimred_msr`](notebooks/02_dimred_msr.ipynb)** → geometry (PCA / dPCA / GPFA / RRR / joint
   manifold).
4. **[`03_rslds_triplet`](notebooks/03_rslds_triplet.ipynb)** → the attractor-sequence tests.

`00_data_smoketest` is the original Phase-0 "does data access work" check — a good first run on a
fresh machine, gated behind `DOWNLOAD = False`.

## Repository map

### Notebooks (`notebooks/`)

| Notebook | Does | Status |
|---|---|---|
| `00_preprocessing` | Builds the frozen `.npz` tensors every model reads | **current — run first** |
| `00_data_smoketest` | Phase-0 data-access sanity check (openalyx → tensor → HMM) | current (entry check) |
| `01_hmm_triplet` | Poisson HMM: state sequences, `forward_fraction`, K by stability | **current** |
| `02_dimred_msr` | Consolidated geometry: PCA/dPCA/MDS/Laplacian (3-D) + GPFA/RRR/joint manifold + MSR | **current** |
| `03_rslds_triplet` | rSLDS (exact-EM + `ssm`): recurrence test, order-consistency | **current** |
| `04_manifold_dpca_gpfa_rrr` | dPCA/GPFA/RRR/manifold | legacy — results merged into `02`; kept for the bin×smoothing sweep + method zoo |
| `05_hmm_summary` | HMM strand write-up: what we tested, what failed, what replicates | write-up (no heavy outputs) |
| `06_results_visualization` | Regime-sequence animation / final viz | current (viz) |

### Source (`src/`)

| Module | Purpose |
|---|---|
| `data/access.py` | ONE-api + `brainwidemap` wrappers; reproducibility pins + bug workarounds |
| `data/preprocess.py` | **The single preprocessing pipeline** → frozen `.npz` (config serialised inside) |
| `data/build_matrices.py` | spikes → `(trials, time, neurons)` tensors, rasters, per-region spike trains |
| `data/select_areas.py` | Phase-1 co-occurrence area selection |
| `dynamics/hmm.py` | Poisson HMM: baseline-biased startprob, `forward_fraction`, CV / stability of K |
| `dynamics/rslds.py` | Our exact-EM rSLDS on a PCA latent + sequence statistics (fast; does the screen + recurrence) |
| `dynamics/rslds_ssm.py` | Out-of-process bridge to Linderman `ssm` in the `ndrh_ssm` env (fit of record) |
| `dynamics/dpca.py`, `gpfa.py`, `rrr.py` | Demixed PCA, GPFA timescales, reduced-rank regression |
| `selection/msr.py` | Multiscale relevance + task-selectivity (evaluated, **not adopted** — see synthesis) |
| `geometry/embed.py`, `joint.py` | Manifold embeddings; joint-population PCA with per-area partial projections |
| `viz/style.py` | **Colour-vision-safe** palette (Okabe-Ito / cividis / RdBu_r); `set_style()`, `savefig()` |
| `viz/nap_plots.py`, `rslds_anim.py` | pynapple rasters/PSTHs/correlograms; rSLDS animation |

### Scripts (`scripts/`) — batch/headless versions of the notebook analyses

- **Build data:** `build_processed_40ms.py`, `build_processed_all.py` (all 5 sessions),
  `preprocess_second_session.py`, `survey_triplet_sessions.py`
- **HMM:** `run_hmm_triplet.py`, `run_hmm_sessions.py`, `run_hmm_kselect.py`,
  `run_state_stability.py`, `run_movement_alignment.py`, `run_binsize_check.py`
- **rSLDS:** `run_rslds_screen.py`, `run_rslds_recurrence.py`, `run_rslds_lambda_stability.py`,
  `run_rslds_ssm.py` / `run_rslds_ssm_record.py` (run inside `ndrh_ssm`)
- **Env + validation:** `install_ssm_env.sh`, `validate_hmm_vs_dynamax.py`,
  `validate_rslds_ground_truth.py`
- **Phase 1:** `run_phase1.py`

### Config & docs

```
config/     areas.yaml (candidate/selected regions), params.yaml (bins, HMM, repro pins, cache dir)
docs/knowledge/  method + maths teaching notes; project_synthesis_*.md = the executive status
docs/worklog/    dated session logs — READ NEWEST-FIRST for the running state
docs/project_notes/  static pre-hackathon scientific plan
results/     figures + CSVs (gitignored; regenerated) — only *.md summaries are committed
```

## Conventions (a few that bite)

- Tensors are `(trials, time, neurons)`. The Poisson HMM needs **raw counts** — never smooth
  before fitting. Smoothing is for geometry/plots only.
- Single-trial methods need **simultaneously recorded** populations → work per session, per region;
  never build a cross-session pseudo-population for the HMM/rSLDS (it destroys what a trial is).
  **Pooling means replicate** (fit each session, pool the statistics). dPCA is the exception —
  a trial-averaged pseudo-population is valid there.
- **Bin = 40 ms, project-wide.** 20 ms manufactures fake integrators; 50 ms degrades SCm stability.
- Never take an argmax without checking the winner is **interior** and the criterion is
  **independent of the parameter being chosen** (this trap bit us three times — K, integrators,
  smoothing σ).
- Save figures via `src.viz.style.savefig(fig, "<phase>", "<name>")`.

## Where to read next

- Current state + cross-machine handoff → [`CLAUDE.md`](CLAUDE.md)
- Executive status (certified / lead / retracted) →
  [`docs/knowledge/project_synthesis_2026-07-15.md`](docs/knowledge/project_synthesis_2026-07-15.md)
- Deep method + maths notes → [`docs/knowledge/`](docs/knowledge/)
- What we did and why, session by session → [`docs/worklog/`](docs/worklog/) (newest first)

## Key references

IBL et al. 2025, *Nature* 645:177–189 · Linderman et al. 2017, AISTATS (rSLDS) · Genkin, Shenoy,
Chandrasekaran & Engel 2025, *Nature* 645:168–176 (NeuralFlow) · Langdon, Genkin & Engel 2023,
*Nat Rev Neurosci* 24:363–377 · Semedo et al. 2019, *Neuron* 102:249–259 (communication subspace) ·
Kobak et al. 2016, *eLife* 5:e10989 (dPCA).
