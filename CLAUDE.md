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

## Data — NOT in git

- IBL data is **downloaded & cached inside the repo at `data/ONE/`** (gitignored). It is
  NOT synced by GitHub, so a freshly-pulled machine re-downloads on first notebook run
  (~440 MB for the working session). `config/params.yaml::data.one_cache_dir` points there.
- **No account needed**: ONE-api ships public `openalyx` credentials; `src/data/access.py`
  uses them. A DANDI account is only needed if we pivot to SHIELD.
- Reproducibility pins live in `src/data/access.py` (`PAPER_FREEZE`, `AGGREGATE_TAG`).
  Two upstream `brainwidemap` bugs are worked around there — do not "fix" by passing
  `enforce_version=False` (that silently changes the unit set). See the Phase 0 worklog.

## Layout

```
config/      areas.yaml, params.yaml (bins, HMM, reproducibility pins, cache dir)
src/data/    access.py (ONE + brainwidemap), build_matrices.py (tensor/rasters/stats),
             select_areas.py (Phase 1 co-occurrence)
src/dynamics/ hmm.py (Poisson HMM, baseline-biased startprob, CV of K, directedness)
src/viz/     style.py (colour-vision-safe palette + structured savefig)
scripts/     run_phase1.py, run_hmm_triplet.py   (batch versions of the notebooks)
notebooks/   00_data_smoketest.ipynb, 01_hmm_triplet.ipynb (the live exploration nb)
docs/project_notes/  static scientific plan
docs/knowledge/      recorded_areas_IBL_SHIELD.md (areas, full names, functions)
docs/worklog/        dated session logs (READ THESE for current state)
results/     figures/<phase>/*.png and *.csv  (gitignored; regenerated)
```

## Conventions

- Tensors are `(trials, time, neurons)`. Poisson HMM needs **raw counts** — never smooth
  before fitting. Smoothing is for geometry/plots only.
- Single-trial methods need **simultaneously recorded** populations → work per session,
  per region; pool a session's probes, never cross-session (except final viz).
- Save figures via `src.viz.style.savefig(fig, "<phase>", "<name>")`.
- Verify package APIs against the installed package before using them; the plan docs
  predate install and some APIs differed.

## Current state (keep this section current; details in the latest worklog)

- **Phase 0 done** — env, public data access, scaffold, smoke-test.
- **Phase 1 done** — area selection. IBL co-records ≤3 of our visual→motor axis regions
  per session; the one replicable axis triplet is `{MRN, SCm, VISp}` (5 sessions). A
  clean *visual→higher-order→motor* triplet essentially does not co-occur in IBL (that
  is a real argument for SHIELD). Working session: `5c0c560e` (has VISp, VISpm, SCm, MRN).
- **Phase 2 in progress** — per-region Poisson HMM with pre-stimulus-baseline-biased
  startprob; `forward_fraction` (directedness, validated on synthetic + shuffle null);
  cross-validation of K. Latest result: **CV prefers K=2** for all four regions; motor
  areas more directed than visual. See `results/hmm/phase2_summary.md` and
  `notebooks/01_hmm_triplet.ipynb`.
- **Open**: freeze region set in `config/areas.yaml` (still empty under `selected`);
  IBL-vs-SHIELD pivot (user keeping both open); GPU at Janelia; the SCm↔MRN feedback
  hypothesis → needs a joint multi-region HMM/rSLDS (Phase 3).
- **Dataset default**: IBL primary, SHIELD as a de-risked second substrate (public,
  streamable — verified).

## Note on history/memory portability

Claude Code conversation transcripts and the `memory/` dir are **machine-local**
(`~/.claude/projects/<path-hash>/`), NOT synced by GitHub. Cross-machine continuity comes
from THIS file + `docs/worklog/`, not from chat history. Keep them current.
