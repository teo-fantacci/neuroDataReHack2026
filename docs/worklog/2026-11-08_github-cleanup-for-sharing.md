# 2026-11-08 — Preparing the repo for public review

**Presentation and infrastructure only. No analysis logic or numerical result changed.** The repo is
public and is about to be read by people outside the project, so this session made it safe to share
(privacy), reproducible to install (pinned requirements), checkable (tests + CI) and readable
(README, consistent numbering). Work was done on the branch `cleanup-for-sharing`, on top of a
history-rewritten `main`.

## 1. Privacy check and history rewrite

A read-only scan of `CLAUDE.md`, `docs/worklog/` and `docs/project_notes/` found personal material
that had been public since the first commit (2026-07-08): a personal letter, another participant's
unpublished abstract, personal to-dos, and a personal health detail in `CLAUDE.md`. Nothing else
sensitive: the openalyx password in the Phase 0 worklog is IBL's published public login, committed
notebooks carry no local paths (nbstripout), and no tokens or keys exist.

**Decision (author):** remove those passages from the *whole* history, not just from `HEAD`.

**How:** `git filter-repo --blob-callback` on a throwaway clone. The callback deletes the three
sections from the two project-notes files and rewords the `CLAUDE.md` line to "Use
colour-vision-safe figures." A git bundle of the old history was saved locally, outside the repo,
before anything was replaced.

**Verified before adopting it:**
- author dates, committer dates, authors and messages of every commit are byte-identical;
- every commit differs from its original **only** in those three files;
- none of the removed strings occur anywhere in the new history.

**Consequences.** Every commit hash changed (the one hash quoted in the docs, `fba432f`, is now
`f33e3ed` and was updated). Any other clone must be re-cloned or hard-reset to the new
`origin/main`, never pushed (see `CLAUDE.md`). GitHub can keep the old commits reachable by their
old hashes until it garbage-collects them; removing them for certain needs a GitHub Support request
or deleting and recreating the repository. The repo had no forks, stars or issues at the time.
**Decision (author, 2026-11-09):** leave them; they are not linked from anywhere and are reachable
only by someone who already has the old hashes.

**Second pass (2026-11-09).** A full sweep of every tracked file, notebook and commit message, run
before the first push, found the same health detail once more, in a docstring in
`src/viz/style.py`. It was reworded in every commit the same way (same verification: dates,
authors and messages identical; each commit differs only in that file; old wording gone). Hashes
changed again; the quoted hash above is the current one. Left as is, for the author to decide: the
commit author e-mail (also quoted in the 2026-07-16 worklog), a grant mention in the
project notes, and "PhD student (Italy)" in `CLAUDE.md`.

The colleague who shared the monkey data is referred to as **"colleague D."** in all public files;
their name never reached git history.

## 2. Backlog committed (July work that was still uncommitted)

One commit per group, on `main`: HMM/dPCA code; preprocessing defaults (`pre_s` 0.3 s, display
smoothing 60 ms); figure scripts and the 3-D GIF helper; mouse notebooks; mouse docs; the monkey
strand. Plus the area decision: `config/areas.yaml::selected` = `{VISp, SCm, MRN}` on working
session `5c0c560e` (the session used before the replications). No code reads that block yet.
`tutorials_ndrh/` (hackathon tutorial material, unclear license) was deliberately left out.

## 3. Notebooks renumbered by phase

Numbers had collided (00, 01, 02 were each used twice). New numbers follow the run order and the
phases of `docs/project_notes/Janelia_project_plan_with_2nd_dataset.md`. The rSLDS belongs to phase 2
(the dynamics backbone) in the plan, so its figure folders moved from `phase3_rslds*` to
`phase2_rslds*`. Renamed with `git mv`, so file history follows.

| old | new | phase |
|---|---|---|
| `00_data_smoketest` | `00_data_smoketest` | 0 setup and data |
| `00_preprocessing` | `01_preprocessing` | 0 setup and data |
| `01_hmm_triplet` | `02_hmm_triplet` | 2 dynamics backbone |
| `03_rslds_triplet` | `03_rslds_triplet` | 2 dynamics backbone |
| `02_dimred_msr` | `04_dimred_msr` | 3 geometry |
| `04_manifold_dpca_gpfa_rrr` | `05_manifold_dpca_gpfa_rrr` | 3 geometry (legacy) |
| `07_data_colleagueD` | `06_monkey_data_colleagueD` | 4 second dataset |
| `01_HMM_monkey` | `07_monkey_hmm` | 4 second dataset |
| `02_dimred_monkey` | `08_monkey_dimred` | 4 second dataset |
| `05_hmm_summary` | `09_hmm_summary` | 5 write-up |
| `06_results_visualization` | `10_results_visualization` | 5 write-up |

References were updated in code, notebooks, the README, `CLAUDE.md` and `docs/knowledge/`.
**Dated worklogs keep the old names**: they are records of what was true that day.

## 4. Requirements, tests, CI, README

- **`requirements.txt` pinned** (`==`) to the environment that produced the results, plus `jax`
  (dynamax is sensitive to it), `brainwidemap` pinned to a commit, and `pytest`. Every pin was
  checked on PyPI to ship a Linux / pure-Python wheel for Python 3.11.
- **`tests/`**: the two validation scripts became tests on synthetic data. The hmmlearn HMM must
  match dynamax's held-out log-likelihood to `rel=1e-6` with identical Viterbi paths. The rSLDS
  must recover a planted 0→1→2 sequence at 40 ms with no spurious integrator, using the script's own
  simulator unchanged. Unit tests cover binning, the frozen `PreprocessConfig` (28 bins, 7 baseline
  bins, tag `stimOn_pre300_post800_bin40_rate1`), the `.npz` round trip, HMM read-outs, RRR and MSR.
  Fast suite: 23 tests, ~13 s. Slow suite (`-m slow`): the full ground-truth sweep (40 ms: no
  spurious integrator in any configuration; 20 ms: at least one) and a real fit through the `ssm`
  bridge, ~5.5 min. **All pass locally.** CI runs the fast suite on push and PR.
- **README**: summary, three key-result figures in `docs/figures/` with honest captions (the dPCA
  gradient labelled a lead, since `111c1762` does not reproduce it), phase/notebook table, tests,
  data on DANDI, an "How AI coding agents were used" section, license.

### Finding while writing the tests (not fixed — it is analysis behaviour)

`build_matrices.bin_spikes_by_event` documents its window as right-open, `[event − pre, event +
post)`, and says a spike exactly at `+post_s` is excluded. Because of floating-point rounding such a
spike can land in the **last** bin instead (`(t − start)/bin` evaluates to 5.9999… rather than 6).
Practical impact is negligible: spike times (30 kHz probe clock) and behavioural event times come
from different clocks, so an exact coincidence essentially never happens. Recorded so nobody is
surprised; changing it would change the binning code, which this session was not allowed to do.

## 5. NWB / DANDI route — loader built as a documented alternative

- All five triplet sessions are in **DANDI:000409** as processed NWB (0.6–2 GB each): `sub-NR-0028`
  (`5c0c560e`), `sub-CSH-ZAD-025` (`90c61c38`), `sub-UCLA033` (`111c1762`), `sub-DY-010`
  (`90d1e82c`), `sub-UCLA034` (`8a3a0197`). Five different mice.
- The files were **created with NeuroConv v0.9.4** (NWB 2.9.0; extensions `ndx-ibl`, `ndx-events`,
  `ndx-pose`, `ndx-anatomical-localization`).
- For `5c0c560e`, the NWB units table (2,962 units) carries a `cluster_uuid` column, and **all
  2,962 UUIDs match the BWM aggregate cluster table** the ONE route uses; all 277 good units are
  present. So the spike sorting is the same one the paper froze.
- The NWB trials table (1,100 trials) uses NeuroConv names: `gabor_stimulus_onset_time`
  (stimOn), `wheel_movement_onset_time` (firstMovement), `mouse_wheel_choice`, `is_mouse_rewarded`,
  `probability_left`, `quiescence_period`.

**Decision (author): option B**, a loader that returns the same structures as the ONE route, with
the differences measured rather than required to vanish. An exact match (option A) turned out to be
impossible with this data, not just slow: spike times are bit-identical, but the NWB trials table
differs from ONE's on a few event times.

- `src/data/load_nwb.py`: DandiAPIClient → remfile → h5py → pynwb (`load_namespaces=True`), the
  pattern in the pynwb streaming tutorial and CatalystNeuro's IBL-to-NWB README. Trials renamed to
  ALF names; the mask is computed by `brainwidemap.load_trials_and_mask` itself (it accepts a
  pre-loaded table); units = `ibl_quality_score >= 1`, Beryl from `max_electrode` →
  `ElectrodesIBLBregma.beryl_location`. Trap found on the way: in pynwb, `units["spike_times"]`
  returns the ragged **index**, not the times; read `units["spike_times_index"].target`.
- `scripts/compare_nwb_vs_one.py` → `results/nwb_vs_one/5c0c560e.json`. Result: **stimulus-aligned
  tensors bit-identical on the 842 shared trials**; each route keeps one trial the other drops
  (66 and 827); movement-aligned tensors differ in 0.4–2.4 % of entries (≤ 2 spikes) because
  movement onsets differ by ~0.4 ms. The two big trial-table differences (trial 66 stimulus onset,
  trial 1099 feedback) look like anomalies in the **ONE** table. Full write-up:
  [`nwb_vs_one_comparison.md`](../knowledge/nwb_vs_one_comparison.md).
- Tests: fast mapping tests in `tests/test_load_nwb.py`; one slow network test streams the session.

## 6. TODO

- **Citation and contact section** for the README (and a `CITATION.cff`): deferred by the author.
- Report the two ONE trial-table anomalies (§5) to IBL.
- Run `compare_nwb_vs_one.py` on the four replicate sessions (only `5c0c560e` was compared).
- The rSLDS strand still runs on the `pre400` tensors while the rest uses `pre300` (open since
  2026-07-16).
- The FEF drift diagnostics (2026-07-17) and the PSTH checks cite scripts that only existed in a
  session scratchpad; promote them to `scripts/` if they become standing controls.
