# 2026-07-17 — Monkey task events arrived: delay-aligned HMM + dPCA + RRR

## What changed

The per-animal pickle **`notebooks/data/000620_sub-Elgar_hmm.pkl.xz`** arrived, adding a
`trial_times` field: **(n_trials, 7)** monotonic event times per trial. This unblocks the
state↔phase HMM (it needed events) and dPCA (it needed a behavioural condition). A 2nd animal is
coming. The old `000620_hmm.pkl.xz` (no events) is superseded.

## Task structure (decoded + confirmed by the user)

**6 phases**: fixation → stimulus → **delay** → cue → response → reward. The 7 `trial_times` columns
are the phase **onsets** (col6 = reward/trial end). Median onsets (s from trial start):

| col | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| event | fixation | stimulus | **delay** | cue | response | reward | end |
| time | 0.00 | 0.40 | **1.42** | 2.45 | 2.70 | 3.15 | 3.52 |

Stimulus and delay phases are each ~1.0 s; only the response/reward epoch varies trial-to-trial.

## Alignment (user request): align to DELAY onset

Trials are now aligned to **delay onset (col 2)**, t = 0, with window **`[-1.4, +2.2] s`** @ 100 ms
(`src/data/monkey.py::load_monkey_aligned`, per-trial re-zeroing). Fixation+stimulus sit at negative
times; the **working-memory delay period `[0, ~+1.0] s`** is front-and-centre with a bold dashed line
at 0. Bonus: a delay-aligned window ending ~+2.2 s stays inside every trial, so **~all 414 trials are
kept** (the old trial-start `[0,4] s` window dropped the ~10 % shortest).

New code: `monkey.py::{load_monkey_aligned, area_tensor_aligned}` (+ `PHASE_NAMES`, `EVENT_NAMES`,
`DELAY_COL`); `dpca.py::balanced_by_label` (general 1-factor C×time design; the existing
`balanced_conditions` is hard-wired to the mouse 2×2). Both notebooks rebuilt delay-aligned.

**Memory caching (a real constraint here — this laptop had ~0.7–1 GB free).** Unpickling the whole
59 MB / 26-session blob peaks at ~1–2 GB and **OOM'd nbconvert**. Two-layer cache in `monkey.py`,
written to `notebooks/data/_session_cache/` (gitignored under `/notebooks/data/`):
`load_session_dict` caches one session's raw dict (~4.5 MB); `load_monkey_aligned` additionally
caches its **binned output** as a dense `.npz` (~0.4 MB, `allow_pickle=False`) keyed by the window
params — so a notebook reload never re-unpickles the ~40 k spike arrays. First call still pays the
full load once (run it as a standalone script, not in nbconvert, if RAM is tight). GIF rendering is
also memory-heavy under nbconvert, so the dim-red GIF cells display a small static **preview PNG** and
leave the full `.gif` on disk (`rotate_{pca,dpca}.gif` + `_preview.png`).

## Results (ses-2022-08-20, provisional — one session, one animal)

**HMM (delay-aligned).**
- **DMFC**: reproducible **K = 3** (genuine interior ARI peak: 0.44 → **0.90** → decline). A
  **persistent state configuration is held across fixation→stimulus→delay**, switching to a distinct
  state only at the **response/reward** (~+1.3 s) — a working-memory maintenance-then-transition
  signature, now clearly visible against the delay line. switch ≈ 6.4/trial.
- **FEF**: nearly **static within-trial** (switch ≈ 0): its K = 3 states cluster *trials* by firing
  level, not time. Not "no states" — may be gaze/saccade-locked, off these phase times.

**State ↔ phase.** DMFC states tile the epochs descriptively (**Cramér's V ≈ 0.18**) but the
trial-permutation null shows **no beyond-clock effect** (z ≈ 0): because the task timing is
near-deterministic, "tracks phase" ≈ "tracks time". Reported both ways, honestly. (FEF: V ≈ 0.01.)

**dPCA** (`balanced_by_label`, `labels="ct"`; condition = response time = reward−response onset,
median-split short/long, 207/207). Variance: **time ~94–95 %**, **response-time ~3–4 %**, interaction
~1–3 %. Time dominates (fixed-timing task); dPCA isolates the small behavioural component — the honest
answer. No stimulus×choice factorial exists, so response time is the one behavioural factor.

**RRR** (FEF↔DMFC, grouped CV): weak and **asymmetric** — **DMFC→FEF R² ≈ 0.11 (rank 5) > FEF→DMFC
≈ 0.05 (rank 6)**; FEF low-dimensional to itself (self-rank ~3), DMFC not (~8). Unchanged by alignment.

## Deliverables

- `notebooks/01_HMM_monkey.ipynb` (delay-aligned): load → ARI → fit (occupancy with phase lines +
  delay line) → **state↔phase** → read-outs.
- `notebooks/02_dimred_monkey.ipynb` (delay-aligned): PCA (+ rotating GIF), **dPCA (+ rotating GIF)**,
  RRR. GIFs → `results/figures/monkey_dimred/rotate_{pca,dpca}.gif`.
- Figures → `results/figures/monkey_hmm/ses-2022-08-20_*` and `results/hmm/monkey_*_readouts.csv`.

## HMM under 3 independent alignments (stimulus / delay / response)

`scripts/run_monkey_hmm_alignments.py` fits the HMM per area three times, re-zeroing every trial on
**stimulus onset (col1)**, **delay onset (col2)**, and **response onset (col4)** independently, each
in a window ~covering the whole task. Result (ses-2022-08-20, K fixed FEF 4 / DMFC 3):

| alignment | FEF fwd_frac | FEF switch | DMFC fwd_frac | DMFC switch |
|---|--:|--:|--:|--:|
| stimulus | 0.585 | 0.23 | 0.528 | 6.04 |
| delay    | 0.562 | 0.23 | 0.529 | 5.94 |
| response | 0.526 | 0.23 | 0.524 | 6.21 |

**The three alignments give essentially the same fit.** Because the task timing is near-deterministic
(fixation 0.4 s, two ~1.0 s intervals, near-fixed response onset), aligning to any of the three events
is ≈ a constant shift, so the HMM recovers the same states and the same read-outs — DMFC's occupancy
just re-references the same maintenance-through-delay + switch-at-response pattern, FEF stays static.
This is the *opposite* of the mouse, where large RT variability made stimOn-vs-firstMovement alignment
decisive. Saved: `results/figures/monkey_hmm/ses-2022-08-20_alignment_comparison.png` (3×2 occupancy),
`results/hmm/monkey_ses-2022-08-20_alignments.csv`.

## Still open

- **Replicate across the 14 usable sessions** (pool per session × area), then add the **2nd animal**.
- A **stimulus/choice** label (if it exists) would enable the mouse-style 2×2 sensory/decision dPCA;
  the current dPCA uses response time as the only available behavioural factor.
- Consider a higher K for DMFC to try to separate stimulus-encoding from delay-maintenance within the
  sustained regime (ARI says K=3 is the reproducible count, so treat higher K as exploratory).
