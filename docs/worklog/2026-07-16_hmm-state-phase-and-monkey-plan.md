# 2026-07-16 — HMM state↔task-phase alignment (IBL), and the plan for the monkey dataset

## Context

A new dataset landed (`notebooks/07_data_colleagueD.ipynb`): macaque **working-memory** task,
DANDI **000620**, two areas **FEF** (frontal eye field) + **DMFC** (dorsomedial frontal cortex).
User wants to (a) fit an HMM to it, (b) know whether to *merge sessions but keep areas separate*,
and (c) correlate inferred states with the task phases — noting this is a good analysis for IBL
too. Implemented the IBL version; wrote the monkey version up as a scoped, blocked TODO.

## What colleague D.'s pickle actually is (inspected, not assumed)

`data/000620_hmm.pkl.xz` (loaded via `lzma` + `dill`), decompressed structure:

- `dict[session] -> dict[area] -> object ndarray (n_trials, n_neurons)`, each cell a 1-D array of
  **spike times (s), relative to trial start** (`t` runs 0 → ~5–25 s, median max ≈ 7 s).
- **26 sessions, 14 usable** (the other 12 have neuron columns but **0 trials** — empty).
- Every usable session has **both** FEF and DMFC with **identical trial counts** (249–510 trials)
  → the two areas are **simultaneously recorded** within a session.
- Neuron yield: FEF small (3–48/session), DMFC larger (25–101/session); totals over usable
  sessions ≈ 261 FEF, 735 DMFC.
- **No task-event timestamps in the pickle** — only spike times. Phase labels need those.

## The design answer (merge sessions? keep areas separate?)

Same rule as IBL (worklog 2026-07-14 §5): **fit one HMM per session × area, pool the statistics**.
- *Keep areas separate* — yes. Fit FEF and DMFC independently (per the per-region convention).
- *Merge sessions* — **no, not into one population.** Neurons differ across sessions, so a merged
  tensor would be a **pseudo-population**: it destroys the single-trial joint activity a state-space
  model consumes. "Merge" = pool the read-outs across sessions as a **replication** ("K=3 in n/14,
  forward_fraction …"), which is the animal-independence argument made properly. 14 sessions is a
  stronger replication substrate than IBL's 5.
- FEF+DMFC *are* simultaneous, so a **joint two-area HMM** (concatenated population, ask whether the
  transition matrix carries cross-area structure) is available later — the analogue of the SCm↔MRN
  joint-rSLDS idea. Not needed for the first pass.

## Implemented (IBL) — state ↔ task-phase alignment

New in `src/dynamics/hmm.py`:
- `phase_labels(bin_centers, landmark_times)` — per-(trial, bin) phase = # landmarks passed.
- `state_phase_alignment(states, bin_centers, landmark_times, …)` — NMI(state; phase) referenced
  to a **trial-permutation null** (shuffle each trial's landmark times across trials). Returns
  `nmi`, null mean/std, `nmi_z`, permutation `nmi_p`, Cramér's V, and `P(state|phase)`.
- `state_locked_occupancy(states, bin_centers, event_offset, …)` — re-lock state occupancy onto a
  per-trial event (a state-PSTH); shows whether a boundary locks to the event.

**Why the null matters:** state and phase are both strong functions of absolute time, so raw NMI is
high for free. The trial-permutation null keeps the absolute-time structure and destroys only the
per-trial coupling (chiefly to RT), so alignment *above* it is what the clock can't explain.

Script `scripts/run_hmm_state_phase.py` → `results/hmm/state_phase.{csv,json}` +
`results/figures/hmm_state_phase/<eid>_<region>.png`. Ran all 5 triplet sessions, K=3 all regions
(5-session common K), stimOn tensors, 200-shuffle null. **Pooled (beats trial-permutation null):**

| region | beats null | mean z | Cramér's V |
|---|:--:|--:|--:|
| VISp | 3/5 | 12 ± 20 | 0.15 ± 0.10 |
| SCm  | 5/5 | 16 ± 13 | 0.37 ± 0.16 |
| MRN  | 5/5 | 13 ± 13 | 0.34 ± 0.17 |

**Reading.** Motor areas' states track the trial phase above the clock-null in every session; VISp
weakly, 3/5 — the same motor-directed / visual-undirected dissociation the directedness read-out
found, via a second metric. **Honest caveat:** the movement-locked occupancy panel shows most of the
alignment *is* the baseline→evoked switch snapping to movement onset, so it inherits the "window is
mostly movement" confound — it beats a clock-only account, it does **not** separate movement from
decision. The `firstMovement`-aligned tensors are the control (a natural extension: run the script on
the `firstMovement_*` npz and label phases with `[−RT, 0]` = [stimOn, move]).

Recorded as **Part 7** of `notebooks/01_hmm_triplet.ipynb`.

## Monkey HMM — first fit built and run (`notebooks/01_HMM_monkey.ipynb`)

Essential single-session pipeline mirroring the mouse `01_hmm_triplet` (load → tensor → **ARI
stability** → fit → read-outs), executed end-to-end. Session `ses-2022-08-20` (43 FEF, 56 DMFC,
415 trials). Trials are near-fixed **~4.3 s** (>95 % reach 4.1 s), so a **`[0, 4] s` @ 100 ms**
window keeps all 415 (λ/bin 0.31 FEF / 0.53 DMFC — the mouse's Poisson regime). Spike-times→counts
loader is **inline** in the notebook (short). `bias_startprob=False` (no verified baseline in a
trial-start-aligned window); **no** state is tagged "baseline/early" (removed — it mislabelled a
late state). Window/bins are **provisional** (chosen for λ, not task structure).

First results (**provisional, one session, no epoch alignment — not to quote**):
- **DMFC**: genuine **interior ARI peak at K = 3** (≈ 0.89, up from the trivial K = 2 ≈ 0.61) — real
  reproducible within-trial structure — with a **late-trial transition into state 2 (~3.2 s)**.
  Switches fast (**~7/trial**) with only a slight net-forward drift (`forward_fraction` ≈ 0.51): the
  same directedness ≠ switch-rate dissociation as the mouse, now in monkey cortex.
- **FEF**: nearly **static within-trial** (**~0.2 switches/trial**); its K = 4 states separate
  *trials* by firing level more than they carve *time*, and ARI just declines from K = 2 (no interior
  peak). Could be real, or FEF structure is **event-locked (saccades) and smeared in trial-time** —
  event alignment would tell. Do **not** read this as "FEF has no states".

Figures: `results/figures/monkey_hmm/ses-2022-08-20_{activity,stability,FEF_hmm,DMFC_hmm}.png`;
read-outs `results/hmm/monkey_ses-2022-08-20_readouts.csv`.

## Dimensionality reduction — monkey notebook + rotating GIFs (mouse & monkey)

New reusable code:
- `src/viz/anim3d.py::rotating_3d_gif(panels, ...)` — slow-rotating 3-D GIF for slides (draws the
  scene once, sweeps azimuth). CVD-safe: cividis for scalar/time, Okabe-Ito categorical, marker
  *shape* for dPCA condition; `depthshade=False` so colours don't flicker through the turn.
- `src/data/monkey.py` — spike-times → `(trials, bins, neurons)` loader for DANDI 000620
  (`load_monkey_tensors`, `usable_sessions`), shared duration-filtered trial set across FEF/DMFC.

`notebooks/02_dimred_monkey.ipynb` (new, executed): **PCA** (+ rotating GIF, cloud + trial-mean
trajectory) and **RRR** (FEF↔DMFC communication subspace) — both need no labels. **dPCA is BLOCKED**
(needs stimulus/decision condition labels the pickle lacks); section is wired with the stub, and a
stand-in GIF is deliberately NOT shown. First RRR (ses-2022-08-20, trial-time window): cross-area
prediction weak — **DMFC→FEF R²≈0.11 (sat rank 5) > FEF→DMFC R²≈0.05 (rank 6)**; FEF self-rank ≈3,
DMFC self-rank ≈8 (DMFC higher-dimensional). PCA: FEF PC1≈25% (coherent low-D arc), DMFC diffuse.

Mouse `notebooks/02_dimred_msr.ipynb`: appended **§10** with the two rotating GIFs (PCA + dPCA)
reusing the notebook's existing `emb` / `dpca_Z`. dPCA GIF reproduces the decision-variance gradient
(VISp 1% → SCm 3% → MRN 9%). GIFs land in `results/figures/{phase3_dimred,monkey_dimred}/rotate_*.gif`.

**Verification:** all three GIFs + the RRR were generated on real data and frame-inspected before
being written into the notebooks (the mouse notebook's heavy cells — raw load, GPFA, MDS — were NOT
re-run; the appended GIF cells reuse already-computed variables and match the verified standalone run).

## dPCA blocker (both datasets need this from the data provider)

The monkey dPCA and the state↔phase analysis are blocked on the **same** missing metadata: per-trial
**condition labels** (stimulus / decision) and **event timestamps**. One request covers both. With
them: dPCA + its GIF is one call, and `state_phase_alignment` runs on FEF/DMFC unchanged.

## Still open

- **Event timestamps** are the gate for everything interpretive: ask the data provider for FEF/DMFC
  task-event times (fixation/sample/delay/go/response). Then re-window to the epochs (or keep this
  window and label bins by phase) and run `state_phase_alignment` (mouse Part 7) on FEF/DMFC.
- **Replicate across the 14 usable sessions** and pool the statistics (per session × area, never a
  cross-session pseudo-population) — the `run_hmm_5sessions.py` pattern. The notebook does one
  session by design; a batch script is the next build once the window is settled.
- Revisit the **provisional `[0,4] s` / 100 ms** choice against the real epoch timing.
