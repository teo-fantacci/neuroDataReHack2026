# 2026-07-17 — Why FEF's HMM looked "flat", and the drift that hid its states

Follow-up to `2026-07-17_monkey-events-delay-aligned.md`. The delay-aligned monkey HMM
(`notebooks/01_HMM_monkey.ipynb`) gave a **flat FEF occupancy** (`switch_rate≈0.26/trial` at K=4,
`0.00` at K=3) while DMFC carved time normally. Investigated end-to-end. Verdict below.

## The FEF "flatness" is REAL and has a specific cause: across-session drift

Session `ses-2022-08-20`, sub-Elgar. Diagnostics (`scratchpad/diag_fef*.py`):

- **FEF is strongly non-stationary.** Population rate falls 157→117 sp/s first→second half;
  **median FEF unit changes rate by ~200 %** across the session; `corr(trial#, pop rate) = −0.51`.
  DMFC is nearly stationary (20 % median unit change, `+0.32`).
- **The FEF HMM spends all its states on session epochs, not on within-trial time.**
  `corr(decoded state, trial index) = +0.87 (K=2), +0.91 (K=3), +0.81 (K=4)`; within a trial the
  state is constant (`frac_modal≈1.0`, `switch_rate≈0`). At K=3 the occupancy has **zero** temporal
  swing (three flat lines) and the states are three contiguous trial-index blocks
  (`0–198 / 199–245 / 246–413`). The "sorted-by-first-change" raster is identical to the chronological
  one — because there is no within-trial change to sort by.
- **It is NOT the inference.** Default (50 it / tol 1e-2) and long (500 it / tol 1e-4) give
  byte-identical LL and switch rate; converges in ~29 it. Running longer does nothing. Not a K
  problem either — K=2..5 all trial-cluster on the full session.
- **The ARI K-selection is "stably wrong" here** (the confound `hmm.py` already warns about): a
  trial-clustering partition is trivially reproducible, so ARI is high without a real temporal state.

## ⚠️ CORRECTION to the first read: FEF *does* have within-trial task-locked states

The first-pass conclusion ("FEF has no shared within-trial sequence") was **too strong**. It held for
the *full session* but reversed once the drift range was limited.

- **Poisson-thinning to remove rate/gain did NOT reveal states** (`scratchpad/fef_dedrift.py`).
  Gain-normalising every trial to a common total flattened `corr(trial#,rate)` to −0.04, yet the HMM
  **still** produced flat occupancy and the same trial-block partition (`corr(state,trial#) = −0.66`).
  Lesson: the drift is in the **population pattern** (which units fire, relative rates), not just the
  scalar gain — and rate-rescaling cannot remove pattern drift.
- **Fitting on a limited-drift BLOCK reveals a clean within-trial sequence**
  (`scratchpad/fef_block_fig.py`, fig `results/figures/monkey_hmm/FEF_block_reveals_states_K3.png`):

  | fit set | occ. swing | switch/trial | Cramér's V (state↔phase) | corr(state, trial#) |
  |---|---|---|---|---|
  | full session (414 tr) | **0.00** | 0.00 | **0.01** | +0.91 |
  | block trials 0–150 | **0.61** | 1.8 | **0.36** | −0.02 |
  | block trials 270–414 | 0.46 | 2.7 | 0.19 | −0.80 (K=3) / +0.47 (K=4) |

  In block 0–150 the occupancy is a real sequence: an **early/baseline state (S0) hands off to a
  delay→response state (S2) that peaks around the response**, S0 returning at reward — and the states
  now carry task-phase information (V 0.01→0.36) instead of session-epoch identity.

**So:** FEF's within-trial states are genuine but **low-variance relative to its slow drift**. On the
full session the drift variance swamps them and the HMM reports session epochs; restrict the drift
range (or, properly, model the drift) and the within-trial sequence dominates the fit.

DMFC is the positive control: de-drifting drops its `corr(state,trial#)` 0.51→0.17 while **preserving**
its within-trial structure (swing ~0.5, switch ~9/trial). The pipeline finds states when they're there.

## What to do (recommended)

1. **Model per-neuron slow drift, don't discard trials.** A GLM-/input-driven Poisson HMM with a
   smooth trial-index covariate on each neuron's log-rate absorbs the drift as an offset and frees the
   states for within-trial dynamics — no thinning, no lost trials. `ssm` (the `ndrh_ssm` env) supports
   input-driven observations; run out-of-process like `rslds_ssm.py`. **Verify Poisson-input support
   against the installed `ssm` before committing.**
2. **Interim / assumption-light:** fit per locally-stationary block and pool the statistics (as here).
   Confirms the states without any drift model.
3. **Fix K reporting for this dataset:** pair ARI with `switch_rate`/occupancy swing; reject any
   "state" with `switch_rate≈0` as a trial label, not a temporal state.
4. **Replicate** across the 14 usable sessions + 2nd animal before any FEF within-trial claim — the
   block-to-block variability (0–150 clean, 270–414 messier) shows one session is not enough.

Figures: `results/figures/monkey_hmm/FEF_K3_vs_K4_drift.png`, `FEF_dedrift_K3.png`,
`FEF_block_reveals_states_K3.png`. Scratch scripts under the session scratchpad (`diag_fef.py`,
`diag_fef2.py`, `fef_dedrift.py`, `fef_block*.py`) — promote to `scripts/` if this becomes a standing
control.
