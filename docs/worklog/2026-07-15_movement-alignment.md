# 2026-07-15 — The HMM's "evoked" state is a MOVEMENT state (+ 40 ms replication plumbing)

New: `scripts/run_movement_alignment.py`, `scripts/run_hmm_sessions.py` (multi-session HMM),
clean `notebooks/01_hmm_triplet.ipynb` (unified preprocessing). This entry is mostly the
movement-alignment finding, which sharpens the 2026-07-14 §1 warning into a measured result.

---

## 1. The test: stimulus-locked or movement-locked?

The 2026-07-14 worklog warned that the `[0,+0.8]s` window is *mostly movement and reward*
(median RT 171 ms). This makes it concrete. For each region we fit the **same** Poisson HMM
twice — on the **stimOn**-aligned tensor and on the **firstMovement**-aligned one — and ask, per
trial, when the population first leaves its baseline set. Then we compare the **across-trial
jitter** (SD) of that exit time between the two alignments.

The logic is the standard RT-jitter argument. Reaction time varies a lot (SD ≈ **333 ms** here):

* a **stimulus-locked** transition, viewed in the movement frame, is smeared by the RT jitter →
  `SD(exit | move) ≈ sqrt(SD_stim² + SD_RT²)` — strictly **worse**;
* a **movement-locked** transition, viewed in the movement frame, has the jitter **removed** →
  `SD(exit | move) < SD(exit | stim)`.

So the sign of `SD_move − SD_stim` names the event the state tracks. Memory-safe: it reads only
the ~1 MB processed tensors, no raw-spike download.

## 2. Result (session `5c0c560e`, 40 ms, both alignments)

| region | SD exit \| stimOn | SD exit \| movement | expected if stim-locked | verdict |
|---|---:|---:|---:|---|
| VISp | 97 ms | **57 ms** | ~347 ms | movement-locked (**soft**, see §3) |
| **SCm** | 117 ms | **44 ms** | ~353 ms | **strongly movement-locked** |
| **MRN** | 189 ms | **153 ms** | ~383 ms | movement-locked (weaker residual) |

Every region's first state transition is **more tightly locked to the wheel-turn than to the
stimulus**, and by a wide margin against the stimulus-locked null (57 / 44 / 153 ms observed vs
~347–383 ms expected if the transition were stimulus-locked).

The figure (`results/figures/phase2_hmm/5c0c560e_occupancy_stim_vs_move.png`) shows it directly.
SCm is the clearest: in the stimulus frame the baseline decline and the green evoked-state rise
are **broad** (peak ≈ 0.6 around +0.2 s); in the movement frame they **snap to t = 0** and the
evoked state peaks **higher** (≈ 0.8). MRN behaves the same, slightly later/broader.

> **So the HMM's post-stimulus "evoked" state is, to first order, a MOVEMENT state.** The
> metastable transition the model finds coincides with movement onset, not with stimulus onset
> or an internal decision. This is the §1 warning made quantitative: **no state after ~0 may be
> called a "decision" state** — at least the *first* one is the movement.

## 3. Caveats — do not over-read this yet

1. **VISp is the soft case.** Its baseline *set* differs between alignments (stimOn → {state 0};
   firstMovement → {0, 2}), because in the movement frame the pre-movement window is
   post-stimulus and an "evoked" state is already occupied. So VISp's exit is defined slightly
   differently in the two frames and its number is not a clean like-for-like. SCm (baseline =
   {0} in both) and MRN (baseline = {1} in both) are clean comparisons — trust those.
2. **"Movement-locked" is not "movement-caused".** In IBL the choice *is* the wheel-turn, and
   mouse cortex carries movement signals broadly. This says the state transition is *timed by*
   movement; it does not say the state encodes movement kinematics rather than choice.
3. **It is the FIRST exit only.** An earlier, weaker stimulus-locked transient could exist and be
   skipped by "first leave baseline". Not evidence against an early sensory state — just silent
   about it.
4. **One session.** Needs the replication (§4).

## 4. 40 ms replication — plumbing done, waiting on tensors

`scripts/run_hmm_sessions.py` runs the clean HMM pipeline over every processed session and
**pools the statistics** (per-session-then-pool; a pseudo-population is invalid for the HMM —
2026-07-14 §5). `scripts/run_movement_alignment.py` does §1–2 for every session that has both
alignments. Both take `--bin-ms` and pin one bin so they never mix timebases.

Current 40 ms coverage: **only `5c0c560e`**. `111c1762` exists at 50 ms only.

### ⚠️ REQUEST TO THE PREPROCESSING CHAT

Please produce, with `src/data/preprocess.py` at **`bin_size_s = 0.04`**, both alignments
(`stimOn_times` **and** `firstMovement_times`), `pre_s = 0.4`, `post_s = 0.8`, `min_rate_hz = 1.0`:

1. **`111c1762`** — re-bin the already-downloaded session to 40 ms (chosen by the user).
2. **One more** for a 3-session replication. Recommendation: **`90c61c38`** (656 usable trials —
   2nd-most, which is what the HMM stability actually needs; `8a3a0197` has more units but only
   290 trials, and trials are the limiting resource for these models — see 2026-07-14 §4).

Expected filenames (what the runners glob for):
`data/processed/<eid8>__stimOn_pre400_post800_bin40_rate1.npz` and `…__firstMovement_…`.

Once they land: `python scripts/run_hmm_sessions.py --bin-ms 40` and
`python scripts/run_movement_alignment.py --bin-ms 40` reproduce everything, no code changes.

---

## Open

1. **`111c1762` + one more at 40 ms/both alignments** (the request above), then re-run the two
   scripts and pool.
2. The movement-locking result needs the replication before it goes in a talk, but it is already
   strong enough to **retire any "decision state" language for the first evoked transition**.
3. Still open from 2026-07-14: rSLDS ground truth at 40 ms was the reason 40 ms was adopted
   (done — that is why we are here); the recurrence test `delta_recurrent`; `areas.yaml::selected`.
