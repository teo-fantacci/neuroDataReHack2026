# The five {VISp, SCm, MRN} sessions — what each one is actually worth

Generated 2026-07-14 from `bwm_units` + the trials tables (metadata only, no spike downloads).
Raw table: `results/triplet_sessions_survey.csv`. Script: `scripts/survey_triplet_sessions.py`.

These are the 5 IBL sessions that co-record all three of our frozen regions with ≥20 good
units each. We have so far used **only** `5c0c560e` for everything (Phases 2–3).

## Overview

| eid | subject / lab | probes | trials tot → usable | perf | perf(easy) | VISp | SCm | MRN | units (≥1 Hz) | K (visible) |
|---|---|---|---|---|---|---|---|---|---|---|
| **5c0c560e** | NR_0028 / steinmetz | 2 | 1100 → **843** | .84 | .98 | 36 (22) | 59 (51) | 63 (62) | 158 (135) | **10** |
| **90c61c38** | CSH_ZAD_025 / zador | 1 | 804 → **656** | .84 | .99 | 20 (17) | 20 (19) | 36 (29) | 76 (65) | **33** |
| **111c1762** | UCLA033 / churchland | 2 | 724 → 364 | .75 | .99 | 32 (32) | 53 (49) | 53 (47) | 138 (128) | 15 |
| **90d1e82c** | DY_010 / danlab | 1 | 541 → 353 | .83 | 1.00 | 28 (26) | 32 (30) | 22 (19) | 82 (75) | 23 |
| **8a3a0197** | UCLA034 / churchland | 2 | 417 → 290 | .78 | 1.00 | 33 (27) | **81** (74) | **74** (67) | **188** (168) | 15 |

`(n)` = units with session firing rate ≥ 1 Hz, i.e. what survives our `MIN_RATE_HZ` cut.
`K (visible)` = trials in the smallest (stimulus side × choice) cell, **excluding 0 % contrast**.

Pooled over all 5: **2506 usable trials**, **149 VISp / 245 SCm / 248 MRN** good units
(124 / 223 / 224 active). All 5 mice are engaged (easy-trial performance .98–1.00).

## 1. Our working session is the best for trials and the WORST for VISp

`5c0c560e` has the most trials (843) — which is why it was picked — but its VISp is the
weakest region in the entire set:

| session / probe | region | n | ≥1 Hz | median FR | median presence_ratio | median contamination |
|---|---|---|---|---|---|---|
| 5c0c560e / d5e5311c | **VISp** | 27 | 16 | **1.09 Hz** | **0.304** | **0.118** |
| 5c0c560e / 478de1ce | **VISp** | 9 | 6 | 1.67 Hz | 0.563 | 0.036 |
| 5c0c560e / 478de1ce | MRN | 50 | 49 | 7.00 Hz | 0.960 | 0.016 |
| 90d1e82c / b40d8c65 | VISp | 28 | 26 | 4.64 Hz | 0.968 | 0.012 |
| 111c1762 / 27bac116 | VISp | 32 | 32 | 5.15 Hz | 0.973 | 0.018 |

`presence_ratio` = fraction of **10-second** bins of the whole session containing ≥1 spike
(`brainbox/metrics/single_units.py:971-974`, `presence_window = 10`). A median of **0.30**
means the median VISp unit is *undetectable in 70 % of the session*. Only **22 of 36** VISp
units clear 1 Hz, against 62/63 for MRN.

### This is not a low-firing-rate artefact — the units really do disappear

The obvious objection is that a slow unit trivially misses some 10-s bins. It does not, at
these rates: a **stationary** Poisson unit at 1.09 Hz expects 10.9 spikes per bin, so
P(empty bin) = e^−10.9 ≈ 2·10⁻⁵ and its presence_ratio must be ≈ 1.0. Observed vs that
expectation, in `5c0c560e`:

| region | n | median FR | presence_ratio observed | expected if stationary | deficit |
|---|---|---|---|---|---|
| MRN | 63 | 8.6 Hz | 0.977 | 1.000 | 0.02 |
| SCm | 59 | 8.0 Hz | 0.930 | 1.000 | 0.07 |
| **VISp** | 36 | 1.2 Hz | **0.352** | 1.000 | **0.61** |

And restricting to the 22 VISp units that **pass our ≥1 Hz cut** — the ones we actually
analyse — **17 of 22 still have presence_ratio < 0.8**, median 0.58. These are not slow
units; they are units that fire at a normal rate *while they are present* and are simply
gone for much of the session. Their low mean rate is a **consequence** of the absence, not
the cause of the low presence_ratio.

**Two consequences, one of which we have not accounted for anywhere:**

1. **The dPCA confound.** The decision−stimulus gradient (VISp −6.8 → SCm +1.8 → MRN +6.0)
   is measured in the one session where VISp is sparse, intermittent and contaminated while
   MRN is clean. "VISp carries little decision variance" and "VISp is badly recorded here"
   are not currently distinguishable.
2. **The VISp population is not the same population across trials.** A unit present in 35 %
   of the session contributes real spikes in only ~35 % of trials and structural zeros in
   the rest. Our tensors and the Poisson HMM/rSLDS emission model assume a **fixed**
   population with stationary rates. For VISp that assumption is violated at the session
   timescale. This inflates across-trial variance and depresses every VISp statistic —
   and VISp is exactly the region with the chance-level `forward_fraction` (.53). Whether
   the intermittency *produced* that number is untested and should be: refit VISp on the
   sub-window where its units are actually present, or carry a per-unit presence mask.
   (Note the dropout is slow, on a session timescale, while HMM states are within-trial —
   so it cannot manufacture *within-trial* dynamics, but it can certainly wash them out.)

Every other triplet session has a better VISp than this one.

## 2. The dPCA K=25 is inflated by 0 %-contrast trials

`notebooks/04` builds the stimulus factor as `stim_right = ~np.isnan(contrastRight)`. On IBL
0 %-contrast trials a side is still *nominally* assigned, so those trials are labelled
left/right even though **there is no visible grating**. They then fall into the rare error
cells and pad them:

| eid | K as run | K, visible stimuli only | 0 %-contrast share of the limiting cell |
|---|---|---|---|
| **5c0c560e** | **25** | **10** | **15 of 25 (60 %)** |
| 90c61c38 | 46 | 33 | 13 of 46 (28 %) |
| 111c1762 | 16 | 15 | 1 of 16 (6 %) |
| 90d1e82c | 28 | 23 | 5 of 28 (18 %) |
| 8a3a0197 | 20 | 15 | 5 of 20 (25 %) |

So in the session we actually ran, **60 % of the limiting cell is trials the mouse could not
see** — guesses. Two consequences: the *stimulus* marginalisation is diluted by trials
carrying no stimulus, and the *decision* marginalisation leans on the cell those guesses
pad. The honest balanced-design number for `5c0c560e` is **K = 10, not 25**.

Decide deliberately which design we want (both are defensible, they answer different
questions) and say which one the figure used:
- **drop 0 % contrast** → clean stimulus factor, K = 10 here;
- **keep them** → K = 25, but "stimulus" means *nominal side*, and it must be reported that way.

The reason `5c0c560e` is worst on this axis is a **side bias**: its error cells are
wildly asymmetric (stimL/chR = 98 vs stimR/chL = **10**). The mouse rarely erred to one side.

## 3. What pooling actually buys (and what it does not)

- **Trials: 843 → 2506.** Real, and it is what rSLDS needs (it is trial-limited, not
  neuron-limited — see CLAUDE.md).
- **Units: VISp 36 → 149.** Real *for trial-averaged / pseudo-population methods* (dPCA,
  RRR, geometry). **Not** for single-trial state-space fits — those need simultaneously
  recorded populations, so HMM/rSLDS must still be fit **per session** and the *statistics*
  combined across sessions, never the units concatenated.
- **dPCA K does not pool.** A pseudo-population needs K trials per cell from *every*
  contributing session, so K_pooled = min over sessions = **10** (visible-stimulus design),
  set by `5c0c560e`. Dropping `5c0c560e` from the dPCA pool raises K to **15** while
  *increasing* VISp units to 113. That is a trade worth considering — and it is the opposite
  of the intuition that our best session must be in every analysis.

## 4. Practical notes per session

- **90c61c38** — the behavioural gem: single probe, 656 trials, best balanced design
  (K = 33), clean units. But only 76 units (20 VISp / 20 SCm), so it is thin for geometry.
- **8a3a0197** — the population gem: 188 units (81 SCm, 74 MRN), but only 290 trials, and
  MRN presence_ratio .69 on the high-yield probe. SCm and VISp sit on **different probes**
  (still simultaneous, so this is fine — just be aware for any depth/probe covariate).
- **111c1762** — the trial mask throws away **half** the session (724 → 364) and performance
  is lowest (.75). Units are excellent (128/138 active).
- **90d1e82c** — unremarkable and clean; the best VISp presence_ratio (.97) after 111c1762.
- **5c0c560e** — most trials, cleanest MRN/SCm, **weakest VISp, worst balanced design.**

## 5. OPEN — contrast is a variable we are currently throwing away (revisit)

Parked deliberately on 2026-07-14; written down so it is not lost.

Right now contrast enters our analyses **only** as a nuisance: `stim_right` is a binary
side label, 0 % trials are silently folded in, and everything else (6 %, 12 %, 25 %, 100 %)
is collapsed together. IBL's task actually gives us a **graded stimulus-strength axis**, and
we are not using it. Two distinct things to decide, and they should not be conflated:

**(a) What to do with 0 % contrast.** These are trials with *no perceivable stimulus* but a
*nominal* side label and a real choice. Three defensible treatments, and they answer
different questions:
  - **exclude** — the clean "stimulus" factor (what we called K_visible above);
  - **keep, labelled by nominal side** — what notebook 05 does now, which quietly redefines
    "stimulus" as "side the trial was scored against" and pads the error cells with guesses;
  - **treat as their own condition** — arguably the *most interesting* option: a 0 %-contrast
    trial is a **choice without a stimulus**. If the metastable sequence still runs on those
    trials, the sequence is not stimulus-driven — it is decision//motor-driven. That is close
    to a direct test of the project's central claim, and it is free (85 trials in `5c0c560e`,
    ~10 % of every session).

**(b) Contrast-graded comparisons.** The natural questions, none of which we have asked:
  - Does the **HMM state sequence** change with contrast — same states, earlier onset (a
    latency shift), or *different* states (a qualitative change)? Baseline-exit latency vs
    contrast is a one-line extension of what we already compute.
  - Does the **VISp → SCm → MRN** decision/stimulus gradient depend on contrast? The
    prediction is that VISp's stimulus variance should scale with contrast while MRN's
    decision variance should not.
  - dPCA supports a **3-factor design** (time × contrast × choice), not just 2 × 2 — but the
    balanced-cell cost is brutal: splitting the stimulus factor into 4–5 contrast levels
    multiplies the number of cells and K is set by the smallest. **Check the cell counts
    before committing**; this may only be affordable on the pooled 5 sessions, if at all.

The trap to avoid: contrast and reaction time are correlated (low contrast → slow RT), so
any contrast comparison on *stimulus-aligned* data is partly an RT comparison. Align to
movement, or control RT, before claiming a contrast effect.

## 6. How to read IBL's unit-quality columns (and one to distrust)

- **`presence_ratio`** — fraction of 10-s bins of the session with ≥1 spike. **Trustworthy**,
  and the metric that carried the VISp finding above. Only confounded with rate below
  ~0.5 Hz (where e^−rate·10 stops being negligible), so always sanity-check it against the
  stationary-Poisson expectation `1 − exp(−fr · 10)` rather than reading it raw.
- **`contamination`** — estimated refractory-period violation rate, i.e. fraction of spikes
  from *other* neurons. Interpretable as-is. VISp in `5c0c560e` has median .118 on one probe
  (vs .016 for MRN), and several of its ≥1 Hz units run .34–.83 — those units are largely
  *not one neuron*.
- **`firing_rate`** — mean rate over the **whole session**, so it is depressed by absence.
  For an intermittent unit it is *not* the rate you would measure while it is present. Our
  `MIN_RATE_HZ = 1.0` cut therefore silently discards well-isolated units that were merely
  absent for half the session — and keeps intermittent ones that fired hard while present.
- **`drift` — DO NOT USE for comparing regions.** It is computed as
  `sum(|diff(spike_depths)|) / duration * 3600` (`single_units.py:1010`), i.e. the *total
  variation* of the unit's estimated depth summed **over every spike**, in µm/hour. Because
  it is a sum over spikes rather than a mean, it scales with spike count:
  **Spearman(drift, firing_rate) = 0.86** across all 62,990 BWM units (0.85 vs spike_count).
  Normalising per spike collapses the correlation to 0.23. So the raw column mostly measures
  *how much the unit fired*, not how much the probe moved — the 10⁵–10⁶ values in the survey
  table are meaningless as a quality ranking, and the fact that VISp shows the *lowest* drift
  (74 k vs MRN 378 k) is purely its lower rate. If drift is ever needed, use
  `drift / spike_count`. This is why the quality claims in this document rest on
  `presence_ratio` and `contamination` only.

## 7. UPDATE 2026-07-14 — 111c1762 downloaded and frozen (our 2nd session)

Downloaded and run through the frozen pipeline (`scripts/preprocess_second_session.py`);
tensors at `data/processed/111c1762__{stimOn,firstMovement}_pre400_post800_bin50_rate1.npz`.
`src/data/preprocess.py::TRIPLET_EIDS` now holds all five (was a `(WORKING_EID,)` stub).

**It confirms its promise: VISp is clean here.** Frozen tensor is **VISp 30 / SCm 49 / MRN 47**
active units (843-session was 19 / 44 / 54). The 1 Hz cut keeps **99.2 / 99.7 / 99.8 %** of
spikes — as free as in the working session. VISp median rate 4.5 Hz (vs 1.1) and only 2 of 32
units fall below 1 Hz. This is the well-recorded VISp we picked it for.

**But two cautions, one of them new and important:**

1. **The movement/feedback confound (worklog §1) is WORSE here, not better.** This mouse is
   even faster: median RT **110 ms** (vs 171), median feedback **285 ms** (vs 427). Over the
   [0, +0.8] s window, movement has begun on **97.8 %** of trials and feedback has arrived on
   **95.9 %**. So the post-stimulus window is *almost entirely* movement + outcome here.
   Any "decision" state is even more entangled — the `firstMovement`-aligned tensor (also
   frozen) is not optional for this session, it is the primary read.
2. **Strong side bias again, and in the same direction.** choice L/R = **277 / 87**, perf .75.
   The rare-error cells that cap dPCA's K are small for the same reason as in `5c0c560e`.
   Both of our two sessions are left-biased — pooling them does **not** average the bias out,
   it compounds it. A right-biased session (if any of the remaining three is) would be worth
   adding specifically to balance this.

Baseline is clean: `quiescencePeriod` min 0.401 s, so the 0.4 s pre-window is inside the
enforced-still period on **100 %** of trials here too — the frozen `pre_s=0.4` transfers.

**Next:** fit the per-region HMM on 111c1762 and compare *statistics* to 5c0c560e (K,
`forward_fraction`, `switch_rate`, baseline-exit ordering) — this is the replication test,
not a pseudo-population (worklog §5). If VISp's undirectedness (ff ≈ .59) reproduces on a
*clean* VISp, that kills the "VISp looks undirected because it's badly recorded" alternative.

## The one-line takeaway

We chose `5c0c560e` on trial count, which was the right criterion for the HMM. But it is the
**worst** of the five for the two things the geometry/dPCA claim depends on — VISp recording
quality and stimulus×choice balance. The cross-area gradient must be reproduced in at least
one other session before it is claimed.
