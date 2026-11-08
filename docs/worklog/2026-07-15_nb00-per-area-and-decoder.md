# 2026-07-15 — `00_data_smoketest`: per-area examples + a choice/stimulus decoder

Extended the Phase-0 smoke-test notebook (still explicitly *not* a scientific result) with two
teaching sections, run end-to-end on session **8a3a0197** (UCLA034, 290 usable trials / 287 go).
Both use the **unified preprocessing** (`preprocess_session`, stimOn, pre 0.4 / post 0.8, **40 ms**,
≥ 1 Hz) and pool **both probes**, so all three axis regions appear — steps 4–6 only ever saw SCm.

## §7 — one PSTH per area, with an error band (`results/figures/smoketest/psth_per_area.png`)

VISp / SCm / MRN population PSTHs, each with a **±1 SEM-across-trials** band and a per-panel note
of *what the band is over* (n trials, n units). The band = std over trials / √n_trials of the
per-trial population rate (rate = mean over that region's units). Physiology is sane: SCm shows a
sustained post-stimulus rise, MRN a sharp ~150–200 ms transient (median RT here is 171 ms), VISp is
weak and noisy (only 23 units ≥ 1 Hz — this session's VISp is thin).

## §8 — decoding **choice** vs **stimulus** at different contrast ("coherence")

Answers a colleague's doubt: *"decoding accuracy doesn't make sense with coherence."* The trap is
decoding the **wrong label**. Balanced accuracy, 5-fold stratified CV, `class_weight='balanced'`
(so majority-guessing can't inflate the score), decoder = standardise → L2 logistic on mean counts
in a window. `results/figures/smoketest/decoding_vs_contrast.png`.

- **"Coherence" here = Gabor contrast** ∈ {0, 6.25, 12.5, 25, 100} %. Signed contrast
  `= contrastRight − contrastLeft` (+ = right stimulus). Empirically **`choice == −1` ⟺ mouse chose
  the RIGHT stimulus** (at |c| ≥ 50 % choice=−1 co-occurs with side=+1). 3 no-go trials
  (`choice == 0`) are dropped before choice decoding — they were the cause of a NaN in the first
  pass (min-class < n_splits).
- **Stimulus-side decoding rises with contrast** (the sensory signal): SCm 0.36 → 0.93,
  MRN 0.56 → 0.86 across |c| = 6 → 100 %. Early window [50, 200] ms.
- **Choice decoding is high and ~flat**, and **above chance already at 0 % contrast**
  (SCm/MRN ≈ 0.58 with no stimulus), rising to ~0.95. Broad window [0, 400] ms. The mouse commits
  to a wheel-turn on every trial (at 0 % it guesses, biased by `probabilityLeft`); motor areas
  encode the executed act, so choice decodes at all contrasts and **need not track contrast**.
- **Divergence is the point**: at low contrast stimulus decoding falls to chance while choice
  stays high; at high contrast the two converge (choice ≈ stimulus because the mouse is ~always
  correct). SCm at 6 %: stim 0.36 vs choice 0.81; at 100 %: 0.93 vs 0.95.
- **VISp near chance here** for both — thin/weak in this single session; the standing reason we
  replicate over the 5 triplet sessions. SCm decodes the *stimulus* well because the superior
  colliculus is itself a visual structure.

**Takeaway for the decoder question:** a *choice* decoder should NOT scale with coherence; a
*stimulus* decoder should. Window choice matters (early ≈ sensory, broad/peri-movement ≈ action),
and this window entanglement is the same movement caveat flagged project-wide (movement has begun
on 92 % of trials by +0.8 s → the `firstMovement`-aligned tensors exist for exactly this).

Nothing here changes the pipeline; the notebook was re-executed clean (0 errors) so its committed
outputs are trustworthy this time.

---

## Update (2026-07-16) — `00` is now the **visual data-inspection** notebook

Reworked §7–§10 into a coherent "first look at the data" block. Every figure has a meaningful
title and an italic one-line caption below it (`fig.text`, so it travels with the saved PNG);
figures land in `results/figures/smoketest/`. Re-executed end-to-end, **0 errors**.

- **§7 · PSTH per area (juxtaposed)** — unchanged content, now captioned; the ±1 SEM-across-trials
  band and per-panel annotation stay.
- **§8 · Spike rasters, two cuts.**
  - **8a neurons × time, one example trial** (`region_event_spikes`): the whole population in a
    single high-contrast correct trial, one column per area. Shows VISp sparse, SCm/MRN dense.
  - **8b trials × time, one example neuron** via **pynapple `compute_perievent`**, rows sorted by
    RT. ⚠️ **Pick the example unit by task *modulation*, not raw rate** — the most-active SCm/MRN
    units fire so densely the raster is a solid black block. `most_modulated_unit` scores
    `(peak−base)/(base+2)` and skips units with baseline ≥ 12 Hz. With that, all three areas show a
    clean band that **bends along the RT curve → movement-locked**, the visual form of the
    project-wide "the window is a movement window" caveat.
- **§9 · Two pynapple plots** (the "we used what the hackathon suggested" beat): **SCm↔MRN
  cross-correlogram** (sharp zero-lag peak → common drive, reproduces 2026-07-12 §6c on the real
  triplet-session data) and **contrast tuning** (SCm tracks |contrast|, MRN flatter, VISp sparse).
  Both via the existing `src/viz/nap_plots.py` helpers. Cross-correlogram over all SCm×MRN pairs
  takes ~35 s.
- **§10 · Decoder decluttered** from a 2×3-with-text-panel to a clean **2×2** (psychometric +
  one panel per area); the "how to read it" prose moved to the §10 markdown + a caption. Same
  numbers as the 07-15 run.

Data-prep note: cell 21 now also derives `event_times`, `rt`, and `region_trains`
(`region_spike_trains` per region) once, reused by every raster/pynapple cell. `keep_idx` maps a
kept-tensor unit back to its `region_trains` key — needed so the example-neuron raster plots the
*same* unit the tensor scored.
