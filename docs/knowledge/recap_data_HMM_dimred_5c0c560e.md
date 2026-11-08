# Recap (single session) — from raw activity to HMM and geometry · session `5c0c560e`

*Written 2026-07-17. A **single-session** version of
[`recap_data_HMM_dimred.md`](recap_data_HMM_dimred.md), for IBL session
**`5c0c560e-9e1f-45e9-b66e-e4ee7855be84`** (NR_0028, steinmetzlab — the project's working
session and the strongest of the five triplet recordings: 843 usable trials, VISp 19 / SCm 44 /
MRN 54 units ≥ 1 Hz). Everything here is computed on **this session alone** — no 5-session
pooling. It is a companion to the parallel doc for session
[`111c1762`](recap_data_HMM_dimred_111c1762.md). Numbers come from
`results/session_recap/5c0c560e.json`; figures from `results/figures/recap_5c0c560e/`
(gitignored — regenerate with `python scripts/build_session_recap.py --eid 5c0c560e`).*

> **Why two single-session docs?** The pooled recap deliberately averages the five replicate
> sessions. This one asks the narrower question — *what does one strong session say on its own?*
> `5c0c560e` reproduces the canonical picture cleanly; `111c1762` reproduces the directedness
> result but **not** the dPCA gradient (see its doc). Reading the two together is the honest
> per-session view behind the pooled numbers.

**The question.** Is a decision a **temporal sequence of stereotyped population states**,
**shared across brain areas** but drawn on **area-specific geometry** — rather than a clean relay
from sensory to motor cortex? Three regions along a visual→motor axis: **VISp** (primary visual),
**SCm** (motor superior colliculus), **MRN** (midbrain reticular nucleus), recorded
simultaneously.

---

## 1. The raw activity

**PSTH shapes differ by area** (`psth_per_area.png`, population rate ± 1 SEM across trials). In
this session: **SCm** rises from 16.9 → 28.6 sp/s and stays up; **MRN** 13.2 → 20.6 sp/s with a
sharp early transient near the median reaction time (**RT 171 ms**); **VISp** is weak and noisy
(2.6 → 7.3 sp/s, only 19 units).

![Population PSTH per area](../../results/figures/recap_5c0c560e/psth_per_area.png)

**A visible sequence of activation** (`macro_raster.png`, units peak-sorted, cividis =
peak-normalised rate). SCm and MRN show a clean activation diagonal; VISp's is weaker.

![Macro raster (this session)](../../results/figures/recap_5c0c560e/macro_raster.png)

**Decoder — stimulus vs choice** (`decoding_vs_contrast.png`). A *stimulus-side* decoder scales
with contrast (sensory); a *choice* decoder is high and roughly flat and is **above chance even
at 0 % contrast** (85 zero-contrast trials): the mouse commits to a turn on every trial. Choice
accuracy at 0 % / mean-over-contrast / stimulus at 100 %:

| area | choice @ 0 % contrast | choice (mean) | stimulus @ 100 % |
|---|---|---|---|
| VISp | 0.52 (≈ chance) | 0.58 | 0.72 |
| SCm | **0.83** | 0.89 | 0.98 |
| MRN | **0.73** | 0.88 | 1.00 |

![Decoding stimulus vs choice](../../results/figures/recap_5c0c560e/decoding_vs_contrast.png)

*(The single-neuron RT-sorted raster and the SCm↔MRN cross-correlogram from the pooled recap need
the raw session (~0.4 GB of spikes) and were not regenerated here under this machine's memory
limit; `python scripts/build_session_recap.py --eid 5c0c560e --only-one --no-gpfa` produces them.
The common-drive point they illustrate is carried quantitatively by the RRR result in §3.)*

---

## 2. The HMM — a sequence of states, and its directedness

Poisson HMM, fit per area on **raw 40 ms counts**, baseline-biased startprob. Two read-outs
(numbers from the frozen `results/hmm_5sessions/` and `results/hmm/state_phase.csv`, so they match
the pooled recap exactly — not re-fit).

**(a) Reproducible state count.** State-stability (ARI over disjoint fits) is high across all
three areas at the canonical K — **VISp 0.88 (K=3), SCm 0.92 (K=4), MRN 0.87 (K=3)** — i.e. the
states are reproducible. The honest common count is **K ≈ 3** (SCm supports 4 in this large
session). No state-count gradient.

![HMM state-stability scan](../../results/figures/recap_5c0c560e/hmm_stability.png)

**(b) Directedness (`forward_fraction`, split-half).** The headline result reproduces cleanly in
this one session:

| area | forward_fraction (split-half) | reading |
|---|---|---|
| **VISp** | **0.55 ± 0.01** | ≈ chance — undirected (still switches states) |
| **SCm** | **0.68 ± 0.04** | directed |
| **MRN** | **0.73 ± 0.07** | directed |

![Directedness (forward_fraction)](../../results/figures/recap_5c0c560e/hmm_forward_fraction.png)

> **Motor areas traverse their states directedly; visual cortex does not** — a clean
> visual→motor gradient within this session.

**(c) State ↔ task-phase (independent confirmation).** NMI of the decoded state with the trial
phase, referenced to a trial-permutation null that preserves the clock: all three beat the null
here (z = 47 / 37 / 25 for VISp / SCm / MRN), with SCm and MRN carrying much larger alignment
(Cramér's V ≈ 0.26 / 0.53 / 0.43) — the same motor-directed / visual-weak dissociation via a
second metric.

---

## 3. Dimensionality reduction — what drives each area's geometry

- **dPCA — the core prediction reproduces.** Demixed variance split by cause (balanced design,
  **K = 25 trials/cell**):

  | area | stimulus % | decision % | time % | **decision − stimulus** |
  |---|---|---|---|---|
  | VISp | 7.4 | 0.9 | 77.5 | **−6.5** (stimulus-dominated) |
  | SCm | 2.6 | 3.9 | 61.8 | +1.3 |
  | MRN | 5.0 | 9.0 | 49.1 | **+4.0** (decision-dominated) |

  A clean visual→motor gradient (−6.5 → +1.3 → +4.0), and the time (condition-independent) share
  falls 77.5 → 61.8 → 49.1 % along the axis — reproducing the pooled recap's headline dPCA result
  on this session. (The 0.4 s-pre pooled recap quoted MRN +6.4; here at the current 0.3 s-pre
  window it is +4.0 — same direction, same gradient.)

  ![dPCA variance by cause](../../results/figures/recap_5c0c560e/dpca_variance_split.png)

- **RRR / communication subspace — SCm and MRN share, they don't relay.** Cross-area held-out R²
  reaches each area's own within-area control, both directions:

  | prediction | best R² | sat. rank | | control | best R² | rank |
  |---|---|---|---|---|---|---|
  | SCm → MRN | 0.448 | 7 | | SCm self-pred. | 0.469 | 6 |
  | MRN → SCm | 0.443 | 7 | | MRN self-pred. | 0.442 | 7 |
  | VISp → SCm | 0.132 | 4 | | VISp self-pred. | 0.203 | 3 |

  **MRN predicts SCm (0.443) about as well as SCm predicts itself (0.469)** → no restricted
  channel between the motor areas: two readouts of a common signal. VISp→motor is the contrasting
  weak channel. (See also the dedicated mouse-vs-monkey RRR illustration,
  `results/figures/rrr_illustration/rrr_mouse_monkey.png`.)

  ![RRR communication subspace](../../results/figures/recap_5c0c560e/rrr_communication.png)

- **Joint manifold — each area's share of the shared latent.** Joint PCA of all 117 units; per-area
  partial projections (exact by linearity): **VISp 0.065, SCm 0.225, MRN 0.306**, summing to 0.596
  — the missing **~0.40 is cross-area covariance** (the shared signal). Same structure as the
  pooled recap.

*(GPFA timescales are omitted from this single-session pass by request; the pooled recap's fitted
τ = 107–283 ms stands.)*

---

## 4. The through-line (this session)

The pictures, the HMM and the geometry agree: the SCm/MRN raster diagonal ↔ their directed
`forward_fraction`; the choice-decodable-at-0 %-contrast decoder ↔ dPCA's decision-over-stimulus
gradient; RRR + joint-PCA ↔ common drive between the motor areas. **The one caveat qualifies all
of it:** median RT is **171 ms**, so over [0, +0.8] s the window is dominated by movement and
outcome — the HMM's late states, dPCA's "decision" variance and the motor directedness are
entangled with movement. *No state after ≈ 0 s is a pure "decision" state without the
`firstMovement`-aligned control.* In IBL the choice **is** the wheel-turn.

---

## 5. Status at a glance (single session `5c0c560e`)

| finding | this session |
|---|---|
| A reproducible temporal sequence of states (high ARI) | ✅ VISp 0.88 / SCm 0.92 / MRN 0.87 |
| Motor areas directed, visual undirected (`forward_fraction`) | ✅ 0.55 / 0.68 / 0.73 |
| State ↔ task-phase above the clock-null | ✅ all three (z 47 / 37 / 25) |
| Visual→motor dPCA gradient (decision − stimulus) | ✅ −6.5 → +1.3 → +4.0 |
| SCm↔MRN common drive, not a relay (RRR) | ✅ cross ≈ within control |
| Movement-window caveat (RT 171 ms) | ⚠️ applies |

**One-line synthesis.** *In this session the three areas hold a similar number of reproducible
states, but the motor areas traverse them more directedly and carry more decision variance — a
shared sequence on area-specific geometry, with SCm and MRN reading a common signal.* This is the
clean single-session case; contrast it with [`111c1762`](recap_data_HMM_dimred_111c1762.md), where
the directedness reproduces but the dPCA gradient does not.
