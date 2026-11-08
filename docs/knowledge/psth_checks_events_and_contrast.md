# PSTH sanity checks — averaging, the MRN error band, event timing, and contrast

*Written 2026-07-17. Follow-up to the population PSTH in
[`recap_data_HMM_dimred.md`](recap_data_HMM_dimred.md) §1. Answers four questions about that
figure on session **8a3a0197** (UCLA034, 290 usable trials; VISp 23 / SCm 59 / MRN 53 units
≥ 1 Hz). Figures live under `results/figures/smoketest/` (gitignored; regenerated).*

---

## 1. Is the PSTH averaged over neurons *and* trials?

**Yes — over both, in that order.** For each area, per trial we take the **mean firing rate
over that area's units** (a single "population rate" time-course per trial), then average those
per-trial curves **over trials**. The shaded band is **±1 SEM across trials** =
`std_over_trials / √(n_trials)` of the per-trial population rate. So the line is a mean over
neurons and trials; the band measures **how reproducible that mean is from trial to trial**
(it is *not* a spread over neurons).

---

## 2. The thin MRN error band — effect, not bug

It is a **real effect**. The band uses the identical formula and the identical **n = 290
trials** in all three areas, so it cannot be a per-area computation error. Two genuine reasons
make MRN's band thin, plus a minor third:

| area | units | mean rate (sp/s) | mean ±1 SEM (sp/s) | **SEM / mean** | **trial-to-trial CV** |
|---|---:|---:|---:|---:|---:|
| VISp | 23 | 4.8 | 0.203 | **4.2 %** | **0.72** |
| SCm | 59 | 15.6 | 0.224 | 1.4 % | 0.25 |
| MRN | 53 | 12.7 | **0.149** | **1.2 %** | **0.20** |

1. **Scale.** MRN fires ~2.6× faster than VISp. Its ±1 SEM (0.15 sp/s) is actually the
   *smallest of the three in absolute terms*, and against a mean of ~12.7 sp/s it is only
   **1.2 %** — so on a y-axis spanning 12–16 it looks like a hairline. VISp's band is ~4.2 %
   of a small mean, so it looks fat.
2. **Reproducibility.** MRN's trial-to-trial coefficient of variation is **0.20** (SCm 0.25)
   versus VISp's **0.72**. MRN neurons fire much the same on every trial, so the trial average
   is tight; **VISp is genuinely noisy** — a thin, low-rate, 23-unit recording in this session.
3. **Unit count (minor).** More units average away more per-trial noise: MRN's SEM falls
   0.20 → 0.15 sp/s going from 23 to 53 units. At a *matched* 23 units MRN and VISp have nearly
   identical absolute SEM (~0.20), confirming the thin *appearance* is mostly scale +
   reproducibility, not the unit count.

**Bottom line:** MRN's tiny shade means "the MRN population does the same thing every trial,
and it fires fast"; VISp's fat shade means "VISp is noisy here." Both are honest features of
the data.

---

## 3. PSTH with the behavioural event each area tracks

To see *what* each response is locked to, below each stimulus-aligned PSTH is a raster where
**each dot is one trial's event time** (sorted): **reward (feedback) time under SCm**,
**movement onset under MRN** (and under VISp for reference). The grey dotted line on the PSTH
marks the median event time.

![PSTH per area with aligned event rasters](../../results/figures/smoketest/psth_with_event_rasters.png)

- **MRN's sharp transient sits exactly on movement onset** (peak ≈ median movement, 164 ms).
  Together with the earlier RT-sorted single-neuron rasters (which bend along the RT curve),
  this is the visual case that **MRN's evoked response is movement-locked**, not stimulus-locked.
- **SCm's sustained plateau spans the reward cloud.** Reward arrives late and spread out
  (median 530 ms; only 74 % within the 0.78 s window), and SCm stays elevated across that
  whole period — consistent with SCm tracking the movement-through-outcome epoch rather than a
  brief sensory transient.
- **VISp** has no clear evoked feature to lock to anything — the flat, noisy trace again.

**Honest caveat:** these rasters overlay the *timing distribution* of each event on the same
stimOn-aligned axis — they show **coincidence**, not proof of locking. The clean test is to
**re-align the PSTH to the event itself** (a reward-aligned SCm PSTH, a movement-aligned MRN
PSTH); the `firstMovement`-aligned tensors already exist for the movement half.

---

## 4. Is VISp noisy because low-contrast trials are included?

**No.** Restricting to **|contrast| = 100 %** (68 of 290 trials) does **not** clean VISp up.
Grey = all trials, colour = 100 % only:

![PSTH all trials vs 100% contrast only](../../results/figures/smoketest/psth_100pct_contrast.png)

- With only 68 trials the SEM band **widens** by ≈ √(290/68) ≈ **2.1×** — fewer trials is
  *noisier*, not cleaner — and VISp's trace stays just as ragged. **VISp's noise is the thin
  recording (23 units, low rate, high trial-to-trial CV), not low-contrast contamination.**
- High contrast does slightly **sharpen the evoked response in SCm and MRN** (MRN's transient
  peaks a touch higher, SCm's rise is crisper), which is the expected sensory effect — but it
  does not rescue VISp.

This is exactly why the project **replicates across the 5 triplet sessions** rather than
leaning on this one: 8a3a0197's VISp is the weak end, and no trial-subset fixes that — more
VISp units (a better-recorded session) is what's needed. Sessions `111c1762` and `90d1e82c`
have the clean VISp recordings for this.

---

*Reproduced by `scratchpad/psth_checks.py`: counts from the frozen tensor
`data/processed/8a3a0197__stimOn_pre400_post800_bin40_rate1.npz` (matches the notebook §7
figure exactly); event times pulled live from ONE's trials table (no spikes loaded).*
