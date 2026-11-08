# Recap (single session) — from raw activity to HMM and geometry · session `111c1762`

*Written 2026-07-17. A **single-session** version of
[`recap_data_HMM_dimred.md`](recap_data_HMM_dimred.md), for IBL session
**`111c1762-7908-47e0-9f40-2f2ee55b6505`** (UCLA033, churchlandlab). This is the second-best
triplet recording and the one with the **cleanest VISp** (364 usable trials, VISp 30 / SCm 49 /
MRN 47 units ≥ 1 Hz — the best-balanced full population of the five). Everything is computed on
**this session alone** — no pooling. Companion to the doc for the working session
[`5c0c560e`](recap_data_HMM_dimred_5c0c560e.md). Numbers from
`results/session_recap/111c1762.json`; figures from `results/figures/recap_111c1762/`
(gitignored — regenerate with `python scripts/build_session_recap.py --eid 111c1762`).*

> **Read this one for where a single session diverges.** `111c1762` **reproduces** the HMM
> directedness result — and even sharpens the SCm↔MRN common-drive result — but it **does not**
> reproduce the dPCA visual→motor gradient (its decision variance is high in *every* area,
> including VISp). That divergence, on only 364 trials, is exactly why the pooled recap flags dPCA
> as "a strong lead, not yet a claim, needs the sessions pooled." A clean recording of a fragile
> statistic can still disagree with the average.

---

## 1. The raw activity

**PSTH shapes** (`psth_per_area.png`, population rate ± 1 SEM across trials): SCm 10.9 → 18.9 sp/s
(sustained), MRN 15.1 → 21.4 sp/s (sharp early transient near the median RT), and here **VISp is
better recorded** — 30 units, 5.2 → 10.1 sp/s — though still the weakest of the three. Median
reaction time in this session is **110 ms** (faster than 5c0c560e's 171 ms).

![Population PSTH per area](../../results/figures/recap_111c1762/psth_per_area.png)

**Macro raster** (`macro_raster.png`, units peak-sorted): SCm and MRN show the activation diagonal.

![Macro raster (this session)](../../results/figures/recap_111c1762/macro_raster.png)

**Decoder — stimulus vs choice** (`decoding_vs_contrast.png`). Choice is strongly decodable in the
motor areas across contrasts; the stimulus decoder scales with contrast.

| area | choice (mean over contrast) | stimulus @ 100 % |
|---|---|---|
| VISp | 0.78 | 0.93 |
| SCm | **0.93** | 0.76 |
| MRN | **0.94** | 0.96 |

*(This session has only 40 zero-contrast trials and a choice bias at 0 %, so the single 0 %-contrast
choice point was too imbalanced to cross-validate stably — the mean-over-contrast column is the
robust summary. The pooled recap's "choice decodable at 0 % contrast" is cleaner in
[`5c0c560e`](recap_data_HMM_dimred_5c0c560e.md), which has 85 zero-contrast trials.)*

![Decoding stimulus vs choice](../../results/figures/recap_111c1762/decoding_vs_contrast.png)

*(RT-sorted single-neuron raster and SCm↔MRN cross-correlogram need the raw session (~0.4 GB) and
were not regenerated under this machine's memory limit;
`python scripts/build_session_recap.py --eid 111c1762 --only-one --no-gpfa` produces them.)*

---

## 2. The HMM — a sequence of states, and its directedness

Poisson HMM per area on raw 40 ms counts (numbers from the frozen `results/hmm_5sessions/` and
`results/hmm/state_phase.csv`).

**(a) Reproducible state count.** State-stability (ARI) is strong in the motor areas but weak in
VISp here — **VISp 0.50 (K=3), SCm 0.78 (K=4), MRN 0.71 (K=3)**. VISp's ARI curve peaks at the
trivial K=2 and does not hold a clean interior K; even with 30 units, VISp's *within-trial state
structure* is weak in this session (its single-trial choice is still decodable — §1 — but the
temporal trajectory is not stereotyped). SCm/MRN carry reproducible structure at K ≈ 3–4.

![HMM state-stability scan](../../results/figures/recap_111c1762/hmm_stability.png)

**(b) Directedness (`forward_fraction`, split-half) — reproduces cleanly:**

| area | forward_fraction (split-half) | reading |
|---|---|---|
| **VISp** | **0.50 ± 0.01** | exactly at chance — undirected |
| **SCm** | **0.72 ± 0.02** | directed |
| **MRN** | **0.66 ± 0.01** | directed |

![Directedness (forward_fraction)](../../results/figures/recap_111c1762/hmm_forward_fraction.png)

> The visual→motor **directedness** gradient is if anything *cleaner* here than in the working
> session — VISp sits exactly on chance (0.50) with the tightest error bar, SCm/MRN clearly
> directed. This is the project's certified result, holding on an independent animal.

**(c) State ↔ task-phase.** Against the trial-permutation (clock-preserving) null: **SCm beats it
strongly (z = 7.3, Cramér's V 0.48), MRN modestly (z = 2.4, V 0.25), VISp does not (z = 0.7,
not significant)** — the same motor-directed / visual-undirected dissociation as `forward_fraction`,
via a second metric, and here VISp fails the null despite its good unit count.

---

## 3. Dimensionality reduction — where this session diverges

- **dPCA — the visual→motor gradient does NOT reproduce here.** Demixed variance (balanced design,
  only **K = 16 trials/cell** — half the working session's, because 364 trials leave few error
  trials for the off-diagonal cells):

  | area | stimulus % | decision % | time % | **decision − stimulus** |
  |---|---|---|---|---|
  | VISp | 1.8 | 8.1 | 67.3 | **+6.3** |
  | SCm | 0.4 | 7.1 | 53.7 | **+6.7** |
  | MRN | 2.2 | 6.5 | 49.8 | **+4.3** |

  Decision variance exceeds stimulus variance in **every** area, including VISp — there is **no
  monotone visual→motor gradient** in this session, unlike `5c0c560e` (−6.5 → +1.3 → +4.0). With
  K = 16 trials/cell the decision components are noisy (the module warns of exactly this), and the
  window is movement-entangled (RT 110 ms), so VISp's apparent "decision" variance is not
  trustworthy on its own. **This is the honest reason dPCA is a pooled claim, not a single-session
  one.** (The time/condition-independent share *does* still fall along the axis, 67 → 54 → 50 %.)

  ![dPCA variance by cause](../../results/figures/recap_111c1762/dpca_variance_split.png)

- **RRR / communication subspace — common drive, even more starkly.** Cross-area held-out R²
  **reaches or exceeds** each motor area's own within-area control:

  | prediction | best R² | sat. rank | | control | best R² | rank |
  |---|---|---|---|---|---|---|
  | MRN → SCm | 0.420 | 7 | | SCm self-pred. | 0.349 | 6 |
  | SCm → MRN | 0.392 | 8 | | MRN self-pred. | 0.369 | 8 |
  | VISp → SCm | 0.152 | 5 | | VISp self-pred. | 0.162 | 5 |

  Here **MRN predicts SCm (0.420) *better* than SCm predicts itself (0.349)** — the strongest form
  of the "no restricted channel between the motor areas" result. VISp→SCm stays weak.

  ![RRR communication subspace](../../results/figures/recap_111c1762/rrr_communication.png)

- **Joint manifold — area shares.** Joint PCA of all 126 units; per-area partial projections:
  **VISp 0.096, SCm 0.255, MRN 0.222**, sum 0.573 → **~0.43 cross-area covariance**. Same
  structure as the pooled recap and `5c0c560e`.

*(GPFA omitted by request.)*

---

## 4. The through-line (this session)

Two strands agree with the project claim (HMM directedness and RRR common drive, both **stronger**
here), and one diverges (dPCA shows no clean gradient). The divergence is informative, not a
failure: it is a low-trial, movement-entangled estimate of a variance split that the pooled recap
already marks as fragile. The **movement-window caveat is sharper here** — median RT **110 ms**, so
almost the entire post-stimulus window is movement/outcome.

---

## 5. Status at a glance (single session `111c1762`)

| finding | this session |
|---|---|
| Reproducible state sequence (ARI) | ✅ SCm 0.78 / MRN 0.71; ⚠️ VISp weak (0.50) |
| Motor directed, visual undirected (`forward_fraction`) | ✅ 0.50 / 0.72 / 0.66 (cleanest VISp = chance) |
| State ↔ task-phase above the clock-null | ✅ SCm/MRN; ❌ VISp (n.s.) |
| Visual→motor dPCA gradient (decision − stimulus) | ❌ **does not reproduce** (all decision-dominated, K=16/cell) |
| SCm↔MRN common drive, not a relay (RRR) | ✅✅ cross ≥ within control |
| Movement-window caveat (RT 110 ms) | ⚠️ applies, sharply |

**One-line synthesis.** *This independent animal reproduces — and sharpens — the certified results
(motor areas directed, visual undirected; SCm and MRN read a common signal), while its dPCA
variance split disagrees with the pooled gradient, which is the clearest single-session
demonstration of why that gradient must be pooled before it is claimed.* Read alongside the clean
case, [`5c0c560e`](recap_data_HMM_dimred_5c0c560e.md).
