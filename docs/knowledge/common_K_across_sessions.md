# Is there a common HMM state count across sessions? Yes — K = 3

Generated 2026-07-15 by `scripts/run_hmm_kselect.py` (per-session stability, then a maximin
common-K rule), all 5 triplet sessions, `stimOn`, 40 ms, units ≥ 1 Hz, 3 repeats × 2 EM restarts.
Raw: `results/hmm/k_selection_stimOn_bin40.json`, figure
`results/figures/phase2_hmm/kselect_stimOn_bin40.png`.

Supersedes the single-session `best_K_per_region.md` (5c0c560e only) as the K decision for the
**replication**: there K was chosen on one session and imposed on the rest, which can neither
discover a common K nor explain a divergence. This runs the stability analysis **independently
on each session** and compares.

---

## 1. Per-session best (non-trivial) K

| session | usable trials | VISp | SCm | MRN |
|---|---:|---:|---:|---:|
| 5c0c560e | 843 | 3 | **4** | 3 |
| 90c61c38 | 656 | 3 | 3 | 3 |
| 111c1762 | 364 | **5** | 3 | **6** |
| 90d1e82c | 353 | 3 | 3 | 3 |
| 8a3a0197 | 290 | 3 | 3 | 3 |

**K = 3 is the mode in every region** (VISp 4/5, SCm 4/5, MRN 4/5).

## 2. The two departures are both explained — neither is biology

- **`111c1762`: VISp = 5, MRN = 6.** The *only* session giving high K, and it does so with **low
  ARI** (0.735, 0.706) against everyone else's 0.8–0.95 at K = 3. Its ARI-vs-K curves are flat and
  noisy (see the figure), so the argmax wanders. It is the fewest-trial 2-probe session (364).
  **This is a low-power argmax artifact, i.e. the standing "trials, not neurons" limit** — not a
  real extra state count. At the common K = 3 its ARI is still 0.69–0.70, only ~0.04 below its
  own (noisy) peak.
- **`5c0c560e`: SCm = 4.** A genuine interior peak in the best-powered session (843 trials) — but
  **K = 3 is within noise even here** (ARI 0.929 at K=3 vs 0.931 at K=4, against fold SDs ~0.03).

## 3. The maximin verdict, and where the automated rule needs a human

`consensus_for_region` picks the K ≥ 3 that maximises the **minimum** ARI across sessions (good
for the *worst* session), then checks it is within one fold-SD of every session's own peak:

| region | maximin says | reading |
|---|---|---|
| SCm | **K = 3, defensible** | correct — all 5 within noise at K = 3 |
| MRN | **K = 3, defensible** | correct — all 5 within noise at K = 3 |
| VISp | K = 4, "weak / diverge" | ⚠️ **rule artifact** — see below |

**The VISp flag is the rule being dragged by the outlier.** `111c1762` prefers K = 4–5, so the
"minimum across sessions" is highest around K = 4 — but K = 4 is far from `5c0c560e`'s K = 3 peak
(0.723 vs 0.898). Reading the actual per-session bests, **VISp is also K = 3** (4/5 sessions, the
mode), and the disagreement is entirely `111c1762`'s noise. *Lesson: a maximin over sessions is
dominated by the least-reliable session; report it alongside the modal K and the trial counts,
never alone.*

## 4. Decision

> **Common state count K = 3 for VISp, SCm, MRN, across all 5 sessions.** Defensible: every
> session is within fold-noise of K = 3 in every region. The apparent divergence is a function of
> **trial count**, not area or animal — it appears only in the lowest-power session and carries
> low reproducibility where it appears.
>
> **SCm supports K = 4 as an equally-good refinement in the highest-trial session** (5c0c560e);
> quote K = 3 for the replication and note K = 4 for SCm's single best-powered fit.

The replication (`run_hmm_sessions.py --common-k 3`) therefore fits **K = 3 everywhere**. Because
`forward_fraction`/`switch_rate` are flat across K (2026-07-14 §6.1), the directedness result does
not depend on the 3-vs-4 choice.

## 5. Caveats

1. **`stimOn` alignment only so far.** Re-run at `--align firstMovement` — the movement-aligned K
   could differ, and *that difference would itself be informative* (2026-07-15 movement result).
2. Screen setting (3 repeats, K ≤ 6). The K = 3 conclusion is robust; if a definitive number is
   needed for a figure, re-run `run_hmm_kselect.py --repeats 5 --kmax 8`.
3. The automated consensus rule (§3) should be read with the per-session table and trial counts,
   not taken as an oracle.
