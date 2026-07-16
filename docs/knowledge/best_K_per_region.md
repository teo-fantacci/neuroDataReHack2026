# Best K per region — the numbers to hand to the rSLDS

**Status: FINAL for session `5c0c560e`** (2026-07-14). All **843 usable trials**, 20 ms bins,
units filtered at `MIN_RATE_HZ = 1.0`, **5 repeats × 2 EM restarts**.

Method: [`ARI_and_state_stability.md`](ARI_and_state_stability.md).
Why not likelihood: [`choosing_K_for_the_HMM.md`](choosing_K_for_the_HMM.md).
Runner: `scripts/run_state_stability.py` → `results/hmm/state_stability_maxtrials.json`,
figure `results/figures/phase2_cv/state_stability_final.png`.

---

## 1. The recommendation

| region | units | **K for the rSLDS** | ARI at that K | note |
|---|---:|---:|---|---|
| **VISp** | 20 | **3** | 0.883 ± 0.019 | tight |
| **SCm** | 44 | **4** | **0.920 ± 0.031** | the most reproducible structure in the dataset |
| **MRN** | 55 | **3** | 0.882 ± 0.052 | its non-trivial peak *beats* the trivial split |

## 2. ⚠️ Do NOT read "peak K = 2" off the raw curve

The raw ARI peaks at **K = 2** for VISp and SCm (0.976, 0.977). **That is a trap, not a
result** — the failure documented in §8(a) of the method note: *stability is not correctness;
a model can be stably wrong.*

A K=2 model only has to find the **baseline-vs-evoked rate step**: one huge, obvious event
that *both* independent fits recover every time. A near-perfect ARI there is close to
guaranteed and says nothing about metastable dynamics. Taking it at face value would repeat
the very mistake this analysis exists to kill — the likelihood argmax gave **K=2** off a
column of `-inf`, then **K=10** off the top of the range. Two opposite non-answers, both
boundaries.

**The question is not "is the ARI high" but "how far up in K does it STAY high".** So the
table reports the best **non-trivial** (K ≥ 3) peak. (Note **MRN's raw peak already IS K=3** —
its three-state structure edges out the trivial split on its own.)

## 3. ⚠️ Correction: the preliminary run was WRONG about VISp

The first pass (`n_repeats=2`, **`n_restarts=1`**) reported *"VISp: nothing reproduces beyond
K=2; best non-trivial K=5 at ARI 0.680 ± 0.176 — a negative result."* **That was wrong.**

With EM restarts, **VISp K=3 gives ARI 0.883 ± 0.019** — tight and clearly reproducible.

This is precisely the bias predicted in §8(c) of the method note: EM is non-convex, so
without restarts two fits disagree from bad **luck** rather than different **data**, and the
metric charges that to *"the states are not real"*. It biases ARI — and therefore K —
**downward**. The prediction was made *before* the run, and the run confirmed it. The other
two regions were unaffected:

| region | preliminary (1 restart) | final (2 restarts) | verdict |
|---|---|---|---|
| VISp | K=5, ARI 0.680 ± 0.176 | **K=3, ARI 0.883 ± 0.019** | **CHANGED** — restarts fixed it |
| SCm | K=4, ARI 0.928 ± 0.003 | K=4, ARI 0.920 ± 0.031 | CONFIRMED |
| MRN | K=3, ARI 0.870 ± 0.044 | K=3, ARI 0.882 ± 0.052 | CONFIRMED |

**Lesson worth keeping: never run a stability analysis with a single EM fit per model.**

## 4. ⚠️ Correction: the "state-count gradient" does NOT hold

On the preliminary numbers I claimed a gradient — *VISp (2) < MRN (3) < SCm (4)*, "motor
midbrain carries richer state structure than visual cortex" — and said three independent
methods agreed on it. **The state-count part of that claim is retracted.**

The final counts are **VISp 3, MRN 3, SCm 4**. VISp and MRN are equal. There is **no clean
visual→motor gradient in the number of reproducible states**; only SCm stands slightly apart,
and 4-vs-3 is not much to hang a story on.

What *does* still hold — and is measured independently of state count:

| method | VISp | SCm | MRN |
|---|---|---|---|
| **number of reproducible states** (this doc) | 3 | 4 | 3 | ← **no gradient** |
| `forward_fraction` (HMM directedness) | 0.59 | 0.83 | 0.79 |
| **`forward_fraction`, split-half ordering** ← **quote this one** | **0.59** | **0.83** | ⚠️ **0.67 ± 0.10** |
| `switch_rate` (state changes / trial) | 1.59 | 2.22 | 1.46 |
| `baseline_exit_latency` | ~90 ms | ~170 ms | ~450 ms |
| dPCA decision−stimulus variance | −6.8 | +1.8 | +6.0 |

*(Updated 2026-07-14: measured at the **final** K = 3/4/3, on the unified preprocessing
(`pre_s` 0.4 s, 50 ms bins, units ≥ 1 Hz) — `scripts/run_binsize_check.py`. The values are
essentially **flat across 20–50 ms bins**, so none of this is a bin-size artefact. The earlier
0.53 / 0.77–0.80 / 0.69–0.80 came from fits at K=2 and K=10, neither of which is the chosen K.)*

So the honest statement is: **the areas hold a similar number of states, but the motor areas
traverse them far more directedly.** That is arguably a *better* fit to the project's framing
("shared temporal sequence, area-specific geometry") than a state-count gradient would have been
— the *sequence* differs, not the alphabet.

**Three things now qualify that, all new on 2026-07-14:**

1. ⚠️ **MRN's directedness is partly circular.** `forward_fraction` needs a state ordering, and we
   take it from occupancy centre-of-mass on *the same trials* whose transitions we then score.
   With the ordering derived on **held-out** trials, VISp and SCm are unchanged (0.59, 0.83) but
   **MRN falls from 0.79 to 0.67 ± 0.10**. Quote the split-half number for MRN.
2. **VISp is UNDIRECTED, not static.** It switches state as often as MRN (1.59 vs 1.46
   changes/trial) — it simply switches with no preferred direction. `forward_fraction` is an
   *asymmetry*, never a *frequency*; the two dissociate cleanly here.
3. ⚠️ **The latency cascade may be the task, not the brain.** VISp ~90 ms → SCm ~170 ms →
   MRN ~450 ms is the predicted visual→motor ordering — but **SCm's exit sits on the median
   reaction time (171 ms) and MRN's on the median feedback time (427 ms)**. Each area may simply
   be locking to the event it is already known to lock to. The movement-aligned control
   (`data/processed/*firstMovement*`) is what separates these, and it has not been run.

## 5. The likelihood elbow is now definitively useless here

At full trials the elbow gives **VISp 7, SCm 10, MRN 10** — SCm and MRN pinned to the top of
the search range *again*. Compare the stability answer (3, 4, 3). **Prefer stability.** The
elbow was only ever a patch on a criterion that does not work on spike counts (see
`choosing_K_for_the_HMM.md` §2).

## 6. Remaining caveats

1. **One session.** Repeat on the 5 replicate `{VISp, SCm, MRN}` sessions before a talk.
2. **Trials, not neurons, are the limit** (same conclusion the rSLDS work reached). Each fit
   sees ~1/3 of 843 trials ≈ 281. If a peak moves right when trials grow, we are data-limited.
3. SCm's K=3 is *unstable* (ARI 0.721 ± 0.284) while K=4 is tight — the classic signature of
   EM arbitrarily splitting one real state at K=3. Trust K=4, not the K=3 dip.

## 7. For the rSLDS chat, concretely

**VISp K=3 · SCm K=4 · MRN K=3.**

* **SCm K=4 is the anchor** — the most reproducible structure found by any method here.
* **Do not expect a state-count difference between VISp and MRN**; they are both 3. If an
  rSLDS reports a large difference in the number of well-separated regimes between them, be
  suspicious — an independent, chance-corrected criterion says they have the same number.
* Expect the areas to differ in **how they traverse** the states (directedness), not in how
  many they have.
