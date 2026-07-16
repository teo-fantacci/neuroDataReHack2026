# 2026-07-15 — The first real rSLDS test: what it certified, and what it could not

The rSLDS finally fitted on real frozen data (40 ms, both alignments), with the recurrence test
run for the first time and the Poisson `ssm` fit of record as arbiter. New:
`scripts/{run_rslds_screen,run_rslds_ssm_record}.py`; results in `results/rslds/`.

**Short version:** the *sequence* and *recurrence* claims are certified and robust. The
*attractor geometry* claim is **not**, and the `ssm` fit that was meant to settle it instead
showed it cannot be settled from a single session of 30-bin trials. That second half is the more
important finding, and it is a good one.

---

## 1. The setup

- **Frozen 40 ms tensors**, both alignments (`stimOn`, `firstMovement`), units ≥ 1 Hz:
  VISp 19 / SCm 44 / MRN 54.
- **K per region 3 / 4 / 3** (HMM stability, `best_K_per_region.md`).
- **Two backends, on purpose.** Fast exact-EM (PCA latent, seconds) does the screen +
  cross-validation; Poisson `ssm` (joint latent, ~18–22 min/region) is the fit of record.

---

## 2. ✅ CERTIFIED: it is an attractor **sequence**, not a stopwatch

The recurrence test — held-out `LL(R free) − LL(R = 0)`, the *only* statistic that separates an
attractor sequence from a timed script — is **positive in every region, on both alignments**:

| region | delta_recurrent (stimOn) | delta_recurrent (firstMovement) |
|---|---:|---:|
| VISp | +0.0034 | +0.0039 |
| SCm | +0.0076 | +0.0129 |
| MRN | **+0.0247** | +0.0146 |

> **Switching depends on where the latent is, not on elapsed time.** And because it survives the
> `firstMovement` alignment, it is **not** a pure movement-timing artefact.

`order_consistency` beats its shuffle null everywhere, with large effect sizes, on **both**
backends (the `ssm` numbers are, if anything, higher):

| region | OC (exact-EM) | OC (ssm) | null | effect |
|---|---:|---:|---:|---:|
| VISp | 0.751 | 0.859 | ~0.57 | +0.18–0.29 |
| SCm | 0.930 | 0.979 | ~0.58 | +0.35–0.40 |
| MRN | 0.888 | 0.994 | ~0.55 | +0.33–0.44 |

**A reliable temporal sequence of metastable states exists.** That is the project's core
scientific claim, and it is now supported by two independent inference schemes and a shuffle
null. MRN carries the strongest recurrence and the tightest sequence; VISp the weakest but still
significant.

⚠️ ~~One honesty caveat on the recurrence magnitude...~~ **RESOLVED — see §2b.**

## 2b. The recurrence result, error-barred and null-tested (`run_rslds_recurrence.py`)

The earlier caveat (VISp's +0.003 came from only 3 folds) is closed. Repeated CV (5 folds × 3
repeats = 15 paired per-fold deltas) plus a **position-shuffle null** — hold the fitted R-free
model fixed, score held-out data with the latent positions feeding the transitions *permuted*,
which breaks the position→switch coupling while changing no parameter. `results/rslds/recurrence_test.json`:

| align | region | delta_recurrent | % folds > 0 | **real vs position-shuffle** | % folds real > shuffled |
|---|---|---:|---:|---:|---:|
| stimOn | VISp | +0.0057 ± 0.0078 | **100 %** | +0.008 | **100 %** |
| stimOn | SCm | +0.0157 ± 0.0080 | 93 % | +0.041 | **100 %** |
| stimOn | MRN | +0.0247 ± 0.0055 | **100 %** | +0.050 | **100 %** |
| firstMovement | VISp | +0.0047 ± 0.0031 | 93 % | +0.006 | **100 %** |
| firstMovement | SCm | +0.0102 ± 0.0085 | 87 % | +0.037 | **100 %** |
| firstMovement | MRN | +0.0146 ± 0.0027 | **100 %** | +0.036 | **100 %** |

Two things, and the second is the clean one:

1. **The paired delta is positive on essentially every fold** (VISp now 100 %, not the fragile
   3-fold +0.003). The Wilcoxon p's are all < 0.001 — but folds share training data, so those p's
   are **anticonservative and are not the evidence**; the fold-sign consistency is.
2. **The position-shuffle null is decisive and clean.** In **100 % of folds, in every region and
   both alignments**, the R-free model scores higher on held-out data with the *real* latent
   positions than with *shuffled* ones. Since this holds the model fixed and only breaks the
   position→switch link, it isolates recurrence from the "R is just extra parameters" confound.
   **Switching genuinely depends on where the latent is.**

The position-shuffle gap is **ordered MRN ≈ SCm ≫ VISp** (0.04–0.05 vs 0.008) — the same
visual→motor gradient as everywhere else. So VISp *has* recurrence (reliably positive), but it is
**weak**; SCm and MRN carry it strongly. And it survives the movement alignment, so it is not a
movement-timing artefact.

> **Certified, now with an error bar and a clean null: decision-making traverses a
> position-dependent sequence of states — an attractor sequence, not a timed script — in all
> three regions, strongest in the motor areas.** This is the project's core claim, and it is the
> part of the rSLDS that does *not* depend on the (artefactual) flow geometry.

---

## 3. ❌ NOT CERTIFIED: the attractor **geometry** — and the arbiter could not arbitrate

The exact-EM screen flagged SCm and MRN as **flow-SUSPECT**: 2 "line attractor / integrator"
regimes each, `|λ| ≈ 0.99–1.00`, above the ground-truth-admissible band (0.88–0.90). VISp was
clean (`|λ| = 0.925`, point/rotational attractors). The whole point of the `ssm` fit was to
resolve this — a Poisson joint latent does not inherit the PCA latent's noise, so it *should*
either certify the slow modes or show clean contracting attractors instead.

**It did neither. It landed in the same danger zone, or worse:**

| region | exact-EM \|λ\| | **ssm \|λ\|** | ssm converged? | implied τ (ssm) |
|---|---:|---:|:--|---|
| VISp | 0.925 (clean) | **1.012** | ❌ **no** (drift 1.9e-3) | unstable — *discard* |
| SCm | 1.000 | **1.032** | ✅ yes | **> 1 → unstable, not a line attractor** |
| MRN | 0.985 | **0.998** | ✅ yes | τ ≈ 20 s (≫ 1.2 s trial) → integrator |

**Why this is NOT "confirmed real slow mode"** (my script's first auto-verdict said that; it is
wrong and has been corrected):

1. **The ground truth calibrates what real looks like, and this isn't it.** At 40 ms a *genuine*
   150 ms attractor gives `|λ| = 0.88` cleanly — even at VISp's 19-unit count. Shot noise / model
   mismatch gives `|λ| → 1`. Real data lands at ~1.0, i.e. on the **failure** side of the
   calibration, not the clean-attractor side.
2. **`|λ| > 1` is not a line attractor — it is an *unstable* (expanding) mode.** SCm's 1.032 and
   VISp's 1.012 are the signature of a linear-regime model overfitting a bounded trajectory, not
   of integration.
3. **VISp's `ssm` fit did not converge** (tail ELBO drift 1.9e-3 > 1e-3). Its flow reading is
   discarded outright.
4. **Agreement between two estimators that are both in their failure regime is not evidence.**
   That is the trap the admissibility gate exists to catch, and it caught it.

**But it is not simply shot noise either.** SCm and MRN *converged*, and the exact-EM backend was
clean on synthetic VISp-like data at 40 ms. So real motor dynamics are **genuinely very slow /
near-integrator** — which agrees with GPFA's independent fitted timescales (VISp ≤176 ms,
SCm 226 ms, **MRN 283 ms**, motor slowest). The honest statement:

> **Real SCm/MRN dynamics are slow enough that, on 30-bin single-session trials, a true line
> attractor cannot be distinguished from slow non-stationarity or model mismatch.** The
> classification "line attractor / integrator" is *plausible and consistent with GPFA*, but it is
> **not certifiable** from this fit, and `|λ| > 1` in two regions means it is currently
> inadmissible. **No line-attractor / integrator claim may be made yet.**

---

## 4. The gradient that survives everything

One thing is consistent across **three independent methods**: the motor areas run slower / more
integrator-like than visual cortex.

| method | VISp | SCm | MRN |
|---|---|---|---|
| rSLDS \|λ\| (both backends) | 0.93 (clean attractors) | ~1.0 | ~1.0 |
| GPFA slowest τ | 176 ms | 226 ms | 283 ms |
| HMM forward_fraction (split-half) | 0.59 | 0.83 | 0.67 |

VISp is a genuine *contracting attractor* region; SCm/MRN sit at the integrator boundary. That is
suggestive and method-robust — but per §3 it is a **lead**, not a certified attractor claim.

---

## 5. What the "test" actually taught us

The test answered the question it *could* answer and correctly refused the one it could not:

- **Answered YES:** is there a position-dependent, reliable temporal sequence? (recurrence + OC,
  both backends, both alignments, vs null.)
- **Refused:** what is the attractor geometry? Both backends land at `|λ| ≈ 1`; the arbiter (`ssm`)
  did not break the tie because a single session of 30-bin trials is **insufficient to certify
  flow geometry** — exactly the "trials, not neurons" limit, arriving from yet another direction.

This is the right outcome. A year of project-time nearly rested "we found the integrator a
decision variable lives on" on a 20 ms fit; the discipline built since (reproducible ground truth,
the admissibility gate, two backends) caught it *twice more* here, on 40 ms data and on the
published Poisson method.

---

## 6. ✅ RESOLVED: the `|λ| ≈ 1` is the SMOOTHER, not the dynamics

`scripts/run_rslds_lambda_stability.py` → `results/rslds/lambda_stability.json`. Three tests on the
fast backend (8 seeds each), and they agree in all three regions:

| region | \|λ\| real | seed spread | \|λ\| **shuffle-null** | gap | \|λ\| at 25→100% trials |
|---|---:|---:|---:|---:|---|
| VISp | 0.925 | **0.000** | 0.915 | **+0.011** | 0.939 → 0.925 (flat) |
| SCm | 1.000 | **0.000** | 0.992 | **+0.008** | 1.004 → 1.000 (flat) |
| MRN | 0.985 | **0.000** | 0.951 | **+0.034** | 0.997 → 0.985 (flat) |

1. **Seed stability = perfect (spread 0.000).** So `|λ|` is *not* EM local-optimum noise — it is a
   deterministic property of the pipeline. Good, but it rules *in* both "real" and "artefact".
2. **Flat with data.** `|λ|` does not fall as trials grow (25% → 100% moves it < 0.015). So it is
   *not* small-sample autoregressive bias either.
3. **⚠️ THE DECISIVE ONE — the shuffle null.** Bin-shuffle within each trial (destroy the temporal
   order, keep the rate content) and refit: `|λ|` barely moves (gap **+0.01 to +0.03**). **If the
   near-unit-root reflected real within-trial dynamics, destroying the order should collapse it. It
   does not.**

> **Conclusion: the `|λ| ≈ 1` — and therefore the "line attractor / integrator" classification —
> is >97% reproduced on temporally-scrambled data. It is manufactured by the smoothing (and by the
> PCA latent's cross-trial structure), NOT by the neural dynamics. The flow classification from a
> pre-smoothed Gaussian latent is not a dynamical read-out.**

This is stronger than "uncertifiable": we now *know* the integrator labels are an artefact of the
latent construction, not a property of the brain. It also explains the `ssm` result by analogy —
`ssm`'s Laplace posterior mean is *also* smooth (the dynamics prior pulls it smooth), so it reaches
`|λ| ≈ 1` by the same route, which is why the "arbiter" could not arbitrate. **At these firing
rates, every route to a single-trial latent imposes smoothness, and the AR fit reads that smoothness
back as a slow mode. `|λ|` is essentially not identifiable here.**

The small residual gap (MRN +0.034, the largest) is the *only* part of `|λ|` that reflects real
temporal structure — and MRN having the largest gap is consistent with it being the slowest region
(strongest recurrence, longest GPFA τ). But it is tiny, and no attractor-geometry claim rests on it.

### The constructive path: GPFA already gave the honest timescale

GPFA **infers** the latent timescale instead of imposing a kernel, and it fitted **finite** τ =
107–283 ms (motor slowest), *not* integrators. Since our smoothing (σ = 40 ms) is well below those
τ, GPFA's timescale is not a smoother artefact. So the trustworthy dynamical statement is:

> **The dynamics are slow but *contracting* (finite τ, motor slower than visual), traversed as a
> reliable position-dependent sequence. They are NOT line attractors / integrators — that reading
> was a smoothing artefact, now retired.**

`RSLDSResult.classify()` / `.admissibility()` remain useful as *guards* (they correctly flag the
danger zone), but the **flow labels themselves must not be reported** from this backend on this
data. The rSLDS's certified contribution here is the **sequence + recurrence** (which use regime
*order* and position-dependent *switching*, not the flow matrix `A`), not the geometry.

---

## 7. Open — what is left

1. ✅ ~~Report per-fold `delta_recurrent`~~ **DONE (§2b).** Recurrence is reliably positive on
   ~every fold and passes a clean position-shuffle null in 100 % of folds, both alignments.
2. **Pool the 5 replicate sessions** (`TRIPLET_EIDS` populated) — for the recurrence/sequence claim,
   as replication (5/5 sessions), not to rescue the geometry (which the smoothing artefact caps).
3. **The joint SCm+MRN rSLDS** — still the direct test of exchange-vs-common-drive; must overturn
   three converging pieces of common-drive evidence.
4. **Do NOT keep chasing attractor geometry with a smoothed/inferred-smooth latent.** §6 shows it is
   not identifiable at these rates. If the geometry is wanted, it needs a fundamentally different
   handle (e.g. jointly modelling many sessions, or a non-smoothing latent), not more fits.

`ssm` fits took 18–22 min each (VISp non-converged at 50 iters → raise SSM_ITERS for VISp *if* the
geometry is ever revisited — but §6 says it should not be, on this data).
Committed `results/rslds/{screen_40ms,ssm_record_40ms,lambda_stability}.json`.
