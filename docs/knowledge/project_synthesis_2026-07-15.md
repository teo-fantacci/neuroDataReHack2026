# Project synthesis — where everything stands (2026-07-15)

**Purpose.** One integrated picture, pulled from every parallel work-stream (HMM, rSLDS, dPCA /
GPFA / RRR / manifold, MSR, preprocessing, movement alignment, session survey), written for the
**cleanup toward a final version**. It says what may go in the talk (**certified**), what may not
(**retracted / artefact**), what is suggestive but not yet a claim (**lead**), and the decisions
still open — headed by **area/session selection**.

This is the executive layer. The deep methods/maths teaching doc is
[`where_we_are_HMM_rSLDS_dimred.md`](where_we_are_HMM_rSLDS_dimred.md) (written 2026-07-14) — still
correct on *how the methods work*, but its "known inconsistencies" and "biggest gap" sections are
**superseded here** by the 07-15 results (recurrence run, |λ| resolved, K/window fixed, 40 ms, all
5 sessions). When the two disagree on *status*, this file wins.

---

## 1. The scientific claim, and its status in one paragraph

> *Decision-making is a **temporal sequence of metastable/attractor states**, **shared across
> areas** but realised on **geometrically distinct, area-specific manifolds**; distributed temporal
> sequences within populations, not a clean anatomical relay.*

As of 2026-07-15: the **sequence** half is **certified** — there is a reliable, position-dependent
temporal sequence of states in all three regions, strongest in the motor areas, and it is *not* a
timed script and *not* a movement-timing artefact. The **attractor-geometry** half (that the states
are line attractors / integrators) is **not certified and currently inadmissible** — it was a
smoothing artefact, now retired; the honest replacement is "slow but *contracting* dynamics, motor
slower than visual." The **area-specific manifold** half has strong single-session support (dPCA
gradient, RRR common-drive) that **must be replicated across sessions** before it is claimed.

---

## 2. ✅ CERTIFIED — may go in the talk (with "one session, N sessions pending")

1. **It is an attractor *sequence*, not a stopwatch.** The recurrence test — held-out
   `LL(R free) − LL(R = 0)`, the only statistic separating a position-dependent sequence from a
   timed script — is **positive in every region, both alignments**, and passes a clean
   **position-shuffle null in 100 % of folds** (hold the model fixed, permute the latent positions
   feeding the transitions → the real positions always score higher). Ordered MRN ≈ SCm ≫ VISp.
   `results/rslds/recurrence_test.json`. *(2026-07-15 rSLDS worklog §2b.)*

2. **A reliable temporal sequence exists.** `order_consistency` beats its shuffle null in every
   region on **both** backends (exact-EM and Poisson `ssm`), large effect sizes: SCm/MRN ~0.93–0.99
   vs null ~0.57. Direction-agnostic (tests *sequence*, not *the* direction). *(rSLDS §2.)*

3. **The HMM's first "evoked" state is a MOVEMENT state.** Fitting the same HMM on stimOn- vs
   firstMovement-aligned tensors, the first state-exit jitter *drops* under movement alignment
   (SCm 117→**44 ms**, MRN 189→**153 ms**) vs ~350 ms expected if stimulus-locked. So **no state
   after ~0 may be called a "decision" state** — the first transition is timed by the wheel-turn.
   *(movement-alignment worklog; makes the 07-14 "movement window" warning quantitative.)*

4. **Motor areas traverse their states far more directedly than visual cortex.** HMM
   `forward_fraction`: VISp ≈ 0.53–0.59 (at chance — genuinely *undirected*, not static: it
   switches as often as MRN), SCm 0.77–0.83, MRN 0.67–0.79. VISp/SCm survive the split-half control;
   **MRN's drops to 0.67 ± 0.10 split-half**, so quote the split-half number for MRN. *(07-14 §6.)*

5. **`ssm` is usable, in an isolated env.** Two stacked install errors (build-time numpy; no C++
   compiler), both fixed with conda's mingw toolchain, **no admin**. Lives in `ndrh_ssm` (numpy 1.x)
   and is called out-of-process; never enters `neurodatarehack`. *(rSLDS 07-13.)*

---

## 3. 🔶 LEADS — method-robust and suggestive, but NOT yet claimable

- **The visual→motor gradient** (§5) — same ordering from **four** independent methods, but every
  number is one session. **Replicate before it is a figure.**
- **dPCA reproduces the core prediction**: decision−stimulus variance VISp −6.8 → SCm +1.8 →
  MRN +6.0, time-share falls 74→64→49 %. But decision components rest on **25 trials/cell** (10 with
  0 %-contrast excluded), and it is measured in the session with the **worst VISp recording**. *(dPCA
  worklog; five-sessions §1–2.)*
- **SCm↔MRN is common drive, not a feedback loop.** RRR finds **no restricted communication
  subspace** (MRN predicts SCm about as well as SCm predicts itself); the pynapple correlogram is
  **zero-lag**; the HMM finds both similarly directed. **Three independent methods** agree. A joint
  rSLDS must now *overturn* this, not confirm a loop. *(dPCA worklog §4; RRR.)*
- **The dynamics are slow but contracting** (GPFA τ = 107–283 ms, motor slowest), traversed as a
  reliable sequence. This is the honest dynamical statement that replaces "line attractor."

---

## 4. ❌ RETRACTED / ARTEFACT — must NOT be said

- **"We found the integrator / line attractor a decision variable lives on."** The `|λ| ≈ 1` that
  produced this is **>97 % reproduced on temporally-shuffled data** (`run_rslds_lambda_stability.py`)
  — it is manufactured by the **smoother** (and PCA cross-trial structure), not the neural dynamics.
  Both backends land at |λ| ≈ 1 (SCm/MRN even `> 1` = *unstable*, not integrating); `ssm` could not
  arbitrate because its Laplace posterior mean is *also* smooth. **|λ| is essentially not
  identifiable at these firing rates.** Flow labels from a pre-smoothed latent are not a dynamical
  read-out. *(rSLDS 07-15 §3, §6.)*
- **A state-count gradient (VISp 2 < MRN 3 < SCm 4).** With EM restarts the counts are **3 / 4 / 3**
  — VISp = MRN. Only **SCm K = 4** is a genuine interior ARI peak; MRN's K=3 beats the trivial split
  by 0.0017 (statistically empty); VISp's ARI *declines* from K=2 (no richer structure). *(07-14 §5;
  best_K_per_region.)*
- **"|λ| = 0.875 at 50 ms"** (old rSLDS docstring) — had no reproducible code behind it. Retracted.
- **The old bin-size / smoothing story.** The original validation swept sigma *in bins*, so the
  bin-size sweep was secretly a smoothing sweep; "50 ms is best" was partly its own smoothing.

---

## 5. The spine of the story: one axis, four methods, one retraction

| method | currency | VISp | SCm | MRN | status |
|---|---|---|---|---|---|
| **dPCA** | decision − stimulus variance | **−6.8** | +1.8 | **+6.0** | lead (1 session) |
| **HMM** | `forward_fraction` (split-half) | 0.53–0.59 (undirected) | 0.83 | 0.67 | certified |
| **rSLDS** | recurrence / position-shuffle gap | +0.006–0.008 (weak) | +0.037–0.041 | +0.036–0.050 | certified |
| **GPFA** | slowest fitted τ | 176 ms | 226 ms | **283 ms** | lead |
| ~~stability~~ | ~~#reproducible states~~ | ~~3~~ | ~~4~~ | ~~3~~ | **no gradient (retracted)** |

**Honest synthesis:** *the areas hold a similar number of states, but the motor areas traverse them
more directedly, carry more decision variance, and run slower.* — "shared alphabet, area-specific
sequence." VISp is the consistent weak/undirected end; **MRN the strongest** on recurrence and τ.

---

## 6. The method spine — decisions now frozen (don't re-litigate)

- **Bin = 40 ms, project-wide.** Passes the (rebuilt, reproducible) rSLDS ground truth — 0 spurious
  integrators, correct switch count, OC 1.0 — *and* preserves SCm's HMM stability (ARI 0.938 at
  40 ms vs 0.826 at 50 ms). 20 ms is banned (manufactures integrators). ⚠️ `PreprocessConfig` still
  **defaults to 0.05**; 40 ms is applied by build scripts overriding it — pass `--bin-ms 40`.
- **One preprocessing pipeline** (`src/data/preprocess.py`, frozen `.npz` with config serialised
  inside): stimOn **+ firstMovement** alignment, `pre_s = 0.4` (inside IBL's enforced quiescence
  floor), `post_s = 0.8`, units ≥ 1 Hz. **All 5 triplet sessions built at 40 ms, both alignments.**
- **K is chosen by state stability (ARI), not likelihood.** Likelihood failed twice (K=2 off −inf,
  K=10 off the range top). Final K: **VISp 3, SCm 4, MRN 3** (best_K_per_region; note VISp=2 in
  older notes is retracted). K is handed to the rSLDS as a *principled prior*, not a measurement
  (different model / timebase / units).
- **Two rSLDS backends by design.** Fast exact-EM (PCA latent) does the screen + recurrence test +
  K; Poisson `ssm` (joint latent, ~18–22 min/region) is the fit of record. `ssm`'s ELBO is a lower
  bound — never argmax it over K or D.
- **Trials, not neurons, are the limiting resource.** The rSLDS payload is estimated from regime
  *transitions* (2–4/trial); dPCA caps at 25 trials/cell; each ARI fit sees ~1/3 of trials. **Never
  prune units to "afford" a model** (MSR is *not adopted* for that — it is largely a rate filter,
  and beat neither a random nor a rate-matched half where the control was valid). To strengthen a
  claim, **pool the 5 sessions**.
- **Pooling means REPLICATE, not pseudo-population.** HMM/rSLDS: fit each session separately, pool
  the *statistics* ("K=4 in 5/5, ff 0.8 ± 0.04"). A pseudo-population destroys the simultaneity these
  sessions were selected for. dPCA: pseudo-population *is* valid (trial-averaged) and fixes the
  25-trials/cell problem.
- **The window is a MOVEMENT+REWARD window** (median RT 171 ms, feedback 427 ms). Every "decision"
  read-out must survive the firstMovement-aligned control. The confound is in the task, not the
  window (shortening doesn't help).
- **Anscombe does not stabilise variance here** (needs λ≳4; IBL is 0.05–0.83). Kept, but it is a
  second, independent argument that the Poisson `ssm` backend is principled and the PCA latent is a
  concession — and it explains why the 20 ms latent manufactured fake attractors.

**The recurring trap, extracted:** *never take an argmax without checking the winner is **interior**
and the criterion is **independent of the parameter being chosen**.* (Bit it three times: HMM K on a
boundary; rSLDS integrators from noise; smoothing-σ chosen by a metric the smoother inflates.)

---

## 7. Open decisions & next steps (ranked)

1. **▶ AREA / SESSION SELECTION — the gating decision (see §8). `config/areas.yaml::selected` still
   empty after 6 worklogs.**
2. **Pool the 5 sessions** for the recurrence/sequence replication and the dPCA gradient. The single
   highest-value analysis step; four strands independently point here. (All 5 are downloaded at
   40 ms.)
3. **Joint SCm+MRN rSLDS** — the direct exchange-vs-common-drive test; must overturn three
   converging pieces of common-drive evidence, not assume a loop.
4. **0 %-contrast trials as their own condition** — a *choice without a stimulus*. If the sequence
   still runs there, it is decision/motor-driven, not stimulus-driven — nearly a direct test of the
   claim, and free (~10 %/session). *(five-sessions §5.)*
5. Change `PreprocessConfig` default to 40 ms (shared decision — re-tags future `.npz`).
6. Do **not** keep chasing attractor geometry with a smoothed/inferred-smooth latent — §4 shows it
   is not identifiable at these rates.

---

## 8. ▶ THE PENDING AREA/SESSION DECISION — laid out so it can be made

**Region set is effectively settled: `{VISp, SCm, MRN}`** (the one visual→motor axis IBL co-records
in ≥5 sessions; VISpm dropped as a fake "higher-order" tier; a clean sensory→association→motor
triplet doesn't exist in IBL — that's the SHIELD argument). What remains is to (a) **write it into
`config/areas.yaml::selected`**, and (b) choose the **session strategy**, which is the real open
question because the five sessions trade off badly:

| eid8 | trials | VISp quality | balance (K visible) | best for |
|---|---:|---|---:|---|
| **5c0c560e** (current) | **843** | **worst** (VISp 1.1 Hz, present 30 % of session, contaminated) | worst (10) | trials → HMM/rSLDS; **cleanest SCm/MRN** |
| **90c61c38** | 656 | ok (20 units) | **best (33)** | clean balanced design; but only 76 units total |
| **111c1762** | 364 | **clean** (VISp 4.5 Hz) | 15 | a well-recorded VISp; but **worst movement confound** (RT 110 ms) |
| **90d1e82c** | 353 | good | 23 | clean all-round; weakest MRN (19 u) |
| **8a3a0197** | 290 | ok | 15 | **most units** (188); fewest trials |

The tension: **we chose `5c0c560e` for trial count (right for the HMM/rSLDS) but it is the *worst*
session for the two things the geometry/dPCA claim leans on — VISp recording and stimulus×choice
balance.** Both of our two fully-built sessions are also **left-biased**, so pooling compounds rather
than cancels the bias.

**Three coherent strategies to choose between:**
- **(A) Keep `5c0c560e` primary, pool all 5 for replication.** Simplest; leans on the certified
  sequence/recurrence result (which is robust) and treats dPCA as replicated-or-bust.
- **(B) Switch the primary/geometry session to a clean-VISp one** (`111c1762` or `90d1e82c`) and use
  `5c0c560e` for the trial-hungry dynamics only. Fixes the dPCA VISp confound; costs the movement
  confound (111c1762) or trials.
- **(C) Two designated roles, written down:** dynamics session (trials) = `5c0c560e`; geometry/dPCA
  pool = the sessions with clean VISp + better balance (possibly *excluding* `5c0c560e`, which
  *raises* dPCA K from 10 → 15 and adds VISp units). *(five-sessions §3 shows this counter-intuitive
  trade.)*

I'd lean **(C)** — it matches "trials for dynamics, balance/quality for geometry" to the data — but
this is your scientific call, and it's the one thing blocking a clean final version.

---

*Provenance: worklogs 2026-07-13 (rSLDS, dPCA/GPFA/RRR, MSR), 2026-07-14 (preprocessing), 2026-07-15
(rSLDS first real test, movement alignment, all-sessions-40ms); knowledge docs best_K_per_region,
the_five_triplet_sessions, ARI_and_state_stability, where_we_are_HMM_rSLDS_dimred.*
