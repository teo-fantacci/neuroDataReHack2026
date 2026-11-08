# Project recap and data

**Written 2026-07-16.** A single, self-contained snapshot of the whole project: the data we
work on, where each analysis strand stands, what we **tried and abandoned** and **why we chose
what we chose**, and the decisions still open. It consolidates the deeper docs — read those for
the maths and the full argument:

- [`knowledge/project_synthesis_2026-07-15.md`](knowledge/project_synthesis_2026-07-15.md) — the executive "certified / lead / retracted" layer.
- [`knowledge/where_we_are_HMM_rSLDS_dimred.md`](knowledge/where_we_are_HMM_rSLDS_dimred.md) — how the three methods work and interact (with concept checklists).
- [`knowledge/project_explained.md`](knowledge/project_explained.md) — the pedagogical Q&A (preprocessing + HMM in full).
- [`worklog/`](worklog/) — the dated running record, newest first.

Status labels used throughout: **✅ certified** (defensible, one session, replication pending) ·
**🔶 lead** (method-robust and suggestive, not yet claimable) · **❌ retracted** (was on the
table, now known to be an artefact).

---

## 0. The scientific claim, in one paragraph

> *Decision-making is a **temporal sequence of metastable/attractor states**, **shared across
> areas** but realised on **geometrically distinct, area-specific manifolds** — distributed
> temporal sequences within populations, not a clean anatomical relay.*

As of today: the **sequence** half is **certified** — there is a reliable, position-dependent
temporal sequence of states in all three regions, strongest in the motor areas, and it is *not* a
timed script and *not* a movement-timing artefact. The **attractor-geometry** half (that the
states are line attractors / integrators) is **retracted** — it was a smoothing artefact; the
honest replacement is "slow but *contracting* dynamics, motor slower than visual." The
**area-specific manifold** half has strong single-session support (dPCA gradient, RRR
common-drive) that **must be replicated across the 5 sessions** before it is claimed.

---

## 1. The data

**Dataset.** IBL Brain-Wide Map (DANDI:000409), accessed through ONE-api with the public
`openalyx` credentials (no account needed). Reproducibility pins (`PAPER_FREEZE`,
`AGGREGATE_TAG = 2024_Q2_IBL_et_al_BWM`) and two upstream `brainwidemap` bug workarounds live in
`src/data/access.py`. **SHIELD (DANDI:001051)** is verified public/streamable and kept as a
de-risked second substrate, not yet used.

**Regions — frozen to `{VISp, SCm, MRN}`** (visual cortex → superior colliculus motor →
midbrain reticular nucleus): the one visual→motor axis IBL co-records in ≥5 sessions. `VISpm`
was dropped — it was a stand-in for a "higher-order" tier that IBL does not actually supply. A
clean *sensory→association→motor* triplet essentially does not co-occur in IBL, which is itself
the strongest argument for pivoting to SHIELD.

**The 5 replicate triplet sessions** (all downloaded and preprocessed at 40 ms, both alignments):

| eid8 | trials | VISp quality | balance (design cells) | best for |
|---|---:|---|---:|---|
| **5c0c560e** (current primary) | **843** | **worst** (VISp 1.1 Hz, present ~30 % of session) | worst (10) | trials → HMM/rSLDS; cleanest SCm/MRN |
| **90c61c38** | 656 | ok (20 units) | **best (33)** | clean balanced design; only 76 units total |
| **111c1762** | 364 | **clean** (VISp 4.5 Hz) | 15 | well-recorded VISp; **worst movement confound** (RT 110 ms) |
| **90d1e82c** | 353 | good | 23 | clean all-round; weakest MRN (18–19 u) |
| **8a3a0197** | 290 | ok | 15 | **most units** (188); fewest trials |

**Preprocessing is unified and frozen** (`src/data/preprocess.py` → `data/processed/*.npz`, with
the config serialised *inside* each file so a tensor can never be mis-described later). Every
model eats the same tensors. Frozen configuration:

| knob | value | chosen because |
|---|---|---|
| alignment | `stimOn_times` **+ `firstMovement_times`** | the post-stim window is a movement window (§4) — the second alignment is a standing control |
| `pre_s` | **0.4 s** | inside IBL's enforced quiescence period on 100 % of trials (it is drawn from [0.400, 0.700] s; 0.5 s stepped outside it on 44 %) |
| `post_s` | **0.8 s** | kept for continuity — but it is a *movement + reward* window, not a decision window |
| `bin_size_s` | **0.04 s (40 ms)** | passes the rSLDS ground truth *and* preserves SCm's HMM stability; 20 ms manufactures fake attractors, 50 ms degrades SCm |
| `min_rate_hz` | **1.0** | nearly free (keeps 96.7 / 99.5 / 99.4 % of spikes); 2 Hz halves VISp |
| transform on disk | **none** | the Poisson HMM needs raw counts; Anscombe is applied downstream only |

Frozen tensors for the primary session: **VISp 19 / SCm 44 / MRN 54** units, 843 trials, 30 bins.

**Not in git.** IBL data is downloaded and cached inside the repo at `data/ONE/` (gitignored,
~440 MB for the working session), so a freshly-pulled machine re-downloads on first run.
Cross-machine continuity is `CLAUDE.md` + `docs/worklog/`, *not* chat history (which is
machine-local).

---

## 2. Where each part stands

### 2.1 Preprocessing / data pipeline — ✅ done and frozen

One function, one frozen dataset, config-inside-file. Two findings from building it changed how
everything downstream is read:

- **The window is a MOVEMENT window.** Median reaction time is **171 ms** and median feedback
  **427 ms**, so over `[0, +0.8] s` movement has begun on **92 %** of trials and the outcome has
  arrived on **83 %**. In IBL the choice *is* the wheel-turn direction. Consequence, now a
  standing rule: **no state after ~+0.2 s may be called a "decision" state** without the
  firstMovement-aligned control. Shortening the window does not help — the confound is in the
  task, so we built the movement-aligned tensor instead.
- **Anscombe does not stabilise variance here.** `2√(x+3/8)` reaches Var≈1 only at λ≳4 spikes/bin;
  IBL runs at λ = 0.05–0.83. At λ≪1 a count is ~Bernoulli and its variance is irreducible — no
  monotone transform fixes it. Kept (it compresses the high-count tail, and the rSLDS ground
  truth was calibrated with it), but this is a **second, independent** argument that the Poisson
  `ssm` backend is principled and our PCA latent is a concession.

### 2.2 HMM — *where* the population is — ✅ the directedness result is solid

Per-region Poisson HMM (`src/dynamics/hmm.py`), with two additions: a **baseline-biased initial
state** (install the pre-stimulus occupancy as π, so the trajectory reads as departure from rest)
and **time-ordering of states by occupancy centre-of-mass** (so "forward" is well defined).

**The result that survives everything: motor midbrain traverses its states far more directedly
than visual cortex.** `forward_fraction` (asymmetry of Viterbi transitions, validated against
synthetic + shuffle nulls): **VISp ≈ 0.53–0.59** (at chance — genuinely *undirected*, not static:
it switches state as often as MRN, ~1.5×/trial, just with no preferred direction), **SCm 0.77–0.83**,
**MRN 0.67–0.79**. VISp and SCm survive the split-half control; MRN drops to **0.67 ± 0.10**
split-half (part of its apparent directedness was the order-then-test-the-order circularity), so
we quote the split-half number for MRN.

**K (number of states) is chosen by state *stability* (ARI), not likelihood** — see §4 for why
likelihood fails structurally. In the primary session: **VISp 3, SCm 4, MRN 3**. Nuance from the
5-session replication (`results/hmm/k_selection_stimOn_bin40.json`): **SCm and MRN land on K=3
defensibly across sessions**; **VISp has no defensible consensus** (best-K scatters 3–5), which
matches its weak, non-peaking ARI curve. Only **SCm's interior ARI peak** (dip at K=3, recover to
0.92 at K=4) is a genuinely solid state-structure result; MRN's K=3-vs-K=2 gap is statistically
empty (0.0017) and VISp's curve *declines* from K=2.

### 2.3 rSLDS — *what kind of dynamics* the population runs — ✅ sequence certified, ❌ geometry retracted

Recurrent switching linear dynamical system, two backends by design: our fast **exact-EM** model
on a PCA latent (does model selection, the recurrence test, the screen) and **Linderman's `ssm`**
Poisson backend (the fit of record, ~18–22 min/region, isolated in the `ndrh_ssm` numpy-1.x env,
called out-of-process). First real fit ran end-to-end at 40 ms, both alignments.

- **✅ CERTIFIED — it is an attractor *sequence*, not a stopwatch.** The recurrence test —
  held-out `LL(R free) − LL(R = 0)`, the only statistic separating a position-dependent sequence
  from a timed script — is **positive in every region, both alignments**, and beats a
  position-shuffle null in **100 % of folds** (`results/rslds/recurrence_test.json`; delta
  ordered MRN ≈ SCm ≫ VISp: real-minus-null 0.050 / 0.041 / 0.008 stimOn). `order_consistency`
  (direction-agnostic sequence test) beats its shuffle null in every region on **both** backends,
  large effect sizes (SCm/MRN ~0.93–0.99 vs null ~0.57).
- **❌ RETRACTED — "we found the integrator / line attractor a decision variable lives on."** Both
  backends hit |λ| ≈ 1 (SCm/MRN even > 1 = *unstable*, not integrating), but the |λ|-stability
  test proved it is **>97 % reproduced on temporally-shuffled data** — the value is manufactured
  by the **smoother** (and PCA cross-trial structure), not the neural dynamics. |λ| is essentially
  **not identifiable** at these firing rates. The honest replacement comes from GPFA: **slow but
  *contracting* dynamics, τ = 107–283 ms, motor slowest.**

### 2.4 Dimensionality reduction / geometry — 🔶 the strongest leads live here

Runs on the unified 40 ms pipeline in `04_dimred_msr.ipynb` (05's results merged in and re-run at
40 ms — every number reproduced the 50 ms values, so none is a bin-size artefact).

- **🔶 dPCA reproduces the project's core prediction.** Decision − stimulus variance runs
  **VISp −6.8 → SCm +1.8 → MRN +6.0**, and the condition-independent (time) share falls
  **74 → 64 → 49 %** along the visual→motor axis. This is the strongest single result — but the
  decision components rest on only **25 trials/cell** (off-diagonal design cells are error trials,
  mice are ~85 % correct), measured in the session with the *worst* VISp recording. **Pool the 5
  sessions before it is a figure.**
- **🔶 SCm↔MRN is common drive, not a feedback loop.** RRR finds **no restricted communication
  subspace** — MRN predicts SCm (R²=0.366, rank 7) about as well as SCm predicts *itself*
  (0.383, rank 7). Three independent methods now agree: RRR, the zero-lag pynapple
  cross-correlogram, and the HMM (both areas similarly directed). **A joint rSLDS must now
  *overturn* this, not confirm a loop.**
- **🔶 GPFA — the dynamics are slower than any bin we argued about.** Inferred (not imposed)
  timescales τ = 107–283 ms, **motor slowest** (MRN 283 ms vs VISp 176 ms) — an independent echo
  of the directedness ordering, and proof our smoothed trajectories are not a smoother artefact.
- **The joint manifold** (PCA of all 158 units, per-area partial projections exact by linearity):
  area shares VISp 0.06 / SCm 0.22 / MRN 0.31, summing to **0.585** — the missing **~0.42 is
  cross-area covariance**, i.e. the shared signal the whole project argues about, quantified.
- **Decoder sanity-check** (`00` notebook, session 8a3a0197): a *stimulus* decoder scales with
  contrast (SCm 0.36→0.93); a *choice* decoder is high and flat and **above chance already at 0 %
  contrast** — the mouse commits to a wheel-turn every trial. Confirms the movement entanglement
  and resolves a colleague's "decoding vs coherence" doubt.

### 2.5 MSR (multiscale relevance) — ❌ tested, not adopted

Attractive because model-free/unsupervised, but: (1) it is largely a **firing-rate filter in
disguise** (Spearman ρ with rate 0.55–0.64 in VISp/SCm), so its rate-matched control silently
degenerated into "the slow half"; (2) in MRN — the one region with a valid control — the top-MSR
half decoded **no better than a random half**. **Standing decision: not adopted.** It did yield
one load-bearing rule, though: halving the population is nearly free for *choice* but costly for
the *temporal trajectory* → **selection is fine for decoding, forbidden for dynamics. Never prune
units to "afford" the HMM or rSLDS.**

---

## 3. The spine of the story: one axis, four methods, one retraction

| method | currency | VISp | SCm | MRN | status |
|---|---|---|---|---|---|
| **dPCA** | decision − stimulus variance | **−6.8** | +1.8 | **+6.0** | 🔶 lead (1 session) |
| **HMM** | `forward_fraction` (split-half) | 0.53–0.59 (undirected) | 0.83 | 0.67 | ✅ certified |
| **rSLDS** | recurrence / position-shuffle gap | +0.006–0.008 (weak) | +0.037–0.041 | +0.036–0.050 | ✅ certified |
| **GPFA** | slowest fitted τ | 176 ms | 226 ms | **283 ms** | 🔶 lead |
| ~~stability~~ | ~~# reproducible states~~ | ~~3~~ | ~~4~~ | ~~3~~ | ❌ **no gradient (retracted)** |

**Honest synthesis:** *the areas hold a similar number of states, but the motor areas traverse
them more directedly, carry more decision variance, and run slower* — "shared alphabet,
area-specific sequence." VISp is the consistent weak/undirected end; **MRN the strongest** on
recurrence and τ.

---

## 4. What we tried that didn't work — and why we chose what we chose

This is the compressed history of dead-ends. Each one is a decision that would otherwise get
silently re-litigated.

**Choosing the number of HMM states (K).**
- ❌ *Held-out likelihood.* It climbs with K **whether or not there are more real states**,
  because a Poisson HMM's only way to express noise correlation / over-dispersion is to add a
  state. Argmax gave K=2 (a `-inf`-column bug), then K=10 (top of the range) — both boundaries.
- ✅ *State stability (ARI).* Fit two independent HMMs on disjoint trial-thirds, decode both on an
  unseen third, score agreement with the label-free, chance-corrected Adjusted Rand Index. It has
  a **peak** (too few states → trivial agreement; too many → surplus states split differently by
  each fit → agreement collapses), which likelihood does not. Validated the `hmmlearn`
  implementation against dynamax first (100 % Viterbi agreement).

**Bin size.**
- ❌ *20 ms* (the HMM's original timebase): the rSLDS's PCA latent becomes shot-noise dominated,
  |λ|→1, and *every* regime gets mislabelled "line attractor" — a spurious result that *looks
  like a finding*.
- ❌ *50 ms*: degrades SCm's HMM state stability (ARI 0.938→0.826 — only 24 bins to fit K=4 into).
- ✅ *40 ms, project-wide*: passes the rSLDS ground truth (0 spurious integrators), preserves SCm
  stability, and sits below the fastest GPFA timescale. The value the user intuited.

**The attractor geometry (|λ|, line-attractor labels).**
- ❌ Reading flow labels off the fitted latent. A Gaussian smoother *imposes* the temporal
  autocorrelation an AR fit reads as a slow mode, so |λ|≈1 is >97 % reproduced on shuffled data.
  Retired. GPFA's finite, inferred τ is the honest dynamical statement instead.

**Smoothing σ.**
- ❌ Choosing it by maximising held-out *time*-decoding R². The metric rises monotonically with σ
  and its argmax follows the sweep's ceiling — because the kernel *imposes* exactly the
  autocorrelation the metric rewards. Nearly circular. σ = 40 ms was chosen on external grounds
  (rSLDS ground truth + GPFA timescales). Saving grace: choice AUC is flat across the whole sweep,
  so nothing downstream rests on the knob.

**Unit selection.**
- ❌ *MSR* — a rate filter in disguise; beat neither a random nor a rate-matched half (§2.5).
- ❌ *2 Hz rate floor* — halves VISp, the weakest and most load-bearing region, for no gain
  elsewhere.
- ✅ *1 Hz rate floor* — transparently arbitrary, keeps ~99 % of spikes, and its one failure mode
  fits in a sentence.

**Adding trials by pooling sessions.**
- ❌ *Pseudo-population* (stitching neurons from different sessions into one vector) — invalid for
  the HMM/rSLDS: those neurons were never recorded simultaneously, so there is no single-trial
  joint activity, which is exactly what a state-space model consumes. It doesn't add trials; it
  destroys what a trial *is*.
- ✅ *Replication* — fit each session separately and pool the **statistics** ("K=3 in 5/5,
  forward_fraction 0.8 ± 0.04"). This *is* the animal-independence argument, made properly.
  (Pseudo-population *is* valid for dPCA, which works on trial-averaged activity, and fixes its
  25-trials/cell problem.)

**Two software/method traps worth remembering.**
- `ssm` was written off as "uninstallable" on one error message; there were **two** stacked,
  unrelated errors (missing build-time numpy, then no C++ compiler), both fixable with conda's
  mingw toolchain, no admin. A tested method was nearly dropped for a day.
- The rSLDS `sticky` prior was being set at initialisation and silently overwritten by the first
  M-step; regimes flickered 8×/trial where the truth was 2. A prior must be applied in **every**
  M-step.

**The one rule these share:** *never take an argmax without checking the winner is **interior**
and the criterion is **independent of the parameter being chosen**.* It bit us three times — HMM K
on a boundary, rSLDS integrators from noise, smoothing σ from a metric the smoother inflates.

---

## 5. Open decisions & next steps (ranked)

1. **▶ Area / session selection — the gating decision.** The region set is settled
   (`{VISp, SCm, MRN}`) but `config/areas.yaml::selected` is *still empty* after 6 worklogs, and
   the session strategy is a genuine scientific trade-off: `5c0c560e` was chosen for trial count
   (right for HMM/rSLDS) but is the *worst* session for the two things the geometry/dPCA claim
   leans on (VISp recording, stimulus×choice balance). Three coherent strategies are laid out in
   [`project_synthesis_2026-07-15.md` §8](knowledge/project_synthesis_2026-07-15.md); the lean is
   **(C): two written-down roles — trials for dynamics (`5c0c560e`), balance/quality for geometry
   (a clean-VISp pool, possibly excluding `5c0c560e`).** This is the user's call and the one thing
   blocking a clean final version.
2. **Pool the 5 sessions** for the recurrence/sequence replication and the dPCA gradient — the
   single highest-value step; four strands independently point here. All 5 are already built.
3. **Joint SCm+MRN rSLDS** — the direct exchange-vs-common-drive test. It must now *overturn*
   three converging pieces of common-drive evidence, not assume a loop.
4. **0 %-contrast trials as their own condition** — a *choice without a stimulus*. If the sequence
   still runs there, it is decision/motor-driven, not stimulus-driven — nearly a direct test of
   the claim, and free (~10 % of trials/session).
5. Do **not** keep chasing attractor geometry with a smoothed/inferred-smooth latent — it is not
   identifiable at these rates.

---

*Provenance: `CLAUDE.md`; knowledge docs (project_synthesis_2026-07-15, where_we_are_HMM_rSLDS_dimred,
project_explained, best_K_per_region, the_five_triplet_sessions); worklogs 2026-07-08 → 2026-07-15;
results/ (recurrence_test, k_selection_stimOn_bin40, ground_truth_binsize).*
</content>
</invoke>
