# HMM analysis — mouse first session vs. monkey session

*Written 2026-07-17. Self-contained comparison of the Poisson-HMM read-outs for the project's
strongest mouse session and the (preliminary) monkey session. Regenerate everything here with
`python scripts/compare_hmm_mouse_monkey.py`. Figures are `.png` in this folder; this `.md` is the
only tracked file (the repo gitignores `results/**` except `*.md`).*

---

## What the HMM is, and what these two numbers mean

We fit a **Poisson hidden Markov model** to each area separately, on **raw, unsmoothed spike
counts** in `(trials × time-bins × neurons)` tensors. At every time bin the population is assumed
to sit in one of **K** stereotyped firing-rate patterns ("states") and to hop between them. The
states are then **ordered in time** by their average time of occurrence, so that a
forward-flowing sequence puts its transitions in the upper triangle of the transition matrix.

Two read-outs carry the science, and they are *deliberately kept separate* because they are
constantly conflated:

- **`forward_fraction` — directedness.** Of all the state changes on single trials, what fraction
  go "forward" (to a later-ordered state) vs. "backward". **0.5 = undirected** (the population
  wanders among its states with no preferred order); **→1.0 = a strict forward chain**. This is the
  shape of the transition-probability structure.
- **`switch_rate` — how often the state changes at all** (state changes per trial). A region can
  switch constantly with *no* net direction (high switch, `forward_fraction ≈ 0.5`) or switch twice
  per trial *always in the same order* (low switch, `forward_fraction ≈ 1`). **They are orthogonal.**

`forward_fraction` is reported two ways: the **pooled** value (bar) and a **split-half** value
(◇ ± SD) that derives the time-ordering on one half of trials and scores the transitions on the
other — this removes the mild circularity of ordering-then-testing-the-same-trials. When the two
agree, the pooled number is honest.

**Configuration (frozen tensors, re-fit here so the figures are internally consistent):**

| | mouse | monkey |
|---|---|---|
| dataset | IBL Brain-Wide Map, session `5c0c560e` | DANDI 000620, `ses-2022-08-20` (sub-Elgar) |
| areas | VISp, SCm, MRN (visual→motor axis) | FEF, DMFC (frontal) |
| task | 2-alternative visual choice (wheel turn) | working-memory delay task |
| alignment | stimulus onset, 40 ms bins, `[−0.3, +0.8] s` | **delay onset**, 100 ms bins, `[−1.4, +2.2] s` |
| K (per area) | VISp 3, SCm 4, MRN 3 | FEF 4, DMFC 3 |
| startprob | baseline-biased (real pre-stim baseline) | none (no verified baseline) |

K is chosen by **state-stability** (do two independent fits recover the same partition of time,
measured by Adjusted Rand Index), *not* by held-out likelihood — on spike counts likelihood climbs
forever because a Poisson HMM can only mimic neural correlations by adding states.

---

## The read-outs

`readouts.csv` in this folder; the mouse values reproduce `results/hmm/hmm_readouts_5c0c560e.csv`
and the monkey values reproduce `results/hmm/monkey_ses-2022-08-20_readouts.csv`.

| species | area | K | units | trials | `forward_fraction` | split-half ± SD | `switch_rate` |
|---|---|--:|--:|--:|--:|--:|--:|
| mouse | **VISp** | 3 | 19 | 843 | 0.55 | 0.55 ± 0.01 | 1.74 |
| mouse | **SCm** | 4 | 44 | 843 | **0.70** | 0.68 ± 0.04 | 2.74 |
| mouse | **MRN** | 3 | 54 | 843 | **0.77** | 0.73 ± 0.07 | 1.45 |
| monkey | **FEF** | 4 | 43 | 414 | 0.55 | 0.52 ± 0.02 | **0.26** |
| monkey | **DMFC** | 3 | 56 | 414 | 0.52 | 0.52 ± 0.00 | **6.38** |

---

## Figure 1 — `01_directedness_and_switching.png`

**The headline contrast.** In the **mouse** the three areas span a clear **directedness gradient**:
VISp sits at chance (0.55 — its states are reproducible but its transitions are *symmetric*, it
wanders forward and backward equally), while the motor-axis areas **SCm (0.70)** and **MRN (0.77)**
traverse their states in a directed forward sequence. The split-half diamonds track the bars, so
this is not a circularity artefact.

In the **monkey**, **both frontal areas sit at chance** (FEF 0.55, DMFC 0.52; split-half **0.52 /
0.52**). The transition probabilities are **flat / undirected** — there is no preferred temporal
order to the state changes, in either area. This is the "**more constant transition probabilities**"
observation.

The `switch_rate` panel shows these are two *different* flavours of "flat": **FEF barely switches at
all (0.26/trial)** — within a trial its state is essentially constant — whereas **DMFC switches a
lot (6.38/trial)** but with no net direction (`forward_fraction 0.52`): it hops back and forth among
its states. So "undirected" in the monkey is not "quiet"; it is *symmetric*.

## Figure 2 — `02_state_occupancy.png`

**The state sequences themselves** — `P(state | time)`. The mouse motor areas show clean directed
hand-offs (SCm: baseline S0 → S2 → S3; MRN: S1 → S2), VISp rises into an evoked state and *returns*
(the symmetric signature), and — critically — the **monkey FEF panel is three flat lines**: the
states do not move with within-trial time at all. **DMFC** holds a sustained maintenance
configuration across fixation→stimulus→delay and switches into a distinct state only near the
**response (~+1.5 s)** — a working-memory *maintain-then-transition* signature (the one genuinely
interpretable monkey result so far).

## Figure 3 — `03_transition_flux.png`

**Net transition flux** `Φ[i,j] = N[i,j] − N[j,i]` (red = net forward, blue = net backward; RdBu_r
is red–blue, CVD-safe). SCm and MRN are strongly **asymmetric** (a directed cascade, hundreds of net
forward transitions); VISp is weaker and more balanced; **FEF has almost no transitions at all**
(total net flux ≈ ±10 — the numeric echo of `switch_rate = 0.26`); DMFC has moderate flux that is
nearly **symmetric** (`forward_fraction 0.52`).

## Figure 4 — `04_monkey_drift_confound.png` — *the colleague's point, made concrete*

Why is the monkey HMM flat? At least for **FEF**, it is partly an **artefact of an across-session
drift** that dominates the population and hijacks the inference:

- **(a)** FEF's population rate **falls across the session** (`r(trial#, rate) = −0.51`; ~3.7 → 2.7
  sp/s/unit), while DMFC is stationary (`+0.32`).
- **(b)** A single **PC1 explains ~42 %** of FEF's trial-to-trial variance and **correlates with
  trial index at r = +0.94** — i.e. the dominant axis of FEF activity is *when in the session the
  trial happened*, splitting the trials into **two firing-rate regimes** (early/high, late/low). This
  is exactly the *"strong first PCA component representing two main firing-rate patterns"*.
- **(c)** Consequently the FEF HMM spends its states on **contiguous trial-index blocks** (session
  epochs) rather than on within-trial phase. Within a trial the state is constant (`switch 0.26`);
  across trials the modal state jumps in blocks that follow the drift. (The plotted
  `r(state, trial#) = −0.53` *understates* this — the block structure in the panel is the real
  evidence; the sign and magnitude are muddied because states are relabelled by within-trial time,
  not by trial order.)

**The consequence for interpretation.** The HMM has a fixed variance budget. When a slow drift is the
largest source of variance, the model spends its states describing the drift and reports a *flat,
undirected* within-trial sequence — **not because the within-trial sequence is absent, but because it
is small next to the drift.** The standing FEF control (`docs/worklog/2026-07-17_fef-drift-vs-within-
trial-states.md`) confirms this directly: restricting the fit to a limited-drift block of trials
**recovers a clean within-trial FEF sequence** (occupancy swing 0.00 → 0.61, `switch 0.0 → 1.8`,
state↔phase Cramér's V 0.01 → 0.36). So FEF's flatness on the full session is a drift confound, not a
biological null. DMFC, with no strong drift, is the positive control — it carves within-trial time
normally.

---

## The comparison in one paragraph

**Mouse (`5c0c560e`).** The three areas hold a similar number of reproducible states but differ
sharply in *how* they traverse them: a clean **visual→motor directedness gradient** (VISp 0.55 ≈
chance / undirected → SCm 0.70 → MRN 0.77), with the motor areas running a directed forward cascade
and VISp switching symmetrically. This is the project's core HMM result — *a shared alphabet of
states realised as an area-specific, increasingly directed sequence.* (Caveat carried from the main
analysis: median RT is 171 ms, so the post-stimulus window is a **movement** window — the late states
are entangled with the wheel-turn and reward.)

**Monkey (`ses-2022-08-20`, preliminary — one session, one animal).** Both frontal areas show
**more constant / undirected transition probabilities** (`forward_fraction ≈ 0.5`, split-half 0.52).
But this is at least partly **confounded by inference**, not a biological flatness: a strong
across-session drift (FEF PC1 = 42 % variance, r = 0.94 with trial index — "two firing-rate
patterns") makes the HMM partition **trials into session epochs** rather than within-trial phase, and
removing the drift range recovers a genuine within-trial FEF sequence. DMFC, drift-free, does show a
real working-memory maintain-then-transition state, though still without a net *direction*.

**Bottom line.** The mouse gives a directed, area-differentiated state sequence; the monkey's
apparently flat sequence should **not** be read as "monkey frontal cortex has no state sequence"
until the drift is modelled — the honest next steps are a drift-aware (input-driven / GLM-HMM) fit
and replication across the 14 usable monkey sessions and the second animal.

---

### Provenance / caveats

- Mouse numbers reproduce the frozen `results/hmm/hmm_readouts_5c0c560e.csv` and the 5-session
  pooled `results/hmm_5sessions/` exactly; monkey numbers reproduce
  `results/hmm/monkey_ses-2022-08-20_readouts.csv`.
- **Not a like-for-like task.** Different species, tasks, alignments (stimulus vs. delay onset), and
  bin sizes (40 vs. 100 ms). The comparison is of *transition structure* (directedness, switching),
  which is dimensionless and alignment-robust — not of absolute timing.
- The monkey strand is **one session, one animal**, and explicitly **preliminary**. The mouse strand
  is one session of five replicates (this is the strongest single session; the directedness gradient
  replicates 5/5, the pooled magnitude is milder — VISp 0.52 / SCm 0.64 / MRN 0.64).
- Related docs: `docs/knowledge/recap_data_HMM_dimred_5c0c560e.md` (mouse single session),
  `docs/worklog/2026-07-17_monkey-events-delay-aligned.md` and
  `docs/worklog/2026-07-17_fef-drift-vs-within-trial-states.md` (monkey).
