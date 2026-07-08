# IBL Brain-Wide Map — Decision Dynamics: Project Plan (Janelia Hackathon)

*Dataset: IBL DANDI:000409 / IBL "Brain Wide Map" (IBL et al. 2025, Nature 645:177-189).
Draft v1. This document is written to be actionable by Claude Code: it contains a
task breakdown, verified repository links, install commands, a verified data-access
snippet, and a proposed repository structure. All non-obvious claims are referenced;
items flagged "verify" still need checking against the archive/repo at build time.*

---

## Level 1 — The project in brief

**Scientific question.** While a mouse performs a standardized visual decision task,
can we describe decision-making as a **temporal sequence of attractor / metastable
states** that unfolds from stimulus onset across a set of brain areas spanning the
visual->motor axis, and show that the **same shared latent decision dynamics** are
implemented on **geometrically distinct, area-specific manifolds**? A secondary,
higher-risk aim is to characterize the **functional direction of signal flow**
between areas (which is far less settled than the anatomy) and compare it to known
mesoscale connectivity.

**How we address it, in words.** We take a handful of simultaneously recorded
populations (one brain region each, within a session), align activity to stimulus
onset, and (1) infer the sequence of discrete states / dynamical regimes the
population passes through, relating that sequence and its timing to behaviour
(reaction time, correct vs error); (2) embed each area's population activity to
measure and compare the *geometry* of its neural manifold; and (3) use NeuralFlow to
show the decision variable behaves like an attractor and that what differs across
areas is the tuning geometry, not the dynamics. If time allows, we estimate directed
interactions between area pairs and check them against the Allen mouse connectome.

**Expected contribution.** A concrete, quantitative demonstration that a shared
decision dynamic is realized through area-specific neural geometry, framed as
*distributed temporal sequences within populations* rather than a clean anatomical
relay — directly engaging the open question (distributed vs staged processing) that
the brain-wide map itself raises.

---

## Level 2 — Approach and technique mapping

### Problem -> technique

| Problem | Primary tool(s) | Role |
|---|---|---|
| Which units to include (small sets) | Multiscale Relevance (MSR) | Unsupervised pre-filter; cross-check vs task selectivity |
| Sequence of states over time (backbone) | HMM; rSLDS | Metastable states / dynamical regimes from stimulus onset |
| Attractor nature of the decision variable | NeuralFlow (1D) | Show shared attractor dynamics; area-specific tuning |
| Area-specific manifold geometry | PHATE / T-PHATE; CEBRA | Quantitative geometry + cross-area aligned embedding |
| Single-trial confirmation | AutoLFADS | Confirm trial-averaged geometry reflects single trials |
| Directed functional flow (stretch) | Communication subspace; DLAG | Directed inter-areal signalling; validate vs anatomy |
| Structural nested-basin view (optional) | Energy Landscape Analysis | Follow-up only |

### Data and areas

- Restrict to **well-isolated ("good") units**; the released dataset has 621,733
  units total, 75,708 good, spanning 279 areas (241 with sufficient counts for
  analysis) (IBL et al. 2025).
- Methods that model single-trial dynamics need **simultaneously recorded**
  populations, so work **per session, per region** (or pooling a session's ~2 probes),
  **not** cross-session pseudopopulations. Pseudopopulations are acceptable only for
  final visualization.
- Practical constraint: ~108 good units/probe on average, ~2 probes/session, so
  per-session per-region yields are often tens of neurons; and DLAG needs two target
  areas in the **same** session -> selection is really *area-set-and-pair* selection
  by co-occurrence (see Data Access section for the query).

### Two prongs + support

- **Prong 1 (backbone):** HMM and rSLDS -> state/regime sequence and timing, aligned
  to stimulus onset; compare to RT and correct-vs-error; read cross-area latency
  structure as a signature of distributed flow.
- **Prong 2:** PHATE/T-PHATE (trajectory geometry) + CEBRA (cross-area aligned,
  quantitatively comparable embedding) + geometry metrics (participation ratio,
  intrinsic dimensionality, curvature) against a shuffle null.
- **Support:** NeuralFlow (attractor demonstration), AutoLFADS (single-trial),
  communication subspace / DLAG (directed flow, stretch), ELA (structural, optional).

**Framing guardrails.** Distributed temporal sequences within populations, *not* a
clean anatomical pipeline. Headline claim: *same latent attractor dynamics,
geometrically distinct area-specific realizations.*

---

## Background — brain areas and the visual->motor flow (from the literature)

*This section grounds area selection and interpretation. Region acronyms follow the
Allen/Beryl parcellation used by IBL (region table:
https://github.com/int-brain-lab/paper-brain-wide-map/blob/main/brainwidemap/meta/region_info.csv).*

### Anatomy / structural connectivity (what is well established)

- **Retina -> dLGN -> V1** is very well characterized by tract tracing and physiology.
- The mouse mesoscale connectome (Oh et al. 2014, Nature 508:207-214) and its
  layer/cell-class-resolved extension (Harris et al. 2019, Nature 575:195-202) map
  cortico-cortical and cortico-thalamic wiring and assign feedforward/feedback roles
  by layer (L2/3, L4 feedforward; L6 feedback; L5 both). Key point: the mouse
  cortico-thalamic hierarchy is **shallow**, with dense reciprocal connectivity — not
  a strict primate-like ladder.
- Indirect **transthalamic** routes (via LP / pulvinar) relay signals between cortical
  areas and carry ascending superior-colliculus and retinal input.
- What is *less* settled: the **directed, functional** flow of task signals through
  this wiring — precisely the secondary target of this project.

### Functional roles along the axis (what the task data show)

From the brain-wide map (IBL et al. 2025), using decoding, controlled single-cell
statistics, population-trajectory distance, and an encoding GLM:

- **Visual stimulus** appears first in the classical cascade — visual thalamus (LGd,
  LP) and V1 (VISp) show early, short-latency trajectory divergence, followed by
  VISpm/VISam/VISl and prefrontal MOs; it then extends as later ramping into midbrain
  (SCm, MRN, APN, SNr, NOT) and hindbrain (GRN, PRNr). Even in visual cortex, at most
  ~10% of neurons are stimulus-selective; only ~0.5% brain-wide when controlling for
  correlated variables.
- **Choice** is broadly distributed, with the **strongest** signals subcortical:
  hindbrain (GRN, IRN, PRNc, PRNr, VII, MARN), midbrain (SCm, MRN, APN, SNr),
  thalamus (CL, LD, SPF), cerebellum (CENT2); ~4% of neurons brain-wide. The earliest
  choice signals emerge nearly simultaneously in thalamus (LD) and cortex (VISl,
  VISam, ECT), then spread. Choice is also decodable from cortical MOs, MOp, PL,
  ORBvl, VISp, SSp-ul. In an earlier dataset, a lead-lag (jPECC) analysis found visual
  cortex leading frontal cortex by ~40 ms only in a narrow ~200 ms pre-movement
  window, with no consistent midbrain-frontal lead-lag (Steinmetz et al. 2019, Nature
  576:266-273) — a caution that "flow" here is distributed and largely simultaneous.
- **Feedback** (reward vs error) is nearly ubiquitous, strongest in thalamus,
  midbrain and hindbrain; the correct-feedback kernel is the single largest
  variance contributor; reward triggers a ~10 Hz oscillation broadcast across areas
  and tied to licking.
- **Wheel movement** is near-ubiquitous (speed decodable from 81% of regions),
  strongest in LPO, MARN, GRN, CLI, thalamus (CL, PF), ORBvl; PRNc and GRN stand out
  in encoding.

Classical framework for reference: evidence accumulation to a decision bound with
bound-crossing signatures across many motor-planning regions (Gold & Shadlen 2007,
Annu Rev Neurosci 30:535-574). Motor-side nodes with causal roles in rodents include
the superior colliculus (direct retinal input; major visual-cortex output; causal in
visual decisions) and secondary/anterior-lateral motor cortex, with a causally tested
M2->SC cortico-collicular pathway (Duan et al. 2021, Nat Commun 12:2727).

**Take-away for the project.** The evidence favours **distributed, parallel** decision
processing, not a serial relay; a good candidate axis of well-sampled regions is
**VISp -> LP/LGd -> MOs -> CP -> SCm/MRN -> GRN**, to be pruned by the co-occurrence
query.

---

## Level 3 — Techniques in detail (with repositories)

*Install commands and minimal API patterns are given so Claude Code can scaffold each
step. "verify" = confirm the repo/API at build time.*

### Environment (suggested)

```bash
# Core scientific stack
pip install numpy scipy scikit-learn pandas matplotlib

# IBL data access
pip install ONE-api
pip install git+https://github.com/int-brain-lab/paper-brain-wide-map  # brainwidemap
# (brainbox / iblatlas come via the IBL environment; verify exact install)

# Latent dynamics / sequences
pip install ssm            # Linderman lab: HMM, ARHMM, SLDS, rSLDS  (verify: may need git install)
pip install dynamax        # JAX HMM/LGSSM (maintained alternative for HMM)

# Geometry
pip install phate
pip install tphate
pip install cebra

# Single-trial (GPU recommended) — install from source
# git clone https://github.com/arsedler9/lfads-torch

# NeuralFlow
pip install git+https://github.com/engellab/neuralflow
```

### Unit selection — Multiscale Relevance (MSR)

Non-parametric, model-free metric computed from spike-time stamps alone; ranks
neurons by the richness of their spike statistics integrated across time scales
(Cubero, Marsili & Roudi 2020, J Comput Neurosci 48:85-102; arXiv:1802.10354).
Use as an **unsupervised** pre-filter to reduce a large good-unit pool to the small
set that data-hungry methods (esp. ELA) can handle, then **cross-check** the selected
set against task selectivity (IBL condition-combined Mann-Whitney) to guard against
picking units rich for non-task reasons.

- Repo: no official package located in this search — **implement from the paper**
  (MSR from the "relevance" H[K] across resolutions; verify).
- Inputs: spike times per unit. Outputs: an MSR score per unit -> ranking.
- Gotcha: high MSR = dynamically rich, not guaranteed task-relevant; keep the
  task-selectivity check on held-out data to avoid double-dipping.

### Latent dynamics — NeuralFlow

Infers a 1D latent decision variable governed by Langevin dynamics (a potential
landscape whose minima are attractors) plus single-neuron tuning to that latent
(Genkin, Shenoy, Chandrasekaran & Engel 2025, Nature 645:168-176). Use to show the
decision dynamics are attractor-like and shared across areas, while tuning geometry
is area-specific.

- Repo: https://github.com/engellab/neuralflow (`pip install git+...`). Verified.
- Inputs: single-trial spike data. Outputs: inferred potential/drift + tuning
  functions.
- Gotcha: 1D only (2D in development); not a sequence or cross-area model.

### Temporal sequence — HMM and rSLDS (backbone)

- **HMM** segments population activity into discrete metastable states (interpretable
  as attractors of the dynamics); robust to trial-to-trial timing jitter.
- **rSLDS** (Linderman et al. 2017, AISTATS) adds a continuous latent with
  locally-linear regimes whose switches depend on the state -> globally nonlinear
  dynamics; decision-specific state-space framework in Zoltowski, Pillow & Linderman
  2020; multi-population extension in Glaser et al. 2020 (NeurIPS).
- Repos:
  - https://github.com/lindermanlab/ssm — canonical NumPy implementation with HMM,
    ARHMM, SLDS, **rSLDS** and example notebooks (recommended for rSLDS). Verified.
  - https://github.com/probml/dynamax — maintained JAX implementation (HMM, LGSSM;
    SLDS support more limited; see lindermanlab/dynamax-slds). Verified.
- Inputs: binned (optionally smoothed) single-trial population activity, aligned to
  stimulus onset. Outputs: per-trial state/regime sequences, transition times, latent
  trajectories.
- Analyses: relate state sequence and transition timing to RT and correct-vs-error;
  compare sequences/timing across areas.
- Gotcha: choose state count by cross-validated log-likelihood; HMM states lack
  internal dynamics (rSLDS mitigates).

### Neural geometry — PHATE / T-PHATE and CEBRA

- **PHATE** (Moon et al. 2019, Nat Biotechnol 37:1482-1492): diffusion-geometry
  embedding using a "potential distance"; preserves local + global structure and
  continuous trajectories (unlike t-SNE/UMAP, which shatter trajectories).
  - Repo: https://github.com/KrishnaswamyLab/PHATE (`pip install phate`;
    `phate.PHATE().fit_transform(X)`). Verified. See also
    https://github.com/KrishnaswamyLab/Multiscale_PHATE for a tree of granularities.
- **T-PHATE** (Busch et al. 2023, Nat Comput Sci): dual-kernel PHATE adding a temporal
  view from the signal's autocorrelation; recovers latent temporal trajectories.
  - Repo: https://github.com/KrishnaswamyLab/TPHATE (`pip install tphate`;
    `tphate.TPHATE().fit_transform(X)`). Verified. Analysis code:
    https://github.com/ericabusch/tphate_analysis_capsule.
  - Note: validated on fMRI; on spikes, bin/smooth so the temporal autocorrelation is
    meaningful (pilot first).
- **CEBRA** (Schneider, Lee & Mathis 2023, Nature 617:360-368): contrastive
  (InfoNCE) encoder; CEBRA-Behavior (label-guided) or CEBRA-Time (self-supervised);
  produces embeddings consistent across sessions/subjects and alignable across areas
  even with different neurons; the consistency score is a quantitative metric.
  - Repo: https://github.com/AdaptiveMotorControlLab/CEBRA (`pip install cebra`;
    scikit-learn-style). Verified. Demos (incl. Allen V1 decoding, consistency,
    multisession across animals): https://github.com/AdaptiveMotorControlLab/CEBRA-demos.
  - Note: not a dynamics model; use for aligned embedding + quantitative geometry +
    visualization, paired with rSLDS/LFADS for the temporal story.
- Geometry metrics: participation ratio, intrinsic dimensionality, curvature — test
  differences across areas against a shuffle null. Anchors: Fortunato et al. (Gallego
  lab, bioRxiv 2023.07.18.549575 — manifold nonlinearity differs across regions);
  Langdon, Genkin & Engel 2023 (dynamics-vs-geometry).
- Inputs: binned/smoothed rate vectors (samples x neurons). Outputs: low-D embeddings
  + geometry metrics.
- Gotcha: UMAP/t-SNE for visualization only, never for quantitative geometric claims.

### Single-trial confirmation — AutoLFADS

Sequential VAE/RNN inferring single-trial latent dynamics (temporal factors +
per-trial initial conditions + inferred inputs) from simultaneously recorded spikes
(Pandarinath et al. 2018, Nat Methods 15:805-815); AutoLFADS automates the
hyperparameter search (Keshtkaran et al. 2022, Nat Methods 19:1572-1577).

- Repo: https://github.com/arsedler9/lfads-torch (PyTorch; LFADS + AutoLFADS; includes
  a multisession stitching guide). Verified. TF2 alternative:
  https://github.com/snel-repo/autolfads-tf2.
- Inputs: single-trial binned spikes. Outputs: single-trial factors/rates.
- Gotcha: GPU strongly recommended; heavier setup than HMM/rSLDS -> "if time remains."

### Directed functional flow (stretch) — communication subspace and DLAG

- **Communication subspace** (Semedo et al. 2019, Neuron 102:249-259): the low-D
  subspace through which one area predicts a downstream one; implement via
  **reduced-rank regression** (straightforward in Python/scikit-learn).
- **DLAG** (Gokcen et al. 2022, Nat Comput Sci 2:512-525): disentangles concurrent,
  bidirectional flow between two simultaneously recorded populations, assigning a
  direction via the sign of a time delay and characterizing temporal evolution.
  - Repos: https://github.com/egokcen/DLAG (MATLAB), https://github.com/egokcen/mDLAG
    (multi-population, MATLAB), https://github.com/egokcen/fast-mDLAG (accelerated).
    Verified. **DLAG is MATLAB** — a friction point for a Python/Claude-Code workflow;
    for a pure-Python sprint prefer reduced-rank-regression communication subspace or
    Granger causality (statsmodels), and treat DLAG as a stretch.
- Validation: compare inferred directed interactions to Allen anterograde connectivity
  (Oh 2014) and Harris-2019 feedforward/feedback assignments. Precedent on IBL: SPIDER
  (2026 preprint, arXiv:2606.22695) — directed flow on the BWM validated vs the Allen
  atlas (passive epochs).

### Structural nested-basin view (optional / follow-up) — ELA

Fit a pairwise maximum-entropy (Ising) model to binarized activity; local minima =
attractors; a disconnectivity graph gives nested basin structure (Watanabe et al.
2014; Ezaki et al. 2017; tutorial + code: Masuda et al. 2024, arXiv:2411.16979).
Binarize spikes as 1/0 per bin (~10-20 ms; report sensitivity). Data-hungry -> small
variable sets only (e.g., top latent dims or a modest unit subset). Keep as a
structural follow-up; the time-resolved version is better done via input-driven
dynamics than sliced Ising models.

---

## Data access (verified) — getting a few areas to test

*Mechanism verified from the IBL docs and the paper-brain-wide-map repo. Requires
network access to the IBL/AWS servers (not available in every sandbox), so run this in
the build environment.*

```python
from one.api import ONE
from brainwidemap import bwm_query, bwm_units
from brainbox.io.one import SpikeSortingLoader

one = ONE(base_url='https://openalyx.internationalbrainlab.org')

# 1) Tables of released sessions/probes and of all analysis-grade units
bwm_df   = bwm_query(one)     # sessions + probe insertions (pid, eid, ...)
unit_df  = bwm_units(one)     # every good unit used in the BWM, with region
                              # (Beryl acronym / atlas_id) and CCF coordinates

# 2) Count good units per region and find co-occurring target regions
#    (group unit_df by [eid/pid, region] to get yields; intersect sessions across
#     the target-region set to find sessions where >=2 targets are recorded together).

# 3) Find all insertions hitting a specific region (Allen parcellation is
#    hierarchical: 'TH' returns all thalamic children, etc.)
ins_visp = one.search_insertions(atlas_acronym='VISp',
                                 datasets='spikes.times.npy', project='brainwide')

# 4) Load spike sorting for one insertion; per-unit region is in clusters['acronym']
pid = ins_visp[0]
sl = SpikeSortingLoader(pid=pid, one=one)
spikes, clusters, channels = sl.load_spike_sorting()
clusters = sl.merge_clusters(spikes, clusters, channels)
```

**Conclusion: yes, 2-3 (and up to ~6) areas are downloadable and selectable by
region.** Also load per session: `trials` table (stimulus side/contrast, choice,
feedback, first-wheel-move time) for the behavioural alignment in Prong 1.

---

## Proposed repository structure (for Claude Code)

```
ibl-decision-dynamics/
  README.md
  environment.yml / requirements.txt
  config/
    areas.yaml            # target regions + selection thresholds
    params.yaml           # bin width, smoothing, state counts, CV folds
  src/
    data/
      access.py           # ONE-api wrappers: bwm tables, region search, loaders
      select_areas.py     # yields per region, co-occurrence -> chosen area set
      build_matrices.py   # spikes -> binned/smoothed (trials x time x neurons)
    selection/
      msr.py              # Multiscale Relevance + task-selectivity cross-check
    dynamics/
      hmm.py              # HMM fit + state sequences (ssm/dynamax)
      rslds.py            # rSLDS fit + regimes (ssm)
      neuralflow_run.py   # NeuralFlow wrapper
      lfads_run.py        # AutoLFADS (optional)
    geometry/
      embeddings.py       # PHATE / T-PHATE / CEBRA
      metrics.py          # participation ratio, intrinsic dim, curvature, nulls
    flow/
      comm_subspace.py    # reduced-rank regression (stretch)
      dlag_bridge.py      # optional MATLAB DLAG bridge (stretch)
    analysis/
      behavior_link.py    # sequences/timing vs RT, correct-vs-error
      figures.py
  notebooks/
    00_data_smoketest.ipynb
  results/
```

---

## Phased plan / milestones

**Phase 0 — De-risk (day 1)**
- [ ] Set up env; authenticate ONE-api; run `00_data_smoketest` on ONE session.
- [ ] Load one insertion's spikes + trials; build a (trials x time x neurons) matrix
      for one region aligned to stimulus onset.
- [ ] Fit a quick HMM on that one region end-to-end (proves the whole pipeline).

**Phase 1 — Area selection**
- [ ] From `bwm_units`, compute good-unit yields per region; compute co-occurrence of
      the candidate axis (VISp, LP/LGd, MOs, CP, SCm/MRN, GRN).
- [ ] Pick 2-3 (up to ~6) regions/pairs with the best simultaneous yields; freeze in
      `config/areas.yaml`.

**Phase 2 — Prong 1 (backbone)**
- [ ] MSR ranking + task-selectivity cross-check for each region.
- [ ] HMM and rSLDS per region; select state count by cross-validated LL.
- [ ] Extract state/regime sequences + transition times; align to stimulus onset.
- [ ] Link to behaviour (RT, correct-vs-error); compare sequences/timing across areas.

**Phase 3 — Prong 2 (geometry)**
- [ ] PHATE / T-PHATE embeddings per region; CEBRA aligned embedding across regions.
- [ ] Geometry metrics (participation ratio, intrinsic dim, curvature) vs shuffle null.
- [ ] Quantify area differences; produce the shared-dynamics/distinct-geometry figure.

**Phase 4 — Support / stretch**
- [ ] NeuralFlow: attractor demonstration (shared dynamics, area-specific tuning).
- [ ] AutoLFADS single-trial confirmation (if GPU/time).
- [ ] Communication subspace (RRR) between pairs; optional DLAG; validate vs Allen.

**Phase 5 — Write-up**
- [ ] Consolidate figures; document decisions; note follow-ups (ELA, DLAG, anatomy).

---

## Success criteria (define before looking at results)

- **Prong 1:** a reproducible, cross-validated state/regime sequence per region whose
  timing relates to behaviour (RT and/or correct-vs-error) above a shuffle baseline;
  a cross-area latency ordering with uncertainty.
- **Prong 2:** an area difference in a geometry metric (e.g., participation ratio or
  intrinsic dimensionality) that exceeds a shuffle null; a CEBRA consistency score
  quantifying cross-area/session comparability.
- **Support:** NeuralFlow potentials that are attractor-shaped and similar across
  areas while tuning differs; (stretch) directed-flow estimates consistent (or
  informatively inconsistent) with Allen connectivity.
- Pre-specify the comparison and correct for multiple regions to avoid p-hacking.

---

## References (see project_notes.md for the full categorized list)

Core, in project knowledge: IBL et al. 2025 (Nature 645:177-189); IBL 2021 (eLife
e63711); Genkin/Shenoy/Chandrasekaran/Engel 2025 (Nature 645:168-176); Langdon/
Genkin/Engel 2023 (Nat Rev Neurosci 24:363-377); Linderman et al. 2017 (AISTATS);
Zoltowski/Pillow/Linderman 2020 (ICML); Glaser et al. 2020 (NeurIPS); Schneider/Lee/
Mathis 2023 (Nature 617:360-368; CEBRA); Fortunato et al. (bioRxiv 2023.07.18.549575);
Semedo et al. 2019 (Neuron 102:249-259); Gokcen et al. 2022 (Nat Comput Sci 2:512-525;
DLAG); Cubero/Marsili/Roudi 2020 (J Comput Neurosci 48:85-102; MSR); Oh et al. 2014
(Nature 508:207-214); Harris et al. 2019 (Nature 575:195-202); Steinmetz et al. 2019
(Nature 576:266-273); Siegle et al. 2021 (Nature 592:86-92); Findling et al. 2025
(Nature; prior companion).

**To add to project knowledge:** Moon et al. 2019 (Nat Biotechnol 37:1482-1492; PHATE);
Busch et al. 2023 (Nat Comput Sci; T-PHATE); Pandarinath et al. 2018 (Nat Methods
15:805-815; LFADS) + Keshtkaran et al. 2022 (Nat Methods 19:1572-1577; AutoLFADS).

Other cited here: Gold & Shadlen 2007 (Annu Rev Neurosci 30:535-574); Duan et al. 2021
(Nat Commun 12:2727); SPIDER 2026 preprint (arXiv:2606.22695). ELA (optional): Watanabe
et al. 2014; Ezaki et al. 2017; Masuda et al. 2024 (arXiv:2411.16979).
```
