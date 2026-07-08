# Project Index & Claude Code Handoff — IBL / SHIELD Decision Dynamics

*Entry point for a fresh Claude Code / chat session (targeting NeuroDataReHack 2026,
HHMI Janelia). Read this first, then the linked companion docs. This file only
synthesizes decisions already made with the human; it introduces no new scientific
claims.*

---

## 0. Standing rules for this project (carry into every session)

- **Science-based; do not invent.** If something is uncertain, say so or ask — do not
  fabricate facts, numbers, citations, or APIs.
- **Reference important claims** (paper + venue/year, or repo URL).
- Distinguish **verified** facts from items flagged **"verify"** (to check against the
  archive/repo/docs at build time).
- **Python only** (little MATLAB available). **CPU-first** (GPU not yet confirmed).
- Prefer editing real files and running real smoke-tests over assuming behavior.

---

## 1. Project in one paragraph

While a mouse performs a visual decision task, characterize decision-making as a
**temporal sequence of attractor / metastable states** that unfolds from stimulus
onset across a few simultaneously recorded brain areas, and show that the **same
shared latent decision dynamics** are realized on **geometrically distinct,
area-specific manifolds**. A secondary, higher-risk aim is the **functional direction
of signal flow** between areas (less settled than the anatomy). Framing guardrail:
*distributed temporal sequences within populations, not a clean anatomical relay.*
Grounding papers: IBL et al. 2025 (Nature 645:177-189; the dataset); Genkin, Shenoy,
Chandrasekaran & Engel 2025 (Nature 645:168-176; NeuralFlow); Langdon, Genkin & Engel
2023 (Nat Rev Neurosci 24:363-377; the dynamics-vs-geometry distinction).

---

## 2. Document map (companion notes)

*Filenames were changed by the human — fill in the right-hand column.*

| Role / contents | Original name | Your filename |
|---|---|---|
| **Master project plan** — 3 levels (brief / technique-mapping / detailed-with-repos), background on brain areas & flow, verified repo links, install commands, verified IBL data-access snippet, proposed repo structure, phased checklist, success criteria | `project_plan.md` | `__________` |
| **Running notes** — concise concept summaries with pros/cons per technique; the human's own project notes woven in; full categorized reference list | `project_notes.md` | `__________` |
| **Dimensionality-reduction reference** — MDS backbone + Isomap, Laplacian eigenmaps, diffusion maps, PHATE, T-PHATE, CEBRA, LFADS/AutoLFADS; spike-data applicability; what is quantitative vs visualization-only | `dimensionality_reduction_methods.md` | `__________` |
| **This index / handoff** | `project_index.md` | `__________` |

Read order for a new session: **this file -> project plan -> running notes ->
DR reference** (the last two are look-ups, not front-to-back reads).

---

## 3. Current status & decisions

**Decided:**
- **Backbone:** temporal sequence of states via **HMM + rSLDS** (Prong 1).
- **Geometry:** **PHATE** (primary) + **CEBRA** (cross-area aligned, quantitative) +
  geometry metrics (Prong 2). **T-PHATE optional** (paper inaccessible; unvalidated on
  spikes) -> use repo docs only.
- **NeuralFlow (1D):** demonstrates the decision variable is attractor-like and shared;
  tuning geometry is area-specific. Not the cross-area link.
- **Unit selection:** **Multiscale Relevance (MSR)**, unsupervised, cross-checked
  against task selectivity.
- **Directed flow (stretch):** **Python-native only** — reduced-rank-regression
  communication subspace + **MRM-GP**; DLAG (MATLAB) is NOT ported.
- **ELA / nested basins:** optional structural follow-up, not
  core.
- **Language:** Python. **Compute:** CPU-first.

**Pending (not blocking Phase 0/1):**
1. **Dataset pivot (A vs B).** Default: **IBL 000409 primary**, **SHIELD 001051** as a
   replication. Flip to SHIELD-primary *only if* the core claim requires many areas
   recorded **simultaneously in one session** (IBL has ~2 probes/session; SHIELD up to
   6). See project plan "Dataset decision".
2. **GPU at the event** (human will confirm) -> gates AutoLFADS + large CEBRA.

---

## 4. Constraints

~5 usable days; team ~3 (1 PhD + possible MSc); Python only; CPU-first (GPU TBD);
data are NWB on DANDI. **Single-trial methods (HMM, rSLDS, geometry, flow) require
simultaneously recorded populations** -> work per session, per region; do **not** pool
cross-session pseudopopulations except for final visualization.

---

## 5. The analyses (compact)

- **Prong 1 — sequence of states:** HMM + rSLDS aligned to stimulus onset; relate
  state/regime sequence and transition timing to behaviour (reaction time; correct vs
  error); read cross-area latency structure as distributed flow.
- **Prong 2 — area-specific geometry:** PHATE (+ optional T-PHATE) and CEBRA; quantify
  with participation ratio / intrinsic dimensionality / curvature vs a shuffle null.
  Anchors: Fortunato et al. (bioRxiv 2023.07.18.549575, manifold nonlinearity differs
  by region); Langdon/Genkin/Engel 2023.
- **Support:** NeuralFlow (attractor demo); AutoLFADS (single-trial confirmation;
  optional/GPU); RRR + MRM-GP (directed flow; validate vs Allen connectivity); ELA
  (optional).
- **Headline claim:** *same latent attractor dynamics, geometrically distinct
  area-specific realizations.*

---

## 6. First test dataset — IBL 000409 access (verified from IBL docs)

*Requires an IBL/openalyx + DANDI account and ONE-api credentials (Phase 0). Run in an
environment that can reach the IBL/AWS servers.*

```python
from one.api import ONE
from brainwidemap import bwm_query, bwm_units
from brainbox.io.one import SpikeSortingLoader

one = ONE(base_url='https://openalyx.internationalbrainlab.org')

bwm_df  = bwm_query(one)      # released sessions + probe insertions (pid, eid, ...)
unit_df = bwm_units(one)      # analysis-grade units w/ region (Beryl acronym) + coords

# all insertions hitting a region (Allen parcellation is hierarchical):
ins = one.search_insertions(atlas_acronym='VISp',
                            datasets='spikes.times.npy', project='brainwide')

# load spikes for one insertion; per-unit region is clusters['acronym']:
pid = ins[0]
sl = SpikeSortingLoader(pid=pid, one=one)
spikes, clusters, channels = sl.load_spike_sorting()
clusters = sl.merge_clusters(spikes, clusters, channels)
# also load the session `trials` table: stim side/contrast, choice, feedback,
# first-wheel-move time -> for stimulus-onset alignment and the behaviour link.
```

**SHIELD 001051 alternative (if promoted):** stream NWB from DANDI via `pynwb` +
`remfile` + `h5py`, or use the AllenSDK path. DRAFT dandiset, 1.8 TB — stream
selectively, do not bulk-download.

---

## 7. Verified repositories & install

```bash
# Core
pip install numpy scipy scikit-learn pandas matplotlib

# IBL data
pip install ONE-api
pip install git+https://github.com/int-brain-lab/paper-brain-wide-map  # brainwidemap
# (brainbox / iblatlas via the IBL environment; verify exact install)

# Dynamics / sequences
pip install ssm         # Linderman lab: HMM, ARHMM, SLDS, rSLDS (verify; may need git)
pip install dynamax     # JAX HMM/LGSSM (maintained; SLDS limited)

# Geometry
pip install phate       # primary
pip install cebra       # cross-area aligned embedding (GPU optional)
pip install tphate      # OPTIONAL only

# Directed flow (Python-native)
# git clone https://github.com/WeihanLikk/MRM-GP     # MRM-GP (ICML 2024)
# reduced-rank regression: implement with numpy/scikit-learn

# NeuralFlow
pip install git+https://github.com/engellab/neuralflow

# Single-trial (OPTIONAL; GPU recommended)
# git clone https://github.com/arsedler9/lfads-torch
```

Repo URLs (all verified in prior sessions): ssm `github.com/lindermanlab/ssm`;
dynamax `github.com/probml/dynamax`; PHATE `github.com/KrishnaswamyLab/PHATE`;
T-PHATE `github.com/KrishnaswamyLab/TPHATE`; CEBRA
`github.com/AdaptiveMotorControlLab/CEBRA`; MRM-GP `github.com/WeihanLikk/MRM-GP`;
NeuralFlow `github.com/engellab/neuralflow`; AutoLFADS
`github.com/arsedler9/lfads-torch`; IBL BWM `github.com/int-brain-lab/paper-brain-wide-map`.
No official MSR package located -> implement from Cubero, Marsili & Roudi 2020
(J Comput Neurosci 48:85-102).

---

## 8. Proposed repo structure

See the master project plan for the full tree. Top level:

```
ibl-decision-dynamics/
  config/ (areas.yaml, params.yaml)
  src/data/ (access.py, select_areas.py, build_matrices.py)
  src/selection/ (msr.py)
  src/dynamics/ (hmm.py, rslds.py, neuralflow_run.py, lfads_run.py)
  src/geometry/ (embeddings.py, metrics.py)
  src/flow/ (comm_subspace.py, mrmgp.py)
  src/analysis/ (behavior_link.py, figures.py)
  notebooks/00_data_smoketest.ipynb
  results/
```

---

## 9. Start here — first Claude Code tasks (Phase 0 -> Phase 1)

Suggested opening instruction to Claude Code:

> "Read the project plan and this index. Scaffold the repo structure above with a
> `requirements.txt` (Python, CPU-first). Then implement `src/data/access.py` and a
> `notebooks/00_data_smoketest.ipynb` that: (1) authenticates ONE-api to openalyx;
> (2) pulls `bwm_units` and prints good-unit counts per region; (3) loads ONE probe
> insertion's spikes + the session `trials` table; (4) builds a
> (trials x time x neurons) matrix for one region aligned to stimulus onset; (5) fits
> a quick HMM (via `ssm`) end-to-end as a pipeline smoke-test. Flag anything that needs
> credentials or that you cannot verify. Do not fabricate APIs — check the installed
> package if unsure."

Then **Phase 1 (area selection):** from `unit_df`, compute good-unit yields per region
and co-occurrence across the candidate axis (VISp, LP/LGd, MOs, CP, SCm/MRN, GRN);
pick 2-3 (up to ~6) regions/pairs with the best **simultaneous** yields; freeze in
`config/areas.yaml`.

Full phased checklist and success criteria: master project plan, "Phased plan" and
"Success criteria".

---

## 10. References

Full categorized list: **running notes** doc. Most-core (all in the human's project
knowledge unless noted): IBL et al. 2025 (Nature 645:177-189); IBL 2021 (eLife
e63711); Genkin/Shenoy/Chandrasekaran/Engel 2025 (Nature 645:168-176); Langdon/Genkin/
Engel 2023 (Nat Rev Neurosci 24:363-377); Linderman et al. 2017 (AISTATS); Zoltowski/
Pillow/Linderman 2020 (ICML); Glaser et al. 2020 (NeurIPS); Schneider/Lee/Mathis 2023
(Nature 617:360-368; CEBRA); Fortunato et al. (bioRxiv 2023.07.18.549575); Semedo et
al. 2019 (Neuron 102:249-259); Cubero/Marsili/Roudi 2020 (J Comput Neurosci 48:85-102;
MSR); Oh et al. 2014 (Nature 508:207-214); Harris et al. 2019 (Nature 575:195-202);
Steinmetz et al. 2019 (Nature 576:266-273). Not in project knowledge (to add / partly
inaccessible): Moon et al. 2019 (PHATE); Busch et al. 2023 (T-PHATE; inaccessible);
Pandarinath et al. 2018 + Keshtkaran et al. 2022 (LFADS/AutoLFADS); Li et al. 2024
(MRM-GP, ICML). Dataset for the alternative: SHIELD — Bennett et al., Neuron 2024
(DANDI:001051).
```
