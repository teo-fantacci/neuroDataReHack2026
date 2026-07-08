# IBL Decision Dynamics

**NeuroDataReHack 2026 — HHMI Janelia, 11–17 July 2026**

Decision-making as a **temporal sequence of attractor / metastable states**, unfolding
from stimulus onset across simultaneously recorded brain areas — and the claim that the
**same shared latent decision dynamics** are realized on **geometrically distinct,
area-specific manifolds**.

Dataset: IBL Brain-Wide Map ([DANDI:000409](https://dandiarchive.org/dandiset/000409);
IBL et al. 2025, *Nature* 645:177–189).

> Framing guardrail: *distributed temporal sequences within populations, not a clean
> anatomical relay.*

## Approach

| Prong | Question | Methods |
|---|---|---|
| **1 — Dynamics** | What sequence of states does the population traverse? | HMM (`hmmlearn`), rSLDS (`ssm`, Phase 2) |
| **2 — Geometry** | Is that sequence realized on area-specific manifolds? | PHATE, CEBRA, participation ratio / intrinsic dim vs shuffle null |
| **Support** | Is the decision variable attractor-like? | NeuralFlow (1D) |
| **Stretch** | Directed inter-areal flow? | Reduced-rank regression (communication subspace), MRM-GP |

## Setup

```bash
conda create -n neurodatarehack python=3.11 -y
conda activate neurodatarehack
pip install -r requirements.txt
```

**No IBL or DANDI account is required.** ONE-api ships public `openalyx` credentials as
defaults. Verified 2026-07-08 — see the [worklog](docs/worklog/2026-07-08_phase0-setup.md).

## Quick start

```bash
jupyter lab notebooks/00_data_smoketest.ipynb
```

The smoke-test goes openalyx → frozen unit set → simultaneous-region selection →
stimulus-aligned tensor → Poisson HMM → state sequence vs reaction time.

Its download cell is **gated behind `DOWNLOAD = False`**; flipping it fetches ~150–280 MB
per probe insertion (a good-units-only spike table, not raw ephys).

## Layout

```
config/     areas.yaml   candidate regions + selection thresholds
            params.yaml  binning, HMM settings, reproducibility pins
src/data/   access.py         ONE-api + brainwidemap wrappers
            build_matrices.py spikes -> (trials x time x neurons)
notebooks/  00_data_smoketest.ipynb
docs/project_notes/  pre-hackathon scientific plan (static)
docs/worklog/        what we did and why (running record)
results/
```

## Reproducibility

Analyses are pinned to the published data freeze `2023_12_bwm_release`, with aggregate
tables pinned to the matching tag `2024_Q2_IBL_et_al_BWM`. `bwm_units` asserts a hash of
the unit UUIDs, so we analyse **exactly** the paper's 62,990 units across 210 Beryl
regions. Two upstream `brainwidemap` bugs required a workaround; both are documented in
the [worklog](docs/worklog/2026-07-08_phase0-setup.md).

## Status

- [x] **Phase 0** — env, public data access, repo scaffold, smoke-test
- [ ] **Phase 1** — area selection; freeze the region set in `config/areas.yaml`
- [ ] **Phase 2** — HMM + rSLDS state sequences; link to behaviour
- [ ] **Phase 3** — geometry (PHATE / CEBRA) vs shuffle null
- [ ] **Phase 4** — NeuralFlow; directed flow (stretch)
- [ ] **Phase 5** — write-up

## Key references

IBL et al. 2025, *Nature* 645:177–189 · Genkin, Shenoy, Chandrasekaran & Engel 2025,
*Nature* 645:168–176 (NeuralFlow) · Langdon, Genkin & Engel 2023, *Nat Rev Neurosci*
24:363–377 · Linderman et al. 2017, AISTATS (rSLDS) · Schneider, Lee & Mathis 2023,
*Nature* 617:360–368 (CEBRA) · Semedo et al. 2019, *Neuron* 102:249–259.

Full list in [`docs/project_notes/`](docs/project_notes/).
