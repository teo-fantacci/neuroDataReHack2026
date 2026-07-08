# 2026-07-08 — Phase 0: environment, data access, repo scaffold

Three days before the hackathon (starts 11 July, dinner). Goal: get Phase 0 **and**
Phase 1 finished *before* travelling, so the 5 days on site are analysis, not `pip install`.

---

## 1. Environment

Conda env **`neurodatarehack`**, Python 3.11.

```bash
conda create -n neurodatarehack python=3.11 -y
conda activate neurodatarehack
pip install -r requirements.txt
```

### Decision: drop `ssm` from Phase 0

The plan docs assumed `ssm` (Linderman lab) for both HMM and rSLDS. But **only rSLDS
needs `ssm`.** A Poisson HMM over binned spike counts needs nothing exotic.

`ssm` is Cython-built, effectively unmaintained, and pins old numpy — a real risk of
burning a day on a build failure. So:

- **Phase 0/1 HMM** → `hmmlearn` (verified: v0.3.3 ships `PoissonHMM`; pure pip; CPU).
- **`ssm` / rSLDS** → deferred to Phase 2, where a failure is survivable.
- Fallback if `ssm` won't build: `dynamax` (JAX), or stay with `hmmlearn` for the
  discrete-state story and drop the continuous-latent regime story.

Poisson emissions are the correct likelihood for spike counts, so `PoissonHMM` is not a
compromise — it is arguably the more principled default. **Feed it raw counts, never
smoothed data.**

---

## 2. openalyx access — no account required

**This was the biggest assumed schedule risk, and it does not exist.**

The plan docs flagged "create IBL/openalyx access and a DANDI account — not yet done."
In fact ONE-api ships the public mirror's URL *and* a shared login as defaults
(`one/params.py::default`):

```python
{'ALYX_URL': 'https://openalyx.internationalbrainlab.org',
 'ALYX_LOGIN': 'intbrainlab',
 'HTTP_DATA_SERVER': 'https://ibl.flatironinstitute.org/public'}
```

Password `international`. Verified end-to-end on 2026-07-08: constructed `ONE`,
made an authenticated REST call, and ran `search_insertions(atlas_acronym='VISp', ...)`
→ **71 insertions**. Zero registration.

The three layers, since they are easy to conflate:

| Layer | What it is | Holds |
|---|---|---|
| **Alyx** | IBL's experiment-metadata DB (Django REST) | subjects, sessions, insertions, dataset records |
| **openalyx** | public read-only Alyx mirror | same, public subset |
| **ONE-api** | the Python client | queries openalyx, fetches arrays from Flatiron, caches to disk |

Mnemonic: **openalyx = the catalogue. Flatiron = the warehouse. ONE-api = the clerk.**

A **DANDI account is still not needed** either, unless we pivot to SHIELD 001051.

---

## 3. Two bugs in `brainwidemap` (upstream, not ours)

Both are worked around in [`src/data/access.py`](../../src/data/access.py). Both are
worth reporting upstream.

### Bug A — `KeyError: 'trials'`

`download_aggregate_tables` defaults to `tag='2026_Q2_IBL_et_al_BWM'`
([`bwm_loading.py:377`]). Its `md5_expected` dict for that tag has keys for `clusters`
and waveform tables, but **no `'trials'` key** ([`bwm_loading.py:406-415`]). The trials
file downloads successfully (24.2 MB), then the integrity check does
`md5_expected[tag]['trials']` and dies.

So: the *download* works; the *checksum lookup* crashes.

### Bug B — the default tag silently breaks the paper freeze

Working around Bug A is not enough. `bwm_units(freeze='2023_12_bwm_release')` asserts a
hash of the unit UUIDs against the paper's frozen set. With the 2026_Q2 aggregates that
assertion **fails** — because the default aggregate tag no longer matches the default
freeze. The assertion is correct and protective: it says *these are not the paper's units.*

It would have been easy to "fix" this by passing `enforce_version=False`. That would have
silently given us a different unit set from the published analysis. **Don't.**

### The fix

Pin the aggregate tag to the one that matches the freeze:

```python
AGGREGATE_TAG = "2024_Q2_IBL_et_al_BWM"   # matches PAPER_FREEZE = "2023_12_bwm_release"
```

`access.py::_pin_aggregate_tag()` monkeypatches `bwm_loading.download_aggregate_tables`
to force this tag (idempotent; applied by `get_one()`).

**Verified:** the unit-UUID hash then matches
`d16d0b38d392b18c0ce8b615ec89d60d7c901df2eeb3432986b62130af28ef01`, i.e. exactly the
paper's units.

> Note: the local aggregate filename is derived from `type` only, not from `tag`
> (`bwm_tables/trials.pqt`). So if you ever switch tags, **delete `bwm_tables/` first**
> or you will silently reuse the other tag's cached file.

### Resulting numbers (verified)

| Quantity | Value |
|---|---|
| Insertions / sessions (`bwm_query`) | 699 / 459 |
| Analysis-grade units (`bwm_units`) | 62,990 |
| Beryl regions | 210 |
| Sessions after unit criteria | 454 (690 pids) |

These are *smaller* than the paper's headline "75,708 good units / 279 areas" because
`bwm_units` additionally applies the session- and trial-level criteria
(`min_qc=1.0`, `min_units_sessions=(5, 2)`, `rt_range`, `min_errors`). Not a contradiction.

---

## 4. Beryl vs Allen acronyms — a silent-failure trap

`clusters['acronym']` (from `load_good_units`) is the **fine-grained Allen** acronym:
`SCig`, `SCiw`, `VISp2/3`, `VISp5`. The `Beryl` column in `bwm_units` is a **coarser
rollup**: `SCm`, `VISp`.

Therefore `clusters[clusters.acronym == 'SCm']` matches **nothing** and fails silently
with an empty population. Map first:

```python
from iblatlas.regions import BrainRegions
br = BrainRegions()
clusters["Beryl"] = br.id2acronym(clusters["atlas_id"].values, mapping="Beryl")
```

Verified: `['SCig','SCiw','VISp2/3','VISp5','MRN'] -> ['SCm','SCm','VISp','VISp','MRN']`.

---

## 5. Candidate-axis yields (metadata only, verified)

Good units pooled over all sessions, for the axis in `config/areas.yaml`:

| Region | good units | sessions |
|---|---:|---:|
| CP | 2878 | 64 |
| MRN | 2643 | 99 |
| LP | 1780 | 58 |
| SCm | 1645 | 54 |
| MOs | 1270 | 36 |
| VISp | 844 | 35 |
| GRN | 555 | 14 |
| LGd | 551 | 22 |

**But pooled counts are the wrong statistic.** Single-trial methods need
*simultaneously recorded* populations, so what matters is co-occurrence within a session.

With thresholds ≥20 good units/region and ≥2 axis regions/session:
**43 qualifying sessions**, and **no session has more than 3** axis regions.

Best sessions:

| eid | n_reg | composition |
|---|---:|---|
| `8a3a0197-b40a-449f-be55-c00b23253bbf` | 3 | SCm:81, MRN:74, VISp:33 |
| `5d01d14e-aced-4465-8f8e-9a1c674f62ec` | 3 | CP:110, MRN:42, VISp:34 |
| `5c0c560e-9e1f-45e9-b66e-e4ee7855be84` | 3 | MRN:63, SCm:59, VISp:36 |
| `4a45c8ba-db6f-4f11-9403-56e06a33dfa4` | 3 | MRN:76, SCm:60, LGd:20 |

**Implication for the project.** The 3-region ceiling is exactly the ~2-probes/session
constraint the plan anticipated. It bites: the full `VISp→LP/LGd→MOs→CP→SCm/MRN→GRN`
axis is **never** recorded at once. Realistic targets are triplets, most commonly
`{VISp, SCm, MRN}` or `{VISp, CP, MRN}`. This is the evidence the "dataset pivot"
decision (IBL vs SHIELD) was waiting on. **Open question for Phase 1.**

---

## 6. Download budget (metadata only — nothing downloaded)

With `qc=1.0`, `load_good_units` sets `good_units=True`, which makes
`SpikeSortingLoader.load_spike_sorting` fetch `['passingSpikes', 'clusters', 'channels']`
([`brainbox/io/one.py:942`]) — a **good-units-only spike table**, not the full
`spikes.*` arrays and *not* the raw ephys.

For the best session (`8a3a0197…`, both probes, revision `2024-05-06`):

| Item | Size |
|---|---:|
| `passingSpikes.table.pqt` probe00 | 153.0 MB |
| `passingSpikes.table.pqt` probe01 | 262.3 MB |
| `clusters.*` + `channels.*` (both) | 26.1 MB |
| `_ibl_trials.*` (behaviour) | 0.1 MB |
| **Total, one session, both probes** | **≈ 442 MB** |

For scale, what we are **not** downloading from that same session:

| | |
|---|---:|
| Everything registered for the session | 138.1 GB |
| Raw ephys `.ap.cbin`, both probes | 96.3 GB |
| `waveforms.traces.npy` | 6.4 GB |

One-off metadata already fetched and cached: the BWM aggregate tables, ≈ 100 MB.

**Rule of thumb: ~150–280 MB per probe insertion.** All 43 qualifying sessions would be
roughly 19 GB — feasible, but not needed before Phase 1 decides the region set.

---

## 7. Repo scaffold

```
config/    areas.yaml (candidate axis + thresholds), params.yaml (bins, HMM, freeze pins)
src/data/  access.py (ONE + brainwidemap wrappers, tag pin), build_matrices.py (tensor)
notebooks/ 00_data_smoketest.ipynb   <- download gated behind DOWNLOAD = False
docs/worklog/
```

`00_data_smoketest.ipynb` runs: openalyx → frozen unit set → co-occurrence → *(gated
download)* → stimulus-aligned tensor → `PoissonHMM` → state sequence vs reaction time.

The HMM there is a **canary, not a result**. `n_states=4` is arbitrary; choosing K by
cross-validated log-likelihood is a Phase 2 job.

---

## Open questions

1. **Data cache location.** ONE defaults to `~/Downloads/ONE`. Move it if disk is tight.
2. **The 3-region ceiling** (§5) — does it force the SHIELD pivot, or do we commit to
   triplets? This is now a data-backed decision, not a speculative one.
3. **GPU at Janelia** — still unanswered; gates AutoLFADS and large CEBRA.
4. Report bugs A and B to `int-brain-lab/paper-brain-wide-map`.
