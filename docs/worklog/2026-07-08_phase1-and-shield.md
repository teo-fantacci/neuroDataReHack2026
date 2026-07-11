# 2026-07-08 — Phase 1 (area selection) + SHIELD investigation

Both are **metadata only**. Total neural data downloaded across everything below: **0 bytes**
of IBL spikes; **0.36 MB** streamed from one SHIELD NWB (HDF5 metadata, lazy reads).

Regenerate the IBL side with `python scripts/run_phase1.py` → `results/phase1_summary.md`.

---

## The question

The plan left one decision explicitly open: **IBL 000409 vs SHIELD 001051**, with the
criterion *"flip to SHIELD-primary only if the core claim requires many areas recorded
simultaneously in one session."* Phase 0 measured the IBL ceiling at 3 axis regions.
This entry turns that criterion from hypothetical into decided.

---

## Finding 1 — the 3-region ceiling is the *axis*, not IBL

I ran the co-occurrence sweep two ways: over our 8-region candidate axis, and over **all
210 Beryl regions**.

Candidate axis (≥20 good units/region):

| min_units | ≥2 reg | ≥3 reg | ≥4 reg | max |
|---:|---:|---:|---:|---:|
| 10 | 72 | 12 | 1 | 4 |
| 20 | 43 | 7 | 0 | **3** |
| 30 | 29 | 4 | 0 | 3 |

All 210 regions (same threshold):

| min_units | ≥4 reg | ≥6 reg | ≥8 reg | max |
|---:|---:|---:|---:|---:|
| 20 | 106 | 30 | 4 | **9** |
| 30 | 43 | 4 | 0 | 7 |

So IBL sessions *routinely* record 6–9 well-sampled regions simultaneously — just not the
visual→motor regions we pre-selected. Example (session `d33baf74`): **9 regions ≥20 units**
— PO:89, ORBvl:82, MOs:72, LD:68, ORBm:53, AON:51, VPM:31, DG:21, DP:20.

**This reframes the whole dataset decision.** The plan assumed IBL's ~2-probes/session
caps simultaneity. It does cap it — but at ~6–9 regions, not 3. The "3" was an artifact of
insisting on one specific anatomical axis that the IBL probe-placement grid simply doesn't
sample together. The honest options are:

1. **Keep IBL, drop the fixed axis.** Let probe geometry choose the regions: pick a session
   with rich simultaneous coverage and analyse whatever it recorded well. This *strengthens*
   the "distributed, not a relay" framing — we stop cherry-picking a pipeline.
2. **Keep IBL + the axis, accept triplets.** Most reliable axis triplet is
   **{MRN, SCm, VISp}: 5 sessions** (see below), so sessions become replicates.
3. **Pivot to SHIELD** (see Finding 2).

The recurring axis combinations (sessions as replicates):

| combination | sessions |
|---|---:|
| MRN + SCm | 23 |
| MRN + VISp | 7 |
| SCm + VISp | 7 |
| CP + MOs | 5 |
| **MRN + SCm + VISp** | **5** |

Only one triplet recurs meaningfully. `{MRN, SCm, VISp}` (midbrain / colliculus / V1) is a
defensible visual→motor sub-axis and gives 5 replicate sessions. If we stay on the axis,
this is the target.

Trial counts are not a constraint: median **395** usable trials/session, **446/459**
sessions have ≥200.

---

## Finding 2 — SHIELD is public, streamable, and much richer per session

The plan flagged SHIELD as "DRAFT, 1.8 TB, thinner tooling, under NIH-review banner."
Checked each:

- **Public.** DANDI:001051, `embargo_status: None`, version `draft`, status `Valid`.
  666 assets, **1.808 TB**, 27 subjects. Session NWBs ~2–2.6 GB; separate
  `_probe-N_ecephys.nwb` ~3 GB each.
- **Streamable with 3 lines.** `remfile.File(url)` → `h5py.File(...)`. Reading the full
  units + electrodes metadata of a **3.04 GB** 6-probe session cost **0.36 MB** on the
  wire. The "thinner tooling" concern is real but small: no ONE-api niceties, and the
  region label lives on `/general/extracellular_ephys/electrodes[location]`, joined to
  `/units` via `peak_channel_id` (units table itself has no region column).

One representative session (`sub-612090_ses-20220524T212631`, 6 probes):

- **3086 units, 2553 good** (quality label `good`).
- **40 areas**; **29 areas ≥20 good units**, **17 areas ≥50**.
- Top: SSp:304, VISp:201, PL:201, ACAv:150, ACAd:143, LSc:134, ILA:125, CP:117, MOs:112,
  SUB:111, POL:105, VISrl:72 …
- 598 trials, 332 change trials (go/no-go visual change detection).

So a *single* SHIELD session offers ~5× the simultaneous regional coverage of the best IBL
session, with hundreds of good units each. If the science wants many areas at once,
SHIELD delivers it in one session instead of stitched replicates.

### The cost of SHIELD

- Different task (**go/no-go change detection**, not IBL's 2AFC contrast task). The
  NeuralFlow "decision-variable" framing was built around 2AFC; it must be re-justified.
- No `brainwidemap` conveniences: unit QC, region rollup (Beryl), and trial masks are all
  hand-rolled. `quality == 'good'` exists but is not the same QC pipeline as IBL's.
- Draft dandiset → could change under us.
- 2–3 GB per session download vs IBL's ~0.15–0.26 GB per probe.

---

## Recommendation (for the human to decide)

The pivot criterion was "does the claim *require* many simultaneous areas." My read of the
evidence:

- **The claim does not strictly require it.** "Same dynamics, area-specific geometry,
  distributed not relayed" is demonstrable with **3 areas × 5 replicate IBL sessions**
  ({MRN, SCm, VISp}), on the published, low-friction, task-matched dataset. That is the
  safe Phase 2 path and I would start there.
- **But SHIELD is a far stronger *stretch* substrate** than the plan assumed — public,
  streamable, 17 areas ≥50 units in one session — and the tooling cost is a day, not a
  wall. It is the right replication/extension dataset, and if the {MRN,SCm,VISp} result
  lands early it is where the ambitious version of this project lives.

Net: **IBL {MRN, SCm, VISp} as primary for Phase 2; SHIELD promoted from "maybe replicate"
to "serious second substrate," de-risked and ready.** This matches the plan's default but
now rests on measured numbers.

Not yet frozen in `config/areas.yaml` — that is the human's call.

---

## Artifacts

- `scripts/run_phase1.py` → `results/phase1_summary.md`, `results/phase1_sessions.csv`
- `src/data/select_areas.py` — the reusable selection functions
- Env additions: `remfile`, `h5py` (SHIELD streaming); `jupyter`, `ipykernel`, `tabulate`

## Open questions

1. **Freeze the region set** — {MRN, SCm, VISp}, or let probe geometry pick? Human's call.
2. SHIELD QC: is `quality == 'good'` strict enough, or replicate IBL's metric thresholds?
3. Task mismatch: can the NeuralFlow decision-variable story survive go/no-go? (Phase 4.)
