# NWB (DANDI:000409) vs ONE: are they the same data?

*Written 2026-11-08. Session `5c0c560e` (`sub-NR-0028`). Regenerate with*

```bash
python scripts/compare_nwb_vs_one.py --one-alf-dir data/ONE/steinmetzlab/Subjects/NR_0028/2023-03-07/001/alf
```

*which writes `results/nwb_vs_one/5c0c560e.json`. The first run streams the spike slices it
needs (about 10 minutes); `data/nwb_cache/` makes later runs faster.*

**Short answer.** The spike data are identical. The trial tables are not quite. Stimulus-aligned
tensors built from NWB are bit-identical to the frozen ONE tensors on every trial both routes keep,
but each route keeps one trial the other drops. Movement-aligned tensors differ slightly
everywhere, because movement-onset times differ by about 0.4 ms. Every result in this repository
comes from the ONE route.

## How the comparison is made

`src/data/load_nwb.py` streams the processed NWB file (DandiAPIClient, remfile, h5py, pynwb) and
returns the same `(trials, mask, probe_loads)` as the ONE route. The trials mask is computed by
`brainwidemap.load_trials_and_mask` itself. The tensors then go through the **unchanged**
`preprocess_session` with the configuration of the frozen files (pre 0.3 s, post 0.8 s, 40 ms bins,
units at least 1 Hz). Trials are paired by their original index and units by their original position
in the region, so a trial or unit kept by only one route is reported, not mis-paired.

## Units and spikes: identical

| Check | Result |
|---|---|
| Units in the NWB units table with the paper's IBL `cluster_uuid` | 2,962 of 2,962 |
| Paper "good" units (`label == 1`) present in NWB (`ibl_quality_score == 1`) | 277 of 277 |
| Spike times of 8 randomly drawn good units | bit-identical |
| Units per region after preprocessing, stimulus-aligned (VISp / SCm / MRN) | 19 / 44 / 54 in both |

## Trial tables: three kinds of difference

| Event | Trials differing (of 1,100) | Size |
|---|---|---|
| Stimulus onset | 1 (trial 66) | 0.66 s |
| Movement onset | 1,098 (every trial with a value) | median 0.4 ms, 7 trials above 1 ms, max 2.4 ms |
| Feedback | 1 (trial 1099) | about 1,336 s |
| Go cue, choice, feedback type, contrasts, block probability, trial start | 0 | identical |

The two large differences are **anomalies in the ONE table**, judged by internal consistency:

- **Trial 66.** In IBL data the stimulus appears about 1 ms before the go cue (median −1.0 ms over
  the session). NWB's stimulus onset is 0.9 ms before this trial's go cue. ONE's is 0.66 s before
  it, which also gives a reaction time of 2.63 s and excludes the trial from the analysis.
- **Trial 1099.** NWB's feedback time is 0.1 ms after the response, as on other trials. ONE's is
  about 22 minutes later, after the trial's own end time.

The 0.4 ms movement-onset shift is systematic, which suggests the two tables come from different
versions of the wheel-movement detection. Which version is better cannot be decided from the data
here.

## Effect on the tensors

The paper's mask keeps 843 trials in both routes, but not the same 843: trial 66 passes only in NWB
(reaction time 1.97 s rather than 2.63 s) and trial 827 passes only in ONE (reaction time 1.9999 s
in ONE, 2.0003 s in NWB, either side of the 2 s limit). The 842 shared trials compare as follows:

| Alignment | Region | Units compared | Count entries differing | Trials with any difference | Largest difference |
|---|---|---|---|---|---|
| Stimulus onset | VISp | 19 | 0 % | 0 | 0 |
| Stimulus onset | SCm | 44 | 0 % | 0 | 0 |
| Stimulus onset | MRN | 54 | 0 % | 0 | 0 |
| Movement onset | VISp | 20 | 0.45 % | 596 | 1 spike |
| Movement onset | SCm | 43 | 2.41 % | 842 | 2 spikes |
| Movement onset | MRN | 54 of 55 | 1.63 % | 842 | 2 spikes |

Movement-aligned, one MRN unit sits at the 1 Hz threshold: ONE keeps it, NWB does not. Trial
labels (choice, feedback type, contrasts) are identical on every shared trial in both alignments.

## What this means for the results

- Stimulus-aligned analyses would differ by one trial out of 843 (trial 66 in, trial 827 out). That
  is far below the trial-to-trial variability every statistic here is tested against.
- Movement-aligned tensors shift a few percent of spikes between neighbouring 40 ms bins. That is
  well below the GPFA timescales (107–283 ms) and the HMM state durations, so no conclusion should
  change; this has not been re-tested by refitting.
- The frozen results stay on the ONE route because it is what the Brain-Wide-Map paper used. The
  NWB route is a documented alternative, not a replacement.
- The two ONE anomalies are worth reporting to IBL. They do not affect the paper's conclusions, but
  they are exactly the kind of difference a format conversion should surface.
