# DRAFT, not sent: two trial-table anomalies in IBL session 5c0c560e

*Status: draft written 2026-11-09, **not reported**. Before sending, confirm the right channel with
IBL (for example an issue on the Brain-Wide-Map code repository, or IBL's data-release contact) and
re-check the values against the latest data release. The evidence is in
[`nwb_vs_one_comparison.md`](nwb_vs_one_comparison.md); `scripts/compare_nwb_vs_one.py` reproduces it.*

---

**Subject:** Two trial-event values in session 5c0c560e (NR_0028, 2023-03-07) differ between ONE
and the DANDI:000409 NWB file, and look erroneous in the ONE trials table

Hello,

While comparing the ONE and DANDI (NWB) versions of a Brain-Wide-Map session, I found two trials
whose event times differ between the two sources. In both cases the ONE value is inconsistent with
the other events of the same trial, and the NWB value is consistent.

**Session.** eid `5c0c560e-9e1f-45e9-b66e-e4ee7855be84`, subject NR_0028, 2023-03-07. ONE trials
table `alf/_ibl_trials.table.pqt` as loaded by `brainwidemap.load_trials_and_mask`; NWB file
`sub-NR-0028_ses-5c0c560e-..._desc-processed_behavior+ecephys.nwb` on DANDI:000409 (draft, accessed
2026-11-08).

| Trial | Event | ONE | NWB | Other events of the trial |
|---|---|---|---|---|
| 66 | `stimOn_times` | 407.3987 s | 408.0589 s | go cue 408.0598 s (identical in both); trial start 405.9087 s |
| 1099 | `feedback_times` | 6542.6738 s | 5207.0891 s | response 5207.0890 s; trial end 5208.6656 s |

- **Trial 66.** Across this session the stimulus onset precedes the go cue by about 1 ms (median
  −1.0 ms). The NWB value is 0.9 ms before the go cue; the ONE value is 0.66 s before it. With the
  ONE value the reaction time becomes 2.63 s, so the Brain-Wide-Map mask (reaction time ≤ 2 s)
  excludes this trial; with the NWB value it is 1.97 s and the trial is kept.
- **Trial 1099 (the last trial).** The ONE feedback time is about 22 minutes after the response and
  after the trial's own end time. The NWB value is 0.1 ms after the response, as on other trials.

A smaller, systematic difference: `firstMovement_times` differs on every trial (median 0.4 ms,
maximum 2.4 ms), which suggests the two tables come from different versions of the movement-onset
detection. Through the 2 s reaction-time limit this flips one more trial (827). All other trial
fields (go cue, choice, feedback type, contrasts, block probability, trial intervals) are identical,
and spike times are bit-identical.

Happy to share the comparison script. Thank you for the data and the NWB release.

Teo Fantacci
teo.fantacci.job@protonmail.com
