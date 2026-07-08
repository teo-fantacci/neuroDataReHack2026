# Worklog

Running record of what we did, what we found, and *why* we chose what we chose.
One dated file per working session. Read newest-first.

The point of this folder is that a decision made on day 1 and forgotten by day 4 is a
decision that gets silently re-litigated. If something here surprised us, it is written
down with the evidence that surprised us.

Distinct from [`../project_notes/`](../project_notes/), which holds the pre-hackathon
scientific plan and does not change.

| Date | Entry | Summary |
|---|---|---|
| 2026-07-08 | [Phase 0 setup](2026-07-08_phase0-setup.md) | Env, public openalyx access, two `brainwidemap` bugs, repo scaffold, download budget |

## Standing decisions

Carried from `project_notes/`, plus what we've settled since:

| Decision | Status | Where |
|---|---|---|
| Python only, CPU-first | from plan | `project_notes/` |
| IBL 000409 primary; SHIELD 001051 replication only | from plan | `project_notes/` |
| Backbone = HMM + rSLDS | from plan | `project_notes/` |
| **Phase 0 HMM uses `hmmlearn.PoissonHMM`, not `ssm`** | 2026-07-08 | [entry](2026-07-08_phase0-setup.md) |
| **`ssm` deferred to Phase 2; only rSLDS needs it** | 2026-07-08 | [entry](2026-07-08_phase0-setup.md) |
| **Aggregate tables pinned to `2024_Q2_IBL_et_al_BWM`** | 2026-07-08 | [entry](2026-07-08_phase0-setup.md) |
| **No IBL account needed; openalyx is public** | 2026-07-08 | [entry](2026-07-08_phase0-setup.md) |
| Region set frozen in `config/areas.yaml` | **open** — Phase 1 | — |
| GPU availability at Janelia | **open** — ask organizers | `project_notes/Janelia pre-hack meeting.md` |
