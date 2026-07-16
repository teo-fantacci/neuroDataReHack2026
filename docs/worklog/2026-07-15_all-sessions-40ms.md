# 2026-07-15 — All 5 triplet sessions downloaded & preprocessed at 40 ms (both alignments)

New: `scripts/build_processed_all.py`. Confirms the 40 ms gate, then produces the full
replication set. This is the note the **other chats** should read before assuming what is in
`data/processed/`.

---

## 1. ✅ 40 ms passes the ground-truth gate — re-verified today, not just trusted

The 2026-07-14 §6.1 caveat ("40 ms is untested against the rSLDS ground truth; verify before
adopting") is now closed with a fresh run of `scripts/validate_rslds_ground_truth.py`
(→ `results/rslds/ground_truth_binsize.json`). Truth: 3 attractors, τ = 150 ms, generated in
**continuous** time, binned after — so 20/40/50 ms are the *same* truth, only the observation
differs. Fair setting (σ fixed at 40 ms in **ms**, which is what we actually use):

| bin | regime acc | switches (true 2.0) | spurious integrators | order consistency |
|---|---:|---:|---:|---:|
| 20 ms | 0.69 | **4.91** | **3** | 0.749 |
| 30 ms | 0.86 | 2.18 | 1 | 0.997 |
| **40 ms** | **0.87** | **2.06** | **0** | **1.000** |
| 50 ms | 0.88 | 2.02 | 0 | 1.000 |

**40 ms recovers the correct switch count, zero spurious integrators, and perfect order
consistency** — clean, and comparable to 50 ms. 20 ms manufactures 3 integrators (the standing
reason it is banned). Reproduced across all config blocks in the JSON.

Combined with the 2026-07-14 §6.1 HMM check (SCm state-stability ARI **0.938 at 40 ms** vs
**0.826 at 50 ms**), **40 ms is validated for BOTH strands** and is the project-wide bin.

> ⚠️ `PreprocessConfig` still **defaults to `bin_size_s = 0.05`**. 40 ms is applied by the build
> scripts overriding it, not by the default. If the project has truly frozen 40 ms, the default
> should change — flagged, not done (it would silently re-tag every future `.npz`; a shared
> decision). For now: **always pass `bin_ms=40` / `--bin-ms 40`.**

## 2. What is now in `data/processed/`

`scripts/build_processed_all.py` downloads (ONE cache) and preprocesses **all five** triplet
sessions through the unified `src/data/preprocess.py`, at **40 ms, both alignments**
(`stimOn_times` + `firstMovement_times`), `pre 0.4 / post 0.8`, units ≥ 1 Hz, `qc == 1.0`.
Sequential, releasing each session's ~0.5 GB of spikes before the next (the machine is
memory-tight); per-session `try/except` so one failure does not sink the batch.

Target set (from `docs/knowledge/the_five_triplet_sessions.md`):

| eid8 | lab | usable trials | note |
|---|---|---:|---|
| 5c0c560e | steinmetz | 843 | most trials; weakest VISp |
| 90c61c38 | zador | 656 | 2nd-most trials; 1 probe |
| 111c1762 | churchland | 364 | re-built here at 40 ms (was 50 ms) |
| 90d1e82c | danlab | 353 | weakest MRN (19 units) |
| 8a3a0197 | churchland | 290 | fewest trials; most units (188) |

Filenames the analysis runners glob for:
`data/processed/<eid8>__stimOn_pre400_post800_bin40_rate1.npz` (and `__firstMovement_…`).

**Rebuild / extend:** `python scripts/build_processed_all.py --bin-ms 40`
(add `--eids 8a3a0197` to target one; `--force` to overwrite).

## 3. For the other chats

- **The 50 ms tensors are now superseded for the HMM.** 40 ms is the bin. `111c1762` exists at
  both; prefer bin40. The runners (`run_hmm_sessions.py`, `run_movement_alignment.py`) pin one
  bin via `--bin-ms`, so mixing cannot happen silently, but point them at 40 ms.
- **rSLDS**: the frozen 40 ms tensors it already uses for `5c0c560e` are the same pipeline; the
  other four sessions are now available at 40 ms for the replication the rSLDS worklog wants
  (`trials, not neurons` — these add trials).
- **Downloads are cached** in `data/ONE/` (gitignored). The five sessions total ~2 GB on disk.

## Open

1. Per-session **K comparison** (stability run on *each* session, not one session's K imposed on
   all) — `scripts/run_hmm_kselect.py`, feeding the chosen K back into `run_hmm_sessions.py`.
   This is the immediate next step and the reason the sessions were downloaded.
2. Decide whether `PreprocessConfig` default should become 40 ms (§1 flag).
3. Any session that OOM'd during the batch: retry when RAM frees (`--eids <that one>`).
