# 2026-07-12 — New machine: conda/VSCode setup, pynapple adopted, requirements gap fixed

First session on the second laptop (fresh `git pull`, no chat history — exactly the
handoff scenario the 2026-07-11 entry described; `CLAUDE.md` + this folder carried the
state, as intended).

---

## 1. Environment on the new machine

Miniconda installed; env created as **`neuroDataReHack`** (camelCase) — note the docs
everywhere say `neurodatarehack` (lowercase). **Conda env names are case-sensitive**, so
copy-pasting `conda activate neurodatarehack` from the docs fails on this machine.
*Open: rename the env, or update the docs. Not yet decided.*

`conda` was not on PowerShell's PATH (miniconda does not add it). Two changes, both
CurrentUser scope, both reversible:

1. `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` — was effectively `Restricted`,
   which would have blocked the conda hook from ever running at shell startup.
2. `conda init powershell` — wrote the hook into
   `~/Documents/WindowsPowerShell/profile.ps1` (the **AllHosts** profile, so it loads in
   VSCode's integrated terminal too).

Verified in a fresh shell: `conda activate neuroDataReHack` → `python` resolves to
`…/envs/neuroDataReHack/python.exe`, Python 3.11.15.

---

## 2. pynapple — adopted, with a bounded role

Presented at the hackathon; evaluated and **kept**.

**What it is.** A time-series data model (`Ts` / `Tsd` / `TsdFrame` / `TsGroup` +
`IntervalSet`), where each object carries its own *time support*. Gives rasters, PSTHs,
tuning curves and cross-correlograms cheaply, and removes a class of epoch-alignment bugs.

**What it is not.** It does **no state-space modelling** — no HMM, no rSLDS, no manifold
geometry. The scientific core stays `hmmlearn` (later `ssm`/`dynamax`). pynapple is a
wrangling + basic-analysis layer.

**Decision: use it for exploration/plots and as an independent check on
`src/data/build_matrices.py` — do NOT rewrite the working HMM pipeline around it.**
`build_matrices.py` is verified and feeding real Phase 2 results; refactoring it mid-project
buys elegance and costs days.

**The check that makes it worth having.** We hand-rolled `bin_spikes_by_event`,
`region_tensor_from_probes`, `firing_rate_table`, `drop_silent_units` and never validated
them against an outside implementation. pynapple's `TsGroup.count()` covers the same
ground, so it is a real second opinion. Verified against the installed v0.11.3 (not from
memory):

| Property | Result |
|---|---|
| `TsGroup.count(bin, ep)` vs `np.histogram` | **bit-for-bit identical**, per bin |
| dtype | `int64` — **raw counts**, safe for PoissonHMM (never smooth before fitting) |
| shape | `(time, neurons)` — matches our `(trials, time, neurons)` convention |
| API present | `Ts`, `Tsd`, `TsdFrame`, `TsGroup`, `IntervalSet`, `compute_perievent`, `compute_1d_tuning_curves`, `compute_crosscorrelogram` |

**Dependency risk checked first** (this is what killed `ssm` in Phase 0): `pip install
--dry-run` showed the install is **purely additive** — no downgrade of `numpy` (2.3.5),
`numba` (0.66.0), `pandas` (3.0.3) or `scipy`. Confirmed post-install: `hmmlearn.PoissonHMM`
and ONE-api still import.

Honest cost: it is not featherweight — pulls `pynwb`, `hdmf`, `neo`, `xarray`, `rich`.
No conflicts, just bulk.

**Next use:** assert our Phase 2 count tensor equals the pynapple one on the working
session (`5c0c560e`). If they diverge, `build_matrices.py` has a bug and the Phase 2
numbers are suspect.

---

## 3. Bug found: `requirements.txt` was missing the SHIELD deps

The Phase 1 entry (2026-07-08) states *"Env additions: `remfile`, `h5py` (SHIELD
streaming)"* — **but neither was ever added to `requirements.txt`.** They were installed
ad-hoc on the first laptop only. On this fresh machine `remfile` was absent, so any SHIELD
streaming code would have failed with `ImportError`.

`h5py` happened to be present, but only as a *transitive* dependency of pynapple — pure
luck, and fragile.

Both are now pinned explicitly in `requirements.txt`, alongside `pynapple`.

**Lesson:** an env addition is not done until it is in `requirements.txt`. The worklog
saying so is not the same as the file saying so.

---

## 4. Env renamed to lowercase (`neurodatarehack`)

Done — the docs were right, the env was wrong. Two Windows-specific traps worth recording:

- **Case-only renames collide on Windows.** NTFS is case-insensitive, so
  `conda rename neuroDataReHack -> neurodatarehack` sees the destination as *already
  existing*. Route through a temp name: `-> tmp_ndrh -> neurodatarehack`.
- **conda could not delete the old env** because two matplotlib `.ttf` files were held open
  by two live `python.exe` processes (VSCode's Python/Jupyter). The leftover directory then
  blocked the rename. Kill the env's python processes first, remove the stale dir, then
  rename. Verified afterwards that matplotlib still renders and every package survived.

Jupyter kernel re-registered as `neurodatarehack`. **VSCode's selected interpreter still
points at the old path — re-select it** (Ctrl+Shift+P -> Python: Select Interpreter).

## 5. pynapple plot catalogue -> `src/viz/nap_plots.py`

Six plots, all routed through `src/viz/style.py` (Okabe-Ito / cividis / RdBu_r), demoed on
synthetic IBL-like data -> `results/figures/pynapple_demo/`:

| # | Plot | Why it earns its place here |
|---|---|---|
| 1 | peri-event raster, **trials sorted by RT** | separates stimulus-locked from movement-locked: a sensory response stays a vertical band, a motor one bends with the RT curve |
| 2 | PSTH per region (mean ± SEM) | the latency *ordering* across areas — first check on any propagation claim |
| 3 | **neuron × time heatmap, sorted by peak latency** | the display version of the metastable-sequence claim |
| 4 | contrast tuning | IBL's actual stimulus variable; sensory areas track \|contrast\|, motor areas should not |
| 5 | **SCm↔MRN cross-correlogram** | model-free first pass at the feedback hypothesis |
| 6 | autocorrelogram grid | QC: no refractory dip ⇒ merged multi-unit ⇒ Poisson-HMM states on it are suspect |

**Honesty flag on #3.** Sorting neurons by their *own* peak latency manufactures a diagonal
even from pure noise. It is a display, not evidence — the evidence is the HMM. Sort on
held-out trials before it goes anywhere near a figure in the write-up.

### pynapple API gotchas (found by running it, v0.11.3)

1. **`nap.Ts` silently re-sorts unsorted timestamps** (only a `UserWarning`). Passing
   RT-ordered event times to `compute_perievent` therefore produced a **silently
   mislabelled raster** — the trials came back in *time* order. Always pass events in time
   order and reorder the *rows* instead. This is a quiet-wrong-answer trap, not a crash.
2. **`compute_crosscorrelogram(..., ep=None)` raises `TypeError`**, despite the signature
   declaring `ep: Optional[IntervalSet] = None`. The kwarg must be *omitted*, not passed
   as `None`.
3. **`nap.compute_perievent_continuous` does not exist** in 0.11.3.

### Palette issue surfaced

MRN is Okabe-Ito **yellow** (`#F0E442`). As a thin line on white it is very low contrast —
a legibility problem for *all* viewers, independent of colour-vision deficiency. Also
learned the hard way: overlays must never reuse a region colour (the movement-onset line
was drawn in SCm's own vermillion and vanished into SCm's spikes). Overlays are now black.
*Open: restyle MRN, or stroke yellow lines with a dark outline?*

## 6. ⚠️ The blank CV plot was the Phase-2 result being wrong — "K=2" is RETRACTED

The `# CV curves` cell rendered a figure with a title, axes and a **full legend over no
data** (axes stuck at matplotlib's default ±0.05 limits). I first concluded this was a
*stale output from a broken run*. **That conclusion was wrong**, and the guard I added to
"fix the cosmetics" is what proved it — on a real run it fired immediately:

```
RuntimeError: VISp: mean_test_ll is non-finite at K=[2,3,4,5,6,7]
```

### What actually happened

`mean_test_ll` was **`-inf` for every K**, in every region:

```
 n_states  mean_test_ll  std_test_ll  mean_train_ll
        2          -inf          NaN      -4.821953
        3          -inf          NaN      -4.674429
        ...
argmax over all -inf -> index 0 -> K = 2
```

**`-inf` is non-finite but it is not NA**, so `idxmax` does not raise — it happily returns
**index 0**, i.e. the *first* entry of `CV_KS`, which was 2. **"CV prefers K=2 for all four
regions" was never a model-selection result.** It was argmax over a column of `-inf`. My
earlier reasoning ("NaN would make idxmax raise, therefore the outputs must be stale") was
defeated by exactly this: I checked NaN and forgot `-inf`.

The *train* LL was finite and rose with K, which is why nothing else looked broken.

### Root cause

`hmmlearn` fits `lambdas_ == 0` for a (state, neuron) pair that never spiked in training. A
Poisson with rate exactly 0 gives probability 0 — log-likelihood `-inf` — to **any** count
> 0. VISp has units with **~2 spikes in the entire window**; under the 250-trial CV
subsample such a unit is easily **silent across a whole training fold**, so every state
learns λ=0 for it, and one held-out spike kills the fold.

(`drop_silent_units` did not save us: it runs on the *full* trial set, before the CV
subsample, so an all-zero unit can still enter the CV tensor. Observed per-unit totals in
the VISp CV tensor: `[0, 1, 2, 2, 59, …]` — a unit with **zero** spikes.)

### Fix (`src/dynamics/hmm.py`)

1. `drop_units_silent_in()` — drop units all-zero in *this* tensor, after subsampling.
2. `LAMBDA_FLOOR = 1e-6`, applied via `_floor_lambdas()` after every fit. A rate of exactly
   zero is a degenerate MLE; 1e-6 spikes/bin is far below anything observable at 20 ms bins,
   so it cannot manufacture structure — it only prevents `-inf`.
3. `cross_validate_states` now **raises** if every K is non-finite, instead of taking an
   argmax over garbage.
4. `best_k` no longer travels in `df.attrs` (**pandas drops `attrs` through most
   operations**); callers recompute it from the table.

### The corrected picture (and it is NOT settled)

Held-out LL is now finite and **rises monotonically with K** — the opposite of the retracted
"parsimonious K=2" story. But the argmax landed on **K=7, the top of the search range**: a
boundary, not a choice. `CV_KS` is now `[2…10]` with an explicit boundary warning.
**If it still has not plateaued by K=10, that is evidence the approach is wrong, not a
licence to raise K further** (user's call, and the right one).

> **Two standing lessons.**
> 1. *A plot that silently draws nothing is a bug reporting itself.* The blank figure WAS
>    the `-inf`. It was read as a rendering glitch for a full session.
> 2. *`-inf` is not NaN.* Guard model-selection code against **non-finite**, not just NA —
>    and never take an argmax without checking the winner is finite and not on a boundary.

## 6b. The binning is CORRECT — pynapple cross-check passed

The reason pynapple was adopted, and it paid off by *narrowing* the bug. On the real session:

| region | ours | pynapple | per-(trial,bin) identical |
|---|---:|---:|---|
| VISp | 42,085 | 42,085 | **yes** |
| SCm | 500,404 | 500,404 | **yes** |
| MRN | 445,569 | 445,569 | **yes** |

~1M spikes, agreeing on every bin. So `build_matrices.py` is sound and the tensor feeding the
HMM was never the problem — the bug was confined to CV scoring.

## 6c. First look at the real correlograms

- **SCm↔MRN cross-correlogram: a sharp peak at ZERO lag** on a broad elevated baseline
  (~1.09 → 1.11). That is the signature of **common drive**, not a directed SCm→MRN
  exchange. It does not refute the feedback hypothesis — HMM states live at 20 ms–100s of ms,
  a different timescale from this millisecond measure — but the **spike-timing evidence does
  not support a lead–lag**, and the joint multi-region model will have to carry the argument
  on its own. Worth knowing *before* Phase 3.
- **VISp autocorrelograms: clean refractory dips** on all inspected units. The units are
  well-isolated and our loading path preserved that, so merged-unit over-dispersion is not a
  source of spurious HMM states.
- **PSTH**: MRN shows an early transient (~150 ms) while SCm ramps and stays elevated; VISp
  is genuinely sparse (~2–4 Hz).

## 7. VISpm dropped from the notebook

`REGIONS = ["VISp", "SCm", "MRN"]`. VISpm was a stand-in for a "higher-order" tier that did
not actually supply one — it added a region without adding a hierarchy, and it was the
weakest link in the story. The replicable IBL axis is the triplet (5 sessions). A real
visual→higher-order→motor hierarchy remains an argument for SHIELD.

## 8. Colour: outline lines, do NOT use the global rcParam

Okabe-Ito is built for **hue** separation under red-green deficiency, not **luminance**
contrast against white — its yellow (`#F0E442`, = MRN) is nearly invisible as a thin line.
A black outline fixes it *without changing a single hue*, so the CVD-safe property is
preserved exactly.

**Tried the obvious thing and it was wrong:** matplotlib's global `path.effects` rcParam
strokes **text** as well, thickening every glyph until titles, ticks and legends are bold
and smeared. Rendered it, looked at it, reverted. The outline is now per-artist via
`style.stroke()`; only *marker* edges are set globally (safe).

## 9. `.conda` deleted — and a correction

An empty **Python 3.14** env (only `pip`, 168 MB, 4,888 files) that VSCode's "Create
Environment" dropped in the repo root. Deleted.

**Correction to my own earlier claim:** I said it was a git-bloat risk. **It was not** —
conda writes a `.gitignore` containing `*` *inside* the env dir, so it self-ignores and
`git status` never saw it. The real hazard was different: **VSCode prefers a workspace-local
`.conda` over any named env** when selecting an interpreter, so it could silently hijack the
kernel and make the project look broken.

## Open

1. MRN is still Okabe-Ito yellow; the outline makes it legible, but a darker hue would be
   better if a slot frees up.
2. Freeze `config/areas.yaml::selected` (now clearly `{VISp, SCm, MRN}`).
3. IBL-vs-SHIELD; Phase 3 joint multi-region HMM/rSLDS for SCm↔MRN.
4. Re-run Phase 2 at `MAX_TRIALS=None` and **read the CV curve, not the argmax** — if the
   held-out LL is flat across K, "K=2 for every region" is noise, not a result.
