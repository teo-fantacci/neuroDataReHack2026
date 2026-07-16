"""ONE preprocessing pipeline, so every model in the project eats the same data.

Why this module exists
----------------------
Until now each notebook built its own tensors, and they silently disagreed:

* ``01_hmm``      : window [-0.5, +0.8] s, 20 ms bins, units >= 1 Hz
* ``03_rslds``    : window [-0.2, +0.8] s (it read ``params.yaml``), 20 + 50 ms, ALL units
* ``02`` / ``04`` : window [-0.5, +0.8] s, 50 ms bins, ALL units

So "K from the HMM" was handed to an rSLDS fitted on a different window and a different unit
set, and nobody had written that down. This module is the single place the choice is made.

The decisions, and the evidence for each
----------------------------------------
Measured on session ``5c0c560e`` (843 usable trials) -- see ``notebooks/00_preprocessing.ipynb``,
which produces every number below and the figures behind them.

**pre_s = 0.4 s.** IBL enforces a *quiescence period* (the wheel must be held still) before
stimulus onset. In this session it is drawn from **[0.400, 0.700] s** (median 0.515). So:

* a **0.4 s** baseline window lies inside the enforced-still period on **100 %** of trials;
* the old **0.5 s** window steps outside it on **44 %** of trials, i.e. for nearly half the
  trials part of the "baseline" is a period in which the mouse was free to move the wheel.

Contamination from the *previous* trial is a non-issue: the minimum gap between the previous
feedback and this stimulus onset is **2.7 s**.

**post_s = 0.8 s -- but read the warning.** This window is NOT a decision-formation window.
Median reaction time in this session is **171 ms** and median feedback is at **427 ms**, so
over [0, +0.8] s movement has begun on **92 %** of trials and reward/error feedback has already
arrived on **83 %**. Most of the post-stimulus window is *movement and outcome*, not deciding.
This does not invalidate anything, but it means a late state is not automatically a "decision"
state, and the alternative alignment below exists for exactly that reason.

**align = "stimOn" (primary) and "firstMovement" (secondary).** IBL choice *is* the wheel-turn
direction, so stimulus-locked "decision" variance is entangled with movement direction. Build
both; disagreement between them is information, not a nuisance.

**bin_size_s = 0.04 (updated 2026-07-15).** Fixed by external results, none a free parameter of
the analysis it constrains. The rSLDS ground-truth (``scripts/validate_rslds_ground_truth.py``,
truth generated in continuous time so 20/40/50 ms compare the *same* attractors) recovers the
correct switch count with **zero spurious integrators at 40 ms**, while 20 ms manufactures three;
GPFA fits latent timescales of 107-283 ms, so 40 ms bins lose nothing the data contains. **40 ms
is preferred over 50 ms because it preserves SCm's HMM state-stability** (ARI 0.938 at 40 ms vs
0.826 at 50 ms -- at 50 ms there are only 24 bins and K=4 no longer fits reproducibly), at no
cost to any other read-out (``forward_fraction``/``switch_rate``/latency are flat 20->50 ms).
The old 0.05 default is superseded; existing 50 ms tensors are retained but not used for the HMM.
See ``docs/worklog/2026-07-15_all-sessions-40ms.md`` and ``2026-07-14_preprocessing-unification.md`` §6.

**min_rate_hz = 1.0.** The 1 Hz cut is nearly **free**: it keeps 19/44/54 units and
**96.7 / 99.5 / 99.4 %** of the spikes -- the dropped units carry almost no data, which is
exactly the case for dropping them. At 2 Hz VISp collapses from 19 units to **10** and loses
**19 %** of its spikes, gutting the region that is already the project's weakest and the one
every cross-area claim is anchored against, while SCm/MRN barely move (44->41, 54->48). So
2 Hz buys nothing where it is safe and costs everything where it is not. See
``unit_yield_table``.

**MSR is NOT used to select units** (standing decision, 2026-07-13): it correlates with firing
rate (rho 0.64/0.55 in VISp/SCm), so an MSR cut in those regions *is* a rate cut with extra
steps, and in the one region where the control was valid it did not beat a random half.

Nothing here smooths or transforms. The Poisson HMM needs **raw integer counts**; the geometry
and the rSLDS apply their own transform downstream (see :func:`anscombe`).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np

# Beryl acronyms, frozen 2026-07-12. VISpm was dropped: it was a stand-in for a
# "higher-order" tier that IBL does not actually supply in the same session.
REGIONS = ("VISp", "SCm", "MRN")

WORKING_EID = "5c0c560e-9e1f-45e9-b66e-e4ee7855be84"

# The 5 sessions in which the whole {VISp, SCm, MRN} triplet co-occurs at >= 20 good units
# each (Phase 1; confirmed 2026-07-14 by scripts/survey_triplet_sessions.py). One authoritative
# list for the "pool the replicates" step. Ordered by usable-trial count (see
# docs/knowledge/the_five_triplet_sessions.md for the per-session quality write-up).
TRIPLET_EIDS = (
    "5c0c560e-9e1f-45e9-b66e-e4ee7855be84",  # NR_0028   843 trials; most trials, WEAKEST VISp
    "90c61c38-b9fd-4cc3-9795-29160d2f8e55",  # CSH_ZAD_025 656; best design (K=33), thin (76 u)
    "111c1762-7908-47e0-9f40-2f2ee55b6505",  # UCLA033   364; clean VISp + full population
    "90d1e82c-c96f-496c-ad4e-ee3f02067f25",  # DY_010    353; clean single-probe
    "8a3a0197-b40a-449f-be55-c00b23253bbf",  # UCLA034   290; most units (188), fewest trials
)
assert WORKING_EID == TRIPLET_EIDS[0]


@dataclass(frozen=True)
class PreprocessConfig:
    """Every knob that decides what the models see. Serialised with the tensors."""

    align_event: str = "stimOn_times"
    pre_s: float = 0.4
    post_s: float = 0.8
    bin_size_s: float = 0.04   # 40 ms (2026-07-15); see the module docstring for why not 50
    min_rate_hz: float = 1.0
    min_rt_s: float = 0.08
    max_rt_s: float = 2.0
    regions: tuple[str, ...] = REGIONS
    qc: float = 1.0

    @property
    def n_bins(self) -> int:
        return int(round((self.pre_s + self.post_s) / self.bin_size_s))

    @property
    def bin_centers(self) -> np.ndarray:
        """Bin centres in seconds relative to the align event (negative = before)."""
        return (np.arange(self.n_bins) + 0.5) * self.bin_size_s - self.pre_s

    @property
    def baseline_mask(self) -> np.ndarray:
        """Boolean over bins: True for pre-event bins. The HMM's baseline prior uses this."""
        return self.bin_centers < 0.0

    def tag(self) -> str:
        """Short filename-safe description, so two configs cannot overwrite each other."""
        ev = self.align_event.replace("_times", "")
        return (f"{ev}_pre{int(self.pre_s*1000)}_post{int(self.post_s*1000)}"
                f"_bin{int(self.bin_size_s*1000)}_rate{self.min_rate_hz:g}")


# --------------------------------------------------------------------------- transforms


def anscombe(counts: np.ndarray) -> np.ndarray:
    """Anscombe transform ``2*sqrt(x + 3/8)``.

    ⚠️ **It is the variance stabiliser for Poisson counts, and at IBL's firing rates it does
    NOT stabilise the variance.** Read that before relying on it.

    What it is *supposed* to do
    ---------------------------
    For a Poisson variable **Var = Mean**, so in raw counts a 20 Hz neuron is not merely louder
    than a 2 Hz neuron -- it is intrinsically *noisier*, and any squared-error method (PCA,
    least squares, a Gaussian likelihood) will chase that neuron's shot noise. Anscombe (1948)
    is the classical fix: a delta-method expansion of ``y = 2*sqrt(x + c)`` gives

        Var[y] = 1 + O(1/lambda),

    with ``c = 3/8`` the offset that cancels the leading 1/lambda term. So *asymptotically*
    every neuron's noise is one unit, whatever its rate.

    Why it fails here (measured, not assumed -- see notebooks/00_preprocessing.ipynb, Part 2)
    -----------------------------------------------------------------------------------------
    ``Var -> 1`` is a **large-lambda** result. On simulated Poisson it is reached only for
    **lambda >~ 4** spikes/bin. IBL runs at::

        VISp @20ms  lambda = 0.046      SCm @20ms  0.331      MRN @20ms  0.276
        VISp @50ms  lambda = 0.116      SCm @50ms  0.827      MRN @50ms  0.689

    -- one to two orders of magnitude below the asymptotic regime. On the real tensor the
    transform yields median Var ~= **0.09**, not 1, and the mean-variance dependence survives
    intact (Spearman ~ +1).

    This is **structural, not a tuning failure**: at lambda << 1 a spike count is essentially
    **Bernoulli**, and Var = p(1-p)*delta^2 depends on p irreducibly. No monotone transform can
    stabilise a Bernoulli. Variance stabilisation is simply unavailable in this data regime.

    So why keep it
    --------------
    Two small honest reasons: it compresses the high-count tail (one 7-spike bin stops
    dominating a squared-error objective), and the rSLDS ground-truth calibration was done with
    it, so changing it now would invalidate that for no gain.

    **The consequence is the actual result.** The Gaussian/PCA latent used by our fast rSLDS
    backend rests on a variance-stabilising step that stabilises nothing here. That is a
    *second*, independent argument (on top of "PCA picks variance, not dynamics") that the
    **Poisson ``ssm`` backend is the principled one and our PCA latent is a concession** -- and
    it explains mechanistically why the 20 ms latent came out shot-noise dominated.

    It is NOT smoothing and NOT normalisation: it is a monotone relabelling (0 -> 1.225,
    1 -> 2.345, 2 -> 3.082). Any continuity you see downstream came from the *smoother*.

    **Never feed it to the Poisson HMM**, which needs raw integer counts by construction: its
    emission model already knows Var = Mean, so pre-stabilising would correct for it twice.
    """
    return 2.0 * np.sqrt(np.asarray(counts, dtype=np.float64) + 0.375)


def sigma_ms_to_bins(sigma_ms: float, bin_size_s: float) -> float:
    """Convert a smoothing sigma in **milliseconds** to bins.

    Always specify smoothing in ms, never in bins: a sigma given in bins silently changes the
    amount of smoothing whenever the bin size changes, which is how a bin-size sweep turns
    into a confounded bin-size-AND-smoothing sweep.
    """
    return (sigma_ms / 1000.0) / bin_size_s


# --------------------------------------------------------------------------- loading


def load_session(one, eid: str = WORKING_EID, *, cfg: PreprocessConfig | None = None):
    """Load trials + both probes' good units for one session. Returns (trials, mask, probe_loads).

    ``clusters['Beryl']`` is added here (the rollup we filter regions on; ``acronym`` is the
    finer Allen label, e.g. SCig).
    """
    from iblatlas.regions import BrainRegions

    from src.data.access import load_trials, load_units, sessions_table

    cfg = cfg or PreprocessConfig()
    trials, mask = load_trials(one, eid, min_rt=cfg.min_rt_s, max_rt=cfg.max_rt_s)

    br = BrainRegions()
    probe_loads = []
    for pid in sessions_table(one).query("eid == @eid")["pid"].tolist():
        spikes, clusters = load_units(one, pid, qc=cfg.qc)
        clusters["Beryl"] = br.id2acronym(clusters["atlas_id"].values, mapping="Beryl")
        probe_loads.append({"spikes": spikes, "clusters": clusters})

    return trials, mask, probe_loads


def align_times(trials, mask, align_event: str) -> np.ndarray:
    """Per-trial alignment times for the *usable* trials, guaranteed NaN-free.

    ``firstMovement_times`` has NaNs even inside the paper's mask, so aligning on it drops a
    few more trials. The returned array is what every tensor for that alignment is built on --
    so callers must use :func:`usable_mask` to keep their trial-level covariates in register.
    """
    t = trials[mask]
    ev = t[align_event].to_numpy(dtype=float)
    return ev[np.isfinite(ev)]


def usable_mask(trials, mask, align_event: str) -> np.ndarray:
    """Boolean over the masked trials: those with a finite alignment time.

    Use this to subset ``choice`` / ``contrast`` / ``feedbackType`` so trial-level covariates
    stay aligned with axis 0 of the tensors. Getting this wrong silently mislabels every trial.
    """
    t = trials[mask]
    return np.isfinite(t[align_event].to_numpy(dtype=float))


# --------------------------------------------------------------------------- the pipeline


def preprocess_session(
    probe_loads: list[dict],
    event_times: np.ndarray,
    cfg: PreprocessConfig,
    *,
    verbose: bool = True,
) -> dict:
    """**The** preprocessing function. Raw spikes -> the count tensors every model consumes.

    One place, one set of decisions, applied identically to every region -- so that "K from the
    HMM" and "the rSLDS fit" refer to the same data.

    Steps, in order:

    1. Bin each region's units into ``(trials, bins, units)`` **integer counts**, pooling both
       probes of the session (they are simultaneous and share the trial events).
    2. Drop units with **zero** spikes in the window (a safety net: lambda == 0 is a degenerate
       Poisson MLE and was what produced the ``-inf`` cross-validation disaster).
    3. Apply the activity threshold ``min_rate_hz``.

    No smoothing, no transform, no z-scoring. Those are downstream, model-specific choices.

    Returns
    -------
    dict with, per region, ``counts`` (int32), ``keep_idx`` (indices into the region's original
    unit order -- needed to line up MSR / spike trains / anything computed per unit), plus
    ``bin_centers``, ``baseline_mask``, ``config`` and a per-region ``n_units_dropped`` record.
    """
    from src.data.build_matrices import (
        drop_silent_units,
        region_tensor_from_probes,
        select_active_units,
    )

    out: dict = {
        "bin_centers": cfg.bin_centers,
        "baseline_mask": cfg.baseline_mask,
        "config": asdict(cfg),
        "n_trials": int(len(event_times)),
        "regions": {},
    }

    for region in cfg.regions:
        counts, bc = region_tensor_from_probes(
            probe_loads, region, event_times,
            pre_s=cfg.pre_s, post_s=cfg.post_s, bin_size_s=cfg.bin_size_s,
        )
        n_raw = counts.shape[2]

        counts, keep_nonsilent = drop_silent_units(counts, min_total_spikes=1)
        n_after_silent = counts.shape[2]

        counts, keep_active = select_active_units(
            counts, cfg.bin_size_s, min_rate_hz=cfg.min_rate_hz)

        # compose the two filters back onto the ORIGINAL unit indices of this region
        keep_idx = keep_nonsilent[keep_active]

        out["regions"][region] = {
            "counts": counts.astype(np.int32),
            "keep_idx": keep_idx,
            "n_raw": int(n_raw),
            "n_dropped_silent": int(n_raw - n_after_silent),
            "n_dropped_rate": int(n_after_silent - counts.shape[2]),
            "n_units": int(counts.shape[2]),
        }
        if verbose:
            r = out["regions"][region]
            print(f"  {region:5s}  {r['n_units']:3d} units "
                  f"(raw {r['n_raw']}, -{r['n_dropped_silent']} silent, "
                  f"-{r['n_dropped_rate']} below {cfg.min_rate_hz:g} Hz)   "
                  f"tensor {counts.shape}")

    # sanity: every region must share the trial and bin axes, or nothing downstream is joint
    shapes = {r: v["counts"].shape[:2] for r, v in out["regions"].items()}
    if len(set(shapes.values())) != 1:
        raise AssertionError(f"regions disagree on (trials, bins): {shapes}")

    return out


def save_processed(data: dict, path: str | Path) -> Path:
    """Freeze the processed tensors to .npz, config included, so a model never re-derives them.

    The config travels *with* the data. A tensor whose preprocessing you have to reconstruct
    from memory is a tensor you will eventually mis-describe in a figure caption.
    """
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    flat: dict = {
        "bin_centers": data["bin_centers"],
        "baseline_mask": data["baseline_mask"],
        "config_json": np.array(json.dumps(data["config"])),
        "n_trials": np.array(data["n_trials"]),
        "region_names": np.array(list(data["regions"])),
    }
    for region, r in data["regions"].items():
        flat[f"{region}__counts"] = r["counts"]
        flat[f"{region}__keep_idx"] = r["keep_idx"]

    # Trial-level covariates (choice, contrast, RT...) travel WITH the tensors. Storing them
    # separately is how axis 0 and the labels drift apart and every trial ends up mislabelled.
    for name, v in data.get("trials", {}).items():
        v = np.asarray(v, dtype=float)
        if v.shape[0] != data["n_trials"]:
            raise AssertionError(
                f"trial covariate {name!r} has {v.shape[0]} entries but there are "
                f"{data['n_trials']} trials — it is not in register with axis 0.")
        flat[f"trials__{name}"] = v

    np.savez_compressed(path, **flat)
    return path


def load_processed(path: str | Path) -> dict:
    """Inverse of :func:`save_processed`. Returns the same dict shape ``preprocess_session`` does."""
    import json

    z = np.load(path, allow_pickle=False)
    regions = [str(r) for r in z["region_names"]]
    return {
        "bin_centers": z["bin_centers"],
        "baseline_mask": z["baseline_mask"],
        "config": json.loads(str(z["config_json"])),
        "n_trials": int(z["n_trials"]),
        "regions": {
            r: {"counts": z[f"{r}__counts"], "keep_idx": z[f"{r}__keep_idx"],
                "n_units": int(z[f"{r}__counts"].shape[2])}
            for r in regions
        },
        "trials": {k.split("__", 1)[1]: z[k] for k in z.files if k.startswith("trials__")},
    }


# --------------------------------------------------------------------------- diagnostics


def trial_timing_table(trials, mask):
    """When does the mouse actually move, and when does it get feedback? Returns a DataFrame.

    This is the table that decides whether the analysis window means what you think it means.
    In this session the median RT is 171 ms -- so a 0.8 s post-stimulus window is dominated by
    *movement and outcome*, not by decision formation.
    """
    import pandas as pd

    t = trials[mask]
    stim = t["stimOn_times"].to_numpy(dtype=float)
    rt = t["firstMovement_times"].to_numpy(dtype=float) - stim
    resp = t["feedback_times"].to_numpy(dtype=float) - stim
    quiesc = t["quiescencePeriod"].to_numpy(dtype=float)

    rows = []
    for name, v in [("firstMovement - stimOn (RT)", rt),
                    ("feedback - stimOn", resp),
                    ("quiescencePeriod (enforced still)", quiesc)]:
        v = v[np.isfinite(v)]
        rows.append({
            "quantity": name, "n": int(v.size),
            "min": round(float(v.min()), 3),
            "p05": round(float(np.percentile(v, 5)), 3),
            "median": round(float(np.median(v)), 3),
            "p95": round(float(np.percentile(v, 95)), 3),
            "max": round(float(v.max()), 3),
        })
    return pd.DataFrame(rows)


def window_occupancy(trials, mask, post_s: float, pre_s: float):
    """What fraction of trials have *already moved* / *already got feedback* inside the window?

    And is the baseline window inside the enforced quiescence period? These two numbers are the
    entire justification for the window, so they are computed rather than asserted.
    """
    import pandas as pd

    t = trials[mask]
    stim = t["stimOn_times"].to_numpy(dtype=float)
    rt = t["firstMovement_times"].to_numpy(dtype=float) - stim
    resp = t["feedback_times"].to_numpy(dtype=float) - stim
    quiesc = t["quiescencePeriod"].to_numpy(dtype=float)

    return pd.DataFrame([{
        "pre_s": pre_s,
        "post_s": post_s,
        "frac_moved_within_window": round(float(np.nanmean(rt < post_s)), 3),
        "frac_feedback_within_window": round(float(np.nanmean(resp < post_s)), 3),
        "frac_baseline_outside_quiescence": round(float(np.nanmean(quiesc < pre_s)), 3),
    }])


def unit_yield_table(probe_loads, event_times, cfg: PreprocessConfig,
                     thresholds=(0.0, 0.5, 1.0, 2.0, 3.0, 5.0)):
    """Units kept, and **spikes kept**, at each candidate rate threshold. Returns a DataFrame.

    Report both. "We kept 44 of 59 units" hides whether the dropped units carried any spikes;
    ``frac_spikes_kept`` says how much of the data the threshold actually discards.
    """
    import pandas as pd

    from src.data.build_matrices import region_tensor_from_probes

    rows = []
    for region in cfg.regions:
        counts, _ = region_tensor_from_probes(
            probe_loads, region, event_times,
            pre_s=cfg.pre_s, post_s=cfg.post_s, bin_size_s=cfg.bin_size_s)
        window_s = counts.shape[1] * cfg.bin_size_s
        rate = counts.sum(axis=(0, 1)) / counts.shape[0] / window_s
        total = counts.sum()
        for thr in thresholds:
            keep = rate >= thr
            rows.append({
                "region": region,
                "min_rate_hz": thr,
                "n_units": int(keep.sum()),
                "frac_units_kept": round(float(keep.mean()), 3),
                "frac_spikes_kept": round(float(counts[:, :, keep].sum() / max(total, 1)), 4),
                "median_rate_kept_hz": round(float(np.median(rate[keep])), 2) if keep.any() else np.nan,
            })
    return pd.DataFrame(rows)
