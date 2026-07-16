"""pynapple-based exploratory plots, in the project's colour-vision-safe style.

Scope (see docs/worklog/2026-07-12_pynapple-and-env.md): pynapple is our *exploration and
basic-analysis* layer. It does no state-space modelling — the HMM/rSLDS core stays in
`src/dynamics/`. Everything here takes plain spike times + event times, wraps them in
pynapple objects, and returns a matplotlib Figure.

Every function routes colour through `src.viz.style`, so nothing depends on a red-vs-green
distinction.

Verified against pynapple 0.11.3 (2026-07-12):
    nap.compute_perievent(data, events, window)          # NB: `window`, not `minmax`
    nap.compute_1d_tuning_curves(group, feature, nb_bins, ep=None, minmax=None)
    nap.compute_crosscorrelogram(group_or_pair, binsize, windowsize, ep=None, norm=True)
    nap.compute_autocorrelogram(group, binsize, windowsize, ep=None, norm=True)
    TsGroup.count(bin_size, ep)  -> raw int64 counts, shape (time, neurons)
`nap.compute_perievent_continuous` does NOT exist in 0.11.3.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap

from .style import DIVERGING, SEQUENTIAL, region_color, set_style, stroke


def spikes_to_tsgroup(spike_times: dict[int, np.ndarray]) -> nap.TsGroup:
    """Wrap {unit_id: spike_time_array} in a pynapple TsGroup."""
    return nap.TsGroup({uid: nap.Ts(t=np.asarray(t)) for uid, t in spike_times.items()})


# `region_spike_trains` now lives in src/data/build_matrices.py and is re-exported here for
# the plot functions below. It is deliberately NOT reimplemented: that function keys each
# train by its *tensor column index*, walking probes and units in the same order as
# `region_tensor_from_probes`. A second, independent unit ordering is exactly how you end up
# plotting one unit's correlogram under another unit's label.
from src.data.build_matrices import region_spike_trains  # noqa: E402,F401


# ---------------------------------------------------------------------------
# 1. Peri-event raster — the basic sanity plot, but sorted by reaction time so
#    a movement-locked response separates visibly from a stimulus-locked one.
# ---------------------------------------------------------------------------

def peri_event_raster(
    unit: nap.Ts,
    stim_times: np.ndarray,
    window: tuple[float, float] = (-0.5, 1.0),
    rt: np.ndarray | None = None,
    title: str = "",
    region: str | None = None,
):
    """Raster of one unit around stimulus onset; trials sorted by RT if given.

    Sorting by RT is the point: a stimulus-locked response stays a vertical band, a
    movement-locked one bends with the RT curve. That distinction is invisible in an
    unsorted raster.
    """
    set_style()
    # NB: pynapple silently re-sorts unsorted timestamps. So we must NOT hand it
    # RT-ordered events - we pass them in time order and instead reorder the *rows*.
    # (Passing rt-sorted events here produced a silently mislabelled raster.)
    stim_times = np.asarray(stim_times)
    ev = nap.Ts(t=stim_times)
    peri = nap.compute_perievent(unit, ev, window)  # TsGroup: one "unit" per trial

    keys = list(peri.keys())
    row_of = (np.argsort(np.argsort(rt)) if rt is not None
              else np.arange(len(keys)))  # trial index -> plotted row

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    color = region_color(region) if region else "#0072B2"
    for trial_i, key in enumerate(keys):
        t = peri[key].t
        if len(t):
            ax.plot(t, np.full_like(t, row_of[trial_i]), "|",
                    color=color, markersize=2.5, mew=0.7)

    # Overlays must not reuse a region colour: for SCm the movement line would otherwise be
    # drawn in SCm's own vermillion and vanish into its spikes. Black is safe against every
    # Okabe-Ito hue and carries no red-green information.
    ax.axvline(0, color="0.35", lw=1.2, ls="--", label="stim on")
    if rt is not None:
        ax.plot(np.sort(rt), np.arange(len(rt)), lw=2.0, color="black",
                label="movement onset")
    ax.set(xlabel="time from stimulus (s)", ylabel="trial (sorted by RT)",
           xlim=window, title=title or "peri-event raster")
    ax.legend(frameon=False, fontsize=8, loc="upper right",
              framealpha=0.9, facecolor="white")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. PSTH per region — mean rate +/- SEM, regions overlaid in their fixed colours.
# ---------------------------------------------------------------------------

def psth_by_region(
    region_spikes: dict[str, dict[int, np.ndarray]],
    stim_times: np.ndarray,
    window: tuple[float, float] = (-0.5, 1.0),
    bin_size: float = 0.02,
    title: str = "PSTH by region",
):
    """Trial-averaged population rate per region, aligned to stimulus onset.

    Reads the *latency ordering* across regions — the first thing to check if the claim is
    that a sequence propagates. Shaded band is SEM across units.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    edges = np.arange(window[0], window[1] + bin_size, bin_size)
    centres = edges[:-1] + bin_size / 2

    for region, spikes in region_spikes.items():
        # rate per unit per bin, averaged over trials
        per_unit = []
        for t in spikes.values():
            t = np.asarray(t)
            counts = np.zeros(len(centres))
            for s in stim_times:
                counts += np.histogram(t - s, bins=edges)[0]
            per_unit.append(counts / (len(stim_times) * bin_size))  # -> Hz
        per_unit = np.asarray(per_unit)
        mean = per_unit.mean(0)
        sem = per_unit.std(0) / np.sqrt(len(per_unit))
        c = region_color(region)
        ax.plot(centres, mean, color=c, lw=1.8, label=f"{region} (n={len(per_unit)})",
                path_effects=stroke())
        ax.fill_between(centres, mean - sem, mean + sem, color=c, alpha=0.22, lw=0)

    ax.axvline(0, color="black", lw=1.0)
    ax.set(xlabel="time from stimulus (s)", ylabel="firing rate (Hz)", xlim=window, title=title)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Sequence heatmap — THE plot for the metastable-sequence claim.
# ---------------------------------------------------------------------------

def sequence_heatmap(
    region_spikes: dict[str, dict[int, np.ndarray]],
    stim_times: np.ndarray,
    window: tuple[float, float] = (-0.2, 0.8),
    bin_size: float = 0.02,
    title: str = "neurons sorted by peak latency",
):
    """Neurons x time, z-scored, sorted by peak latency, region blocks marked.

    If decision-making is a *temporal sequence* of states, this is where it should be
    visible as a diagonal: successive neurons peaking in succession. Z-score is signed, so
    it uses the diverging (red-blue) map, not the sequential one.

    Caveat worth stating in any write-up: sorting neurons by their own peak latency will
    manufacture a diagonal even from noise. Treat this as a *display*, not evidence —
    the evidence is the HMM. (Sort on held-out trials to make it honest.)
    """
    set_style()
    edges = np.arange(window[0], window[1] + bin_size, bin_size)
    centres = edges[:-1] + bin_size / 2

    rows, row_region = [], []
    for region, spikes in region_spikes.items():
        for t in spikes.values():
            t = np.asarray(t)
            counts = np.zeros(len(centres))
            for s in stim_times:
                counts += np.histogram(t - s, bins=edges)[0]
            rows.append(counts / (len(stim_times) * bin_size))
            row_region.append(region)
    x = np.asarray(rows)

    sd = x.std(1, keepdims=True)
    z = (x - x.mean(1, keepdims=True)) / np.where(sd == 0, 1.0, sd)
    order = np.argsort(np.argmax(z, axis=1))
    z, row_region = z[order], np.asarray(row_region)[order]

    fig, (cax, ax) = plt.subplots(
        1, 2, figsize=(6.6, 4.4), gridspec_kw={"width_ratios": [0.035, 1], "wspace": 0.02}
    )
    lim = np.nanpercentile(np.abs(z), 98)
    im = ax.imshow(z, aspect="auto", cmap=DIVERGING, vmin=-lim, vmax=lim,
                   extent=[window[0], window[1], len(z), 0], interpolation="nearest")
    ax.axvline(0, color="black", lw=1.0)
    ax.set(xlabel="time from stimulus (s)", title=title)
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, label="z-scored rate", pad=0.02)

    # region membership stripe, so you can see whether the diagonal crosses areas
    for region in dict.fromkeys(row_region):
        idx = np.where(row_region == region)[0]
        cax.barh(idx, 1, height=1.0, color=region_color(region), lw=0)
        cax.text(-0.6, idx.mean(), region, ha="right", va="center", fontsize=8,
                 color=region_color(region))
    cax.set(xlim=(0, 1), ylim=(len(z), 0))
    cax.axis("off")
    return fig


# ---------------------------------------------------------------------------
# 4. Contrast tuning — IBL's actual stimulus variable.
# ---------------------------------------------------------------------------

def contrast_tuning(
    region_spikes: dict[str, dict[int, np.ndarray]],
    stim_times: np.ndarray,
    contrast: np.ndarray,
    resp_window: tuple[float, float] = (0.05, 0.15),
    title: str = "contrast tuning (evoked window)",
):
    """Mean evoked rate vs signed contrast, per region.

    A sensory area should be tuned to |contrast|; a motor area should be flatter, or tuned
    to the *side* rather than the strength. That contrast between areas is a cheap first
    test of the 'shared dynamics, different geometry' claim.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    levels = np.unique(contrast)

    for region, spikes in region_spikes.items():
        means = []
        for lvl in levels:
            ts = np.asarray(stim_times)[contrast == lvl]
            rates = []
            for t in spikes.values():
                t = np.asarray(t)
                n = sum(((t >= s + resp_window[0]) & (t < s + resp_window[1])).sum() for s in ts)
                rates.append(n / (len(ts) * (resp_window[1] - resp_window[0])))
            means.append(np.mean(rates))
        ax.plot(levels, means, "o-", color=region_color(region), lw=1.8, ms=5,
                label=region, path_effects=stroke())

    ax.axvline(0, color="black", lw=0.8, ls=":")
    ax.set(xlabel="signed contrast  (<0 = left, >0 = right)", ylabel="evoked rate (Hz)",
           title=title)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. Cross-correlogram between two regions — the SCm<->MRN feedback hypothesis.
# ---------------------------------------------------------------------------

def region_crosscorrelogram(
    spikes_a: dict[int, np.ndarray],
    spikes_b: dict[int, np.ndarray],
    name_a: str,
    name_b: str,
    binsize: float = 0.005,
    windowsize: float = 0.2,
    ep: nap.IntervalSet | None = None,
):
    """Mean cross-correlogram between all A-B unit pairs.

    This is the cheap, model-free version of your SCm<->MRN question. A peak at positive
    lag means B tends to fire *after* A; a symmetric peak means common drive, not
    directed exchange. It cannot prove direction (common input mimics it), which is
    exactly why the joint HMM/rSLDS in Phase 3 is still needed — but if there is no
    asymmetry here at all, that is an early warning.
    """
    set_style()
    ga, gb = spikes_to_tsgroup(spikes_a), spikes_to_tsgroup(spikes_b)
    # pynapple 0.11.3 rejects ep=None outright (despite the Optional signature), so the
    # kwarg has to be omitted rather than passed as None.
    kw = {"ep": ep} if ep is not None else {}
    cc = nap.compute_crosscorrelogram((ga, gb), binsize, windowsize, norm=True, **kw)

    lags = cc.index.to_numpy()
    vals = cc.to_numpy()
    mean, sem = np.nanmean(vals, axis=1), np.nanstd(vals, axis=1) / np.sqrt(vals.shape[1])

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    c = region_color(name_b)
    ax.fill_between(lags, mean - sem, mean + sem, color=c, alpha=0.25, lw=0)
    ax.plot(lags, mean, color=c, lw=1.8, path_effects=stroke())
    ax.axvline(0, color="black", lw=1.0)
    ax.axhline(1.0, color="grey", lw=0.8, ls=":")  # norm=True -> 1.0 is chance
    ax.set(xlabel=f"lag (s)   [{name_b} relative to {name_a}]",
           ylabel="normalised rate",
           title=f"{name_a} -> {name_b} cross-correlogram  ({vals.shape[1]} pairs)")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 6. Autocorrelogram — unit QC. Cheap, and it catches bad sorting.
# ---------------------------------------------------------------------------

def autocorrelogram_grid(
    spikes: dict[int, np.ndarray],
    region: str,
    n_units: int = 8,
    binsize: float = 0.001,
    windowsize: float = 0.05,
):
    """Autocorrelograms for a few units: a real refractory dip near 0 means real neurons.

    If the centre bin is not depressed, the 'unit' is probably merged multi-unit activity,
    and any Poisson-HMM state built on it is suspect.
    """
    set_style()
    g = spikes_to_tsgroup(spikes)
    ac = nap.compute_autocorrelogram(g, binsize, windowsize, norm=True)

    keys = list(ac.columns)[:n_units]
    ncol = 4
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.0 * ncol, 1.7 * nrow), sharex=True)
    lags = ac.index.to_numpy()
    for ax, k in zip(np.ravel(axes), keys):
        ax.fill_between(lags, 0, ac[k].to_numpy(), color=region_color(region), lw=0)
        ax.axvline(0, color="black", lw=0.6)
        ax.set_title(f"unit {k}", fontsize=8)
        ax.set_yticks([])
    for ax in np.ravel(axes)[len(keys):]:
        ax.axis("off")
    fig.suptitle(f"{region} — autocorrelograms (refractory dip = well-isolated)", fontsize=10)
    fig.supxlabel("lag (s)", fontsize=9)
    fig.tight_layout()
    return fig
