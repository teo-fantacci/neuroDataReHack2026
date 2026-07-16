"""Turn spike trains into the (trials x time x neurons) tensor every method consumes.

Convention used throughout the project: axis 0 = trials, axis 1 = time bins,
axis 2 = neurons. Counts are integers; rates are floats in spikes/s.

Poisson-emission HMMs must be fitted on **raw counts**, never on smoothed data.
Smoothing exists for geometry (PHATE/CEBRA) and for plotting.
"""

from __future__ import annotations

import numpy as np


def bin_spikes_by_event(
    spike_times: np.ndarray,
    spike_clusters: np.ndarray,
    n_clusters: int,
    event_times: np.ndarray,
    pre_s: float = 0.2,
    post_s: float = 0.8,
    bin_size_s: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a spike-count tensor aligned to a per-trial event.

    Parameters
    ----------
    spike_times
        Spike times in seconds, **sorted ascending** (ONE returns them sorted).
    spike_clusters
        Cluster index per spike, in ``[0, n_clusters)``. As returned by
        ``brainwidemap.load_good_units``, these already index the clusters table.
    n_clusters
        Number of units (defines axis 2, so empty units keep their column).
    event_times
        One alignment time per trial, e.g. ``trials['stimOn_times']``. Must not
        contain NaN -- apply the trial mask first.
    pre_s, post_s
        Window around each event. Window is ``[event - pre_s, event + post_s)``.
    bin_size_s
        Bin width in seconds.

    Returns
    -------
    counts : ndarray, shape (n_trials, n_bins, n_clusters), dtype int32
    bin_centers : ndarray, shape (n_bins,)
        Bin centres in seconds **relative to the event** (negative = before).
    """
    event_times = np.asarray(event_times, dtype=float)
    if np.isnan(event_times).any():
        raise ValueError(
            "event_times contains NaN; apply the trial mask from load_trials() first."
        )

    n_bins = int(round((pre_s + post_s) / bin_size_s))
    n_trials = event_times.size
    counts = np.zeros((n_trials, n_bins, n_clusters), dtype=np.int32)

    spike_times = np.asarray(spike_times)
    spike_clusters = np.asarray(spike_clusters).astype(np.int64)

    # searchsorted per trial keeps this O(n_trials * log n_spikes + n_spikes_in_window)
    starts = np.searchsorted(spike_times, event_times - pre_s, side="left")
    stops = np.searchsorted(spike_times, event_times + post_s, side="right")

    for i, (lo, hi) in enumerate(zip(starts, stops)):
        if hi <= lo:
            continue
        rel = spike_times[lo:hi] - (event_times[i] - pre_s)
        b = (rel / bin_size_s).astype(np.int64)
        # Guard the right edge: a spike exactly at +post_s would land on n_bins.
        keep = (b >= 0) & (b < n_bins)
        b = b[keep]
        c = spike_clusters[lo:hi][keep]
        if b.size == 0:
            continue
        flat = np.bincount(b * n_clusters + c, minlength=n_bins * n_clusters)
        counts[i] = flat.reshape(n_bins, n_clusters)

    bin_centers = (np.arange(n_bins) + 0.5) * bin_size_s - pre_s
    return counts, bin_centers


def counts_to_rates(counts: np.ndarray, bin_size_s: float) -> np.ndarray:
    """Convert the count tensor to firing rates in spikes/s."""
    return counts.astype(np.float64) / bin_size_s


def smooth_over_time(x: np.ndarray, sigma_bins: float) -> np.ndarray:
    """Gaussian-smooth along the time axis (axis 1). ``sigma_bins=0`` is a no-op.

    For geometry and plotting only -- never feed the result to a Poisson HMM.
    """
    if sigma_bins <= 0:
        return x
    from scipy.ndimage import gaussian_filter1d

    return gaussian_filter1d(x.astype(np.float64), sigma=sigma_bins, axis=1, mode="nearest")


def stack_for_hmmlearn(counts: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Flatten (trials, time, neurons) into hmmlearn's (samples, features) + lengths.

    hmmlearn treats trials as independent sequences via the ``lengths`` argument,
    which prevents the model from learning a bogus transition across the trial
    boundary.
    """
    n_trials, n_bins, n_units = counts.shape
    X = counts.reshape(n_trials * n_bins, n_units)
    lengths = [n_bins] * n_trials
    return X, lengths


def region_tensor_from_probes(
    probe_loads: list[dict],
    region: str,
    event_times: np.ndarray,
    *,
    pre_s: float,
    post_s: float,
    bin_size_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Pool one Beryl region's units across every probe of a session.

    Both probes of a session are recorded simultaneously and share the same trial
    events, so their units concatenate cleanly along the neuron axis.

    Parameters
    ----------
    probe_loads
        One dict per probe, each with keys ``spikes`` (dict from ``load_good_units``)
        and ``clusters`` (DataFrame with a ``Beryl`` column added by the caller).
    region
        Beryl acronym to select.
    event_times
        Per-trial alignment times (already masked, no NaN).

    Returns
    -------
    counts : (n_trials, n_bins, n_units_total) int32
    bin_centers : (n_bins,)
    """
    blocks = []
    bin_centers = None
    for load in probe_loads:
        clusters = load["clusters"]
        spikes = load["spikes"]
        unit_idx = np.flatnonzero((clusters["Beryl"] == region).values)
        if unit_idx.size == 0:
            continue
        keep = np.isin(spikes["clusters"], unit_idx)
        remap = -np.ones(clusters.shape[0], dtype=np.int64)
        remap[unit_idx] = np.arange(unit_idx.size)
        counts, bin_centers = bin_spikes_by_event(
            spike_times=spikes["times"][keep],
            spike_clusters=remap[spikes["clusters"][keep]],
            n_clusters=unit_idx.size,
            event_times=event_times,
            pre_s=pre_s, post_s=post_s, bin_size_s=bin_size_s,
        )
        blocks.append(counts)

    if not blocks:
        raise ValueError(f"No units found for region {region!r} on any probe.")
    return np.concatenate(blocks, axis=2), bin_centers


def region_spike_trains(
    probe_loads: list[dict],
    region: str,
    *,
    t_start: float | None = None,
    t_stop: float | None = None,
) -> dict[int, np.ndarray]:
    """Full spike train of each unit of ``region``, keyed by its **tensor column index**.

    Companion to :func:`region_tensor_from_probes`, and it walks the probes and units in
    exactly the same order, so key ``j`` here is column ``j`` of that function's neuron
    axis. That alignment is the whole point: a per-unit score (e.g. multiscale relevance)
    computed from these trains can be used to index the tensor directly. Recomputing the
    unit order independently is how you silently keep the wrong half of the population.

    Unlike the tensor builders this does **not** cut the train into trial windows -- it
    returns continuous session time, which is what a multi-timescale statistic needs.
    ``t_start``/``t_stop`` optionally restrict it to the task period.
    """
    trains: dict[int, np.ndarray] = {}
    offset = 0
    for load in probe_loads:
        clusters, spikes = load["clusters"], load["spikes"]
        unit_idx = np.flatnonzero((clusters["Beryl"] == region).values)
        if unit_idx.size == 0:
            continue                      # skipped here AND in region_tensor_from_probes
        st, sc = np.asarray(spikes["times"]), np.asarray(spikes["clusters"])
        if t_start is not None or t_stop is not None:
            lo = -np.inf if t_start is None else t_start
            hi = np.inf if t_stop is None else t_stop
            keep = (st >= lo) & (st < hi)
            st, sc = st[keep], sc[keep]
        for j, u in enumerate(unit_idx):
            trains[offset + j] = st[sc == u]
        offset += unit_idx.size

    if not trains:
        raise ValueError(f"No units found for region {region!r} on any probe.")
    return trains


def region_event_spikes(
    probe_loads: list[dict],
    region: str,
    event_times: np.ndarray,
    *,
    pre_s: float,
    post_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect in-window spike times for a region, for **dot raster** plots.

    Unlike ``region_tensor_from_probes`` (which bins), this keeps exact spike times so a
    raster shows individual spikes. Units are numbered globally across probes.

    Returns three parallel flat arrays, one entry per spike:
    ``trial_idx``, ``unit_idx``, ``t_rel`` (seconds relative to the event).
    """
    trial_idx, unit_idx, t_rel = [], [], []
    unit_offset = 0
    for load in probe_loads:
        clusters, spikes = load["clusters"], load["spikes"]
        uidx = np.flatnonzero((clusters["Beryl"] == region).values)
        if uidx.size == 0:
            continue
        remap = -np.ones(clusters.shape[0], dtype=np.int64)
        remap[uidx] = np.arange(uidx.size) + unit_offset
        keep = np.isin(spikes["clusters"], uidx)
        st = spikes["times"][keep]
        sc = remap[spikes["clusters"][keep]]
        starts = np.searchsorted(st, event_times - pre_s, side="left")
        stops = np.searchsorted(st, event_times + post_s, side="right")
        for i, (lo, hi) in enumerate(zip(starts, stops)):
            if hi <= lo:
                continue
            trial_idx.append(np.full(hi - lo, i))
            unit_idx.append(sc[lo:hi])
            t_rel.append(st[lo:hi] - event_times[i])
        unit_offset += uidx.size

    if not trial_idx:
        raise ValueError(f"No units found for region {region!r} on any probe.")
    return (
        np.concatenate(trial_idx),
        np.concatenate(unit_idx),
        np.concatenate(t_rel),
    )


def firing_rate_table(counts: np.ndarray, bin_size_s: float):
    """Per-unit firing statistics from a (trials x time x neurons) count tensor.

    Answers 'are these units actually active, and how were they doing?'. Returns a
    pandas DataFrame with one row per unit:

    * ``mean_rate_hz``     — mean firing rate over the whole window
    * ``spikes_per_trial`` — mean spike count per trial
    * ``frac_trials_active`` — fraction of trials with >= 1 spike (0 = often silent)
    * ``peak_rate_hz``     — peak of the trial-averaged PSTH
    * ``fano``             — Fano factor of per-trial counts (variance / mean; ~1 Poisson)
    """
    import pandas as pd

    n_trials, n_bins, n_units = counts.shape
    per_trial = counts.sum(axis=1)                    # (trials, units) total spikes/trial
    window_s = n_bins * bin_size_s

    mean_count = per_trial.mean(axis=0)               # per unit
    mean_rate = mean_count / window_s
    frac_active = (per_trial > 0).mean(axis=0)
    psth = counts.mean(axis=0) / bin_size_s           # (bins, units)
    peak_rate = psth.max(axis=0)
    var = per_trial.var(axis=0)
    fano = np.divide(var, mean_count, out=np.full(n_units, np.nan), where=mean_count > 0)

    return pd.DataFrame({
        "unit": np.arange(n_units),
        "mean_rate_hz": np.round(mean_rate, 3),
        "spikes_per_trial": np.round(mean_count, 3),
        "frac_trials_active": np.round(frac_active, 3),
        "peak_rate_hz": np.round(peak_rate, 2),
        "fano": np.round(fano, 2),
    }).sort_values("mean_rate_hz", ascending=False).reset_index(drop=True)


def drop_silent_units(
    counts: np.ndarray, min_total_spikes: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Remove units with too few spikes in the analysis window.

    A unit that never fires inside the window contributes nothing but can make the
    Poisson HMM's rate parameter degenerate (log of zero).

    NOTE: this is the *minimum* safety net, not an activity criterion. For HMM fitting use
    ``select_active_units`` — see the rationale there.

    Returns the trimmed tensor and the integer indices of the units kept.
    """
    totals = counts.sum(axis=(0, 1))
    keep = np.flatnonzero(totals >= min_total_spikes)
    return counts[:, :, keep], keep


def select_active_units(
    counts: np.ndarray,
    bin_size_s: float,
    *,
    min_rate_hz: float = 1.0,
    min_spikes_per_trial: float = 0.0,
    min_frac_trials_active: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep only units active enough to *support* a Poisson-HMM state. Returns (counts, kept).

    Why a rate threshold matters for the HMM specifically
    -----------------------------------------------------
    IBL "good" units are **well-isolated, not active** (`qc == 1.0` is a sorting-quality
    criterion). VISp here contains units firing ~0.004 Hz — a handful of spikes across the
    whole session. Keeping them is not neutral:

    * **They add emission dimensions without adding information.** A Poisson HMM estimates
      one rate per (state x neuron). A near-silent neuron contributes K parameters that are
      fit to a few spikes, i.e. almost pure noise.
    * **They push rates toward zero.** lambda ~= 0 is a degenerate MLE: it assigns
      probability 0 (log-likelihood -inf) to any count > 0. This is precisely what destroyed
      our first cross-validation (see docs/worklog/2026-07-12). Flooring lambda prevents the
      -inf, but it does not make those units *informative*.
    * **They dilute the likelihood.** Held-out log-likelihood is summed over neurons, so
      dozens of near-silent units spend the score on trivially-predictable zeros and mask the
      contribution of the units that actually carry the dynamics. This is a plausible reason
      the CV curve gains so little per added state.

    The threshold is a modelling choice, not a truth: it must be reported, and the result
    should be checked for sensitivity to it (try 0.5 / 1 / 2 Hz).

    Parameters
    ----------
    min_rate_hz
        Mean in-window firing rate. 1.0 Hz is a common, mild floor for population models.
    min_spikes_per_trial
        Optional: mean spikes per trial (a unit can pass a rate cut yet be absent from most
        trials).
    min_frac_trials_active
        Optional: fraction of trials with >= 1 spike. Guards against a unit whose rate comes
        from bursting on a few trials.
    """
    n_trials, n_bins, n_units = counts.shape
    window_s = n_bins * bin_size_s

    spikes_per_trial = counts.sum(axis=(0, 1)) / n_trials
    rate_hz = spikes_per_trial / window_s
    frac_active = (counts.sum(axis=1) > 0).mean(axis=0)

    keep = np.flatnonzero(
        (rate_hz >= min_rate_hz)
        & (spikes_per_trial >= min_spikes_per_trial)
        & (frac_active >= min_frac_trials_active)
    )
    return counts[:, :, keep], keep
