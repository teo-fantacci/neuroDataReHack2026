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


def drop_silent_units(
    counts: np.ndarray, min_total_spikes: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Remove units with too few spikes in the analysis window.

    A unit that never fires inside the window contributes nothing but can make the
    Poisson HMM's rate parameter degenerate (log of zero).

    Returns the trimmed tensor and the integer indices of the units kept.
    """
    totals = counts.sum(axis=(0, 1))
    keep = np.flatnonzero(totals >= min_total_spikes)
    return counts[:, :, keep], keep
