"""Spike-time -> count-tensor helpers (src/data/build_matrices.py)."""

import numpy as np
import pytest

from src.data.build_matrices import bin_spikes_by_event, drop_silent_units, stack_for_hmmlearn


def test_bin_spikes_by_event_places_spikes_in_the_right_bins():
    # Two trials with events at t=10 s and t=20 s; window [-0.2, +0.4) s in 0.1 s bins.
    spike_times = np.array([9.85, 9.95, 10.05, 10.39, 10.41, 19.81, 20.15])
    spike_clusters = np.array([0, 1, 0, 1, 0, 1, 1])
    counts, centers = bin_spikes_by_event(
        spike_times, spike_clusters, n_clusters=2,
        event_times=np.array([10.0, 20.0]), pre_s=0.2, post_s=0.4, bin_size_s=0.1)

    assert counts.shape == (2, 6, 2)
    assert counts.dtype == np.int32
    np.testing.assert_allclose(centers, [-0.15, -0.05, 0.05, 0.15, 0.25, 0.35])
    # trial 0: 9.85 -> bin 0 (unit 0), 9.95 -> bin 1 (unit 1), 10.05 -> bin 2 (unit 0),
    # 10.39 -> bin 5 (unit 1); 10.41 is past the window and is not counted.
    # (A spike at exactly +post_s is avoided on purpose: float rounding decides its bin.)
    expected0 = np.zeros((6, 2), dtype=int)
    expected0[0, 0] = expected0[1, 1] = expected0[2, 0] = expected0[5, 1] = 1
    np.testing.assert_array_equal(counts[0], expected0)
    # trial 1: 19.81 -> bin 0 (unit 1), 20.15 -> bin 3 (unit 1)
    assert counts[1, 0, 1] == 1 and counts[1, 3, 1] == 1 and counts[1].sum() == 2


def test_bin_spikes_by_event_rejects_nan_events():
    with pytest.raises(ValueError, match="NaN"):
        bin_spikes_by_event(np.array([1.0]), np.array([0]), 1, np.array([np.nan]))


def test_bin_spikes_by_event_conserves_spikes_inside_the_window(rng):
    spike_times = np.sort(rng.uniform(0, 100, 5000))
    clusters = rng.integers(0, 7, spike_times.size)
    events = np.arange(5.0, 95.0, 3.0)
    counts, _ = bin_spikes_by_event(spike_times, clusters, 7, events,
                                    pre_s=0.3, post_s=0.8, bin_size_s=0.04)
    # Windows [event - 0.3, event + 0.8] do not overlap, so every spike inside one is counted once.
    rel = spike_times[None, :] - events[:, None]
    in_window = ((rel >= -0.3) & (rel <= 0.8)).sum()
    assert counts.sum() == in_window


def test_stack_for_hmmlearn_is_trial_major():
    counts = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    X, lengths = stack_for_hmmlearn(counts)
    assert X.shape == (6, 4)
    assert lengths == [3, 3]
    np.testing.assert_array_equal(X[3], counts[1, 0])


def test_drop_silent_units_keeps_only_units_that_fire():
    counts = np.zeros((4, 5, 3), dtype=int)
    counts[0, 0, 0] = 2
    counts[2, 4, 2] = 1
    trimmed, keep = drop_silent_units(counts)
    np.testing.assert_array_equal(keep, [0, 2])
    assert trimmed.shape == (4, 5, 2)
