"""HMM read-outs and the end-to-end fit on a planted sequence (src/dynamics/hmm.py)."""

from types import SimpleNamespace

import numpy as np

from src.dynamics.hmm import (
    _forward_fraction, _repair_transmat, _transition_counts, fit_poisson_hmm,
    forward_fraction_per_trial, phase_labels, switch_rate,
)


def test_forward_fraction_limits():
    forward = np.array([[0, 0, 1, 1, 2, 2]] * 5)
    assert _forward_fraction(_transition_counts(forward, 3)) == 1.0
    symmetric = np.array([[0, 1, 2, 1, 0]] * 5)          # 2 forward, 2 backward
    assert _forward_fraction(_transition_counts(symmetric, 3)) == 0.5
    static = np.zeros((4, 6), dtype=int)
    assert _forward_fraction(_transition_counts(static, 3)) == 0.5


def test_switch_rate_counts_changes_not_direction():
    states = np.array([[0, 0, 1, 1], [0, 1, 0, 1], [2, 2, 2, 2]])
    assert switch_rate(states) == (1 + 3 + 0) / 3


def test_forward_fraction_per_trial_is_nan_without_transitions():
    states = np.array([[0, 1, 2], [2, 1, 0], [1, 1, 1]])
    out = forward_fraction_per_trial(states, 3)
    assert out[0] == 1.0 and out[1] == 0.0 and np.isnan(out[2])


def test_phase_labels_count_landmarks_passed():
    centers = np.array([-0.1, 0.05, 0.15, 0.3])
    landmarks = np.array([[0.0, 0.2],          # stimOn at 0, movement at 0.2 s
                          [0.0, np.nan]])      # no movement on this trial
    np.testing.assert_array_equal(phase_labels(centers, landmarks),
                                  [[0, 1, 1, 2], [0, 1, 1, 1]])


def test_repair_transmat_turns_unused_rows_into_self_transitions():
    model = SimpleNamespace(transmat_=np.array([[0.9, 0.1, 0.0],
                                                [0.0, 0.0, 0.0],
                                                [0.2, 0.0, 0.8]]))
    _repair_transmat(model)
    np.testing.assert_array_equal(model.transmat_[1], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(model.transmat_.sum(axis=1), 1.0)


def _planted_sequence(rng, n_trials=60, n_bins=30, n_units=15):
    """Every trial runs state 0 -> 1 -> 2 with jittered switch times; 8 baseline bins."""
    rates = rng.uniform(0.1, 0.6, (3, n_units))
    rates[1, :5] += 1.5
    rates[2, 5:10] += 1.5
    states = np.zeros((n_trials, n_bins), dtype=int)
    for i in range(n_trials):
        s1 = rng.integers(10, 14)
        s2 = rng.integers(19, 23)
        states[i, s1:s2], states[i, s2:] = 1, 2
    counts = rng.poisson(rates[states]).astype(np.int32)
    baseline = np.arange(n_bins) < 8
    return counts, states, baseline


def test_fit_poisson_hmm_recovers_a_planted_forward_sequence(rng):
    counts, true_states, baseline = _planted_sequence(rng)
    res = fit_poisson_hmm(counts, 3, baseline, n_iter=50, seed=0)

    assert res.states.shape == true_states.shape
    # States are returned in time order, so the planted labels should be recovered directly.
    assert (res.states == true_states).mean() > 0.9
    assert res.forward_fraction > 0.9
    np.testing.assert_allclose(res.occupancy.sum(axis=0), 1.0)
