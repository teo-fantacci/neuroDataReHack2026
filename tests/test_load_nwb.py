"""The NWB route (src/data/load_nwb.py).

The fast tests check the NWB-to-IBL trial mapping on hand-made columns (no download). The slow
test streams the working session from DANDI and checks it against facts verified on 2026-11-08.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from src.data.load_nwb import TRIAL_COLUMNS, trials_from_columns


def _columns():
    return {
        "start_time": [0.0, 5.0, 10.0],
        "stop_time": [4.0, 9.0, 14.0],
        "gabor_stimulus_onset_time": [0.5, 5.5, 10.5],
        "gabor_stimulus_offset_time": [2.0, 7.0, 12.0],
        "auditory_cue_time": [0.5, 5.5, 10.5],
        "wheel_movement_onset_time": [0.7, 5.6, np.nan],
        "choice_registration_time": [0.9, 5.9, 13.0],
        "feedback_time": [0.9, 5.9, 13.0],
        "probability_left": [0.5, 0.8, 0.2],
        "reward_volume_uL": [1.5, 0.0, 0.0],
        "mouse_wheel_choice": [b"clockwise", b"counter_clockwise", b"no_go"],
        "is_mouse_rewarded": [True, False, False],
        "gabor_stimulus_contrast": [100.0, 12.5, 0.0],
        "gabor_stimulus_side": [b"left", b"right", b"left"],
    }


def test_trial_columns_are_renamed_to_ibl_names():
    df = trials_from_columns(_columns())
    for alf in TRIAL_COLUMNS.values():
        assert alf in df.columns
    np.testing.assert_array_equal(df["stimOn_times"], [0.5, 5.5, 10.5])
    assert np.isnan(df["firstMovement_times"][2])


def test_choice_feedback_and_contrast_use_ibl_encodings():
    df = trials_from_columns(_columns())
    np.testing.assert_array_equal(df["choice"], [1.0, -1.0, 0.0])        # CW +1, CCW -1, no-go 0
    np.testing.assert_array_equal(df["feedbackType"], [1.0, -1.0, -1.0])
    np.testing.assert_array_equal(df["contrastLeft"], [1.0, np.nan, 0.0])  # percent -> fraction
    np.testing.assert_array_equal(df["contrastRight"], [np.nan, 0.125, np.nan])


def test_the_paper_mask_runs_on_the_mapped_table():
    brainwidemap = pytest.importorskip("brainwidemap")
    df = trials_from_columns(_columns())
    # truncate_to_pass needs hundreds of trials to be meaningful; switch it off for 3 trials.
    trials, mask = brainwidemap.load_trials_and_mask(
        None, "test", sess_loader=SimpleNamespace(trials=df), truncate_to_pass=False)
    # trial 0: RT 0.2 s kept; trial 1: RT 0.1 s kept; trial 2: no movement -> excluded
    np.testing.assert_array_equal(mask.to_numpy(), [True, True, False])


@pytest.mark.slow
def test_working_session_streams_from_dandi():
    """Network test: the working session on DANDI:000409 (about a minute)."""
    from src.data.load_nwb import good_units, open_session, trials_and_mask

    with open_session("5c0c560e-9e1f-45e9-b66e-e4ee7855be84") as s:
        trials, mask = trials_and_mask(s.nwbfile)
        assert len(trials) == 1100
        assert int(mask.sum()) == 843                     # same count as the ONE route
        spikes, clusters = good_units(s.nwbfile, "Probe00", regions=["SCm"])
    assert len(clusters) > 0 and set(clusters["Beryl"]) == {"SCm"}
    assert np.all(np.diff(spikes["times"]) >= 0)          # sorted, as the binning requires
    assert spikes["clusters"].max() == len(clusters) - 1
