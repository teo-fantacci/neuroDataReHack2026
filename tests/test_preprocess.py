"""Preprocessing transforms, the frozen config, and the .npz round trip (src/data/preprocess.py)."""

import numpy as np
import pytest

from src.data.preprocess import (
    PreprocessConfig, anscombe, load_processed, save_processed, sigma_ms_to_bins,
)


def test_frozen_defaults_match_the_documented_pipeline():
    """These values are documented in README/CLAUDE.md; a silent change here is a bug."""
    cfg = PreprocessConfig()
    assert cfg.align_event == "stimOn_times"
    assert cfg.pre_s == 0.3 and cfg.post_s == 0.8
    assert cfg.bin_size_s == 0.04
    assert cfg.min_rate_hz == 1.0
    assert cfg.n_bins == 28
    assert cfg.baseline_mask.sum() == 7          # 7 pre-stimulus bins of 40 ms
    assert cfg.tag() == "stimOn_pre300_post800_bin40_rate1"


def test_anscombe_values_and_monotonicity():
    np.testing.assert_allclose(anscombe(np.array([0, 1, 2])),
                               2 * np.sqrt(np.array([0, 1, 2]) + 0.375))
    y = anscombe(np.arange(50))
    assert np.all(np.diff(y) > 0)


def test_sigma_is_specified_in_ms_not_bins():
    assert sigma_ms_to_bins(40.0, 0.04) == pytest.approx(1.0)
    assert sigma_ms_to_bins(60.0, 0.02) == pytest.approx(3.0)


def _fake_processed(rng, n_trials=6):
    cfg = PreprocessConfig()
    return {
        "bin_centers": cfg.bin_centers,
        "baseline_mask": cfg.baseline_mask,
        "config": {"pre_s": cfg.pre_s, "bin_size_s": cfg.bin_size_s},
        "n_trials": n_trials,
        "regions": {
            "SCm": {"counts": rng.poisson(0.5, (n_trials, cfg.n_bins, 4)).astype(np.int32),
                    "keep_idx": np.array([0, 2, 3, 5])},
            "MRN": {"counts": rng.poisson(0.3, (n_trials, cfg.n_bins, 2)).astype(np.int32),
                    "keep_idx": np.array([1, 4])},
        },
        "trials": {"choice": rng.choice([-1.0, 1.0], n_trials)},
    }


def test_save_load_round_trip_is_exact(tmp_path, rng):
    data = _fake_processed(rng)
    path = save_processed(data, tmp_path / "x.npz")
    back = load_processed(path)

    assert back["config"] == data["config"]
    assert back["n_trials"] == data["n_trials"]
    for region, r in data["regions"].items():
        np.testing.assert_array_equal(back["regions"][region]["counts"], r["counts"])
        assert back["regions"][region]["counts"].dtype == np.int32   # raw counts, never floats
        np.testing.assert_array_equal(back["regions"][region]["keep_idx"], r["keep_idx"])
    np.testing.assert_array_equal(back["trials"]["choice"], data["trials"]["choice"])


def test_save_refuses_covariates_out_of_register_with_trials(tmp_path, rng):
    data = _fake_processed(rng)
    data["trials"]["choice"] = data["trials"]["choice"][:-1]
    with pytest.raises(AssertionError, match="not in register"):
        save_processed(data, tmp_path / "bad.npz")
