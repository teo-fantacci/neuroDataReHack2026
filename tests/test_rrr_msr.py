"""Reduced-rank regression (src/dynamics/rrr.py) and multiscale relevance (src/selection/msr.py)."""

import numpy as np

from src.dynamics.rrr import _r2, rrr_fit
from src.selection.msr import msr, msr_table


def test_rrr_recovers_a_rank_two_mapping(rng):
    n, nx, ny, rank = 400, 12, 10, 2
    X = rng.standard_normal((n, nx))
    W = rng.standard_normal((nx, rank)) @ rng.standard_normal((rank, ny))
    Y = X @ W + 0.1 * rng.standard_normal((n, ny))

    B2 = rrr_fit(X, Y, rank=2, alpha=1e-6)
    B1 = rrr_fit(X, Y, rank=1, alpha=1e-6)
    assert np.linalg.matrix_rank(B2, tol=1e-8) == 2
    r2_2 = _r2(Y, (X - X.mean(0)) @ B2 + Y.mean(0))
    r2_1 = _r2(Y, (X - X.mean(0)) @ B1 + Y.mean(0))
    assert r2_2 > 0.98 and r2_2 > r2_1


def test_msr_is_nan_for_too_few_spikes_and_bounded_otherwise(rng):
    assert np.isnan(msr(np.array([0.1, 0.5, 0.9])))
    poisson = np.sort(rng.uniform(0, 200, 2000))
    value = msr(poisson)
    assert np.isfinite(value) and 0.0 <= value <= 1.0


def test_msr_table_is_ranked_and_reports_rate(rng):
    trains = {u: np.sort(rng.uniform(0, 100, n)) for u, n in [(1, 300), (2, 1500), (3, 5)]}
    table = msr_table(trains, t_start=0.0, t_stop=100.0)
    scored = table["msr"].dropna().to_numpy()
    assert np.all(np.diff(scored) <= 0)                       # ranked high to low
    assert table["msr"].isna().sum() == 1                     # the 5-spike unit is not scored
    assert table.set_index("unit").loc[2, "rate_hz"] == 15.0
