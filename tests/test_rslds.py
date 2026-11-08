"""rSLDS sequence statistics and the ground-truth recovery test (src/dynamics/rslds.py).

The ground-truth tests reuse the simulator in ``scripts/validate_rslds_ground_truth.py``
unchanged: latents simulated at 1 ms with three planted attractors visited 0 -> 1 -> 2, Poisson
spikes, then binned. The fast test runs one realistic configuration at the project's 40 ms bin
with fewer trials; the slow test runs the script's full sweep and checks the two claims the
project rests on.
"""

import numpy as np
import pytest

from src.dynamics.rslds import (
    counts_to_latents, fit_rslds, order_consistency, visit_sequences,
)


def test_visit_sequences_collapse_repeats():
    states = np.array([[0, 0, 1, 1, 2], [2, 2, 2, 0, 0]])
    assert visit_sequences(states) == [[0, 1, 2], [2, 0]]


def test_order_consistency_is_one_for_a_fixed_order_and_near_half_for_random(rng):
    fixed = np.repeat(np.array([[0] * 5 + [1] * 5 + [2] * 5]), 40, axis=0)
    assert order_consistency(fixed, 3) == 1.0

    shuffled = np.stack([rng.permutation(fixed[0]) for _ in range(400)])
    assert 0.45 < order_consistency(shuffled, 3) < 0.6


def _fit_ground_truth(gt, n_units, rate_hz, bin_ms):
    fine, z_fine = gt.simulate(n_units, rate_hz, seed=gt.SEED)
    counts, z_true = gt.rebin(fine, z_fine, bin_ms)
    x, _, _ = counts_to_latents(counts, n_latent=gt.D_TRUE, smooth_sigma_ms=40.0,
                                bin_size_s=bin_ms / 1000.0)
    res = fit_rslds(x, gt.K_TRUE, recurrent=True, n_iter=60, n_restarts=3,
                    seed=gt.SEED, sticky=2.0)
    return {
        "regime_acc": gt.regime_accuracy(res.states, z_true, gt.K_TRUE),
        "switches": float((res.states[:, 1:] != res.states[:, :-1]).sum(axis=1).mean()),
        "n_integrator": sum("integrator" in s for s in res.classify()),
        "order_cons": order_consistency(res.states, gt.K_TRUE),
    }


def test_rslds_recovers_planted_sequence_at_40ms(monkeypatch, ground_truth_script):
    gt = ground_truth_script
    monkeypatch.setattr(gt, "N_TRIALS", 100)          # smaller than the script's 300, for speed
    out = _fit_ground_truth(gt, n_units=44, rate_hz=22.0, bin_ms=40)   # the "SCm-like" config

    assert out["n_integrator"] == 0                    # no spurious line attractor
    assert out["order_cons"] >= 0.95                   # the 0 -> 1 -> 2 sequence is recovered
    assert out["regime_acc"] >= 0.8
    assert 1.5 <= out["switches"] <= 2.6               # truth: 2 switches per trial


@pytest.mark.slow
def test_full_ground_truth_sweep_supports_the_40ms_bin(ground_truth_script):
    """The script's full sweep (all configs x 4 bin sizes, about 5 minutes).

    Checks the two documented findings: at 40 ms no configuration produces a spurious
    integrator, and at 20 ms at least one does (shot noise manufactures |lambda| -> 1).
    """
    gt = ground_truth_script
    rows = [r for r in gt.run() if r["smoothing"] == "sigma_ms"]
    at40 = [r for r in rows if r["bin_ms"] == 40]
    at20 = [r for r in rows if r["bin_ms"] == 20]
    assert at40 and all(r["n_integrator"] == 0 for r in at40)
    assert any(r["n_integrator"] > 0 for r in at20)
