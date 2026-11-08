"""Our hmmlearn Poisson HMM scores data exactly as an independent implementation (dynamax).

Test version of ``scripts/validate_hmm_vs_dynamax.py``. The script runs on the real session
(it downloads IBL data); this test runs the *same check* on synthetic Poisson-HMM data so it is
fast and offline: fit with hmmlearn, inject the fitted parameters into dynamax, and require the
two libraries to give the same held-out log-likelihood and the same Viterbi path.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("dynamax")
jax.config.update("jax_enable_x64", True)   # dynamax defaults to float32; we want a fair test
import jax.numpy as jnp                       # noqa: E402
from dynamax.hidden_markov_model import PoissonHMM as DxHMM   # noqa: E402
from hmmlearn.hmm import PoissonHMM            # noqa: E402

from src.data.build_matrices import stack_for_hmmlearn   # noqa: E402
from src.dynamics.hmm import _floor_lambdas               # noqa: E402

K = 4


def _synthetic_counts(rng, n_trials=90, n_bins=30, n_units=12):
    rates = rng.uniform(0.05, 1.5, (K, n_units))
    T = np.full((K, K), 0.03) + np.eye(K) * (1 - 0.03 * K)
    states = np.zeros((n_trials, n_bins), dtype=int)
    for i in range(n_trials):
        for t in range(1, n_bins):
            states[i, t] = rng.choice(K, p=T[states[i, t - 1]])
    return rng.poisson(rates[states]).astype(np.int64)


def test_hmmlearn_and_dynamax_agree_to_machine_precision(rng):
    counts = _synthetic_counts(rng)
    train, test = np.arange(0, 60), np.arange(60, 90)
    Xtr, ltr = stack_for_hmmlearn(counts[train])
    Xte, lte = stack_for_hmmlearn(counts[test])

    model = PoissonHMM(n_components=K, n_iter=40, random_state=0, tol=1e-2,
                       implementation="log")
    model.fit(Xtr, ltr)
    _floor_lambdas(model)

    dx = DxHMM(num_states=K, emission_dim=counts.shape[2])
    params, _ = dx.initialize(
        jax.random.PRNGKey(0),
        initial_probs=jnp.asarray(model.startprob_),
        transition_matrix=jnp.asarray(model.transmat_),
        emission_rates=jnp.asarray(model.lambdas_),
    )

    held = counts[test]
    ll_hmm = float(model.score(Xte, lte))
    ll_dyn = float(sum(dx.marginal_log_prob(params, jnp.asarray(x)) for x in held))
    assert ll_hmm == pytest.approx(ll_dyn, rel=1e-6)

    vit_hmm = model.predict(Xte, lte).reshape(held.shape[:2])
    vit_dyn = np.stack([np.asarray(dx.most_likely_states(params, jnp.asarray(x))) for x in held])
    assert (vit_hmm == vit_dyn).mean() > 0.999
