"""The out-of-process bridge to Linderman's `ssm` (src/dynamics/rslds_ssm.py).

Needs the optional `ndrh_ssm` conda env (build it with ``bash scripts/install_ssm_env.sh``),
so it is marked slow and skipped when the env is absent. Never runs in CI.
"""

import numpy as np
import pytest

from src.dynamics.rslds_ssm import fit_rslds_ssm, ssm_env_available


@pytest.mark.slow
def test_ssm_bridge_fits_a_small_poisson_rslds(rng):
    if not ssm_env_available():
        pytest.skip("optional `ndrh_ssm` env not built")
    counts = rng.poisson(0.5, (20, 25, 8)).astype(np.int32)
    res = fit_rslds_ssm(counts, 2, n_latent=2, n_iter=5, verbose=False)
    assert res.states.shape == (20, 25)
    assert set(np.unique(res.states)) <= {0, 1}
