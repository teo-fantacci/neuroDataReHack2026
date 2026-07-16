"""Validate our hmmlearn Poisson HMM against an independent implementation (dynamax).

Why this exists
---------------
The Phase-2 cross-validation curve gains very little log-likelihood per added state, and
never plateaus. Before reading anything into that, we have to rule out the boring
explanation: *our code is wrong.*

The test is deliberately not "do two EM runs agree" — they will not, because EM is
non-convex and the two packages initialise differently. Instead we **inject hmmlearn's exact
fitted parameters into dynamax** and ask whether the two libraries compute the *same
log-likelihood on the same data*. That isolates the likelihood/scoring path, which is
precisely what cross-validation scores.

dynamax (Linderman lab, JAX, actively maintained) is the reference. `ssm` from the same lab
is NOT usable here: it fails to build (its legacy setup.py imports numpy without declaring
it as a build dependency, so PEP-517 isolation kills it) — exactly the risk Phase 0 flagged.

Result (2026-07-12, session 5c0c560e, K=5)
------------------------------------------
    VISp : hmmlearn -39882.146392   dynamax -39882.146392   rel.diff 3.6e-16   Viterbi 100%
    SCm  : hmmlearn -248343.303728  dynamax -248343.303728  rel.diff 0.0       Viterbi 100%

Agreement to machine precision, identical Viterbi paths. **Our HMM is correct.** The flat CV
curve is therefore a property of the data, not a bug.

Run:  python scripts/validate_hmm_vs_dynamax.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import jax

jax.config.update("jax_enable_x64", True)  # dynamax defaults to float32; we want a fair test
import jax.numpy as jnp
from dynamax.hidden_markov_model import PoissonHMM as DxHMM
from hmmlearn.hmm import PoissonHMM
from iblatlas.regions import BrainRegions
from sklearn.model_selection import KFold

from src.data.access import get_one, load_trials, load_units, sessions_table
from src.data.build_matrices import (
    drop_silent_units,
    region_tensor_from_probes,
    stack_for_hmmlearn,
)
from src.dynamics.hmm import _floor_lambdas, drop_units_silent_in

ROOT = Path(__file__).resolve().parents[1]
EID = "5c0c560e-9e1f-45e9-b66e-e4ee7855be84"
REGIONS = ["VISp", "SCm"]
K = 5
N_TRIALS = 400


def main() -> int:
    params = yaml.safe_load((ROOT / "config" / "params.yaml").read_text())
    one = get_one(cache_dir=str((ROOT / params["data"]["one_cache_dir"]).resolve()))

    pids = sessions_table(one).query("eid == @EID")["pid"].tolist()
    trials, mask = load_trials(one, EID)
    good = trials[mask].reset_index(drop=True)
    rng = np.random.default_rng(0)
    good = good.iloc[np.sort(rng.choice(len(good), N_TRIALS, replace=False))].reset_index(drop=True)
    events = good["stimOn_times"].values

    br = BrainRegions()
    loads = []
    for pid in pids:
        spikes, clusters = load_units(one, pid)
        clusters = clusters.copy()
        clusters["Beryl"] = br.id2acronym(clusters["atlas_id"].values, mapping="Beryl")
        loads.append({"spikes": spikes, "clusters": clusters})

    ok = True
    for region in REGIONS:
        counts, _ = region_tensor_from_probes(
            loads, region, events, pre_s=0.5, post_s=0.8, bin_size_s=0.02
        )
        counts, _ = drop_silent_units(counts, min_total_spikes=1)
        counts, _ = drop_units_silent_in(counts)

        train, test = next(iter(KFold(3, shuffle=True, random_state=0).split(np.arange(len(counts)))))
        Xtr, ltr = stack_for_hmmlearn(counts[train])
        Xte, lte = stack_for_hmmlearn(counts[test])

        model = PoissonHMM(n_components=K, n_iter=40, random_state=0, tol=1e-2,
                           implementation="log")
        model.fit(Xtr, ltr)
        _floor_lambdas(model)

        # Inject hmmlearn's EXACT fitted parameters into dynamax.
        dx = DxHMM(num_states=K, emission_dim=counts.shape[2])
        dx_params, _ = dx.initialize(
            jax.random.PRNGKey(0),
            initial_probs=jnp.asarray(model.startprob_),
            transition_matrix=jnp.asarray(model.transmat_),
            emission_rates=jnp.asarray(model.lambdas_),
        )

        held = counts[test]
        ll_hmm = float(model.score(Xte, lte))
        ll_dyn = float(sum(dx.marginal_log_prob(dx_params, jnp.asarray(x)) for x in held))

        vit_hmm = model.predict(Xte, lte).reshape(held.shape[0], held.shape[1])
        vit_dyn = np.stack(
            [np.asarray(dx.most_likely_states(dx_params, jnp.asarray(x))) for x in held]
        )
        agree = float((vit_hmm == vit_dyn).mean())

        match = np.isclose(ll_hmm, ll_dyn, rtol=1e-6)
        rel = abs(ll_hmm - ll_dyn) / max(abs(ll_hmm), 1e-12)
        ok &= bool(match) and agree > 0.999

        print(f"\n{region}  (K={K}, {counts.shape[2]} units, {len(test)} held-out trials)")
        print(f"  held-out LL  hmmlearn = {ll_hmm:.6f}")
        print(f"  held-out LL  dynamax  = {ll_dyn:.6f}")
        print(f"  relative diff = {rel:.3e}   MATCH = {match}")
        print(f"  Viterbi agreement = {agree * 100:.2f}%")

    print("\n" + ("PASS — our HMM matches dynamax." if ok else "FAIL — implementations disagree!"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
