"""Does the HMM give a comparable answer at 50 ms as at 20 ms?

The question
------------
The rSLDS *needs* 50 ms bins: its ground truth shows that at 20 ms the PCA latent is shot-noise
dominated, |lambda| -> 1, and every regime gets classified "line attractor / integrator" -- a
spurious result that looks like a finding. The HMM, meanwhile, has produced all of its numbers
so far at 20 ms.

If the HMM's answers survive at 50 ms, the whole project can share one timebase, which is what
the unified preprocessing is for. If they do not, we need two, and we need to say why.

What is measured, at the K chosen by the ARI stability analysis (VISp 3, SCm 4, MRN 3):

* ``ari``                 -- do the states still REPRODUCE across independent fits?
* ``forward_fraction``    -- is the directedness result bin-size dependent?
* ``forward_fraction_split_half`` -- ... and does it survive an ordering derived on held-out trials?
* ``switch_rate``         -- state changes per trial. NOT directedness; reported so the two are
                             never conflated again.
* ``baseline_exit_s``     -- the inter-areal timing read-out (robust version; the old one
                             returned NaN in 2 of 3 regions).

Cost: 4 bin sizes x 3 regions x (1 fit + stability). Expect tens of minutes.
Writes results/hmm/binsize_check.json, which notebooks/00_preprocessing.ipynb plots.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.access import get_one                                    # noqa: E402
from src.data.build_matrices import (                                  # noqa: E402
    drop_silent_units, region_tensor_from_probes, select_active_units,
)
from src.data.preprocess import REGIONS, WORKING_EID, load_session, align_times  # noqa: E402
from src.dynamics.hmm import (                                         # noqa: E402
    baseline_exit_latency_robust, fit_poisson_hmm, forward_fraction_split_half,
    state_stability, switch_rate,
)

# From docs/knowledge/best_K_per_region.md (FINAL, 2026-07-14). NOT re-selected here.
BEST_K = {"VISp": 3, "SCm": 4, "MRN": 3}

BIN_SIZES = [0.020, 0.030, 0.040, 0.050]
PRE_S, POST_S = 0.4, 0.8
MIN_RATE_HZ = 1.0
N_REPEATS = 3        # stability repeats (5 in the definitive run; 3 keeps this tractable)
N_RESTARTS = 2       # EM restarts -- NEVER 1: see best_K_per_region.md section 3
SEED = 0

OUT = ROOT / "results" / "hmm" / "binsize_check.json"


def main() -> None:
    one = get_one(cache_dir=ROOT / "data" / "ONE")
    trials, mask, probe_loads = load_session(one, WORKING_EID)
    events = align_times(trials, mask, "stimOn_times")
    print(f"session {WORKING_EID[:8]}  {len(events)} trials", flush=True)

    rows = []
    for bs in BIN_SIZES:
        n_bins = int(round((PRE_S + POST_S) / bs))
        bc = (np.arange(n_bins) + 0.5) * bs - PRE_S
        baseline_mask = bc < 0.0

        for region in REGIONS:
            K = BEST_K[region]
            t0 = time.time()

            counts, _ = region_tensor_from_probes(
                probe_loads, region, events, pre_s=PRE_S, post_s=POST_S, bin_size_s=bs)
            counts, _ = drop_silent_units(counts, min_total_spikes=1)
            counts, _ = select_active_units(counts, bs, min_rate_hz=MIN_RATE_HZ)

            res = fit_poisson_hmm(counts, K, baseline_mask, seed=SEED, bias_startprob=True)
            stab = state_stability(counts, K, n_repeats=N_REPEATS, seed=SEED,
                                   n_restarts=N_RESTARTS)
            sh = forward_fraction_split_half(counts, K, baseline_mask,
                                             n_repeats=N_REPEATS, seed=SEED)
            exit_ = baseline_exit_latency_robust(res.occupancy, bc)

            row = {
                "bin_ms": int(bs * 1000),
                "region": region,
                "K": K,
                "n_units": int(counts.shape[2]),
                "n_bins": n_bins,
                "forward_fraction": round(res.forward_fraction, 3),
                "forward_fraction_split_half": round(sh["forward_fraction_split_half"], 3),
                "ff_split_half_std": round(sh["std"], 3),
                "switch_rate": round(switch_rate(res.states), 2),
                "ari_mean": round(stab["ari_mean"], 3),
                "ari_std": round(stab["ari_std"], 3),
                "baseline_exit_s": (None if not np.isfinite(exit_["latency_s"])
                                    else round(exit_["latency_s"], 3)),
                "baseline_states": exit_["baseline_states"],
                "baseline_level": exit_["baseline_level"],
                "seconds": round(time.time() - t0, 1),
            }
            rows.append(row)
            print(f"  {bs*1000:.0f}ms {region:5s} K={K}  "
                  f"ff={row['forward_fraction']:.3f} (split-half {row['forward_fraction_split_half']:.3f})  "
                  f"ARI={row['ari_mean']:.3f}  switches/trial={row['switch_rate']:.1f}  "
                  f"exit={row['baseline_exit_s']}  [{row['seconds']:.0f}s]", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "session": WORKING_EID,
        "best_k": BEST_K,
        "pre_s": PRE_S, "post_s": POST_S, "min_rate_hz": MIN_RATE_HZ,
        "n_repeats": N_REPEATS, "n_restarts": N_RESTARTS,
        "rows": rows,
    }, indent=1))
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
