"""The fast rSLDS screen: attractor sequence, or stopwatch? On the real frozen data.

This is the exact-EM backend (`src/dynamics/rslds.py`) -- the fast one. It answers the questions
`ssm` is too slow to cross-validate, and tells us WHICH regions are worth the 15-20 min/region
`ssm` fit of record. Nothing here is the fit of record; it is the screen that decides where to
point the fit of record.

Per region, per alignment (stimOn + firstMovement), at the unified 40 ms / sigma 40 ms:

* **delta_recurrent** -- held-out LL(R free) - LL(R = 0), via `cross_validate_K` at the single K
  the HMM stability analysis chose. **THE test.** > 0 => switching depends on latent POSITION
  (an attractor sequence); ~ 0 => switching is a timed script (a stopwatch). This is the single
  biggest open question in the project.
* **order_consistency vs its own shuffle null** -- is there a reliable SEQUENCE, in either
  direction? Compared to the within-trial bin-shuffle null (which is NOT 0.5), with effect size.
* **admissibility()** -- is the fit in the shot-noise failure mode (|lambda| -> 1, every regime an
  "integrator")? If not admissible, the flow classification is refused -- especially for VISp,
  which the ground truth says recovers only ~0.71 of regimes even at 40 ms.

Why BOTH alignments: the [0, +0.8] s window is a MOVEMENT window (median RT 171 ms). A recurrence
result on stimOn that vanishes on firstMovement is a movement artefact, not a decision dynamic.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.preprocess import WORKING_EID, load_processed              # noqa: E402
from src.dynamics.rslds import (                                         # noqa: E402
    counts_to_latents, cross_validate_K, fit_rslds,
    order_consistency, order_consistency_null,
)

# K per region from docs/knowledge/best_K_per_region.md (FINAL). NOT selected here.
K_PER_REGION = {"VISp": 3, "SCm": 4, "MRN": 3}

BIN_S = 0.04
SIGMA_MS = 40.0
N_LATENT = 3
STICKY = 2.0
SEED = 0

# The switch count the ground truth used, per region, for the flickering check in admissibility.
# The synthetic truth was 2 switches/trial; real data is unknown, so we only use it as a loose
# flag (> 3x would be pathological). Pass None to skip the flickering test on real data.
TRUE_SWITCHES = None

ALIGNMENTS = {
    "stimOn": f"{WORKING_EID[:8]}__stimOn_pre400_post800_bin40_rate1.npz",
    "firstMovement": f"{WORKING_EID[:8]}__firstMovement_pre400_post800_bin40_rate1.npz",
}


def screen_region(counts: np.ndarray, K: int, bc: np.ndarray) -> dict:
    # --- the recurrence test: held-out LL(R free) - LL(R=0), CV by trial ---
    cv = cross_validate_K(
        counts, [K], n_latent=N_LATENT, smooth_sigma_ms=SIGMA_MS, bin_size_s=BIN_S,
        n_folds=3, n_iter=40, n_restarts=2, seed=SEED, compare_recurrent=True, sticky=STICKY)
    row = cv.iloc[0].to_dict()

    # --- one full fit for the sequence statistic + admissibility + flow labels ---
    x, _, ve = counts_to_latents(counts, n_latent=N_LATENT,
                                 smooth_sigma_ms=SIGMA_MS, bin_size_s=BIN_S)
    res = fit_rslds(x, K, recurrent=True, n_iter=60, n_restarts=3, seed=SEED, sticky=STICKY)
    res.bin_centers = bc

    oc = order_consistency(res.states, K)
    null = order_consistency_null(res.states, K, n_shuffles=200, seed=SEED)
    adm = res.admissibility(true_switches=TRUE_SWITCHES)

    return {
        "K": K,
        "n_units": int(counts.shape[2]),
        "var_exp_D3": round(float(ve.sum()), 3),
        "delta_recurrent": round(float(row["delta_recurrent"]), 4),
        "ll_Rfree": round(float(row["mean_test_ll"]), 4),
        "ll_R0": round(float(row["mean_test_ll_norec"]), 4),
        "order_consistency": round(float(oc), 3),
        "oc_null_mean": round(float(null.mean()), 3),
        "oc_null_p95": round(float(np.percentile(null, 95)), 3),
        "oc_effect": round(float(oc - null.mean()), 3),
        "oc_beats_null": bool(oc > np.percentile(null, 95)),
        "flow_admissible": adm["flow_admissible"],
        "sequence_admissible": adm["sequence_admissible"],
        "adm_reason": adm["reason"],
        "max_abs_lambda": adm["max_abs_lambda"],
        "n_integrator_regimes": adm["n_integrator_regimes"],
        "switches_per_trial": adm["switches_per_trial"],
        "flow_labels": res.classify(),
    }


def main() -> None:
    all_rows = []
    for align, fname in ALIGNMENTS.items():
        d = load_processed(ROOT / "data" / "processed" / fname)
        bc = d["bin_centers"]
        print(f"\n########## alignment = {align}  ({d['n_trials']} trials, "
              f"{len(bc)} bins) ##########", flush=True)

        for region, rd in d["regions"].items():
            K = K_PER_REGION[region]
            t0 = time.time()
            r = screen_region(rd["counts"], K, bc)
            r.update({"alignment": align, "region": region})
            all_rows.append(r)

            tag = "flow OK  " if r["flow_admissible"] else "flow SUSPECT"
            rec = ("RECURRENT (attractor seq)" if r["delta_recurrent"] > 0
                   else "no recurrence (stopwatch)")
            seq = "SEQUENCE" if r["oc_beats_null"] else "no sequence"
            print(f"  {region:5s} K={K}  "
                  f"delta_rec={r['delta_recurrent']:+.4f} [{rec}]  "
                  f"OC={r['order_consistency']:.3f} vs null {r['oc_null_mean']:.3f} "
                  f"(eff {r['oc_effect']:+.3f}, {seq})  "
                  f"|lam|={r['max_abs_lambda']:.3f} int={r['n_integrator_regimes']}/{K}  "
                  f"[{tag}]  [{time.time()-t0:.0f}s]", flush=True)
            print(f"        flows: {r['flow_labels']}", flush=True)

    out = ROOT / "results" / "rslds" / "screen_40ms.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "session": WORKING_EID, "bin_s": BIN_S, "sigma_ms": SIGMA_MS,
        "n_latent": N_LATENT, "K_per_region": K_PER_REGION, "rows": all_rows,
    }, indent=1))
    print(f"\nwrote {out.relative_to(ROOT)}")

    print("\n=== READ-OUT ===")
    print("delta_recurrent > 0  => switching depends on latent POSITION (attractor sequence)")
    print("delta_recurrent ~ 0  => switching is a timed script (stopwatch)")
    print("A result that holds on stimOn but NOT on firstMovement is a movement artefact.")
    print("Only regions marked admissible get an ssm fit of record.")


if __name__ == "__main__":
    main()
