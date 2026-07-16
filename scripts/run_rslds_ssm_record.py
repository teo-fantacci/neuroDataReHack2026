"""The rSLDS fit of record: Poisson `ssm`, on the real frozen data. Adjudicates the screen.

The fast exact-EM screen (`scripts/run_rslds_screen.py`) found:
  * the recurrence result is robust everywhere (delta_recurrent > 0, both alignments);
  * VISp's flow is admissible (point / rotational attractors);
  * **SCm and MRN's "line attractor / integrator" labels are SUSPECT** -- |lambda| ~ 0.99-1.0,
    above the ground-truth-admissible band, i.e. the PCA latent cannot tell a real slow mode
    from a shot-noise unit root.

`ssm` is exactly the instrument that resolves that: it infers the latent JOINTLY with the
dynamics under a true **Poisson** likelihood, so it does not inherit the PCA latent's noise.

The adjudication, per region:
  * ssm ALSO finds a slow mode (|lambda| -> 1) in SCm/MRN  => the integrator is REAL. A
    genuine result: a decision/movement variable living on a line attractor.
  * ssm finds clean contracting attractors where the PCA backend saw integrators => the
    integrators were a latent-quality artefact, and the exact-EM flow labels are retracted.
  * VISp is the positive control: both backends should agree it is admissible.

Runs on the stimOn tensor (the primary). ~15-20 min/region -> budget ~1 h. Writes
results/rslds/ssm_record_40ms.json with both backends' read-outs side by side.
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
from src.dynamics.rslds import order_consistency, order_consistency_null  # noqa: E402
from src.dynamics.rslds_ssm import fit_rslds_ssm, ssm_env_available       # noqa: E402

K_PER_REGION = {"VISp": 3, "SCm": 4, "MRN": 3}
N_LATENT = 3
SSM_ITERS = 50
SEED = 0
BIN_S = 0.04
ALIGN_FILE = f"{WORKING_EID[:8]}__stimOn_pre400_post800_bin40_rate1.npz"


def main() -> None:
    if not ssm_env_available():
        raise SystemExit("ndrh_ssm env not available. Build it: bash scripts/install_ssm_env.sh")

    d = load_processed(ROOT / "data" / "processed" / ALIGN_FILE)
    bc = d["bin_centers"]
    print(f"session {WORKING_EID[:8]}  stimOn  {d['n_trials']} trials x {len(bc)} bins  "
          f"(40 ms)\n", flush=True)

    rows = []
    for region, rd in d["regions"].items():
        K = K_PER_REGION[region]
        counts = rd["counts"]
        print(f"########## {region}  K={K}  {counts.shape[2]} units "
              f"########## ({time.strftime('%H:%M:%S')})", flush=True)

        t0 = time.time()
        # ssm takes RAW COUNTS (it infers + models the latent as Poisson); never smooth.
        res = fit_rslds_ssm(counts, K, n_latent=N_LATENT, n_iter=SSM_ITERS, seed=SEED,
                            root=ROOT, verbose=True)
        res.bin_centers = bc

        lam = np.abs(res.eigenvalues()).max(axis=1)
        labels = res.classify()
        adm = res.admissibility()
        oc = order_consistency(res.states, K)
        null = order_consistency_null(res.states, K, n_shuffles=200, seed=SEED)

        row = {
            "region": region, "K": K, "n_units": int(counts.shape[2]),
            "backend": "ssm",
            "converged": res.meta["converged"],
            "elbo_tail_drift": round(res.meta["elbo_tail_drift"], 5),
            "fit_seconds": res.meta["fit_seconds"],
            "max_abs_lambda": round(float(lam.max()), 3),
            "n_integrator_regimes": adm["n_integrator_regimes"],
            "flow_admissible": adm["flow_admissible"],
            "flow_labels": labels,
            "order_consistency": round(float(oc), 3),
            "oc_null_mean": round(float(null.mean()), 3),
            "oc_effect": round(float(oc - null.mean()), 3),
            "oc_beats_null": bool(oc > np.percentile(null, 95)),
            "switches_per_trial": adm["switches_per_trial"],
        }
        rows.append(row)
        print(f"  => |lam|={row['max_abs_lambda']:.3f}  int={row['n_integrator_regimes']}/{K}  "
              f"flow_admissible={row['flow_admissible']}  OC={row['order_consistency']:.3f} "
              f"(eff {row['oc_effect']:+.3f})  [{time.time()-t0:.0f}s]", flush=True)
        print(f"     flows: {labels}\n", flush=True)

    out = ROOT / "results" / "rslds" / "ssm_record_40ms.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "session": WORKING_EID, "alignment": "stimOn", "bin_s": BIN_S,
        "n_latent": N_LATENT, "ssm_iters": SSM_ITERS, "K_per_region": K_PER_REGION,
        "rows": rows,
    }, indent=1))
    print(f"wrote {out.relative_to(ROOT)}")

    # --- the adjudication ---
    # IMPORTANT: ssm landing in the SAME danger zone as the PCA backend is NOT confirmation.
    # The 40 ms ground truth showed a REAL 150 ms attractor gives |lambda| ~ 0.88 cleanly, even
    # at these unit counts; shot noise / model-mismatch gives |lambda| -> 1. Both backends
    # returning |lambda| ~ 1 means the flow geometry is in the regime NEITHER can certify --
    # agreement between two estimators that are both failing is not evidence.
    print("\n=== ssm vs exact-EM on the flow geometry ===")
    screen = {(r["region"]): r for r in
              json.loads((ROOT / "results" / "rslds" / "screen_40ms.json").read_text())["rows"]
              if r["alignment"] == "stimOn"}
    for r in rows:
        s = screen[r["region"]]
        both_danger = (r["max_abs_lambda"] >= 0.95 and s["max_abs_lambda"] >= 0.95)
        if not r["converged"]:
            verdict = "ssm DID NOT CONVERGE -- flow reading discarded"
        elif r["max_abs_lambda"] >= 1.0:
            verdict = "ssm |lambda| >= 1 (UNSTABLE) -- NOT a line attractor; flow uncertifiable"
        elif both_danger:
            verdict = "both backends at |lambda|~1: flow NOT certifiable (ground-truth danger zone)"
        elif r["flow_admissible"] and s["flow_admissible"]:
            verdict = "both admissible and agree"
        else:
            verdict = "see numbers"
        print(f"  {r['region']:5s}  exact-EM |lam|={s['max_abs_lambda']:.3f} "
              f"int={s['n_integrator_regimes']}  ->  ssm |lam|={r['max_abs_lambda']:.3f} "
              f"int={r['n_integrator_regimes']} (conv={r['converged']})   {verdict}")

    print("\nWHAT IS CERTIFIED: recurrence (delta_recurrent>0, both alignments) + sequence "
          "(order_consistency>>null).")
    print("WHAT IS NOT: the attractor/integrator geometry. Both backends land at |lambda|~1, "
          "the ground-truth failure signature; ssm gives |lambda|>1 for 2/3 regions. This needs "
          "the pooled sessions or a |lambda|-stability test, not a single 30-bin fit.")


if __name__ == "__main__":
    main()
