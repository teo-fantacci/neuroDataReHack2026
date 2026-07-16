"""Is the rSLDS's |lambda| ~ 1 a REAL slow mode, or a fitting artefact? Three decisive tests.

Context
-------
Both backends (exact-EM and Poisson ssm) returned |lambda| ~ 1 for SCm/MRN, i.e. "line
attractor / integrator" -- but that is exactly the shot-noise/model-mismatch failure signature
the ground truth warned about, and it cannot be told apart from a genuine slow mode by looking at
the number alone. This script settles it, on the fast exact-EM backend (the only one cheap enough
to refit dozens of times).

The logic: a genuine dynamical property is REPRODUCIBLE, DATA-STABLE, and TEMPORALLY GROUNDED. An
artefact is none of those.

1. **Seed stability** -- refit the SAME data under N random inits (n_restarts=1, so each seed is
   one independent fit and the spread is visible). A real |lambda| is tight; an artefact scatters.

2. **Data-amount trend** -- fit at 25 / 50 / 100 % of trials. Autoregressive estimates near a
   unit root are biased, and the bias SHRINKS with data. So if |lambda| falls toward a smaller
   value as trials grow, the high |lambda| was small-sample bias (an artefact). If |lambda| is
   flat, it is a real property of the dynamics.

3. **Shuffle null** -- bin-shuffle WITHIN each trial (destroys temporal order, preserves each
   trial's rate content), then run the identical pipeline. The Gaussian smoother re-imposes some
   autocorrelation regardless, so this asks the key question directly: **does |lambda| ~ 1 come
   from the smoother or from the dynamics?** If shuffled data ALSO gives |lambda| ~ 1, the flow
   classification is meaningless (it is the kernel). If shuffled |lambda| is clearly lower, the
   real slow mode carries genuine temporal structure.

VERDICT logic (printed at the end):
  real slow mode   <= tight across seeds AND flat-or-rising with data AND real >> shuffle
  artefact         <= any of: scatters across seeds, drops with data, or real ~ shuffle
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
from src.dynamics.rslds import counts_to_latents, fit_rslds             # noqa: E402

K_PER_REGION = {"VISp": 3, "SCm": 4, "MRN": 3}
N_LATENT = 3
BIN_S = 0.04
SIGMA_MS = 40.0
STICKY = 2.0
N_ITER = 40

N_SEEDS = 8
FRACTIONS = [0.25, 0.5, 1.0]
ALIGN_FILE = f"{WORKING_EID[:8]}__stimOn_pre400_post800_bin40_rate1.npz"


def lam_max(counts, K, seed, shuffle=False):
    """|lambda|max of a single exact-EM fit (n_restarts=1, so the seed's spread is visible)."""
    c = counts
    if shuffle:
        rng = np.random.default_rng(seed + 777)
        c = np.stack([trial[rng.permutation(trial.shape[0])] for trial in counts])
    x, _, _ = counts_to_latents(c, n_latent=N_LATENT, smooth_sigma_ms=SIGMA_MS, bin_size_s=BIN_S)
    res = fit_rslds(x, K, recurrent=True, n_iter=N_ITER, n_restarts=1, seed=seed, sticky=STICKY)
    lam = float(np.abs(res.eigenvalues()).max())
    n_int = int(sum("integrator" in s for s in res.classify()))
    return lam, n_int


def main() -> None:
    d = load_processed(ROOT / "data" / "processed" / ALIGN_FILE)
    print(f"session {WORKING_EID[:8]}  stimOn 40 ms  {d['n_trials']} trials\n", flush=True)

    out_rows = []
    summary = {}
    for region, rd in d["regions"].items():
        K = K_PER_REGION[region]
        counts = rd["counts"]
        n_trials = counts.shape[0]
        print(f"########## {region}  K={K}  {counts.shape[2]} units ##########", flush=True)

        # --- 1. seed stability + 3. shuffle null, at full data ---
        real_full, shuf_full = [], []
        for s in range(N_SEEDS):
            lam_r, ni_r = lam_max(counts, K, s, shuffle=False)
            lam_s, ni_s = lam_max(counts, K, s, shuffle=True)
            real_full.append(lam_r)
            shuf_full.append(lam_s)
            out_rows.append({"region": region, "test": "full", "seed": s,
                             "lam_real": round(lam_r, 3), "n_int_real": ni_r,
                             "lam_shuffle": round(lam_s, 3), "n_int_shuffle": ni_s})
        real_full, shuf_full = np.array(real_full), np.array(shuf_full)

        # --- 2. data-amount trend ---
        trend = {}
        for frac in FRACTIONS:
            n = max(int(frac * n_trials), 30)
            vals = []
            for s in range(3):
                rng = np.random.default_rng(1000 + s)
                idx = rng.choice(n_trials, n, replace=False)
                lam_r, _ = lam_max(counts[idx], K, s)
                vals.append(lam_r)
                out_rows.append({"region": region, "test": f"frac{frac}", "seed": s,
                                 "n_trials": n, "lam_real": round(lam_r, 3)})
            trend[frac] = (float(np.mean(vals)), float(np.std(vals)))

        # --- verdict ---
        seed_spread = float(real_full.std())
        real_mean = float(real_full.mean())
        shuf_mean = float(shuf_full.mean())
        gap = real_mean - shuf_mean
        data_drop = trend[FRACTIONS[0]][0] - trend[1.0][0]   # small-data minus full-data |lam|

        tight = seed_spread < 0.03
        flat = data_drop < 0.03                              # |lam| not falling with more data
        grounded = gap > 0.05                                # real clearly above shuffle

        verdict = ("REAL slow mode (tight, data-stable, above shuffle)"
                   if (tight and flat and grounded)
                   else "ARTEFACT / uncertifiable: " + ", ".join(filter(None, [
                       "scatters across seeds" if not tight else "",
                       "drops with more data" if not flat else "",
                       "not above shuffle null (it is the smoother)" if not grounded else "",
                   ])))

        summary[region] = {
            "real_mean": round(real_mean, 3), "seed_spread": round(seed_spread, 3),
            "shuffle_mean": round(shuf_mean, 3), "real_minus_shuffle": round(gap, 3),
            "lam_by_frac": {str(k): round(v[0], 3) for k, v in trend.items()},
            "data_drop_25_to_100": round(data_drop, 3),
            "verdict": verdict,
        }
        print(f"  seed stability:   |lam|_real = {real_mean:.3f} +/- {seed_spread:.3f}  "
              f"(n_int mode {int(np.median([r['n_int_real'] for r in out_rows if r['region']==region and r['test']=='full']))}/{K})")
        print(f"  shuffle null:     |lam|_shuf = {shuf_mean:.3f}   gap = {gap:+.3f}")
        print(f"  data trend |lam|: " + "  ".join(f"{int(f*100)}%={trend[f][0]:.3f}" for f in FRACTIONS)
              + f"   (drop 25->100%: {data_drop:+.3f})")
        print(f"  => {verdict}\n", flush=True)

    out = ROOT / "results" / "rslds" / "lambda_stability.json"
    out.write_text(json.dumps({"session": WORKING_EID, "summary": summary, "rows": out_rows},
                              indent=1))
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
