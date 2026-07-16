"""rSLDS ground truth: can the fit recover attractors we PUT THERE? Swept over bin size.

Why this script exists at all
----------------------------
The table in ``src/dynamics/rslds.py``'s docstring -- the one that says 20 ms bins manufacture
spurious "line attractors" and 50 ms recovers real ones, and on which the standing rule *"any
integrator claim from a 20 ms fit is inadmissible"* rests -- **had no code behind it**. It was
run ad hoc in a session and only the numbers were recorded. The single most load-bearing
methodological decision in the project was therefore **not reproducible**. This script is that
validation, committed.

Two design flaws in the original are fixed here
-----------------------------------------------
**1. The truth must live in CONTINUOUS time.** If you generate the latent directly in bins, then
"20 ms" and "50 ms" are different *ground truths* and the comparison is meaningless. Here the
dynamics are simulated at 1 ms with a time constant in **seconds**, spikes are drawn at 1 ms, and
only then are counts aggregated into 20/30/40/50 ms bins. The truth is identical; only the
observation changes. That is the comparison we actually want to make.

**2. The original confounded bin size with SMOOTHING.** ``counts_to_latents`` takes
``smooth_sigma_bins`` -- sigma in *bins*. The original sweep left it at 1.5, so at 20 ms it
smoothed with sigma = 30 ms and at 50 ms with sigma = **75 ms**. The 50 ms fit was therefore
getting 2.5x more smoothing, and "50 ms is better" may partly have been "more smoothing is
better". This script holds **sigma fixed in milliseconds** (the fair test) and *also* reproduces
the original sigma-in-bins setting, so the two effects can be separated.

The generative model
--------------------
Latent ``x`` in R^3. Three attractors at well-separated points mu_0, mu_1, mu_2. Within regime
k the latent is an Ornstein-Uhlenbeck process pulled toward mu_k with time constant ``TAU_S``::

    dx = (mu_k - x) * dt / TAU_S  +  sigma * sqrt(dt) * eps

Every trial visits the regimes in the order 0 -> 1 -> 2 (so **true switches/trial = 2.0** and the
true ``order_consistency`` = 1.0), with jittered switch times. Spikes are Poisson with
log-rate ``C x + d``, scaled to a target mean firing rate.

Because the OU process is exact, the **true** discrete-time dynamics matrix at bin width D is
``A = exp(-D / TAU_S) * I``, so the true ``|lambda|max`` is known in closed form **for each bin
size** and the fit can be scored against it rather than against a vibe.

What is scored
--------------
* ``regime_acc``     -- Hungarian-matched agreement between fitted and true regime, per bin.
* ``switches``       -- fitted switches/trial (**truth = 2.0**). Flickering => not metastable.
* ``lam_fit`` vs ``lam_true`` -- did we recover the contraction, or invent a unit root?
* ``n_integrator``   -- how many of the K regimes get classified "line attractor / integrator".
  **This is the headline failure mode**: a near-unit-root random walk fitted to shot noise reads
  exactly like "we found the integrator a decision variable lives on".
* ``order_cons``     -- fitted order consistency (**truth = 1.0**).

Configs swept: the original 60-unit / 4 Hz benchmark (to check this rebuild reproduces the
recorded table), plus **each real region's actual unit count and firing rate**, so we learn where
*our* fits are trustworthy rather than where a generic one is.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dynamics.rslds import (  # noqa: E402
    counts_to_latents, fit_rslds, order_consistency,
)

# --- ground truth ---------------------------------------------------------------
TAU_S = 0.150          # attractor time constant (s). GPFA fitted 107-283 ms on the real data.
TRIAL_S = 1.2          # trial length (s) -- matches the frozen window (0.4 pre + 0.8 post)
FINE_DT = 0.001        # simulation timestep (s)
N_TRIALS = 300
D_TRUE = 3
K_TRUE = 3
TRUE_SWITCHES = 2.0    # 0 -> 1 -> 2, every trial
LATENT_SD = 0.35       # OU noise

BIN_SIZES_MS = [20, 30, 40, 50]

# (name, n_units, mean_rate_hz). The region rows are the ACTUAL frozen values
# (data/processed/*, units >= 1 Hz): spikes / trials / window / units.
CONFIGS = [
    ("benchmark(orig)", 60, 4.0),    # the original docstring benchmark -- rebuild check
    ("VISp-like",       19, 3.5),
    ("MRN-like",        54, 16.0),
    ("SCm-like",        44, 22.0),
]

SEED = 0


def simulate(n_units: int, rate_hz: float, seed: int = SEED):
    """Simulate at 1 ms. Returns (fine_counts (trials, n_fine, units), true_z (trials, n_fine))."""
    rng = np.random.default_rng(seed)
    n_fine = int(round(TRIAL_S / FINE_DT))

    # three well-separated attractors
    mu = np.array([[0.0, 0.0, 0.0],
                   [3.0, 0.5, 0.0],
                   [1.5, 3.0, 0.5]])

    # emission map, scaled so the MEAN rate is rate_hz
    C = 0.30 * rng.standard_normal((n_units, D_TRUE))
    d = np.log(rate_hz) - 0.5 * (C @ mu.mean(0)) ** 2 * 0  # placeholder, calibrated below
    d = np.full(n_units, np.log(rate_hz))

    x = np.zeros((N_TRIALS, n_fine, D_TRUE))
    z = np.zeros((N_TRIALS, n_fine), dtype=int)

    a = FINE_DT / TAU_S
    for i in range(N_TRIALS):
        # jittered switch times around 1/3 and 2/3 of the trial
        s1 = int(np.clip(rng.normal(n_fine / 3, 0.05 * n_fine), 0.15 * n_fine, 0.5 * n_fine))
        s2 = int(np.clip(rng.normal(2 * n_fine / 3, 0.05 * n_fine), s1 + 0.15 * n_fine,
                         0.9 * n_fine))
        z[i, :s1], z[i, s1:s2], z[i, s2:] = 0, 1, 2

        xt = mu[0] + 0.3 * rng.standard_normal(D_TRUE)
        for t in range(n_fine):
            k = z[i, t]
            xt = xt + a * (mu[k] - xt) + LATENT_SD * np.sqrt(a) * rng.standard_normal(D_TRUE)
            x[i, t] = xt

    # Poisson spikes at the fine timestep; log-rate = C x + d, calibrated to rate_hz
    log_rate = x @ C.T + d
    rate = np.exp(log_rate - log_rate.mean() + np.log(rate_hz))     # mean == rate_hz
    counts = rng.poisson(rate * FINE_DT)
    return counts.astype(np.int32), z


def rebin(fine_counts: np.ndarray, true_z: np.ndarray, bin_ms: int):
    """Aggregate 1 ms counts into ``bin_ms`` bins. True regime = the majority regime in the bin."""
    step = int(round(bin_ms / (FINE_DT * 1000)))
    n_fine = fine_counts.shape[1]
    n_bins = n_fine // step

    c = fine_counts[:, : n_bins * step, :].reshape(
        fine_counts.shape[0], n_bins, step, fine_counts.shape[2]).sum(axis=2)

    zz = true_z[:, : n_bins * step].reshape(true_z.shape[0], n_bins, step)
    # majority regime within the bin
    z = np.array([[np.bincount(row, minlength=K_TRUE).argmax() for row in trial] for trial in zz])
    return c.astype(np.int32), z


def regime_accuracy(fitted: np.ndarray, true: np.ndarray, k: int) -> float:
    """Hungarian-matched agreement. Regime labels are arbitrary, so match them first."""
    from scipy.optimize import linear_sum_assignment

    conf = np.zeros((k, K_TRUE), dtype=np.int64)
    np.add.at(conf, (fitted.ravel(), true.ravel()), 1)
    r, c = linear_sum_assignment(-conf)
    return float(conf[r, c].sum() / conf.sum())


def run() -> list[dict]:
    rows = []
    for name, n_units, rate_hz in CONFIGS:
        fine, z_fine = simulate(n_units, rate_hz, seed=SEED)
        obs_hz = fine.sum() / (N_TRIALS * TRIAL_S * n_units)
        print(f"\n=== {name}: {n_units} units, target {rate_hz} Hz, observed {obs_hz:.1f} Hz ===",
              flush=True)

        for bin_ms in BIN_SIZES_MS:
            counts, z_true = rebin(fine, z_fine, bin_ms)
            lam_true = float(np.exp(-(bin_ms / 1000.0) / TAU_S))   # closed form: OU at width D

            # Two smoothing conventions, on purpose:
            #   'sigma_ms'   : sigma FIXED at 40 ms  -> the fair bin-size comparison
            #   'sigma_bins' : sigma = 1.5 BINS      -> what the original sweep actually did,
            #                  which confounds bin size with smoothing (30 ms at 20 ms bins,
            #                  75 ms at 50 ms bins)
            for mode in ("sigma_ms", "sigma_bins"):
                t0 = time.time()
                if mode == "sigma_ms":
                    # the honest convention: smoothing fixed in ms, invariant to the bin
                    x, _, ve = counts_to_latents(counts, n_latent=D_TRUE,
                                                 smooth_sigma_ms=40.0,
                                                 bin_size_s=bin_ms / 1000.0)
                    sig_bins = 40.0 / bin_ms
                else:
                    # the OLD convention, reproduced on purpose to show it confounds
                    # smoothing with bin size (this path warns, by design)
                    import warnings
                    sig_bins = 1.5
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", DeprecationWarning)
                        x, _, ve = counts_to_latents(counts, n_latent=D_TRUE,
                                                     smooth_sigma_bins=sig_bins)
                res = fit_rslds(x, K_TRUE, recurrent=True, n_iter=60, n_restarts=3,
                                seed=SEED, sticky=2.0)

                lam = np.abs(res.eigenvalues()).max(axis=1)          # (K,)
                labels = res.classify()
                n_int = sum("integrator" in s for s in labels)
                switches = float((res.states[:, 1:] != res.states[:, :-1]).sum(axis=1).mean())

                row = {
                    "config": name, "n_units": n_units, "rate_hz": rate_hz,
                    "bin_ms": bin_ms, "smoothing": mode,
                    "sigma_ms": round(sig_bins * bin_ms, 1),
                    "n_bins": int(counts.shape[1]),
                    "var_exp": round(float(ve.sum()), 3),
                    "regime_acc": round(regime_accuracy(res.states, z_true, K_TRUE), 3),
                    "switches": round(switches, 2),
                    "lam_fit": round(float(lam.max()), 3),
                    "lam_true": round(lam_true, 3),
                    "n_integrator": int(n_int),
                    "order_cons": round(float(order_consistency(res.states, K_TRUE)), 3),
                    "seconds": round(time.time() - t0, 1),
                }
                rows.append(row)
                flag = "  <<< SPURIOUS INTEGRATOR" if n_int > 0 else ""
                print(f"  {bin_ms:2d}ms {mode:10s} (sig {row['sigma_ms']:4.0f}ms)  "
                      f"acc={row['regime_acc']:.2f}  sw={row['switches']:.1f} (true 2.0)  "
                      f"|lam|={row['lam_fit']:.3f} (true {lam_true:.3f})  "
                      f"int={n_int}/{K_TRUE}  OC={row['order_cons']:.2f}{flag}", flush=True)
    return rows


if __name__ == "__main__":
    rows = run()
    out = ROOT / "results" / "rslds" / "ground_truth_binsize.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "tau_s": TAU_S, "trial_s": TRIAL_S, "n_trials": N_TRIALS,
        "true_switches": TRUE_SWITCHES, "K_true": K_TRUE, "D_true": D_TRUE,
        "rows": rows,
    }, indent=1))
    print(f"\nwrote {out.relative_to(ROOT)}")
