"""Put an error bar on the recurrence result: is delta_recurrent RELIABLY > 0?

The certified claim from the first real fit was "attractor sequence, not stopwatch", resting on
delta_recurrent = held-out LL(R free) - LL(R = 0) > 0. But that was a MEAN over 3 CV folds, and
for VISp it was only +0.003. This script asks whether that survives its own noise.

Two tests, because they answer two different questions:

1. **Repeated cross-validation** (5 folds x N_REPEATS repeats) -> the sampling distribution of the
   PAIRED per-fold delta (same train/test split feeds both the R-free and R=0 fit, so each delta
   is paired and the comparison is powerful). Reports mean, SD, fraction positive, and a Wilcoxon
   signed-rank -- WITH the caveat that CV folds share training data, so that p-value is
   approximate (anticonservative). This gives the error bar.

2. **Position-shuffle null** (the clean significance test). Take the FITTED R-free model and score
   held-out data twice: once with the real latent positions feeding the transition term, once with
   those positions permuted across (trial, bin). Same model, same parameters -- only the
   position->switch coupling is broken. If R genuinely encodes "switch depends on where you are",
   the real positions beat the shuffled ones. This isolates recurrence from the "R is just extra
   parameters" confound, and it is a proper permutation null (p = fraction of shuffles >= observed).

Read-out:
  delta_recurrent > 0, consistent across folds, AND real position beats shuffled position
      => switching genuinely depends on latent position (attractor sequence). CERTIFIED.
  delta ~ 0 or real ~ shuffled
      => the R term is not buying real position-dependence for that region.
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
    _emission_loglik, _forward_backward, _transition_logits,
    counts_to_latents, fit_rslds, score_rslds,
)

K_PER_REGION = {"VISp": 3, "SCm": 4, "MRN": 3}
N_LATENT = 3
BIN_S = 0.04
SIGMA_MS = 40.0
STICKY = 2.0
N_ITER = 40
N_RESTARTS = 2
N_FOLDS = 5
N_REPEATS = 3
N_POSSHUFFLE = 40

ALIGNMENTS = {
    "stimOn": f"{WORKING_EID[:8]}__stimOn_pre400_post800_bin40_rate1.npz",
    "firstMovement": f"{WORKING_EID[:8]}__firstMovement_pre400_post800_bin40_rate1.npz",
}


def score_posshuffle(res, x_te, rng) -> float:
    """Held-out LL of a fitted R-free model with the transition positions PERMUTED.

    Emission/dynamics term uses the REAL x (so the dynamics likelihood is unchanged); only the
    x_{t-1} feeding the recurrent transition logits is shuffled across (trial, bin). Breaks the
    position->switch coupling while holding every parameter fixed.
    """
    n, T, D = x_te.shape
    log_e = _emission_loglik(x_te, res.A, res.b, res.Q)          # real dynamics
    flat = x_te.reshape(n * T, D)
    x_sh = flat[rng.permutation(n * T)].reshape(n, T, D)
    log_T = _transition_logits(x_sh, res.log_pi, res.R)         # shuffled position
    _, _, ll = _forward_backward(log_e, log_T, res.pi0)
    return float(ll / (n * T))


def screen(counts, K, seed0):
    from sklearn.model_selection import KFold

    deltas, real_minus_posnull = [], []
    for rep in range(N_REPEATS):
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed0 + rep)
        for f, (tr, te) in enumerate(kf.split(np.arange(counts.shape[0]))):
            x_tr, pca, _ = counts_to_latents(counts[tr], n_latent=N_LATENT,
                                             smooth_sigma_ms=SIGMA_MS, bin_size_s=BIN_S)
            x_te, _, _ = counts_to_latents(counts[te], n_latent=N_LATENT,
                                           smooth_sigma_ms=SIGMA_MS, bin_size_s=BIN_S, pca=pca)

            m_free = fit_rslds(x_tr, K, recurrent=True, n_iter=N_ITER, n_restarts=N_RESTARTS,
                               seed=seed0, sticky=STICKY)
            m_r0 = fit_rslds(x_tr, K, recurrent=False, n_iter=N_ITER, n_restarts=N_RESTARTS,
                             seed=seed0, sticky=STICKY)

            ll_free = score_rslds(m_free, x_te)
            ll_r0 = score_rslds(m_r0, x_te)
            deltas.append(ll_free - ll_r0)                      # paired, per fold

            # position-shuffle null on the SAME held-out fold, same fitted model
            rng = np.random.default_rng(1000 * rep + f)
            posnull = np.array([score_posshuffle(m_free, x_te, rng) - ll_r0
                                for _ in range(N_POSSHUFFLE)])
            # how far the REAL delta sits above the position-shuffled delta
            real_minus_posnull.append((ll_free - ll_r0) - float(posnull.mean()))

    return np.array(deltas), np.array(real_minus_posnull)


def main() -> None:
    from scipy.stats import wilcoxon

    all_rows = []
    for align, fname in ALIGNMENTS.items():
        d = load_processed(ROOT / "data" / "processed" / fname)
        print(f"\n########## alignment = {align} ##########", flush=True)
        for region, rd in d["regions"].items():
            K = K_PER_REGION[region]
            t0 = time.time()
            deltas, real_vs_posnull = screen(rd["counts"], K, seed0=0)

            frac_pos = float((deltas > 0).mean())
            # approximate significance (folds not fully independent -> anticonservative)
            try:
                w_p = float(wilcoxon(deltas, alternative="greater").pvalue)
            except ValueError:
                w_p = float("nan")
            # position-shuffle: is the real delta reliably above the shuffled-position delta?
            pos_frac_pos = float((real_vs_posnull > 0).mean())

            row = {
                "alignment": align, "region": region, "K": K,
                "n_folds_total": len(deltas),
                "delta_mean": round(float(deltas.mean()), 4),
                "delta_sd": round(float(deltas.std()), 4),
                "delta_sem": round(float(deltas.std() / np.sqrt(len(deltas))), 4),
                "frac_folds_positive": round(frac_pos, 3),
                "wilcoxon_p_approx": round(w_p, 4),
                "real_minus_posnull_mean": round(float(real_vs_posnull.mean()), 4),
                "real_above_posnull_frac": round(pos_frac_pos, 3),
            }
            all_rows.append(row)
            verdict = ("RECURRENT (position-dependent)"
                       if frac_pos >= 0.8 and pos_frac_pos >= 0.8
                       else "weak / not reliably position-dependent")
            print(f"  {region:5s} K={K}  delta = {row['delta_mean']:+.4f} +/- "
                  f"{row['delta_sd']:.4f} (sem {row['delta_sem']:.4f}), "
                  f"{frac_pos:.0%} folds > 0, wilcoxon p~{w_p:.3f}", flush=True)
            print(f"        real vs position-shuffle: +{row['real_minus_posnull_mean']:.4f} "
                  f"({pos_frac_pos:.0%} of folds real > shuffled)  =>  {verdict}  "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    out = ROOT / "results" / "rslds" / "recurrence_test.json"
    out.write_text(json.dumps({
        "session": WORKING_EID, "bin_s": BIN_S, "n_folds": N_FOLDS, "n_repeats": N_REPEATS,
        "n_posshuffle": N_POSSHUFFLE, "K_per_region": K_PER_REGION, "rows": all_rows,
    }, indent=1))
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
