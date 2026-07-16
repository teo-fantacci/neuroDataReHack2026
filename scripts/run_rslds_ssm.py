"""Fit Linderman's rSLDS (the real `ssm`) to a count tensor. Runs INSIDE the `ndrh_ssm` env.

This is the bridge. The notebook lives in `neurodatarehack` (numpy 2.x, ONE-api, pynapple);
`ssm` needs numpy 1.x. Rather than break one env to satisfy the other, the notebook writes a
count tensor to .npz, calls

    conda run -n ndrh_ssm python scripts/run_rslds_ssm.py --in tensor.npz --out fit.npz

and reads the result back. Nothing from `ssm` is ever imported into the working env.

What is fitted
--------------
A **recurrent** SLDS with **Poisson emissions**: the discrete regime z_t has its own linear
flow on a continuous latent x_t, and the probability of switching depends on *where the
latent currently is* (``transitions="recurrent_only"``), not just on a clock. Inference is
Laplace-EM, which — unlike our PCA-latent version in src/dynamics/rslds.py — infers x_t
**jointly** with the dynamics under the true Poisson likelihood. That is the whole reason to
pay for this: with IBL's sparse units, a PCA latent is largely noise, and a noise latent
makes every regime look like a random walk (|lambda| ~ 1).

Outputs (npz)
-------------
    x        (trials, time, D)   posterior mean latent
    states   (trials, time)      most-likely regime
    A, b     (K, D, D), (K, D)   per-regime linear flow  ->  fixed points, eigenvalues
    C, d     (N, D), (N,)        emission loading        ->  which neurons define the latent
    Rs, r    (K, D), (K,)        recurrent transition weights
    elbos    (n_iter,)           Laplace-EM ELBO trace   ->  check it converged
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True, help="npz with counts (trials,time,units)")
    p.add_argument("--out", dest="out", required=True)
    p.add_argument("--n-states", type=int, default=4)
    p.add_argument("--n-latent", type=int, default=3)
    p.add_argument("--n-iter", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    import ssm

    npz = np.load(args.inp)
    counts = npz["counts"].astype(int)                    # (trials, time, units)
    n_trials, T, N = counts.shape

    # ssm wants a list of (time, units) arrays -- one per trial. Trials are separate
    # sequences, exactly as `lengths` does for hmmlearn: this stops the model from learning
    # a transition across the trial boundary.
    datas = [counts[i] for i in range(n_trials)]

    print(f"[ssm] {n_trials} trials x {T} bins x {N} units | "
          f"mean count/bin = {counts.mean():.3f} | K={args.n_states} D={args.n_latent}",
          flush=True)

    np.random.seed(args.seed)
    t0 = time.time()
    rslds = ssm.SLDS(
        N, args.n_states, args.n_latent,
        transitions="recurrent_only",       # p(z_t | z_{t-1}, x_{t-1}) -- the "r" in rSLDS
        dynamics="diagonal_gaussian",
        emissions="poisson_orthog",
        emission_kwargs=dict(link="softplus"),   # softplus: no exp() blow-up on sparse counts
    )
    rslds.initialize(datas)
    elbos, q = rslds.fit(
        datas, method="laplace_em", variational_posterior="structured_meanfield",
        initialize=False, num_iters=args.n_iter, alpha=0.0,
    )
    fit_s = time.time() - t0

    xs = np.array(q.mean_continuous_states)               # (trials, time, D)
    zs = np.array([rslds.most_likely_states(x, d) for x, d in zip(xs, datas)])

    # The ELBO is a *lower bound* and Laplace-EM is not guaranteed monotone. If the last
    # quarter of the trace is still climbing hard, the fit has not converged -- say so
    # rather than let a half-converged model become a result.
    tail = elbos[-max(len(elbos) // 4, 2):]
    drift = float(np.abs(np.diff(tail)).mean() / max(abs(float(elbos[-1])), 1e-9))
    converged = bool(drift < 1e-3)

    print(f"[ssm] laplace-EM {args.n_iter} iters in {fit_s:.0f}s | final ELBO {elbos[-1]:.1f} | "
          f"converged={converged} (tail drift {drift:.2e})", flush=True)

    np.savez_compressed(
        args.out,
        x=xs, states=zs,
        A=np.array(rslds.dynamics.As), b=np.array(rslds.dynamics.bs),
        C=np.array(rslds.emissions.Cs[0]), d=np.array(rslds.emissions.ds[0]),
        Rs=np.array(rslds.transitions.Rs), r=np.array(rslds.transitions.r),
        elbos=np.array(elbos),
        meta=json.dumps({
            "n_states": args.n_states, "n_latent": args.n_latent,
            "n_iter": args.n_iter, "seed": args.seed,
            "fit_seconds": round(fit_s, 1), "converged": converged,
            "elbo_tail_drift": drift,
        }),
    )
    print(f"[ssm] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
