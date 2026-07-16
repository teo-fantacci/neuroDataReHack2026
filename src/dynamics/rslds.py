"""Recurrent switching linear dynamical system (rSLDS) over a low-dimensional latent.

Purpose
-------
The HMM in ``hmm.py`` says *which discrete state* the population is in at each bin. It
says nothing about **what kind of dynamical object** each state is. This module answers
that: each state gets its own **linear flow**, so we can ask whether it is a point
attractor, a rotation, a saddle the trajectory merely passes through, or a slow
(line-attractor / integrator) direction — and whether the states are visited as a
**reliable sequence**, irrespective of any assumed direction.

Model
-----
Latent ``x_t`` in R^D, discrete regime ``z_t`` in {0..K-1}, per trial:

    z_t | z_{t-1}, x_{t-1}  ~  softmax_k( logPi[z_{t-1}, k] + R[k] . x_{t-1} )
    x_t | z_t,     x_{t-1}  ~  N( A[z_t] x_{t-1} + b[z_t],  Q[z_t] )

The ``R[k] . x_{t-1}`` term is what makes it **recurrent** (Linderman et al. 2017,
AISTATS): the probability of switching depends on *where you are* in latent space, not
only on which regime you were in. That is the difference between "the brain runs a fixed
timed script" (a plain ARHMM, R = 0) and "the trajectory switches when it reaches a
region of state space" (rSLDS). We fit both and compare them on held-out trials, so the
recurrence has to earn its parameters.

This is the FAST backend, not the fit of record
-----------------------------------------------
The canonical rSLDS package (lindermanlab/ssm) infers ``x`` **jointly** with the dynamics via
Laplace-EM with **Poisson** emissions -- that is the principled model, and it IS installed here,
in the separate ``ndrh_ssm`` conda env, called out of process via ``src/dynamics/rslds_ssm.py``
(see ``docs/worklog/2026-07-13_rslds.md`` -- the earlier "ssm won't install" verdict was wrong).

This module exists **alongside** it, as the fast screen: it obtains the latent *first*, by PCA on
Anscombe counts, then holds it fixed. That makes the model a recurrent ARHMM with an exact,
cheap EM -- fast enough to cross-validate K and run the R=0 recurrence comparison, which ``ssm``
at ~15-20 min/fit cannot. The two agree on the ``RSLDSResult`` interface, so every figure and
statistic is backend-agnostic; ``ssm`` is fitted at the K this backend selects.

So we make **one explicit modelling concession**: the latent is obtained *first*, by PCA
on Anscombe counts, and then held fixed. Consequences, stated honestly:

* **We gain** exact EM. With ``x`` observed, the model is a recurrent ARHMM: the E-step is
  plain forward-backward and the M-step is weighted least squares plus a concave
  multinomial-logistic fit. No Laplace approximation, no sampling, no build risk, and the
  log-likelihood is exact — so cross-validation and the R=0 comparison are trustworthy.
* **We lose** the Poisson observation model and the feedback from the dynamics onto the
  latent. PCA picks directions of *variance*, not directions the dynamics act on. A real
  rSLDS could find a low-variance but dynamically decisive direction that PCA discards.
  Treat D (the latent dimension) as a *chosen* quantity, never as one selected by
  held-out likelihood — see ``cross_validate_K``.

Consequence you must not forget: the latent log-likelihood **cannot be compared across
different D**, because different D means different data. Only K (and R=0 vs R free) may
be selected by held-out likelihood.

Validated against ground truth — REPRODUCIBLE (2026-07-14)
----------------------------------------------------------
``scripts/validate_rslds_ground_truth.py`` → ``results/rslds/ground_truth_binsize.json``.

.. warning::
   The table that used to sit here **had no code behind it**. It was run ad hoc and only the
   numbers were recorded, so the most load-bearing decision in the project — the bin size, and
   the rule that 20 ms fits are inadmissible — was **not reproducible**. The rebuild below
   matches its 20 ms row closely but **NOT** its 50 ms row. The old numbers, in particular
   "|lambda| = 0.875 at 50 ms", **must not be quoted.**

The benchmark: 3 attractors visited 0 -> 1 -> 2 on **every** trial, simulated as an OU process
in **continuous time** (tau = 150 ms, inside GPFA's fitted 107-283 ms), Poisson spikes drawn at
1 ms, and only then binned. The truth is therefore *identical* across bin sizes and only the
observation changes — which is the comparison we actually want. Because OU is exact, the TRUE
discrete ``|lambda|max`` at bin width D is ``exp(-D/tau)``, known in closed form, so the fit is
scored against a number rather than a vibe.

**Result, with sigma held FIXED at 40 ms.** Spurious integrators, out of 3 regimes (truth = 0):

    bin      benchmark   VISp-like   MRN-like   SCm-like    regime acc (SCm-like)
    20 ms      3/3         1/3         3/3        3/3        0.73    <- FAILS
    30 ms      1/3         0/3         0/3        0/3        0.92    marginal
    40 ms      0/3         0/3         0/3        0/3        0.94    <- SAFE
    50 ms      0/3         0/3         0/3        0/3        0.94    <- SAFE

Three things this bought us:

1. **``sticky`` must be applied in every M-step, not just at initialisation.** With
   ``sticky=0`` the decoded regimes flicker (many switches per trial where the truth is 2) and
   regime recovery collapses. Flickering regimes are not metastable states. ``sticky=2.0``
   recovers 2.0-2.1 switches/trial at 40 ms, so the default is now *validated*, not asserted.

2. **|lambda| -> 1 has TWO causes, and the original validation confounded them.**

   * **Too little signal per bin** (shot noise, at 20 ms). The latent is noise, the AR fit
     returns a near-unit root, and *every* regime is labelled "line attractor / integrator".
     That is not a finding — it is the paper-ready sentence *"we found the integrator a decision
     variable lives on"*, generated out of nothing.
   * **Too much smoothing.** A Gaussian kernel *imposes* temporal autocorrelation, which is
     exactly what an AR fit reads as a slow mode. The old code took ``smooth_sigma_bins`` —
     sigma in **bins** — so a bin-size sweep was silently also a **smoothing sweep**: 30 ms of
     smoothing at 20 ms bins, but **75 ms** at 50 ms bins. Under that convention |lambda| climbs
     monotonically toward 1 as the bin grows (0.951 -> 0.974 -> 0.990 -> 0.996) and **50 ms
     comes out WORSE than 20 ms**. "50 ms is better" was partly an artefact of its own smoothing.

   Hence :func:`counts_to_latents` now takes **``smooth_sigma_ms``** and *requires*
   ``bin_size_s``. Sigma-in-bins is no longer available, because it cannot be swept honestly.

3. **VISp will be the least trustworthy region.** At its real size (19 units, 3.5 Hz) regime
   recovery is only **0.71** even at 40 ms, and switches *undershoot* (1.6 vs a true 2.0) — while
   the *order* is still recovered perfectly (order-consistency 1.00). So in VISp: **trust the
   sequence, distrust the regime boundaries and the regime count.**

**Recommended configuration: 40 ms bins, sigma 40 ms.** It passes the ground truth in every
region, and independently preserves SCm's HMM state-stability, which degrades at 50 ms
(ARI 0.938 -> 0.826). One timebase for the whole project.

The emission model is the weak point, not the bin
-------------------------------------------------
Our latent is PCA on Anscombe counts — a **Gaussian** observation model. Two independent reasons
that is a concession:

* PCA picks directions of **variance**, not directions the **dynamics** act on.
* **Anscombe does not stabilise variance at IBL's firing rates.** It reaches Var = 1 only for
  lambda >~ 4 spikes/bin; IBL runs at lambda = 0.05-0.83 (see ``src/data/preprocess.py::anscombe``
  and ``notebooks/00_preprocessing.ipynb`` Part 2). At lambda << 1 a count is essentially
  Bernoulli, and no monotone transform can stabilise a Bernoulli. So the "Gaussian latent" is a
  **smoothed Bernoulli**, and all of its continuity comes from the smoother.

Both point the same way: **``ssm``'s Poisson rSLDS is the principled model; this one is the fast
screen.** It also explains *mechanistically* why the 20 ms latent failed.

Also learned: the within-trial bin-shuffle null for ``order_consistency`` does **not** sit
at 0.5 — it lands at 0.53-0.62, because shuffling changes which regime happens to be
entered first. Always compare the statistic to its own null, never to 0.5, and report the
*effect size* as well as the p-value (with hundreds of trials, a trivial order bias is
still "significant").
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import logsumexp

# Ridge added to every state's dynamics noise covariance Q before inversion. A regime that
# is visited by only a handful of bins can otherwise produce a singular (or negative-
# definite, through round-off) Q and an infinite log-likelihood — the Gaussian analogue of
# the lambda == 0 trap that destroyed our first HMM cross-validation.
Q_RIDGE = 1e-4

# L2 penalty on the transition parameters. logPi is over-parameterised (adding a constant
# to a row of logPi changes nothing after the softmax), so without a penalty the optimiser
# drifts along that flat direction forever. This pins it down; it is not a scientific prior.
TRANS_L2 = 1e-3

# L-BFGS iterations allowed inside the transition M-step, per EM iteration.
#
# This is a *generalised* EM (GEM) step, and the distinction is worth 6x of wall-clock. EM only
# requires the M-step to *increase* the expected complete-data log-likelihood, not to maximise it:
# the next E-step re-optimises anyway, so driving the inner softmax fit to convergence every
# single iteration is wasted work. Profiled on the real session (843 trials x 50 bins, K=4, D=3):
#
#     L-BFGS budget    fit time    final LL/bin
#     100 (naive)        40.8 s      -0.6362
#      20                18.0 s      -0.6362
#      10 (this)          ~9  s      -0.6362
#       5                 6.4 s      -0.6362
#
# The likelihood is *identical* -- the extra 35 s per fit bought exactly nothing. It was 96% of
# total fit time, and it is what made a cross-validation sweep take 45+ minutes.
TRANS_MAXITER = 10


# --------------------------------------------------------------------------- latents


def counts_to_latents(
    counts: np.ndarray,
    *,
    n_latent: int = 3,
    smooth_sigma_ms: float | None = None,
    bin_size_s: float | None = None,
    smooth_sigma_bins: float | None = None,
    pca=None,
):
    """(trials, time, neurons) integer counts -> (trials, time, n_latent) latent trajectories.

    Three steps, each for a reason:

    1. **Anscombe transform** ``2*sqrt(x + 3/8)``. Intended as the Poisson variance stabiliser.
       ⚠️ **At IBL's firing rates it does NOT stabilise the variance** (it needs lambda >~ 4;
       we run at 0.05-0.83), so what it actually does here is compress the high-count tail and
       monotonically relabel the counts. See ``src/data/preprocess.py::anscombe``. This is a
       *known limitation of the Gaussian latent*, not a step that fixes the counts.
    2. **Gaussian smoothing over time.** At these rates a single bin of the population vector is
       mostly shot noise; smoothing trades time resolution for a latent you can see on a single
       trial. *Legal here, illegal for the HMM* (which needs raw counts).
    3. **PCA** to n_latent dimensions, after z-scoring each neuron.

    Smoothing is specified in **milliseconds** (``smooth_sigma_ms``) and requires ``bin_size_s``.
    -------------------------------------------------------------------------------------------
    This is not a style preference. The ground truth (``scripts/validate_rslds_ground_truth.py``)
    showed that sigma-in-**bins** silently ties the amount of smoothing to the bin size: at
    ``smooth_sigma_bins=1.5`` you get 30 ms of smoothing at 20 ms bins but **75 ms** at 50 ms
    bins. A bin-size sweep then confounds bin size with smoothing, and because a Gaussian kernel
    *imposes* the temporal autocorrelation an AR fit reads as a slow mode, the confound pushes
    ``|lambda| -> 1`` (a spurious integrator) at the large-bin end. Sigma in ms is invariant to
    the bin, so the sweep is honest.

    ``smooth_sigma_bins`` is still accepted for backward compatibility (with a warning); prefer
    ``smooth_sigma_ms``. Pass a fitted ``pca`` to project *new* trials into an existing latent
    space (required for cross-validation, so the test fold does not re-estimate the basis).

    Returns
    -------
    x : (trials, time, n_latent) float64
    pca : the fitted PCA object, with ``mu_``/``sd_`` attached for reuse
    var_explained : (n_latent,) fraction of variance per component
    """
    import warnings

    from sklearn.decomposition import PCA

    from src.data.build_matrices import smooth_over_time

    # Resolve smoothing to a number of bins, preferring the ms spec.
    if smooth_sigma_ms is not None:
        if bin_size_s is None:
            raise ValueError("smooth_sigma_ms requires bin_size_s (sigma in ms -> bins).")
        sigma_bins = (smooth_sigma_ms / 1000.0) / bin_size_s
    elif smooth_sigma_bins is not None:
        warnings.warn(
            "counts_to_latents(smooth_sigma_bins=...) ties smoothing to bin size and makes a "
            "bin-size sweep confound smoothing with bin width -- which manufactures spurious "
            "integrators at large bins. Pass smooth_sigma_ms=... and bin_size_s=... instead.",
            DeprecationWarning, stacklevel=2,
        )
        sigma_bins = smooth_sigma_bins
    else:
        raise ValueError("provide smooth_sigma_ms (+ bin_size_s), or smooth_sigma_bins.")

    a = 2.0 * np.sqrt(counts.astype(np.float64) + 0.375)      # Anscombe (see docstring caveat)
    a = smooth_over_time(a, sigma_bins)                       # geometry only — never the HMM
    n_trials, n_bins, n_units = a.shape
    flat = a.reshape(n_trials * n_bins, n_units)

    if pca is None:
        mu = flat.mean(axis=0)
        sd = flat.std(axis=0)
        sd[sd < 1e-8] = 1.0                                   # a flat unit must not explode
        pca = PCA(n_components=n_latent, random_state=0)
        pca.fit((flat - mu) / sd)
        pca.mu_, pca.sd_ = mu, sd

    z = (flat - pca.mu_) / pca.sd_
    x = pca.transform(z).reshape(n_trials, n_bins, -1)
    return x, pca, pca.explained_variance_ratio_


# --------------------------------------------------------------------------- model


@dataclass
class RSLDSResult:
    """A fitted rSLDS, plus everything the figures and the sequence tests need."""

    x: np.ndarray                 # (trials, time, D) latents the model was fitted to
    states: np.ndarray            # (trials, time) most-likely (Viterbi) regime
    gamma: np.ndarray             # (trials, time, K) posterior P(z_t = k | all data)
    A: np.ndarray                 # (K, D, D) per-regime dynamics matrix
    b: np.ndarray                 # (K, D) per-regime affine offset
    Q: np.ndarray                 # (K, D, D) per-regime noise covariance
    log_pi: np.ndarray            # (K, K) baseline log-transition (unnormalised)
    R: np.ndarray                 # (K, D) recurrence weights (all zeros if recurrent=False)
    pi0: np.ndarray               # (K,) initial regime distribution
    log_likelihood: float         # exact, on the latents
    ll_trace: list = field(default_factory=list)
    bin_centers: np.ndarray | None = None
    recurrent: bool = True

    # --- dynamical-systems read-out (the point of the whole exercise) ---

    @property
    def n_states(self) -> int:
        return self.A.shape[0]

    def fixed_points(self) -> np.ndarray:
        """(K, D) fixed point of each regime's flow: x* solves x* = A x* + b.

        NaN for a regime whose ``I - A`` is singular — that is a *result*, not a failure:
        it means the flow has no isolated fixed point (a perfect line attractor /
        integrator direction), which is exactly the object a decision variable is supposed
        to live on.
        """
        K, D, _ = self.A.shape
        out = np.full((K, D), np.nan)
        for k in range(K):
            M = np.eye(D) - self.A[k]
            if np.linalg.matrix_rank(M, tol=1e-8) == D:
                out[k] = np.linalg.solve(M, self.b[k])
        return out

    def eigenvalues(self) -> np.ndarray:
        """(K, D) complex eigenvalues of each regime's A. |lambda| < 1 = contracting."""
        return np.array([np.linalg.eigvals(self.A[k]) for k in range(self.n_states)])

    def timescales(self, bin_size_s: float) -> np.ndarray:
        """(K,) time constant (s) of each regime's **slowest** mode.

        tau = -dt / ln|lambda_max|. A tau far longer than the trial means the regime is
        effectively an integrator on this timescale; a tau of a few bins means the regime
        is a fast transient that the trajectory passes straight through.
        """
        ev = np.abs(self.eigenvalues()).max(axis=1)
        ev = np.clip(ev, 1e-12, 1 - 1e-12)          # |lambda| >= 1 -> unstable, tau undefined
        return -bin_size_s / np.log(ev)

    def classify(self, tol: float = 0.05) -> list[str]:
        """Name each regime's flow. The one-line summary of what kind of object it is.

        ⚠️ **On IBL data these labels are NOT a dynamical read-out.** The shuffle-null test
        (``scripts/run_rslds_lambda_stability.py``, 2026-07-15) showed ``|lambda|`` is >97%
        reproduced on temporally-scrambled data: it is set by the smoothing of the latent, not by
        the neural dynamics. So "line attractor / integrator" here reflects the kernel, not the
        brain. Use these labels only as an *admissibility guard* (see :meth:`admissibility`), never
        as a reported result. The certified rSLDS outputs are the **sequence** and **recurrence**
        (which use regime order and position-dependent switching, not ``A``'s eigenvalues); the
        honest timescale comes from **GPFA** (which infers, rather than imposes, smoothness).
        """
        labels = []
        for lam in self.eigenvalues():
            mod = np.abs(lam)
            rotational = np.any(np.abs(lam.imag) > 1e-6)
            slow = np.any(np.abs(mod - 1.0) <= tol)
            if np.all(mod < 1.0 - tol):
                labels.append("rotational attractor" if rotational else "point attractor")
            elif np.all(mod > 1.0 + tol):
                labels.append("unstable / repeller")
            elif slow and np.all(mod <= 1.0 + tol):
                labels.append("slow mode (line attractor / integrator)")
            else:
                labels.append("saddle (transient waypoint)")
        return labels

    # Ground-truth-calibrated ceiling on |lambda| for the FLOW CLASSIFICATION to be trusted.
    # At 40 ms with real contracting attractors (tau = 150 ms) the admissible fits in
    # scripts/validate_rslds_ground_truth.py came out at |lambda| = 0.86-0.90. A fit whose
    # |lambda| exceeds ~0.95 is in the band where that same ground truth could NOT tell a real
    # slow mode from a shot-noise unit root -- so its integrator labels are unverifiable here.
    # This is NOT "the dynamics are fake"; it is "this backend cannot certify them". That is
    # precisely the ambiguity the Poisson `ssm` fit exists to resolve.
    SUSPECT_LAMBDA = 0.95

    def admissibility(self, *, true_switches: float | None = None,
                      suspect_lambda: float = SUSPECT_LAMBDA) -> dict:
        """Can we read a dynamical (attractor/integrator) claim off this fit, or not?

        The ground truth (``scripts/validate_rslds_ground_truth.py``) established that an
        under-resolved latent (too few spikes/bin, or too much imposed smoothing) fails in a
        *characteristic and seductive* way: ``|lambda| -> 1``, so regimes are classified
        "line attractor / integrator". That reads exactly like the headline result we want
        (*"a decision variable lives on this integrator"*), which is why it must be caught
        **mechanically** rather than by remembering the rule.

        Two things are graded separately, because they fail separately:

        * **``sequence_admissible``** -- are ``order_consistency`` / entry-times trustworthy? These
          survive a noisy latent (they only use *which regime is entered when*, not the flow), so
          this is almost always True. It is only False if the regimes flicker so hard there is no
          stable "first entry" (``flickering``).
        * **``flow_admissible``** -- is the attractor/integrator CLASSIFICATION trustworthy? False
          if ``|lambda|`` exceeds the ground-truth-calibrated ceiling (``suspect_lambda``) or a
          regime is outright unstable (``|lambda| >= 1``). **When False, do not report an
          integrator/line-attractor claim from this backend** -- escalate to the Poisson ``ssm``
          fit, which is the only thing that can certify a slow mode is real rather than a
          shot-noise unit root.

        ``admissible`` (kept for back-compat) is ``sequence_admissible and flow_admissible``.
        """
        lam = np.abs(self.eigenvalues()).max(axis=1)                 # (K,)
        labels = self.classify()
        n_integrator = int(sum("integrator" in s for s in labels))
        switches = float((self.states[:, 1:] != self.states[:, :-1]).sum(axis=1).mean())

        unstable = bool(np.any(lam >= 1.0))
        suspect = bool(np.any(lam >= suspect_lambda))
        flickering = (true_switches is not None) and (switches > 3 * true_switches)

        flow_ok = not (suspect or unstable)
        sequence_ok = not flickering

        return {
            "admissible": flow_ok and sequence_ok,
            "flow_admissible": flow_ok,
            "sequence_admissible": sequence_ok,
            "max_abs_lambda": round(float(lam.max()), 3),
            "n_integrator_regimes": n_integrator,
            "n_states": self.n_states,
            "switches_per_trial": round(switches, 2),
            "unstable": unstable,
            "suspect_lambda_exceeded": suspect,
            "flickering": flickering,
            "reason": (
                "|lambda| >= 1: an unstable regime (shot-noise unit root)" if unstable else
                f"|lambda| >= {suspect_lambda}: flow classification NOT certifiable by this "
                "backend -- escalate to ssm" if suspect else
                "flickering regimes: no stable sequence" if flickering else
                "ok"
            ),
        }


def fit_rslds(
    x: np.ndarray,
    n_states: int,
    *,
    recurrent: bool = True,
    n_iter: int = 60,
    tol: float = 1e-4,
    seed: int = 0,
    n_restarts: int = 3,
    sticky: float = 2.0,
    trans_maxiter: int = TRANS_MAXITER,
) -> RSLDSResult:
    """Fit the rSLDS to latent trajectories ``x`` of shape (trials, time, D) by exact EM.

    Parameters
    ----------
    recurrent
        If False, ``R`` is held at zero: a plain switching linear dynamical system whose
        transitions do not depend on the latent position. **Fit both and compare on
        held-out trials** — that comparison is the actual test of recurrence.
    sticky
        Self-transition pseudo-counts, as a fraction of the observed transition mass,
        applied in **every** M-step. Without it EM produces regimes that flicker bin to bin,
        which are not metastable states in any useful sense. A prior for *persistence*, not
        for *order*: it adds mass to k->k only, never to k->l, so it cannot manufacture the
        sequence we are testing for. ``sticky=0`` recovers the unregularised fit.
    n_restarts
        EM on a switching model is riddled with local optima. We run several seeds and keep
        the best training log-likelihood.
    trans_maxiter
        L-BFGS steps allowed in the transition M-step per EM iteration (**generalised EM**).
        The default is deliberately small: see ``TRANS_MAXITER``. Running it to convergence is
        6x slower and gives a bit-identical likelihood.

    Regime labels are permuted at the end so that regime 0 is the one occupied earliest —
    the same convention as ``hmm.py``, so figures line up.
    """
    best = None
    for r in range(n_restarts):
        res = _fit_once(
            x, n_states, recurrent=recurrent, n_iter=n_iter, tol=tol,
            seed=seed + 1000 * r, sticky=sticky, trans_maxiter=trans_maxiter,
        )
        if best is None or res.log_likelihood > best.log_likelihood:
            best = res
    return _order_by_time(best)


def _fit_once(x, K, *, recurrent, n_iter, tol, seed, sticky, trans_maxiter) -> RSLDSResult:
    from sklearn.cluster import KMeans

    n_trials, T, D = x.shape
    rng = np.random.default_rng(seed)

    # Initialise regimes by k-means on latent POSITION. Deliberately not on time: chopping
    # each trial into K equal time-chunks would hand the model the temporal sequence we are
    # trying to test for, and it would come back looking like a beautiful confirmation.
    flat = x.reshape(-1, D)
    km = KMeans(n_clusters=K, n_init=5, random_state=seed).fit(flat)
    gamma = np.zeros((n_trials * T, K))
    gamma[np.arange(n_trials * T), km.labels_] = 1.0
    gamma = gamma.reshape(n_trials, T, K)
    gamma = 0.9 * gamma + 0.1 / K                     # soften, so no regime starts empty

    log_pi = np.zeros((K, K)) + sticky * np.eye(K)
    R = np.zeros((K, D))
    A = np.tile(np.eye(D), (K, 1, 1)) + 0.01 * rng.standard_normal((K, D, D))
    b = np.zeros((K, D))
    Q = np.tile(np.eye(D), (K, 1, 1))
    pi0 = np.full(K, 1.0 / K)

    # We enter the loop holding responsibilities (from k-means), not parameters, so each
    # iteration is M-then-E. `xi` only exists after the first E-step, so the first pass
    # keeps the sticky initial transitions.
    xi = None
    ll_trace: list[float] = []
    for it in range(n_iter):
        A, b, Q = _m_step_dynamics(x, gamma)
        log_pi, R = _m_step_transitions(
            x, xi, gamma, log_pi, R, recurrent=recurrent, sticky=sticky,
            maxiter=trans_maxiter)
        pi0 = gamma[:, 0, :].mean(axis=0) + 1e-8
        pi0 /= pi0.sum()

        log_e = _emission_loglik(x, A, b, Q)          # (trials, T, K), zero at t = 0
        log_T = _transition_logits(x, log_pi, R)      # (trials, T, K, K), for t = 1..T-1
        gamma, xi, ll = _forward_backward(log_e, log_T, pi0)
        ll_trace.append(ll)

        if it > 2 and abs(ll_trace[-1] - ll_trace[-2]) < tol * abs(ll_trace[-2]):
            break

    states = _viterbi(log_e, log_T, pi0)

    return RSLDSResult(
        x=x, states=states, gamma=gamma, A=A, b=b, Q=Q, log_pi=log_pi, R=R, pi0=pi0,
        log_likelihood=float(ll), ll_trace=ll_trace, recurrent=recurrent,
    )


# --------------------------------------------------------------------------- E-step


def _emission_loglik(x, A, b, Q):
    """log N(x_t ; A_k x_{t-1} + b_k, Q_k) for every (trial, t>=1, k). Zero at t = 0.

    t = 0 carries no emission term because the model conditions on x_0 rather than
    generating it: there is no x_{-1} to propagate from.
    """
    n_trials, T, D = x.shape
    K = A.shape[0]
    out = np.zeros((n_trials, T, K))
    prev = x[:, :-1, :].reshape(-1, D)                # (n*(T-1), D)
    curr = x[:, 1:, :].reshape(-1, D)

    for k in range(K):
        Qk = Q[k] + Q_RIDGE * np.eye(D)
        L = np.linalg.cholesky(Qk)
        pred = prev @ A[k].T + b[k]
        resid = curr - pred
        sol = np.linalg.solve(L, resid.T)             # (D, n)
        quad = (sol ** 2).sum(axis=0)
        logdet = 2.0 * np.log(np.diag(L)).sum()
        ll = -0.5 * (quad + logdet + D * np.log(2 * np.pi))
        out[:, 1:, k] = ll.reshape(n_trials, T - 1)
    return out


def _transition_logits(x, log_pi, R):
    """(trials, T, K, K) row-normalised log P(z_t = k | z_{t-1} = j, x_{t-1}).

    Slot t holds the transition INTO t, so it is a function of x_{t-1}; slot 0 is unused.
    """
    n_trials, T, D = x.shape
    K = log_pi.shape[0]
    eta = np.zeros((n_trials, T, K, K))
    # log_pi[j, k] + R[k] . x_{t-1}   -> broadcast over j
    rec = x[:, :-1, :] @ R.T                          # (trials, T-1, K)
    eta[:, 1:, :, :] = log_pi[None, None, :, :] + rec[:, :, None, :]
    eta[:, 1:] -= logsumexp(eta[:, 1:], axis=-1, keepdims=True)
    return eta


def _forward_backward(log_e, log_T, pi0):
    """Exact forward-backward with time-varying transitions. Returns gamma, xi, total LL."""
    n_trials, T, K = log_e.shape
    log_alpha = np.zeros((n_trials, T, K))
    log_beta = np.zeros((n_trials, T, K))

    log_alpha[:, 0, :] = np.log(pi0)[None, :] + log_e[:, 0, :]
    for t in range(1, T):
        m = log_alpha[:, t - 1, :, None] + log_T[:, t, :, :]      # (n, K_prev, K_next)
        log_alpha[:, t, :] = logsumexp(m, axis=1) + log_e[:, t, :]

    for t in range(T - 2, -1, -1):
        m = log_T[:, t + 1, :, :] + (log_e[:, t + 1, :] + log_beta[:, t + 1, :])[:, None, :]
        log_beta[:, t, :] = logsumexp(m, axis=2)

    ll_per_trial = logsumexp(log_alpha[:, -1, :], axis=1)
    ll = float(ll_per_trial.sum())

    log_gamma = log_alpha + log_beta - ll_per_trial[:, None, None]
    gamma = np.exp(log_gamma)

    # xi[:, t, j, k] = P(z_{t-1} = j, z_t = k | data), for t = 1..T-1
    log_xi = np.zeros((n_trials, T, K, K))
    log_xi[:, 1:, :, :] = (
        log_alpha[:, :-1, :, None]
        + log_T[:, 1:, :, :]
        + (log_e[:, 1:, :] + log_beta[:, 1:, :])[:, :, None, :]
        - ll_per_trial[:, None, None, None]
    )
    xi = np.exp(log_xi)
    xi[:, 0] = 0.0
    return gamma, xi, ll


def _viterbi(log_e, log_T, pi0):
    """Most-likely regime path per trial."""
    n_trials, T, K = log_e.shape
    states = np.zeros((n_trials, T), dtype=int)
    for n in range(n_trials):
        delta = np.log(pi0) + log_e[n, 0]
        psi = np.zeros((T, K), dtype=int)
        for t in range(1, T):
            m = delta[:, None] + log_T[n, t]
            psi[t] = m.argmax(axis=0)
            delta = m.max(axis=0) + log_e[n, t]
        states[n, -1] = int(delta.argmax())
        for t in range(T - 1, 0, -1):
            states[n, t - 1] = psi[t, states[n, t]]
    return states


# --------------------------------------------------------------------------- M-step


def _m_step_dynamics(x, gamma):
    """Weighted least squares for (A_k, b_k, Q_k): regress x_t on [x_{t-1}, 1]."""
    n_trials, T, D = x.shape
    K = gamma.shape[2]
    prev = x[:, :-1, :].reshape(-1, D)
    curr = x[:, 1:, :].reshape(-1, D)
    Phi = np.concatenate([prev, np.ones((prev.shape[0], 1))], axis=1)   # (n, D+1)
    W = gamma[:, 1:, :].reshape(-1, K)                                  # weight of each pair

    A = np.zeros((K, D, D))
    b = np.zeros((K, D))
    Q = np.zeros((K, D, D))
    for k in range(K):
        w = W[:, k]
        sw = w.sum()
        if sw < D + 2:                       # regime effectively unused -> leave it inert
            A[k] = np.eye(D)
            Q[k] = np.eye(D)
            continue
        Phi_w = Phi * w[:, None]
        G = Phi.T @ Phi_w + 1e-6 * np.eye(D + 1)       # ridge: keeps G invertible
        H = Phi_w.T @ curr
        coef = np.linalg.solve(G, H)                    # (D+1, D)
        A[k] = coef[:D].T
        b[k] = coef[D]
        resid = curr - Phi @ coef
        Q[k] = (resid * w[:, None]).T @ resid / sw + Q_RIDGE * np.eye(D)
    return A, b, Q


def _m_step_transitions(x, xi, gamma, log_pi, R, *, recurrent, sticky=0.0,
                        maxiter: int = TRANS_MAXITER):
    """Fit (log_pi, R) by maximising sum_t sum_jk xi[t,j,k] log softmax_k(eta[t,j,k]).

    This is a weighted multinomial logistic regression and is **concave** in (log_pi, R),
    so L-BFGS on the exact gradient converges to the global optimum of the M-step. When
    ``recurrent=False`` the R block is dropped and the solution is the ordinary
    (closed-form) transition-count estimate.

    ``sticky`` adds self-transition pseudo-counts (a Dirichlet prior on each row), as a
    *fraction* of the observed transition mass. This is what makes the regimes **metastable**
    rather than flickering. It must be applied here, in every M-step — applying it only to
    the initial ``log_pi`` does nothing, because the first M-step overwrites it. (That was a
    real bug: on synthetic data with a known 0->1->2 sequence the decoded paths came out as
    0,1,0,2,1,0,1,2,... and regime recovery sat at 0.55.)

    It is a prior for *persistence*, not for *order*: it adds mass to k->k, never to k->k+1,
    so it cannot manufacture the sequence we are testing for.
    """
    from scipy.optimize import minimize

    if xi is None:
        return log_pi, R

    n_trials, T, D = x.shape
    K = log_pi.shape[0]
    xs = x[:, :-1, :].reshape(-1, D)                  # x_{t-1} for the transition into t
    X = xi[:, 1:, :, :].reshape(-1, K, K)             # (n, j, k)

    if sticky > 0:
        # Spread the pseudo-counts along the observed trajectory so the recurrent fit sees
        # them at representative x values, rather than piling them at one arbitrary point.
        X = X + (sticky * X.sum() / (K * X.shape[0])) * np.eye(K)[None, :, :]

    Nj = X.sum(axis=2)                                # (n, j) mass leaving j at this step

    if not recurrent:
        counts = X.sum(axis=0) + 1e-6                 # (K, K)
        return np.log(counts / counts.sum(axis=1, keepdims=True)), np.zeros((K, D))

    def unpack(theta):
        return theta[: K * K].reshape(K, K), theta[K * K:].reshape(K, D)

    def negobj(theta):
        lp, Rk = unpack(theta)
        eta = lp[None, :, :] + (xs @ Rk.T)[:, None, :]        # (n, j, k)
        lse = logsumexp(eta, axis=2, keepdims=True)
        logp = eta - lse
        obj = (X * logp).sum() - TRANS_L2 * (np.sum(lp ** 2) + np.sum(Rk ** 2))
        g = X - Nj[:, :, None] * np.exp(logp)                 # (n, j, k)
        g_lp = g.sum(axis=0) - 2 * TRANS_L2 * lp              # (K, K)
        g_R = np.einsum("njk,nd->kd", g, xs) - 2 * TRANS_L2 * Rk
        return -obj, -np.concatenate([g_lp.ravel(), g_R.ravel()])

    # Warm-started from the current parameters and given only a few L-BFGS steps: a partial
    # (generalised) M-step. See TRANS_MAXITER -- the full inner maximisation is 96% of runtime
    # and changes the final likelihood by nothing.
    theta0 = np.concatenate([log_pi.ravel(), R.ravel()])
    out = minimize(negobj, theta0, jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter, "ftol": 1e-8})
    return unpack(out.x)


def _order_by_time(res: RSLDSResult) -> RSLDSResult:
    """Relabel regimes so regime 0 is occupied earliest (same convention as hmm.py)."""
    K = res.n_states
    T = res.gamma.shape[1]
    occ = res.gamma.mean(axis=0).T                       # (K, T)
    t = np.arange(T)
    com = (occ * t).sum(axis=1) / np.clip(occ.sum(axis=1), 1e-9, None)
    order = np.argsort(com)
    inv = np.argsort(order)

    res.states = inv[res.states]
    res.gamma = res.gamma[:, :, order]
    res.A, res.b, res.Q = res.A[order], res.b[order], res.Q[order]
    res.R = res.R[order]
    res.log_pi = res.log_pi[np.ix_(order, order)]
    res.pi0 = res.pi0[order]
    return res


def score_rslds(res: RSLDSResult, x_new: np.ndarray) -> float:
    """Exact log-likelihood of held-out latent trajectories under a fitted model.

    Per time-bin, so folds and models with different trial counts are comparable.
    """
    log_e = _emission_loglik(x_new, res.A, res.b, res.Q)
    log_T = _transition_logits(x_new, res.log_pi, res.R)
    _, _, ll = _forward_backward(log_e, log_T, res.pi0)
    return float(ll / (x_new.shape[0] * x_new.shape[1]))


# --------------------------------------------------------------------------- selection


def cross_validate_K(
    counts: np.ndarray,
    candidate_states,
    *,
    n_latent: int = 3,
    smooth_sigma_ms: float = 40.0,
    bin_size_s: float = 0.04,
    n_folds: int = 3,
    n_iter: int = 40,
    n_restarts: int = 2,
    seed: int = 0,
    compare_recurrent: bool = True,
    sticky: float = 2.0,
    trans_maxiter: int = TRANS_MAXITER,
):
    """Choose K by held-out log-likelihood, and test whether recurrence pays for itself.

    **The PCA basis is re-fitted inside every fold, on the training trials only.** Fitting
    it once on all trials would leak the test fold into the latent space and inflate every
    held-out score.

    D (``n_latent``) is **not** selected here and must not be: changing D changes the data
    the likelihood is computed on, so the numbers are not comparable across D. Choose D from
    the PCA variance-explained curve and hold it fixed. (This is the same class of mistake
    as the -inf argmax that produced our bogus "K = 2 everywhere" HMM result — a number that
    is not comparable being compared anyway.)

    Returns a DataFrame with one row per K: ``mean_test_ll`` (recurrent) and, if
    ``compare_recurrent``, ``mean_test_ll_norec`` for the R = 0 model. ``delta_recurrent``
    > 0 means switching genuinely depends on latent position.
    """
    import pandas as pd
    from sklearn.model_selection import KFold

    n_trials = counts.shape[0]
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    rows = []
    for K in candidate_states:
        rec_ll, norec_ll = [], []
        for train_idx, test_idx in kf.split(np.arange(n_trials)):
            x_tr, pca, _ = counts_to_latents(
                counts[train_idx], n_latent=n_latent,
                smooth_sigma_ms=smooth_sigma_ms, bin_size_s=bin_size_s)
            x_te, _, _ = counts_to_latents(
                counts[test_idx], n_latent=n_latent,
                smooth_sigma_ms=smooth_sigma_ms, bin_size_s=bin_size_s, pca=pca)

            m = fit_rslds(x_tr, K, recurrent=True, n_iter=n_iter, n_restarts=n_restarts,
                          seed=seed, sticky=sticky, trans_maxiter=trans_maxiter)
            rec_ll.append(score_rslds(m, x_te))
            if compare_recurrent:
                m0 = fit_rslds(x_tr, K, recurrent=False, n_iter=n_iter, n_restarts=n_restarts,
                               seed=seed, sticky=sticky, trans_maxiter=trans_maxiter)
                norec_ll.append(score_rslds(m0, x_te))

        row = {
            "n_states": K,
            "mean_test_ll": float(np.mean(rec_ll)),
            "std_test_ll": float(np.std(rec_ll)),
        }
        if compare_recurrent:
            row["mean_test_ll_norec"] = float(np.mean(norec_ll))
            row["delta_recurrent"] = row["mean_test_ll"] - row["mean_test_ll_norec"]
        rows.append(row)

    df = pd.DataFrame(rows)
    if not np.isfinite(df["mean_test_ll"].to_numpy()).any():
        raise RuntimeError(
            "Every candidate K scored a non-finite held-out log-likelihood. Do NOT take the "
            "argmax — this is a degenerate-covariance failure, not a model-selection result."
        )
    return df


# --------------------------------------------------------------------------- sequence


def visit_sequences(states: np.ndarray) -> list[list[int]]:
    """Per trial, the regimes visited in order, with consecutive repeats collapsed."""
    out = []
    for row in states:
        seq = [int(row[0])]
        for s in row[1:]:
            if s != seq[-1]:
                seq.append(int(s))
        out.append(seq)
    return out


def precedence_matrix(states: np.ndarray, n_states: int) -> np.ndarray:
    """P[k, l] = P(regime k is first entered before regime l), over trials visiting both.

    This is the **direction-agnostic** description the project wants. It does not ask
    whether the sequence runs "forward" in any pre-imposed ordering; it asks whether, when
    two regimes both occur, one *reliably* comes first. NaN where too few trials visit both.
    """
    K = n_states
    first = np.full((states.shape[0], K), np.nan)
    for k in range(K):
        hit = states == k
        any_hit = hit.any(axis=1)
        first[any_hit, k] = hit.argmax(axis=1)[any_hit]

    P = np.full((K, K), np.nan)
    for k in range(K):
        for l in range(K):
            if k == l:
                continue
            both = np.isfinite(first[:, k]) & np.isfinite(first[:, l])
            if both.sum() < 5:
                continue
            P[k, l] = float((first[both, k] < first[both, l]).mean())
    return P


def order_consistency(states: np.ndarray, n_states: int) -> float:
    """One number for 'is there a reliable sequence, in *either* direction?'

    For each pair of regimes take ``max(P[k,l], 1 - P[k,l])`` — how reproducibly one
    precedes the other, whichever way round — and average over pairs.

    * **0.5** = the two regimes occur in random order across trials: no sequence, just a
      set of states the population wanders among.
    * **1.0** = every trial visits them in the same order: a sequence.

    Deliberately blind to *which* order, because the scientific claim under test is
    "decision-making is a temporal *sequence* of metastable states", not "the sequence runs
    in the direction we happened to label as forward". Compare against
    ``order_consistency_null``: shuffling bins within a trial preserves how much time is
    spent in each regime but destroys the order, and lands near 0.5.
    """
    P = precedence_matrix(states, n_states)
    iu = np.triu_indices(n_states, 1)
    vals = P[iu]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    return float(np.mean(np.maximum(vals, 1.0 - vals)))


def order_consistency_null(
    states: np.ndarray, n_states: int, *, n_shuffles: int = 200, seed: int = 0
) -> np.ndarray:
    """Null for ``order_consistency``: shuffle time bins within each trial.

    Preserves each trial's regime *composition* exactly and destroys only the *order*, so a
    consistency above this null is evidence of sequence and not of, say, one regime simply
    being more common.
    """
    rng = np.random.default_rng(seed)
    T = states.shape[1]
    out = np.empty(n_shuffles)
    for s in range(n_shuffles):
        sh = np.stack([row[rng.permutation(T)] for row in states])
        out[s] = order_consistency(sh, n_states)
    return out


def entry_times(states: np.ndarray, bin_centers: np.ndarray, n_states: int):
    """Median (and IQR) time of first entry into each regime, across trials.

    The cross-region *ordering* of these medians is the inter-areal timing read-out — the
    rSLDS counterpart of ``hmm.baseline_exit_latency``, but per regime.
    """
    import pandas as pd

    rows = []
    for k in range(n_states):
        hit = states == k
        any_hit = hit.any(axis=1)
        if any_hit.sum() < 5:
            rows.append({"state": k, "n_trials": int(any_hit.sum()),
                         "median_entry_s": np.nan, "q25": np.nan, "q75": np.nan})
            continue
        t = bin_centers[hit.argmax(axis=1)[any_hit]]
        rows.append({
            "state": k,
            "n_trials": int(any_hit.sum()),
            "frac_trials": round(float(any_hit.mean()), 3),
            "median_entry_s": round(float(np.median(t)), 3),
            "q25": round(float(np.quantile(t, 0.25)), 3),
            "q75": round(float(np.quantile(t, 0.75)), 3),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- flow field


def flow_field(res: RSLDSResult, state: int, grid=None, n_grid: int = 18):
    """Vector field ``dx = (A_k - I) x + b_k`` of one regime, on a grid over the latent.

    Plots the *flow*, i.e. where the trajectory would be pushed next from each point. The
    fixed point is where the arrows vanish; arrows converging on it = attractor, arrows
    passing by = saddle/transient.

    Uses the first two latent dimensions; higher dims are held at the fixed point (or 0).
    """
    D = res.A.shape[1]
    if grid is None:
        pts = res.x.reshape(-1, D)
        lo, hi = np.percentile(pts[:, :2], [1, 99], axis=0)
        pad = 0.15 * (hi - lo)
        grid = (lo - pad, hi + pad)
    lo, hi = grid

    gx = np.linspace(lo[0], hi[0], n_grid)
    gy = np.linspace(lo[1], hi[1], n_grid)
    GX, GY = np.meshgrid(gx, gy)

    fp = res.fixed_points()[state]
    base = np.zeros(D) if not np.all(np.isfinite(fp)) else fp
    X = np.tile(base, (GX.size, 1))
    X[:, 0] = GX.ravel()
    X[:, 1] = GY.ravel()

    dX = X @ (res.A[state] - np.eye(D)).T + res.b[state]
    return GX, GY, dX[:, 0].reshape(GX.shape), dX[:, 1].reshape(GX.shape)
