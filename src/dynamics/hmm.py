"""Poisson HMM over binned spike counts, with a pre-stimulus baseline prior.

Two design choices, both requested for this project:

1. **A real pre-stimulus baseline in the window.** Each trial's sequence starts well
   before stimulus onset, so the model must represent the resting/baseline regime, not
   only the evoked response. Every trial therefore *begins* in baseline.

2. **Bias the initial condition from the baseline.** After a first fit we measure the
   state distribution during the pre-stimulus bins and install it as the HMM's
   ``startprob_``. Viterbi decoding is then anchored in the baseline state, so the
   post-stimulus trajectory is read as a *departure from baseline* rather than from an
   arbitrary EM initialisation. This is a modelling choice, documented here so it can be
   turned off (``bias_startprob=False``) for comparison.

Directionality
--------------
Order the states by when they occur (occupancy centre-of-mass). In that ordering a
forward-flowing temporal sequence puts its transition mass in the **upper triangle** of
the transition matrix. ``directionality`` quantifies that asymmetry from the empirically
observed (Viterbi) transitions, which is more honest than reading it off the fitted
``transmat_`` alone.

Feed **raw integer counts** — Poisson emissions require them; never smooth first.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from src.data.build_matrices import stack_for_hmmlearn


# A Poisson rate of *exactly* zero is a degenerate maximum-likelihood estimate: it assigns
# probability 0 — log-likelihood -inf — to any count > 0. hmmlearn will happily fit
# lambdas_ == 0 for a (state, neuron) pair that never spiked in training, and then a single
# spike in a held-out fold makes the whole held-out score -inf.
#
# That is not hypothetical: it silently destroyed our first cross-validation. Sparse VISp
# units (2 spikes in the entire window) can be completely silent in a training fold, so
# every state learns lambda = 0 for them, and every candidate K scored -inf. `idxmax` over a
# column of -inf returns index 0, which made "best K" come out as the *smallest* K for every
# region — a pure artifact that looked like a real result. See
# docs/worklog/2026-07-12_pynapple-and-env.md.
#
# Flooring the rate is the standard remedy (equivalently: a weak prior that no neuron is
# truly incapable of spiking). 1e-6 spikes/bin is far below anything observable at 20 ms
# bins, so it cannot manufacture structure — it only prevents -inf.
LAMBDA_FLOOR = 1e-6


def _floor_lambdas(model, floor: float = LAMBDA_FLOOR):
    """Clip fitted Poisson rates away from exactly zero. See LAMBDA_FLOOR."""
    import numpy as _np

    model.lambdas_ = _np.maximum(model.lambdas_, floor)
    return model


def drop_units_silent_in(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop neurons with zero spikes in *this* tensor. Returns (counts, kept_idx).

    Must be re-applied after any trial subsampling: `drop_silent_units` upstream is computed
    on the full trial set, so a unit that survives there can still be all-zero in a
    subsample — and an all-zero unit is exactly the one that produces lambda == 0.
    """
    kept = np.flatnonzero(counts.sum(axis=(0, 1)) > 0)
    return counts[:, :, kept], kept


@dataclass
class HMMResult:
    """Everything downstream code and figures need from one region's fit."""

    model: object                 # fitted hmmlearn.PoissonHMM (states already reordered)
    states: np.ndarray            # (n_trials, n_bins) Viterbi states, in time order
    bin_centers: np.ndarray       # (n_bins,) seconds relative to the align event
    order: np.ndarray             # permutation applied: order[new] = old state index
    occupancy: np.ndarray         # (n_states, n_bins) P(state | time)
    trans_counts: np.ndarray      # (n_states, n_states) observed transition counts
    startprob_baseline: np.ndarray  # (n_states,) baseline state distribution installed
    forward_fraction: float       # share of net off-diagonal flux that flows forward
    log_likelihood: float


def fit_poisson_hmm(
    counts: np.ndarray,
    n_states: int,
    baseline_mask: np.ndarray,
    *,
    n_iter: int = 50,
    seed: int = 0,
    bias_startprob: bool = True,
) -> HMMResult:
    """Fit a Poisson HMM to a (trials x time x neurons) count tensor.

    Parameters
    ----------
    counts
        Integer spike counts, shape (n_trials, n_bins, n_neurons). Raw, unsmoothed.
    n_states
        Number of hidden states K.
    baseline_mask
        Boolean length-n_bins array; ``True`` for pre-stimulus bins. Used to (a) install
        the startprob prior and (b) sanity-check that the earliest state is the baseline.
    bias_startprob
        If True, override ``startprob_`` with the measured pre-stimulus occupancy after
        the first fit, then re-decode. If False, keep the EM-estimated startprob.

    Returns
    -------
    HMMResult with states/transition matrix **reordered** so state 0 is earliest.
    """
    from hmmlearn.hmm import PoissonHMM

    n_trials, n_bins, _ = counts.shape
    X, lengths = stack_for_hmmlearn(counts)

    # implementation="log": both faster than "scaling" here (benchmarked) and numerically
    # stable — "scaling" underflows on the forward pass when scoring held-out data.
    model = PoissonHMM(
        n_components=n_states, n_iter=n_iter, random_state=seed, tol=1e-2,
        implementation="log",
    )
    model.fit(X, lengths)
    _floor_lambdas(model)   # never leave a Poisson rate at exactly 0 (see LAMBDA_FLOOR)
    states = model.predict(X, lengths).reshape(n_trials, n_bins)

    # (2) install the baseline prior and re-decode
    startprob_baseline = _state_distribution(states[:, baseline_mask], n_states)
    if bias_startprob:
        prior = startprob_baseline.copy()
        prior[prior < 1e-6] = 1e-6           # keep every state reachable
        model.startprob_ = prior / prior.sum()
        states = model.predict(X, lengths).reshape(n_trials, n_bins)

    # order states by occupancy centre-of-mass in time, then relabel everything
    occ = _occupancy(states, n_states)               # (K, n_bins)
    order = _order_by_time(occ)
    model, states, occ = _apply_order(model, states, occ, order)

    trans_counts = _transition_counts(states, n_states)
    forward = _forward_fraction(trans_counts)
    # Relabelling states is a permutation, so the data log-likelihood is unchanged.
    ll = float(model.score(X, lengths))

    return HMMResult(
        model=model,
        states=states,
        bin_centers=np.array([]),   # filled in by the caller which knows the timebase
        order=order,
        occupancy=occ,
        trans_counts=trans_counts,
        startprob_baseline=startprob_baseline[order],
        forward_fraction=forward,
        log_likelihood=ll,
    )


# --------------------------------------------------------------------------- helpers


def _state_distribution(states_slice: np.ndarray, n_states: int) -> np.ndarray:
    """Normalised histogram of states over an arbitrary slice."""
    counts = np.bincount(states_slice.ravel(), minlength=n_states).astype(float)
    total = counts.sum()
    return counts / total if total else np.full(n_states, 1.0 / n_states)


def _occupancy(states: np.ndarray, n_states: int) -> np.ndarray:
    """P(state | time bin): shape (n_states, n_bins)."""
    n_bins = states.shape[1]
    occ = np.zeros((n_states, n_bins))
    for t in range(n_bins):
        occ[:, t] = np.bincount(states[:, t], minlength=n_states) / states.shape[0]
    return occ


def _order_by_time(occ: np.ndarray) -> np.ndarray:
    """Permutation ordering states by occupancy centre-of-mass (earliest first)."""
    t = np.arange(occ.shape[1])
    com = (occ * t).sum(axis=1) / np.clip(occ.sum(axis=1), 1e-9, None)
    return np.argsort(com)


def _apply_order(model, states, occ, order):
    """Relabel states everywhere by ``order`` (new label i == old label order[i])."""
    inv = np.argsort(order)                       # old -> new label
    new_states = inv[states]
    new_occ = occ[order]
    model.startprob_ = model.startprob_[order]
    model.transmat_ = model.transmat_[np.ix_(order, order)]
    model.lambdas_ = model.lambdas_[order]
    return model, new_states, new_occ


def _transition_counts(states: np.ndarray, n_states: int) -> np.ndarray:
    """Observed transition counts N[i,j] from Viterbi paths (within trials only)."""
    N = np.zeros((n_states, n_states), dtype=np.int64)
    a = states[:, :-1].ravel()
    b = states[:, 1:].ravel()
    np.add.at(N, (a, b), 1)
    return N


def _forward_fraction(N: np.ndarray) -> float:
    """Share of net off-diagonal transition flux that flows to a later-ordered state.

    States are already time-ordered, so 'forward' == upper triangle. Returns 0.5 for a
    perfectly symmetric (directionless) process, →1 for a strict forward sequence.
    """
    off = N.copy()
    np.fill_diagonal(off, 0)
    fwd = np.triu(off, 1).sum()
    bwd = np.tril(off, -1).sum()
    total = fwd + bwd
    return float(fwd / total) if total else 0.5


def net_flux(N: np.ndarray) -> np.ndarray:
    """Antisymmetric net transition flux Φ[i,j] = N[i,j] − N[j,i] (positive = forward)."""
    return N - N.T


def cross_validate_states(
    counts: np.ndarray,
    candidate_states,
    *,
    n_folds: int = 5,
    n_iter: int = 50,
    seed: int = 0,
):
    """Choose K (number of HMM states) by cross-validated held-out log-likelihood.

    Trials are the unit of splitting: we fit on the training trials and score the
    *held-out* trials, so K is rewarded for generalising, not for memorising. Scores are
    per-sample (total LL / number of time bins scored) so folds are comparable.

    Returns a pandas DataFrame with columns
    ``['n_states', 'mean_test_ll', 'std_test_ll', 'mean_train_ll']`` and the K with the
    highest mean held-out LL as ``df.attrs['best_k']``.

    Note: held-out LL typically rises then plateaus; the plateau's onset is the honest K.
    Read the curve, don't just take the argmax.
    """
    import pandas as pd
    from hmmlearn.hmm import PoissonHMM
    from sklearn.model_selection import KFold

    # A unit that is all-zero in *this* tensor forces lambda == 0 in every state, which makes
    # any held-out spike score -inf. Drop those first (see drop_units_silent_in).
    counts, kept = drop_units_silent_in(counts)

    n_trials, n_bins, _ = counts.shape
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    rows = []
    for K in candidate_states:
        test_lls, train_lls = [], []
        for train_idx, test_idx in kf.split(np.arange(n_trials)):
            Xtr, ltr = stack_for_hmmlearn(counts[train_idx])
            Xte, lte = stack_for_hmmlearn(counts[test_idx])
            model = PoissonHMM(n_components=K, n_iter=n_iter, random_state=seed,
                               tol=1e-2, implementation="log")
            model.fit(Xtr, ltr)
            # Even after dropping all-zero units, a unit can be silent in a *training fold*
            # while spiking in the held-out fold -> lambda == 0 -> -inf. Floor it.
            _floor_lambdas(model)
            test_lls.append(model.score(Xte, lte) / Xte.shape[0])
            train_lls.append(model.score(Xtr, ltr) / Xtr.shape[0])
        rows.append({
            "n_states": K,
            "mean_test_ll": float(np.mean(test_lls)),
            "std_test_ll": float(np.std(test_lls)),
            "mean_train_ll": float(np.mean(train_lls)),
        })

    df = pd.DataFrame(rows)

    # Fail loudly rather than let argmax pick index 0 off a column of -inf, which is exactly
    # how the bogus "K=2 for every region" result was produced.
    ll = df["mean_test_ll"].to_numpy()
    if not np.isfinite(ll).any():
        raise RuntimeError(
            "Every candidate K scored a non-finite held-out log-likelihood. This is the "
            "lambda==0 / -inf failure, not a model-selection outcome — do not take the "
            "argmax. Check for near-silent units in the tensor."
        )
    df.attrs["best_k"] = int(df.loc[df["mean_test_ll"].idxmax(), "n_states"])
    return df


def select_k_by_elbow(cv: "pd.DataFrame") -> int:
    """Choose K as the elbow of the CV curve: the last K that buys more than noise.

    **Do not use the argmax of held-out log-likelihood.** On spike counts it barely ever
    turns over: a Poisson HMM assumes neurons are conditionally independent given the
    state, but real populations are correlated, and the *only* way the model can express
    correlation is to add states. So held-out LL keeps creeping up with K, and the argmax
    just lands on whichever K is largest in the list (verified 2026-07-12: it returned the
    top of the range for every region, having previously returned the bottom of the range
    for every region when the scores were -inf).

    The elbow asks a better question: **is the next state buying more than the fold-to-fold
    noise?** We take the increment `LL(K) - LL(K-1)` and compare it to the standard
    deviation across CV folds at K. The elbow is the smallest K after which *no* later
    increment clears its own noise floor.

    This is a judgement call encoded as a rule, not a truth — report it, and check the
    conclusions are not sensitive to it.
    """
    ll = cv["mean_test_ll"].to_numpy()
    sd = cv["std_test_ll"].to_numpy()
    ks = cv["n_states"].to_numpy()
    if len(ks) < 2:
        return int(ks[0])

    gains = np.diff(ll)
    noise = sd[1:]                       # fold SD at the K we moved *to*
    beats_noise = gains > noise

    for i in range(len(gains)):
        if not beats_noise[i:].any():    # nothing after this point is worth buying
            return int(ks[i])
    return int(ks[-1])                   # still climbing at the top of the range


# ---------------------------------------------------------------------------
# State stability — the criterion that actually matches the scientific claim.
# ---------------------------------------------------------------------------


def _fit_best_of(counts_subset, n_states, n_iter, seed, n_restarts):
    """Fit the HMM ``n_restarts`` times and keep the highest-likelihood solution.

    EM is non-convex. Without restarts, two fits can disagree simply because they fell into
    different local optima from different initialisations — and `state_stability` would
    charge that to "the states are not real". Restarting and keeping the best-LL fit means a
    disagreement reflects *different data*, not *different luck*. Omitting this biases the
    stability estimate DOWNWARD, and therefore biases the selected K down too.
    """
    from hmmlearn.hmm import PoissonHMM

    if n_restarts < 1:
        raise ValueError("n_restarts must be >= 1")

    X, lengths = stack_for_hmmlearn(counts_subset)
    best, best_ll = None, -np.inf
    for r in range(n_restarts):
        m = PoissonHMM(n_components=n_states, n_iter=n_iter, random_state=seed + 1000 * r,
                       tol=1e-2, implementation="log")
        m.fit(X, lengths)
        _floor_lambdas(m)
        ll = float(m.score(X, lengths))
        if ll > best_ll:
            best, best_ll = m, ll
    assert best is not None      # n_restarts >= 1, so the loop always assigns
    return best


def state_stability(
    counts: np.ndarray,
    n_states: int,
    *,
    n_repeats: int = 3,
    n_iter: int = 25,
    seed: int = 0,
    n_restarts: int = 1,
):
    """Do two INDEPENDENT fits recover the same partition of time? Returns a dict of scores.

    Why this, and not log-likelihood
    --------------------------------
    Our scientific claim is that decision-making passes through *metastable states* — that
    there exist identifiable, reproducible regimes. Held-out likelihood does not test that
    claim. It rewards flexibility: a model with more states predicts held-out spikes a
    little better whether or not those states are real, which is why the CV curve never
    turns over.

    Reproducibility is the claim, so measure reproducibility. A state that appears when you
    fit trials 1-100 but not when you fit trials 101-200 is not an attractor — it is a
    fitting artifact. Concretely:

    1. Split trials three ways: **A**, **B**, **C**.
    2. Fit one HMM on **A** and a completely separate HMM on **B**.
    3. Decode *both* models on the *same* held-out trials **C**.
    4. Ask how much the two decoded state sequences agree.

    Both models see C for the first time, so neither is advantaged. If the K states are real,
    two models that never shared a trial should carve C's timeline the same way.

    Unlike likelihood, this has a **peak**: too few states and both models agree trivially
    but describe little; too many and the surplus states get split arbitrarily by each fit,
    so agreement collapses. The turnover is the point.

    Metrics
    -------
    ``ari``
        Adjusted Rand Index between the two decodings of C. **Permutation-invariant** — it
        compares the *partition* of time, not the arbitrary state labels — and chance-
        corrected, so 0 = chance and 1 = identical. This is the headline number.
    ``matched_acc``
        Fraction of time bins on which the two models agree after optimally matching their
        labels (Hungarian algorithm on the confusion matrix). Easier to interpret than ARI,
        but not chance-corrected — it rises trivially as K falls, so read it *with* the ARI.

    Known confounds (read these before believing a low ARI)
    -------------------------------------------------------
    * **Trial-limited, not state-limited.** Each fit sees only ~1/3 of the trials. A real but
      subtle state may simply not be recoverable *twice* from that many trials, and shows up
      as instability — a false negative. Diagnose by re-running at several trial counts: if
      the peak K marches right as trials grow, you are data-limited, not state-limited.
    * **EM local minima.** Two fits can disagree from bad luck rather than different data.
      ``n_restarts`` > 1 fits each model several times and keeps the best-likelihood
      solution. With ``n_restarts=1`` the ARI is an UNDER-estimate of true stability.
    * **Stability is not correctness.** A model can be stably wrong: one strong event (a rate
      step at stimulus onset) reproduces trivially, so a high ARI at K=2 means almost
      nothing. The informative question is not "is ARI high" but "how far up in K does it
      STAY high".

    Returns ``{"ari_mean","ari_std","matched_acc_mean","matched_acc_std"}`` over repeats.
    """
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import adjusted_rand_score

    rng = np.random.default_rng(seed)
    n_trials = counts.shape[0]

    aris, accs = [], []
    for rep in range(n_repeats):
        idx = rng.permutation(n_trials)
        a, b, c = np.array_split(idx, 3)          # disjoint: fit A, fit B, judge on C
        if min(len(a), len(b), len(c)) < 2:
            raise ValueError("Too few trials to split three ways for a stability estimate.")

        models = [
            _fit_best_of(counts[part], n_states, n_iter, seed + rep, n_restarts)
            for part in (a, b)
        ]

        Xc, lc = stack_for_hmmlearn(counts[c])
        s1 = models[0].predict(Xc, lc)
        s2 = models[1].predict(Xc, lc)

        aris.append(adjusted_rand_score(s1, s2))

        # optimal label matching, then agreement
        conf = np.zeros((n_states, n_states), dtype=np.int64)
        np.add.at(conf, (s1, s2), 1)
        row, col = linear_sum_assignment(-conf)   # maximise the matched diagonal
        accs.append(conf[row, col].sum() / conf.sum())

    return {
        "n_states": int(n_states),
        "ari_mean": float(np.mean(aris)),
        "ari_std": float(np.std(aris)),
        "matched_acc_mean": float(np.mean(accs)),
        "matched_acc_std": float(np.std(accs)),
    }


def stability_scan(
    counts: np.ndarray,
    candidate_states,
    *,
    n_repeats: int = 3,
    n_iter: int = 25,
    seed: int = 0,
    n_restarts: int = 1,
    progress: bool = False,
):
    """`state_stability` across a range of K. Returns a DataFrame; peak ARI is the honest K."""
    import pandas as pd

    rows = []
    for K in candidate_states:
        t0 = time.time()
        row = state_stability(counts, K, n_repeats=n_repeats, n_iter=n_iter, seed=seed,
                              n_restarts=n_restarts)
        rows.append(row)
        if progress:
            print(f"    K={K:2d}  ARI={row['ari_mean']:.3f} +/- {row['ari_std']:.3f}"
                  f"   matched_acc={row['matched_acc_mean']:.3f}   [{time.time()-t0:.0f}s]",
                  flush=True)
    return pd.DataFrame(rows)


def baseline_exit_latency(
    occupancy: np.ndarray,
    bin_centers: np.ndarray,
    baseline_state: int = 0,
    thresh: float = 0.5,
    min_consecutive: int = 3,
    require_dominant_baseline: bool = True,
) -> float:
    """DEPRECATED — use :func:`baseline_exit_latency_robust`. Kept so old calls still run.

    This version returned ``nan`` for **VISp and MRN** on the real session, i.e. it failed in
    two of three regions, which made the inter-areal timing read-out useless. Two design bugs,
    both fixed in the robust version:

    1. It hard-codes the baseline as **state 0** (the earliest occupancy centre-of-mass). The
       earliest-on-average state is *not necessarily the pre-stimulus state*, and nothing
       enforced that it was.
    2. It demands that **one single state** hold > 50 % of the pre-stimulus bins. When baseline
       is shared between two states — which is a perfectly ordinary fit, not a pathology — the
       guard fires and the answer is thrown away.

    Both replaced by: identify the baseline **set** empirically, and use a **relative**
    threshold (fall to half of its *own* pre-stimulus level), which needs no assumption about
    how dominant baseline was to begin with.
    """
    pre = bin_centers < 0.0
    if require_dominant_baseline and occupancy[baseline_state, pre].mean() < 0.5:
        return float("nan")

    post = bin_centers >= 0.0
    occ = occupancy[baseline_state, post]
    t = bin_centers[post]
    below = occ < thresh
    # find the first index starting a run of >= min_consecutive True values
    run = 0
    for i, b in enumerate(below):
        run = run + 1 if b else 0
        if run >= min_consecutive:
            return float(t[i - min_consecutive + 1])
    return float("nan")


def baseline_states(occupancy: np.ndarray, bin_centers: np.ndarray,
                    coverage: float = 0.5) -> np.ndarray:
    """Which states *are* the baseline? The smallest set covering ``coverage`` of pre-stim time.

    Do not assume the baseline is one state, and do not assume it is state 0. Measure it: rank
    states by their pre-stimulus occupancy and take the smallest set that accounts for at least
    ``coverage`` of the pre-stimulus bins. Usually that is one state; when the fit splits
    baseline across two, this returns both instead of silently reporting failure.
    """
    pre = bin_centers < 0.0
    occ_pre = occupancy[:, pre].mean(axis=1)              # (K,) share of pre-stim bins
    order = np.argsort(occ_pre)[::-1]
    cum = np.cumsum(occ_pre[order])
    n = int(np.searchsorted(cum, coverage) + 1)
    return np.sort(order[:min(n, len(order))])


def baseline_exit_latency_robust(
    occupancy: np.ndarray,
    bin_centers: np.ndarray,
    *,
    coverage: float = 0.5,
    rel_thresh: float = 0.5,
    min_consecutive: int = 2,
) -> dict:
    """Time (s) at which the population *sustainably* leaves baseline. Works in every region.

    Two changes from the original, each fixing a way the old one returned ``nan``:

    * **The baseline is a SET, found from the data** (:func:`baseline_states`), not hard-coded
      to state 0. Nothing guaranteed the earliest-centre-of-mass state was the pre-stimulus one.
    * **The threshold is RELATIVE to baseline's own pre-stimulus level**, not an absolute 0.5.
      "Baseline occupancy has fallen to half of what it was" is meaningful whether that level
      started at 0.95 or at 0.45; an absolute 0.5 is only meaningful in the first case, and that
      is precisely why VISp and MRN came back ``nan``.

    A *sustained* crossing (``min_consecutive`` bins) is still required, so a single noisy bin
    dipping under the line cannot report a spuriously early exit.

    Returns ``{'latency_s', 'baseline_states', 'baseline_level', 'threshold'}``. ``latency_s``
    is ``nan`` only if the population genuinely never leaves — which is now a *result*, not an
    artefact of the guard.
    """
    pre = bin_centers < 0.0
    post = bin_centers >= 0.0

    bstates = baseline_states(occupancy, bin_centers, coverage=coverage)
    p_base = occupancy[bstates].sum(axis=0)              # P(in ANY baseline state | time)

    level = float(p_base[pre].mean())
    thr = rel_thresh * level

    t = bin_centers[post]
    below = p_base[post] < thr

    latency = float("nan")
    run = 0
    for i, b in enumerate(below):
        run = run + 1 if b else 0
        if run >= min_consecutive:
            latency = float(t[i - min_consecutive + 1])
            break

    return {
        "latency_s": latency,
        "baseline_states": bstates.tolist(),
        "baseline_level": round(level, 3),
        "threshold": round(thr, 3),
    }


# ---------------------------------------------------------------------------
# Directedness, done honestly: per trial, split-half, and separated from switch RATE.
# ---------------------------------------------------------------------------


def switch_rate(states: np.ndarray) -> float:
    """Mean number of state CHANGES per trial. **This is not directedness.**

    Recorded separately because the two are constantly conflated. ``forward_fraction`` measures
    the *asymmetry* of transitions in the time-ordering; ``switch_rate`` measures *how often*
    the population changes state at all. A region can switch constantly with no preferred
    direction (high switch_rate, forward_fraction ~ 0.5), or switch twice per trial always in
    the same order (low switch_rate, forward_fraction ~ 1.0).

    VISp's forward_fraction of 0.53 says its transitions are **symmetric**, NOT that it
    transitions rarely. Report both, always.
    """
    changes = (states[:, 1:] != states[:, :-1]).sum(axis=1)
    return float(changes.mean())


def forward_fraction_per_trial(states: np.ndarray, n_states: int) -> np.ndarray:
    """``forward_fraction`` computed **within each trial** -> (n_trials,), NaN if no transitions.

    The pooled statistic can be high because *most trials* run forward, or because *a few
    trials* contribute most of the transitions. Those are different claims. The distribution
    settles it, and it is the direct answer to "does the sequence hold on single trials, or
    only on average?"
    """
    n_trials = states.shape[0]
    out = np.full(n_trials, np.nan)
    for i in range(n_trials):
        N = _transition_counts(states[i:i + 1], n_states)
        np.fill_diagonal(N, 0)
        fwd = np.triu(N, 1).sum()
        bwd = np.tril(N, -1).sum()
        if fwd + bwd > 0:
            out[i] = fwd / (fwd + bwd)
    return out


def forward_fraction_split_half(
    counts: np.ndarray,
    n_states: int,
    baseline_mask: np.ndarray,
    *,
    n_repeats: int = 5,
    seed: int = 0,
    n_iter: int = 50,
) -> dict:
    """``forward_fraction`` with the state ORDERING derived on held-out trials. Removes the loop.

    The circularity it closes
    -------------------------
    ``forward_fraction`` needs a time-ordering of the states, and we take that ordering from the
    states' occupancy centre-of-mass — measured on **the same trials** whose transitions we then
    score. That is not vacuous (the ordering is a trial-*averaged* quantity while the transitions
    are single-trial, which is exactly why VISp can still come out at 0.53), but it is close
    enough to circular that it deserves a clean test.

    So: fit on half the trials, **fix the state ordering from that half**, then decode the other
    half and score the transitions there. The ordering never sees the transitions it is judged on.

    If the split-half value tracks the pooled one, the pooled number was honest. If it collapses
    to ~0.5, the directedness was an artefact of ordering-then-testing-the-ordering.
    """
    rng = np.random.default_rng(seed)
    n_trials = counts.shape[0]

    vals = []
    for rep in range(n_repeats):
        idx = rng.permutation(n_trials)
        a, b = np.array_split(idx, 2)

        # Fit on A. `fit_poisson_hmm` already relabels states by occupancy COM on A.
        res = fit_poisson_hmm(counts[a], n_states, baseline_mask,
                              n_iter=n_iter, seed=seed + rep, bias_startprob=True)

        # Decode B with that model. The ordering is inherited from A and is NOT re-derived.
        Xb, lb = stack_for_hmmlearn(counts[b])
        states_b = res.model.predict(Xb, lb).reshape(len(b), counts.shape[1])

        N = _transition_counts(states_b, n_states)
        vals.append(_forward_fraction(N))

    vals = np.asarray(vals)
    return {
        "forward_fraction_split_half": float(vals.mean()),
        "std": float(vals.std()),
        "n_repeats": n_repeats,
        "values": vals.tolist(),
    }


def forward_fraction_null(
    counts: np.ndarray,
    n_states: int,
    baseline_mask: np.ndarray,
    *,
    n_shuffles: int = 20,
    seed: int = 0,
) -> np.ndarray:
    """Null distribution of forward_fraction under within-trial time-bin shuffling.

    Shuffling bins within each trial destroys the temporal sequence while preserving
    per-trial rate content, so the null answers: 'how chain-like would this look with no
    real sequence?'. Compare the observed forward_fraction against this.
    """
    rng = np.random.default_rng(seed)
    n_trials, n_bins, _ = counts.shape
    out = np.empty(n_shuffles)
    for s in range(n_shuffles):
        sh = np.empty_like(counts)
        for i in range(n_trials):
            sh[i] = counts[i, rng.permutation(n_bins), :]
        out[s] = fit_poisson_hmm(
            sh, n_states, baseline_mask, seed=seed, bias_startprob=True
        ).forward_fraction
    return out
