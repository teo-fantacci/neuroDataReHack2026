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

from dataclasses import dataclass

import numpy as np

from src.data.build_matrices import stack_for_hmmlearn


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
            test_lls.append(model.score(Xte, lte) / Xte.shape[0])
            train_lls.append(model.score(Xtr, ltr) / Xtr.shape[0])
        rows.append({
            "n_states": K,
            "mean_test_ll": float(np.mean(test_lls)),
            "std_test_ll": float(np.std(test_lls)),
            "mean_train_ll": float(np.mean(train_lls)),
        })

    df = pd.DataFrame(rows)
    df.attrs["best_k"] = int(df.loc[df["mean_test_ll"].idxmax(), "n_states"])
    return df


def baseline_exit_latency(
    occupancy: np.ndarray,
    bin_centers: np.ndarray,
    baseline_state: int = 0,
    thresh: float = 0.5,
    min_consecutive: int = 3,
    require_dominant_baseline: bool = True,
) -> float:
    """Time (s) at which the population *sustainably* leaves its baseline state.

    Defined as the first post-stimulus time where P(baseline_state) stays below ``thresh``
    for at least ``min_consecutive`` consecutive bins. Requiring a sustained crossing (not
    a single-bin dip) avoids the noise artifact where an occupancy that merely hovers near
    the threshold reports a spuriously early exit.

    The *ordering across regions* of this latency is the inter-areal directionality the
    project cares about (does VISp leave baseline before SCm before MRN?).

    Returns ``nan`` if the baseline state is not dominant pre-stimulus
    (``require_dominant_baseline`` — the latency is meaningless if there was no baseline to
    leave) or if the threshold is never sustainably crossed.
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
