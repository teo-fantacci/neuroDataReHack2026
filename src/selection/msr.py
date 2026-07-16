"""Multiscale Relevance (MSR): rank neurons by how informative their spiking is, unsupervised.

The idea (Cubero, Marsili & Roudi 2020, J Comput Neurosci 48:85-102; arXiv:1802.10354)
-------------------------------------------------------------------------------------
Take one neuron's spike train and bin it at resolution dt. Each bin i has a count k_i, and
M = sum_i k_i spikes total. Now ask a question about the *sample*, not about the neuron:
if I pick a spike at random, how much do I learn about **which bin** it came from?

* **Resolution** ``H[s] = - sum_i (k_i/M) log(k_i/M)`` — the entropy over bins. At very
  fine dt every spike sits in its own bin and H[s] is maximal (each spike is fully
  resolved); at very coarse dt all spikes fall in one bin and H[s] -> 0.
* **Relevance** ``H[k] = - sum_k (k*m_k/M) log(k*m_k/M)``, where ``m_k`` is the number of
  bins holding exactly k spikes. This is the entropy of the *frequency of frequencies* —
  how much the data say about the underlying distribution rather than about the sample.

Sweeping dt traces a curve of H[k] against H[s]. **MSR is the area under that curve.** A
neuron whose curve stays high over many resolutions is one whose spike statistics carry
structure at *many timescales* — the opposite of a constant-rate Poisson unit, whose counts
are maximally boring at every scale.

Why this is attractive here
---------------------------
It is **model-free and unsupervised**: it never sees the stimulus, the choice, or the trial
structure. So it cannot manufacture task-relevance out of nothing, and using it to pick
units does not double-dip on the effects we later want to test.

Three honest caveats
--------------------
1. **MSR is not task-relevance.** A neuron can be dynamically rich for reasons that have
   nothing to do with the decision (bursting, drift, licking). ``task_selectivity`` below is
   the cross-check the paper's own authors recommend — and it must be run on *different*
   trials from the ones used for the final model if you want to avoid circularity.
2. **MSR correlates with firing rate**, because rate sets how many spikes M there are to be
   informative with. "MSR-selected units beat random units" is therefore a rigged
   comparison unless the control is **rate-matched** — see ``rate_matched_control``.
3. It is defined per *neuron*, so it is blind to the thing a population model actually
   cares about: whether a unit is redundant with the ones you already have.

Implemented from the paper's definition (there is no official package). Validated on
synthetic trains where the answer is known; see the notebook.
"""

from __future__ import annotations

import numpy as np


def relevance_curve(
    spike_times: np.ndarray,
    *,
    t_start: float | None = None,
    t_stop: float | None = None,
    n_scales: int = 30,
    dt_min: float = 0.002,
    dt_max: float = 1.0,
):
    """Trace (resolution, relevance) as the bin width dt is swept across scales.

    Both entropies are normalised by ``log M`` so they live in [0, 1] and are comparable
    between neurons with different spike counts — without this, a high-rate unit would win
    on total entropy alone.

    Returns
    -------
    dts : (n_scales,) the bin widths used, log-spaced
    h_s : (n_scales,) normalised resolution
    h_k : (n_scales,) normalised relevance
    """
    st = np.asarray(spike_times, dtype=float)
    if t_start is None:
        t_start = st.min() if st.size else 0.0
    if t_stop is None:
        t_stop = st.max() if st.size else 1.0
    st = st[(st >= t_start) & (st < t_stop)]

    M = st.size
    dts = np.geomspace(dt_min, dt_max, n_scales)
    h_s = np.full(n_scales, np.nan)
    h_k = np.full(n_scales, np.nan)
    if M < 10:                      # too few spikes for any entropy to mean anything
        return dts, h_s, h_k

    logM = np.log(M)
    for i, dt in enumerate(dts):
        n_bins = max(int(np.ceil((t_stop - t_start) / dt)), 1)
        idx = np.minimum(((st - t_start) / dt).astype(np.int64), n_bins - 1)
        counts = np.bincount(idx, minlength=n_bins)
        counts = counts[counts > 0]                  # empty bins contribute 0 log 0 = 0

        p = counts / M
        h_s[i] = -np.sum(p * np.log(p)) / logM       # resolution

        # m_k = how many bins hold exactly k spikes; only k >= 1 exists in `counts`
        k_vals, m_k = np.unique(counts, return_counts=True)
        q = k_vals * m_k / M                         # spike-mass sitting at multiplicity k
        h_k[i] = -np.sum(q * np.log(q)) / logM       # relevance

    return dts, h_s, h_k


def msr(spike_times: np.ndarray, **kwargs) -> float:
    """Multiscale Relevance = area under the relevance-vs-resolution curve.

    Integrated with the trapezoid rule over H[s] (the curve is parametric in dt, and H[s] is
    monotonically decreasing in dt, so we sort by H[s] first). Returns NaN for a unit with
    too few spikes to score — do not silently treat that as zero.
    """
    _, h_s, h_k = relevance_curve(spike_times, **kwargs)
    ok = np.isfinite(h_s) & np.isfinite(h_k)
    if ok.sum() < 3:
        return float("nan")
    s, k = h_s[ok], h_k[ok]
    order = np.argsort(s)
    return float(np.trapezoid(k[order], s[order]))


def msr_table(
    spike_trains: dict[int, np.ndarray],
    *,
    t_start: float | None = None,
    t_stop: float | None = None,
    **kwargs,
):
    """MSR for every unit -> DataFrame ranked high to low, with the rate alongside.

    The rate column is not decoration: it is there so you can *see* the rate confound before
    you act on the ranking.
    """
    import pandas as pd

    rows = []
    for uid, st in spike_trains.items():
        st = np.asarray(st, dtype=float)
        lo = t_start if t_start is not None else (st.min() if st.size else 0.0)
        hi = t_stop if t_stop is not None else (st.max() if st.size else 1.0)
        n = int(((st >= lo) & (st < hi)).sum())
        rows.append({
            "unit": uid,
            "msr": msr(st, t_start=lo, t_stop=hi, **kwargs),
            "n_spikes": n,
            "rate_hz": round(n / max(hi - lo, 1e-9), 3),
        })
    return (
        pd.DataFrame(rows)
        .sort_values("msr", ascending=False, na_position="last")
        .reset_index(drop=True)
    )


def task_selectivity(counts: np.ndarray, bin_centers: np.ndarray, labels: np.ndarray,
                     resp_window=(0.05, 0.25)):
    """Per-unit Mann-Whitney U of in-window spike count between two trial groups.

    The cross-check on MSR: a unit can be dynamically rich (high MSR) and completely
    uninterested in the task. ``labels`` is a boolean per trial (e.g. stimulus on the right).

    Returns a DataFrame with the U statistic, the p-value (uncorrected — apply your own
    multiple-comparison correction) and the effect size as AUC (0.5 = no selectivity).
    """
    import pandas as pd
    from scipy.stats import mannwhitneyu

    w = (bin_centers >= resp_window[0]) & (bin_centers < resp_window[1])
    resp = counts[:, w, :].sum(axis=1)                 # (trials, units)
    labels = np.asarray(labels, dtype=bool)

    rows = []
    for u in range(resp.shape[1]):
        a, b = resp[labels, u], resp[~labels, u]
        if a.size < 5 or b.size < 5 or (a.sum() + b.sum()) == 0:
            rows.append({"unit": u, "p": np.nan, "auc": np.nan})
            continue
        U, p = mannwhitneyu(a, b, alternative="two-sided")
        rows.append({"unit": u, "p": float(p), "auc": float(U / (a.size * b.size))})
    return pd.DataFrame(rows)


def rate_matched_control(table, selected_units, *, seed: int = 0, tol: float = 0.25):
    """Pick a control set of the same size whose firing rates match ``selected_units``.

    **This is the control that makes an MSR result mean anything.** MSR rises with firing
    rate, so "the top-M MSR units give a cleaner latent than M units drawn at random" is
    almost guaranteed and proves nothing — the MSR units are simply the loud ones. The real
    question is whether MSR beats *equally loud* units. If it does not, MSR is an expensive
    firing-rate filter and we should say so and use the rate.

    Matches each selected unit to the nearest-rate unmatched unit within ``tol`` (relative),
    without replacement.
    """
    rng = np.random.default_rng(seed)
    sel = set(int(u) for u in selected_units)
    pool = table[~table["unit"].isin(sel)].copy()
    target_rates = table.set_index("unit").loc[list(sel), "rate_hz"].to_numpy()

    chosen = []
    for r in rng.permutation(target_rates):
        if pool.empty:
            break
        d = (pool["rate_hz"] - r).abs()
        rel = d / max(r, 1e-6)
        cand = pool[rel <= tol]
        if cand.empty:
            cand = pool.loc[[d.idxmin()]]             # no match in tolerance: take nearest
        pick = int(cand.sample(1, random_state=int(rng.integers(1 << 31)))["unit"].iloc[0])
        chosen.append(pick)
        pool = pool[pool["unit"] != pick]
    return np.array(chosen, dtype=int)
