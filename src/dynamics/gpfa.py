"""GPFA (Yu et al. 2009, J Neurophysiol 102:614) via `elephant.gpfa` — the tested implementation.

What GPFA adds that nothing in `src/geometry/` has: **a time prior**
--------------------------------------------------------------------
PCA/UMAP/PHATE treat each (trial, bin) population vector as an unordered point. They do not
know that bin 12 comes right after bin 11. Any smoothness you see in their trajectories was
put there by *us*, in the pre-smoothing step.

GPFA does not need that. It models the latent as a **Gaussian process across time** and the
spikes as noisy observations of it, so the smoothing is *inferred* — each latent dimension
gets its own fitted timescale (``tau``) — rather than imposed. Consequences:

* It works on **raw counts** (square-root transformed, per the paper), so no smoothing
  hyper-parameter to sweep and no risk of manufacturing the trajectory you are looking for.
* It gives genuine **single-trial** latents, which is what a metastable-sequence claim needs.
* The fitted ``tau`` per dimension is itself a result: it says what timescale the population
  dynamics actually live on. Compare it against the 20/50 ms bin debate.

Honest limits
-------------
* Latent dimensionality ``x_dim`` must be chosen; elephant offers no built-in CV. Fit a few
  and look at the log-likelihood, and remember K on a boundary is not a choice (Phase 2's
  lesson).
* The observation model is Gaussian on sqrt-counts, not Poisson. Fine at our rates, wrong for
  very sparse units — the same units that already break the Poisson HMM.
* GPFA is *linear* in the latent-to-neuron map. It finds smooth trajectories, not attractors.
  The switching structure is the rSLDS's job.
"""

from __future__ import annotations

import numpy as np


def trials_to_spiketrains(
    trains: dict[int, np.ndarray],
    event_times: np.ndarray,
    *,
    pre_s: float,
    post_s: float,
    units: np.ndarray | None = None,
) -> list[list]:
    """Cut continuous spike trains into per-trial `neo.SpikeTrain`s, aligned to the event.

    ``trains`` is ``build_matrices.region_spike_trains`` output (keyed by tensor column), so
    ``units`` indexes the same columns everything else in the project uses.

    Returns elephant's expected structure: ``[[SpikeTrain] * n_units] * n_trials``, with
    every trial on a common clock ``[0, pre_s + post_s)``.
    """
    import neo
    import quantities as pq

    keys = sorted(trains) if units is None else [int(u) for u in units]
    dur = pre_s + post_s
    out = []
    for e in np.asarray(event_times, dtype=float):
        lo, hi = e - pre_s, e + post_s
        trial = []
        for k in keys:
            st = trains[k]
            s = st[(st >= lo) & (st < hi)] - lo          # -> [0, dur)
            trial.append(neo.SpikeTrain(s * pq.s, t_start=0 * pq.s, t_stop=dur * pq.s))
        out.append(trial)
    return out


def fit_gpfa(spiketrains: list[list], *, bin_size_s: float = 0.02, x_dim: int = 5,
             em_max_iters: int = 50, verbose: bool = False):
    """Fit GPFA. Returns ``(model, latents)`` with ``latents`` = (n_trials, x_dim, n_bins).

    elephant returns a ragged object array of (x_dim, n_bins); we stack it, which is safe
    because every trial here shares one window length.
    """
    import quantities as pq
    from elephant.gpfa import GPFA

    model = GPFA(bin_size=bin_size_s * pq.s, x_dim=x_dim, em_max_iters=em_max_iters,
                 verbose=verbose)
    lat = model.fit_transform(spiketrains)
    return model, np.stack([np.asarray(t) for t in lat])


def latent_timescales(model, bin_size_s: float) -> np.ndarray:
    """Fitted GP timescale ``tau`` of each latent dimension, in **milliseconds**.

    elephant stores ``gamma`` with ``tau = bin_size / sqrt(gamma)``. This is a *result*, not
    a diagnostic: it is the timescale the data say the latent dynamics run at.
    """
    gamma = np.asarray(model.params_estimated["gamma"], dtype=float)
    return 1000.0 * bin_size_s / np.sqrt(gamma)
