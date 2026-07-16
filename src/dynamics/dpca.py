"""demixed PCA (Kobak et al. 2016, eLife 5:e10989) — variance split by *what causes it*.

Why this is the method the project actually needs
-------------------------------------------------
PCA answers "what are the dominant directions of population activity?" and cannot tell you
*why* they vary. Our question is not that. Our question is: **how much of a region's
population geometry is stimulus-driven, how much is decision-related, and how much is just
time?** dPCA answers exactly that, by decomposing the trial-averaged activity into
marginalisations before it reduces dimensionality:

* **t**   — condition-independent: everything that happens at a given moment in every trial
            (the big one, usually — onset transients, arousal, movement). Almost always
            dominates, and it is NOT evidence of anything decision-related.
* **s**   — stimulus (here: side of the grating).
* **d**   — decision (the animal's choice).
* **sd**  — stimulus x decision interaction.
* plus the time-interactions (**ts**, **td**, **tsd**) when the label set includes them.

The prediction that this project lives or dies by: **VISp should carry more stimulus
variance, SCm/MRN more decision variance.** dPCA is how you check that with a number rather
than by squinting at a trajectory.

The trap
--------
dPCA needs a **balanced** condition x condition design, and mice are ~80-90% correct — so
the (stimulus-left, chose-right) cells are *error trials* and are rare. This module
subsamples every cell down to the smallest one (``balanced_conditions``) rather than letting
dPCA average a 300-trial cell against a 30-trial one. That is a real cost: the number of
trials is set by the worst cell, and if it is tiny the decision components are noise. The
count is returned so it can be reported instead of buried.
"""

from __future__ import annotations

import numpy as np


def balanced_conditions(
    counts: np.ndarray,
    stim_right: np.ndarray,
    choice: np.ndarray,
    *,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Build dPCA's arrays for a 2 (stimulus) x 2 (decision) x time design.

    Returns
    -------
    X : (n_units, 2, 2, n_bins)         trial-averaged, mean-centred over conditions
    trialX : (K, n_units, 2, 2, n_bins) K single trials per cell (K = smallest cell)
    K : int                             trials per cell -- REPORT THIS, it caps everything
    """
    rng = np.random.default_rng(seed)
    stim_right = np.asarray(stim_right, dtype=bool)
    ch_right = np.asarray(choice) > 0

    cells = {(s, d): np.flatnonzero((stim_right == s) & (ch_right == d))
             for s in (False, True) for d in (False, True)}
    K = min(len(v) for v in cells.values())
    if K < 5:
        raise ValueError(
            f"smallest stimulus x decision cell has only {K} trials; the design is too "
            f"unbalanced for dPCA on this session. Cell sizes: "
            f"{ {k: len(v) for k, v in cells.items()} }"
        )

    n_trials, n_bins, n_units = counts.shape
    trialX = np.empty((K, n_units, 2, 2, n_bins))
    for (s, d), idx in cells.items():
        take = rng.choice(idx, K, replace=False)
        # (K, bins, units) -> (K, units, bins)
        trialX[:, :, int(s), int(d), :] = np.sqrt(counts[take].astype(float)).transpose(0, 2, 1)

    X = trialX.mean(axis=0)                                   # (units, 2, 2, bins)
    X -= X.reshape(n_units, -1).mean(axis=1)[:, None, None, None]   # dPCA wants it centred
    return X, trialX, int(K)


def fit_dpca(X: np.ndarray, trialX: np.ndarray, *, n_components: int = 4,
             labels: str = "sdt", protect: tuple[str, ...] = ("t",), regularizer=None):
    """Fit dPCA. Returns ``(model, Z)`` where ``Z[key]`` is (n_components, 2, 2, n_bins).

    ``protect=('t',)`` tells the regulariser not to shuffle the time axis when it builds its
    noise estimate — time is not an exchangeable condition.
    """
    from dPCA import dPCA as _dpca

    model = _dpca.dPCA(labels=labels, n_components=n_components, regularizer=regularizer)
    model.protect = list(protect)
    Z = model.fit_transform(X, trialX)
    return model, Z


def variance_by_marginalization(model) -> "object":
    """Total explained variance per marginalisation -> tidy DataFrame, largest first.

    This is the number to quote. ``t`` (condition-independent) will usually dwarf the rest;
    that is normal and is exactly why raw PCA cannot answer the project's question — PC1 of
    a plain PCA is mostly this component.
    """
    import pandas as pd

    evr = model.explained_variance_ratio_
    rows = [{"marginalization": k, "explained_var": round(float(np.sum(v)), 4)}
            for k, v in evr.items()]
    df = pd.DataFrame(rows).sort_values("explained_var", ascending=False).reset_index(drop=True)
    df["pct_of_explained"] = (100 * df["explained_var"] / df["explained_var"].sum()).round(1)
    return df


MARGINALIZATION_NAMES = {
    "t": "time (condition-independent)",
    "s": "stimulus",
    "d": "decision",
    "sd": "stimulus x decision",
    "st": "stimulus x time",
    "dt": "decision x time",
    "sdt": "stimulus x decision x time",
    "ts": "stimulus x time",
    "td": "decision x time",
    "tsd": "stimulus x decision x time",
}
