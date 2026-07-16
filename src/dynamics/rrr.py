"""Reduced-rank regression = the **communication subspace** between two areas.

The question it answers, which no single-area method can
-------------------------------------------------------
The HMM says *when* a region changes state; PCA says what its geometry looks like. Neither
says **what one area actually sends to another**. RRR does: regress area B's population
activity on area A's, restricted to rank ``r``. The rank at which held-out prediction
saturates is the **dimensionality of the subspace through which A influences B** (Semedo
et al. 2019, Neuron 102:249). A small rank means the areas talk through a narrow channel —
B does not see all of A, only a low-dimensional projection of it.

For this project this is the geometric counterpart of the flow claim, and the natural tool
for the open SCm <-> MRN feedback question.

The control that makes it interpretable — and the mistake to avoid
------------------------------------------------------------------
"SCm predicts MRN at rank 3" means nothing on its own: **any** two populations correlate,
and the achievable R^2 depends on how predictable the target is at all. The reference is
Semedo's **within-area control**: split the *target* area's units in half and predict one
half from the other. That tells you how many dimensions the target's own activity needs.
Then:

* cross-area rank **<** within-area rank -> a genuinely *restricted* communication subspace:
  B does not receive all of A's dimensions. This is the interesting result.
* cross-area rank **~=** within-area rank -> the areas share whatever structure is there;
  no evidence of a selective channel.

Both curves are computed here, with the same CV, so they are comparable.

Cross-validation must be **grouped by trial** (bins of one trial are correlated); an
un-grouped split leaks and inflates R^2 -- the same trap as everywhere else in this project.
"""

from __future__ import annotations

import numpy as np


def rrr_fit(X: np.ndarray, Y: np.ndarray, rank: int, alpha: float = 1.0) -> np.ndarray:
    """Ridge solution truncated to ``rank`` -> coefficient matrix B (n_x, n_y).

    Closed form: fit ridge, then keep the top-``rank`` singular directions of the *fitted
    values* (not of B) -- that is what makes it the best rank-r predictor in the
    least-squares sense.
    """
    Xc = X - X.mean(0)
    Yc = Y - Y.mean(0)
    G = Xc.T @ Xc + alpha * np.eye(Xc.shape[1])
    B = np.linalg.solve(G, Xc.T @ Yc)                 # ridge, (n_x, n_y)

    Yhat = Xc @ B
    _, _, Vt = np.linalg.svd(Yhat, full_matrices=False)
    Vr = Vt[:rank].T                                  # (n_y, rank)
    return B @ Vr @ Vr.T


def _r2(Y: np.ndarray, Yhat: np.ndarray) -> float:
    """Variance explained, pooled over target neurons (not a per-neuron mean)."""
    ss_res = float(((Y - Yhat) ** 2).sum())
    ss_tot = float(((Y - Y.mean(0)) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def rrr_cv_curve(
    X: np.ndarray,
    Y: np.ndarray,
    groups: np.ndarray,
    *,
    ranks: list[int] | None = None,
    alpha: float = 1.0,
    n_splits: int = 5,
) -> "object":
    """Held-out R^2 as a function of rank. Returns a DataFrame (rank, mean_r2, sem_r2).

    ``groups`` = trial index per row. Folds are split on trials, never on bins.
    """
    import pandas as pd
    from sklearn.model_selection import GroupKFold

    max_rank = min(X.shape[1], Y.shape[1])
    ranks = ranks or list(range(1, min(max_rank, 10) + 1))

    scores = {r: [] for r in ranks}
    for tr, te in GroupKFold(n_splits=n_splits).split(X, Y, groups=groups):
        mx, my = X[tr].mean(0), Y[tr].mean(0)
        for r in ranks:
            B = rrr_fit(X[tr], Y[tr], rank=r, alpha=alpha)
            scores[r].append(_r2(Y[te], (X[te] - mx) @ B + my))

    rows = [{"rank": r, "mean_r2": float(np.mean(v)),
             "sem_r2": float(np.std(v) / np.sqrt(len(v)))} for r, v in scores.items()]
    return pd.DataFrame(rows)


def within_area_control(
    Y: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int = 0,
    **kwargs,
) -> "object":
    """Semedo's control: predict half of an area's units from its **other** half.

    The number of dimensions the target area needs to predict *itself* is the ceiling any
    cross-area subspace should be judged against.
    """
    rng = np.random.default_rng(seed)
    n = Y.shape[1]
    perm = rng.permutation(n)
    a, b = perm[: n // 2], perm[n // 2:]
    return rrr_cv_curve(Y[:, a], Y[:, b], groups, **kwargs)


def saturation_rank(curve, tol: float = 0.95) -> int:
    """Smallest rank reaching ``tol`` of the curve's own best held-out R^2.

    A blunt summary of "where does it plateau", so two curves can be compared with a number.
    If the best R^2 is <= 0 the regression failed and this returns 0 rather than an argmax
    over noise (the `-inf` lesson from Phase 2: never argmax without checking the winner).
    """
    best = float(curve["mean_r2"].max())
    if best <= 0:
        return 0
    ok = curve[curve["mean_r2"] >= tol * best]
    return int(ok["rank"].min())
