"""One manifold, three areas: a common space in which VISp, SCm and MRN can be compared.

The problem, stated honestly
---------------------------
You cannot simply "embed three areas together". A point in VISp's state space is a vector
over VISp's 36 neurons; a point in MRN's is a vector over MRN's 63. They are different
spaces. Running t-SNE on the two clouds stacked would compare coordinates that mean nothing
to each other, and the areas would separate **trivially** -- not because their dynamics
differ, but because their axes do. That plot would be pure artefact.

What makes a common space legitimate here
-----------------------------------------
The three areas are recorded **simultaneously**, so they share the one thing we need: the
same (trial, time-bin) samples. That gives two principled constructions, and this module
implements the simpler and stricter one.

**Joint PCA + per-area partial projection.** Concatenate the units of all areas into one
population (VISp | SCm | MRN) and take its PCA. That basis is the *shared* latent space --
a single set of axes, defined by the whole triplet at once. Because PCA is **linear**, the
joint trajectory decomposes **exactly** into a sum of per-area contributions:

    Y(t) = (X(t) - mu) V^T = sum_over_areas  (X_a(t) - mu_a) V_a^T = sum_over_areas Y_a(t)

where ``V_a`` are the loadings restricted to area ``a``'s units. So ``Y_a`` is *area a's
contribution to the shared state*, expressed in coordinates every area shares. Plotting the
three ``Y_a`` in one figure is meaningful: same axes, same units, and they literally add up
to the joint state. ``partial_projections`` returns them, and the identity is asserted.

What this buys, scientifically: an area that drives the shared trajectory has a large,
time-structured ``Y_a``; an area that merely follows has a small or temporally flat one.
That is a *geometric* version of the project's flow question -- and it is the honest way to
ask "do the areas live on the same manifold?", because the manifold is defined once, jointly,
rather than separately per area and then wished into alignment.

Caveats, which are real
-----------------------
1. A partial projection is only exact for a **linear** embedding. For a nonlinear common
   manifold, use :func:`joint_latent` to get the shared linear basis first and run the
   nonlinear method on the stacked per-area contributions -- the nonlinear step then acts on
   coordinates that are already comparable across areas.
2. Areas with more units, or louder ones, contribute more variance. Units are z-scored, so
   loudness is handled; unit *count* is not. Compare areas at matched unit count (subsample
   the bigger ones) before making a claim about who drives what.
"""

from __future__ import annotations

import numpy as np


def joint_matrix(
    mats: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, slice]]:
    """Concatenate per-area point clouds along the neuron axis. All must share the rows.

    ``mats[region]`` is (n_points, n_units_region) from ``embed.population_matrix`` -- so the
    rows are the same (trial, bin) samples in the same order for every area. That shared row
    index is what licenses everything else in this module; it holds only because the areas
    were recorded simultaneously.
    """
    regions = list(mats)
    n = {r: m.shape[0] for r, m in mats.items()}
    if len(set(n.values())) != 1:
        raise ValueError(f"areas do not share the same samples: {n}")

    slices, cols = {}, 0
    for r in regions:
        w = mats[r].shape[1]
        slices[r] = slice(cols, cols + w)
        cols += w
    return np.concatenate([mats[r] for r in regions], axis=1), slices


def joint_latent(X: np.ndarray, n_components: int = 10, seed: int = 0):
    """PCA of the concatenated triplet population = the shared basis. Returns (pca, Y)."""
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components, random_state=seed).fit(X)
    return pca, pca.transform(X)


def partial_projections(
    X: np.ndarray,
    pca,
    slices: dict[str, slice],
    *,
    check: bool = True,
) -> dict[str, np.ndarray]:
    """Each area's **contribution** to the shared latent, in the shared coordinates.

    Returns ``{region: (n_points, n_components)}``. By linearity these sum exactly to
    ``pca.transform(X)``; with ``check=True`` that identity is asserted rather than assumed,
    because it is the entire justification for plotting the areas in one figure.
    """
    V = pca.components_                        # (n_components, n_features)
    mu = pca.mean_
    out = {r: (X[:, sl] - mu[sl]) @ V[:, sl].T for r, sl in slices.items()}

    if check:
        total = sum(out.values())
        ref = pca.transform(X)
        if not np.allclose(total, ref, atol=1e-6):
            raise AssertionError(
                "per-area contributions do not sum to the joint latent; the partial "
                "projection is not valid and the joint plot would be meaningless."
            )
    return out


def area_drive(parts: dict[str, np.ndarray]) -> "object":
    """How much of the shared trajectory does each area actually carry?

    Two numbers per area, both in the shared basis so they are directly comparable:

    * ``var_share``  -- fraction of the total variance of the joint latent contributed by
      this area (the areas' contributions are not orthogonal, so these need not sum to 1;
      the residual is cross-area covariance, which is itself informative).
    * ``time_locking`` -- variance of the *trial-averaged* contribution over the variance of
      the raw contribution. High = this area's contribution is reproducible across trials
      (time-locked to the task); low = it is there, but it is not stereotyped.
    """
    import pandas as pd

    tot = sum(parts.values())
    tot_var = float(np.var(tot, axis=0).sum())
    rows = []
    for r, Y in parts.items():
        rows.append({
            "region": r,
            "var_share": round(float(np.var(Y, axis=0).sum()) / max(tot_var, 1e-12), 3),
        })
    return pd.DataFrame(rows)
