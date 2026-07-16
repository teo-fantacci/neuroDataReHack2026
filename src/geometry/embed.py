"""Run the embedding zoo, and report *how good each embedding actually is*.

Why the quality metrics matter more than the pictures
----------------------------------------------------
A 2-D embedding always produces a picture. The picture is not the question; the question is
how much of the real geometry survived the projection into two dimensions. Each method
answers that in its own currency, and they are NOT interchangeable:

* **PCA** -- ``explained_variance_ratio_``. The honest one: "PC1+PC2 hold 18% of the
  variance" tells you 82% of the structure is off-screen. Reported in the axis labels.
* **MDS** -- **Kruskal stress-1**, computed here explicitly rather than read off
  ``mds.stress_`` (whose normalisation has changed across sklearn versions -- do not trust
  it silently). 0 = distances perfectly preserved; > 0.2 is usually called a poor fit.
* **t-SNE** -- ``kl_divergence_``: how badly the low-D neighbourhood distribution matches
  the high-D one. Lower is better, but it is NOT comparable across datasets or perplexities.
* **Isomap** -- ``reconstruction_error()``.
* **UMAP, Laplacian eigenmaps, PHATE** -- expose **no** native goodness-of-fit scalar. That
  is not an oversight on our part: they simply do not define one. Anyone quoting a
  "variance explained" for UMAP is making it up.

Because of that gap, every method also gets two *common* metrics computed the same way, so
they can be compared on one axis:

* **trustworthiness** (sklearn) -- fraction of each point's low-D neighbours that were
  genuinely its high-D neighbours. Penalises *invented* structure, which is exactly the
  failure mode of t-SNE/UMAP. 1.0 = nothing fabricated.
* **stress-1** -- global distance distortion, for every method. Neighbourhood methods do
  badly here by construction, and that is the point: it makes the local/global trade-off
  visible instead of hiding it.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np

METHODS = ["PCA", "MDS", "t-SNE", "UMAP", "Isomap", "Laplacian", "PHATE"]

# Methods that cannot place new points into an existing embedding. Anything here must be
# fitted on the very points it displays -- worth knowing before you "project held-out data".
NO_OUT_OF_SAMPLE = {"MDS", "t-SNE", "Laplacian", "PHATE"}


@dataclass
class Embedding:
    name: str
    coords: np.ndarray                      # (n_points, 2)
    seconds: float
    quality: dict = field(default_factory=dict)

    def axis_labels(self) -> tuple[str, str]:
        """PCA gets variance in the axis labels; the rest have no such quantity."""
        if self.name == "PCA" and "var_pc" in self.quality:
            v = self.quality["var_pc"]
            return (f"PC1 ({100 * v[0]:.1f}% var)", f"PC2 ({100 * v[1]:.1f}% var)")
        return (f"{self.name} 1", f"{self.name} 2")

    def caption(self) -> str:
        """One line for the panel: the native metric (if any) + the common ones."""
        q = self.quality
        native = {
            "PCA":   lambda: f"PC1+2 = {100 * q['var_pc'][:2].sum():.1f}% var",
            "MDS":   lambda: f"stress-1 = {q['stress1']:.3f}",
            "t-SNE": lambda: f"KL = {q['kl']:.2f}",
            "Isomap": lambda: f"recon err = {q['recon_err']:.3f}",
        }.get(self.name, lambda: "no native fit metric")()
        return f"{native}\ntrust={q['trustworthiness']:.2f}  stress-1={q['stress1']:.2f}"


def stress1(X: np.ndarray, Y: np.ndarray, max_points: int = 800, seed: int = 0) -> float:
    """Kruskal stress-1 between high-D X and low-D Y: sqrt(sum (dX-dY)^2 / sum dX^2).

    Y is optimally rescaled first (stress is not scale-invariant, and every method here
    returns coordinates in arbitrary units -- without the rescale we would be measuring the
    method's choice of units, not its distortion).
    """
    from scipy.spatial.distance import pdist

    rng = np.random.default_rng(seed)
    if X.shape[0] > max_points:                    # pdist is O(n^2); subsample
        i = rng.choice(X.shape[0], max_points, replace=False)
        X, Y = X[i], Y[i]
    dx, dy = pdist(X), pdist(Y)
    denom = float(dy @ dy)
    if denom == 0 or float(dx @ dx) == 0:
        return float("nan")
    dy = dy * (dx @ dy) / denom                    # least-squares scale for dy -> dx
    return float(np.sqrt(((dx - dy) ** 2).sum() / (dx ** 2).sum()))


def run_embeddings(
    X: np.ndarray,
    idx: np.ndarray,
    *,
    methods: list[str] | None = None,
    idx_mds: np.ndarray | None = None,
    seed: int = 0,
    trust_k: int = 15,
) -> dict[str, Embedding]:
    """Embed ``X[idx]`` with every method and score each one.

    ``X`` is the full point cloud (PCA is fitted on all of it -- it is cheap and linear);
    every other method is fitted on the subsample it displays, because it has no
    out-of-sample transform. ``idx_mds`` allows MDS a smaller subset (it is O(n^2)).
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import MDS, TSNE, Isomap, SpectralEmbedding, trustworthiness

    import phate
    import umap

    methods = methods or METHODS
    idx_mds = idx if idx_mds is None else idx_mds
    out: dict[str, Embedding] = {}

    for name in methods:
        pts = idx_mds if name == "MDS" else idx
        Xs = X[pts]
        q: dict = {}
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if name == "PCA":
                p = PCA(n_components=min(10, X.shape[1]), random_state=seed).fit(X)
                Y = p.transform(Xs)[:, :2]
                q["var_pc"] = p.explained_variance_ratio_
            elif name == "MDS":
                m = MDS(n_components=2, n_init=1, max_iter=120, random_state=seed, n_jobs=-1)
                Y = m.fit_transform(Xs)
            elif name == "t-SNE":
                m = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=30,
                         random_state=seed)
                Y = m.fit_transform(Xs)
                q["kl"] = float(m.kl_divergence_)
            elif name == "UMAP":
                Y = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                              random_state=seed).fit_transform(Xs)
            elif name == "Isomap":
                m = Isomap(n_components=2, n_neighbors=15).fit(Xs)
                Y = m.transform(Xs)
                q["recon_err"] = float(m.reconstruction_error())
            elif name == "Laplacian":
                Y = SpectralEmbedding(n_components=2, n_neighbors=15,
                                      random_state=seed).fit_transform(Xs)
            elif name == "PHATE":
                Y = phate.PHATE(n_components=2, knn=15, verbose=0, random_state=seed,
                                n_jobs=-1).fit_transform(Xs)
            else:
                raise ValueError(f"unknown method {name!r}")
            secs = time.time() - t0

            # common metrics: same computation for every method, so they are comparable
            n_t = min(600, Xs.shape[0])
            ti = np.random.default_rng(seed).choice(Xs.shape[0], n_t, replace=False)
            q["trustworthiness"] = float(trustworthiness(Xs[ti], Y[ti], n_neighbors=trust_k))
            q["stress1"] = stress1(Xs, Y, seed=seed)

        out[name] = Embedding(name=name, coords=np.asarray(Y), seconds=secs, quality=q)
    return out


def population_matrix(
    counts: np.ndarray,
    units: np.ndarray | None = None,
    *,
    bin_size_s: float,
    smooth_sigma_ms: float = 40.0,
    zscore: bool = True,
) -> np.ndarray:
    """(trials, time, neurons) -> (trials*time, |units|) point cloud, for geometry only.

    ``smooth_sigma_ms`` is specified in **milliseconds, not bins**, on purpose: the whole
    point of sweeping the bin size is to vary the bin size, and a sigma quoted in bins would
    silently change the amount of smoothing along with it, confounding the two.

    sqrt-transform first (variance-stabilises Poisson counts), then smooth, then z-score per
    unit so a single loud neuron cannot dominate the Euclidean distance every embedding is
    built on. **Never feed the output to a Poisson HMM** -- that needs raw counts.
    """
    from src.data.build_matrices import smooth_over_time

    x = counts if units is None else counts[:, :, units]
    x = np.sqrt(x.astype(float))
    sigma_bins = smooth_sigma_ms / 1000.0 / bin_size_s
    x = smooth_over_time(x, sigma_bins)
    X = x.reshape(-1, x.shape[2])
    if zscore:
        X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    return X
