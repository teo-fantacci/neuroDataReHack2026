# Phase 3 — dPCA / GPFA / RRR / common manifold

Session `5c0c560e-9e1f-45e9-b66e-e4ee7855be84`, regions ['VISp', 'SCm', 'MRN'], 843 trials.
Chosen configuration: **50 ms bins, sigma 40 ms**, swept over [10, 20, 50] ms x [0, 20, 40, 80, 160] ms.

**The sweep cannot pick the smoothing, and we did not let it.** `time_R2` rises
monotonically with sigma and its argmax lands on the boundary -- a Gaussian kernel
imposes exactly the temporal autocorrelation that `time_R2` rewards, so maximising it
maximises the smoother. `choice_AUC` is flat across the entire sweep, which is the real
news: **the choice signal does not depend on bin size or smoothing at all.** The
configuration was therefore chosen externally (rSLDS ground-truth bin size + the GPFA
timescales below).

## Bin size x smoothing (mean over regions and bins, by smoothing)

|   sigma_ms |   time_R2 |   choice_AUC |   var_pc12 |
|-----------:|----------:|-------------:|-----------:|
|          0 |     0.319 |        0.858 |      0.157 |
|         20 |     0.389 |        0.859 |      0.225 |
|         40 |     0.422 |        0.86  |      0.269 |
|         80 |     0.451 |        0.86  |      0.301 |
|        160 |     0.479 |        0.854 |      0.319 |

## Embedding quality (there IS no variance-explained for UMAP/Laplacian/PHATE)

| region   | method    |   trustworthiness |   stress1 | native               |
|:---------|:----------|------------------:|----------:|:---------------------|
| VISp     | PCA       |             0.731 |     0.412 | PC1+2 = 23.5% var    |
| VISp     | MDS       |             0.748 |     0.228 | stress-1 = 0.228     |
| VISp     | t-SNE     |             0.882 |     0.471 | KL = 1.30            |
| VISp     | UMAP      |             0.877 |     0.504 | no native fit metric |
| VISp     | Isomap    |             0.745 |     0.421 | recon err = 16.823   |
| VISp     | Laplacian |             0.767 |     0.618 | no native fit metric |
| VISp     | PHATE     |             0.805 |     0.537 | no native fit metric |
| SCm      | PCA       |             0.841 |     0.428 | PC1+2 = 28.8% var    |
| SCm      | MDS       |             0.825 |     0.29  | stress-1 = 0.290     |
| SCm      | t-SNE     |             0.914 |     0.405 | KL = 1.40            |
| SCm      | UMAP      |             0.91  |     0.437 | no native fit metric |
| SCm      | Isomap    |             0.867 |     0.414 | recon err = 46.347   |
| SCm      | Laplacian |             0.87  |     0.535 | no native fit metric |
| SCm      | PHATE     |             0.875 |     0.478 | no native fit metric |
| MRN      | PCA       |             0.832 |     0.411 | PC1+2 = 30.5% var    |
| MRN      | MDS       |             0.837 |     0.286 | stress-1 = 0.286     |
| MRN      | t-SNE     |             0.933 |     0.384 | KL = 1.32            |
| MRN      | UMAP      |             0.932 |     0.388 | no native fit metric |
| MRN      | Isomap    |             0.874 |     0.39  | recon err = 53.216   |
| MRN      | Laplacian |             0.869 |     0.6   | no native fit metric |
| MRN      | PHATE     |             0.891 |     0.406 | no native fit metric |

## dPCA — variance by cause

| region   |   stimulus_% |   decision_% |   time_% |   decision_minus_stimulus |
|:---------|-------------:|-------------:|---------:|--------------------------:|
| VISp     |         11.8 |          5   |     74.1 |                      -6.8 |
| SCm      |          4.5 |          6.3 |     63.5 |                       1.8 |
| MRN      |          8.4 |         14.3 |     49.2 |                       5.9 |

Project prediction: `decision_minus_stimulus` lowest in VISp, highest in SCm/MRN.

## GPFA — fitted latent timescales (ms)

| region   |   tau1_ms |   tau2_ms |   tau3_ms |   tau4_ms |   tau5_ms |
|:---------|----------:|----------:|----------:|----------:|----------:|
| VISp     |     176.2 |     152.3 |     125.3 |     111   |     107   |
| SCm      |     226.2 |     174.4 |     152.7 |     145.2 |     107   |
| MRN      |     283.2 |     232.4 |     228.3 |     198.3 |     116.7 |

## RRR — communication subspace (vs within-area control)

| source      | target   | kind                |   best_r2 |   sat_rank |
|:------------|:---------|:--------------------|----------:|-----------:|
| VISp        | SCm      | cross-area          |     0.116 |          4 |
| VISp        | MRN      | cross-area          |     0.165 |          5 |
| SCm         | MRN      | cross-area          |     0.414 |          7 |
| MRN         | SCm      | cross-area          |     0.366 |          7 |
| VISp (half) | VISp     | within-area control |     0.163 |          4 |
| SCm (half)  | SCm      | within-area control |     0.383 |          7 |
| MRN (half)  | MRN      | within-area control |     0.47  |          8 |

## Joint manifold — each area's share of the shared latent

| region   |   var_share |   n_units |
|:---------|------------:|----------:|
| VISp     |       0.056 |        36 |
| SCm      |       0.222 |        59 |
| MRN      |       0.307 |        63 |

Joint PC1 = 16.7% var, PC2 = 8.0%, PC1-10 = 44.7%.

Figures: `results/figures/phase3_manifold/`.