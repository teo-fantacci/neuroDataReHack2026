# 02 — dimensionality reduction + merged 04 results (40 ms pipeline)

Session `5c0c560e-9e1f-45e9-b66e-e4ee7855be84`, **40 ms unified pipeline** (`5c0c560e__stimOn_pre400_post800_bin40_rate1.npz`), 843 trials, window [-0.4, +0.8] s, geometry smoothing sigma 40 ms.
Units (>= 1 Hz): VISp 19, SCm 44, MRN 54; MSR-top half: VISp 9, SCm 22, MRN 27.

## Visualisation (interactive 3-D Plotly, coloured by time — *.html, drag to rotate)
- `3d_post_<region>.html` / `3d_prestim_<region>.html` — PCA/MDS/Laplacian, post-stim +
  full window (diverging: blue=baseline, red=evoked). all units vs MSR-top half.
- `3d_dpca_demixed.html` — dPCA time/stimulus/decision axes, colour=time, shape=condition.
- `gpfa_latents_3d.html`, `joint_manifold_3d.html` — the GPFA + joint-manifold scenes.
  (2-D: `dpca_variance_split.png`, `rrr_communication.png` stay static matplotlib.)

## dPCA variance by cause (the core prediction)

| region   |   stimulus_% |   decision_% |   time_% |   decision_minus_stimulus |
|:---------|-------------:|-------------:|---------:|--------------------------:|
| VISp     |         10.9 |          4.1 |     76.8 |                      -6.8 |
| SCm      |          4.1 |          5.9 |     64   |                       1.8 |
| MRN      |          7.5 |         13.9 |     48.6 |                       6.4 |

## GPFA fitted timescales (ms)

| region   |   tau1_ms |   tau2_ms |   tau3_ms |   tau4_ms |   tau5_ms |
|:---------|----------:|----------:|----------:|----------:|----------:|
| VISp     |       167 |       144 |       121 |       110 |        99 |
| SCm      |       211 |       156 |       139 |       122 |       113 |
| MRN      |       260 |       220 |       215 |       161 |       110 |

## RRR communication subspace (vs within-area control)

| source      | target   | kind                |   best_r2 |   sat_rank |
|:------------|:---------|:--------------------|----------:|-----------:|
| VISp        | SCm      | cross-area          |     0.137 |          4 |
| VISp        | MRN      | cross-area          |     0.175 |          5 |
| SCm         | MRN      | cross-area          |     0.449 |          7 |
| MRN         | SCm      | cross-area          |     0.446 |          7 |
| VISp (half) | VISp     | within-area control |     0.2   |          3 |
| SCm (half)  | SCm      | within-area control |     0.473 |          6 |
| MRN (half)  | MRN      | within-area control |     0.435 |          7 |

## Joint manifold — each area's share of the shared latent

| region   |   var_share |   n_units |
|:---------|------------:|----------:|
| VISp     |       0.064 |        19 |
| SCm      |       0.225 |        44 |
| MRN      |       0.306 |        54 |

var shares sum to 0.595; the remaining ~0.41 is cross-area covariance.

MSR is kept only as a comparison set, NOT adopted (worklog 2026-07-13). Maths of the dPCA
result and its link to the attractor sequence: `docs/knowledge/dPCA_maths_and_the_attractor_sequence.md`.