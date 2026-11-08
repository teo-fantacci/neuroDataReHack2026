"""Per-session replication of `docs/knowledge/recap_data_HMM_dimred.md`.

    python scripts/build_session_recap.py --eid 5c0c560e
    python scripts/build_session_recap.py --eid 111c1762 --no-gpfa   # skip the slow GPFA fit

Regenerates, for ONE session, the figures + numbers behind the recap doc, so a single-session
version of that write-up can quote *this session's* results rather than the 5-session pool. The
user asked for the two best sessions replicated WITHOUT the multi-session pooling step -- so this
runs everything per session and pools nothing.

What comes from where (honesty about provenance):

  From the frozen 40 ms tensor (`data/processed/<eid>__stimOn_pre300_..._bin40_rate1.npz`):
      - population PSTH per area                     -> psth_per_area.png
      - single-session macro raster (peak-sorted)    -> macro_raster.png
      - stimulus-vs-choice decoding by contrast       -> decoding_vs_contrast.png
      - dPCA variance split (stimulus/decision/time)  -> dpca_variance_split.png
      - RRR communication subspace vs within control  -> rrr_communication.png
      - joint-manifold per-area variance share        (number)

  From the already-computed HMM CSVs (results/hmm_5sessions/, results/hmm/state_phase.csv) so the
  figures match the recap's numbers exactly -- NOT re-fit here:
      - HMM state-stability (ARI vs K)                -> hmm_stability.png
      - HMM forward_fraction (directedness)           -> hmm_forward_fraction.png
      - state x task-phase NMI vs shuffle null        (number)

  From the raw session spikes (cached in data/ONE/; loaded once):
      - SCm<->MRN cross-correlogram                   -> crosscorr_SCm_MRN.png
      - RT-sorted single-neuron raster                -> raster_example_neuron.png
      - GPFA fitted latent timescales                 (number)

Everything lands under a per-session folder so nothing collides with the project figures:
    results/figures/recap_<eid8>/*.png
    results/session_recap/<eid8>.json      (every number the markdown quotes)
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data.build_matrices import counts_to_rates, smooth_over_time  # noqa: E402
from src.data.preprocess import load_processed, sigma_ms_to_bins  # noqa: E402
from src.viz.style import (  # noqa: E402
    OKABE_ITO, SEQUENTIAL, region_color, savefig, set_style, stroke,
)

PROC = ROOT / "data" / "processed"
FIGROOT = ROOT / "results" / "figures"
OUT = ROOT / "results" / "session_recap"

REGIONS = ["VISp", "SCm", "MRN"]
K_CANONICAL = {"VISp": 3, "SCm": 4, "MRN": 3}     # readouts.csv canonical K (SCm supports 4)
K_COMMON = 3                                       # the honest common count (recap Sec 2a)
SMOOTH_MS_GEOM = 40.0                              # standing geometry smoothing (notebook 02)
SMOOTH_MS_RASTER = 60.0                            # display smoothing for the macro raster only
SEED = 0

# short human labels for the two best sessions (subject / lab), for the figure titles
SESSION_LABEL = {
    "5c0c560e": "NR_0028 · steinmetzlab",
    "111c1762": "UCLA033 · churchlandlab",
    "90c61c38": "CSH_ZAD_025 · zadorlab",
    "90d1e82c": "DY_010 · danlab",
    "8a3a0197": "UCLA034 · churchlandlab",
}
FULL_EID = {
    "5c0c560e": "5c0c560e-9e1f-45e9-b66e-e4ee7855be84",
    "111c1762": "111c1762-7908-47e0-9f40-2f2ee55b6505",
    "90c61c38": "90c61c38-b9fd-4cc3-9795-29160d2f8e55",
    "90d1e82c": "90d1e82c-c96f-496c-ad4e-ee3f02067f25",
    "8a3a0197": "8a3a0197-b40a-449f-be55-c00b23253bbf",
}


# --------------------------------------------------------------------------- tensor figures


def fig_psth_per_area(data, eid8, phase):
    """Population PSTH per area, +/-1 SEM across trials (recap Sec 1). Replicates 00 cell 22."""
    bc = data["bin_centers"]
    bin_s = data["config"]["bin_size_s"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), sharex=True)
    out = {}
    for ax, r in zip(axes, REGIONS):
        counts = data["regions"][r]["counts"]
        n_tr, _, n_u = counts.shape
        pop_rate = counts.mean(axis=2) / bin_s
        psth = pop_rate.mean(axis=0)
        sem = pop_rate.std(axis=0, ddof=1) / np.sqrt(n_tr)
        col = region_color(r)
        ax.fill_between(bc, psth - sem, psth + sem, color=col, alpha=0.30, linewidth=0)
        ax.plot(bc, psth, color=col, lw=1.8, path_effects=stroke())
        ax.axvline(0, color="k", ls="--", lw=1)
        ax.set_title(r, fontweight="bold")
        ax.set_xlabel("time from stimOn (s)")
        ax.text(0.03, 0.97, f"±1 SEM over\n{n_tr} trials\n({n_u} units)",
                transform=ax.transAxes, va="top", ha="left", fontsize=8,
                bbox=dict(boxstyle="round", fc="white", ec=col, alpha=0.9))
        out[r] = {"peak_rate_hz": round(float(psth.max()), 2),
                  "baseline_rate_hz": round(float(psth[bc < 0].mean()), 2)}
    axes[0].set_ylabel("population rate (sp/s)")
    fig.suptitle(f"Population PSTH per area · session {eid8} ({SESSION_LABEL.get(eid8,'')})",
                 y=1.04, fontweight="bold")
    fig.tight_layout()
    savefig(fig, phase, "psth_per_area", root=ROOT)
    plt.close(fig)
    return out


def fig_macro_raster(data, eid8, phase):
    """Single-session macro raster: 3 area panels, units peak-sorted, cividis (recap Sec 1)."""
    bc = data["bin_centers"]
    bin_s = data["config"]["bin_size_s"]
    sigma_bins = sigma_ms_to_bins(SMOOTH_MS_RASTER, bin_s)
    blocks = {}
    for r in REGIONS:
        counts = data["regions"][r]["counts"]
        psth = smooth_over_time(counts_to_rates(counts, bin_s), sigma_bins).mean(axis=0).T
        peak = psth.max(axis=1, keepdims=True)
        norm = psth / np.where(peak > 0, peak, 1.0)
        order = np.argsort(np.argmax(norm, axis=1))
        blocks[r] = norm[order]
    heights = [blocks[r].shape[0] for r in REGIONS]
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.8), sharex=True, height_ratios=heights)
    for ax, r in zip(axes, REGIONS):
        M = blocks[r]
        im = ax.imshow(M, aspect="auto", cmap=SEQUENTIAL, vmin=0, vmax=1,
                       extent=[bc[0], bc[-1], M.shape[0], 0])
        ax.axvline(0, color="white", lw=1.2)
        ax.set_ylabel(f"{r}\n({M.shape[0]} units)", fontsize=11, fontweight="bold")
        ax.set_yticks([])
    axes[-1].set_xlabel("time from stimOn (s)")
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="peak-normalised rate")
    fig.suptitle(f"Macro raster — trial-averaged sequence · session {eid8}", y=0.995,
                 fontweight="bold")
    fig.subplots_adjust(left=0.13)
    savefig(fig, phase, "macro_raster", root=ROOT)
    plt.close(fig)


def _covariates(data):
    """stimulus side / signed contrast / choice from the frozen trial covariates (00 cell 21)."""
    tr = data["trials"]
    cL = np.asarray(tr["contrastLeft"], float)
    cR = np.asarray(tr["contrastRight"], float)
    choice = np.asarray(tr["choice"], float)
    signed = np.nan_to_num(cR, nan=0.0) - np.nan_to_num(cL, nan=0.0)
    return {"choice": choice, "signed": signed, "side": np.sign(signed),
            "abs": np.abs(signed), "rt": np.asarray(tr["rt"], float),
            "stim_right": ~np.isnan(cR)}


def fig_decoding(data, cov, eid8, phase):
    """Stimulus-side vs choice decoding by contrast (recap Sec 1). Replicates 00 cell 30."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    bc = data["bin_centers"]
    choice, side, abs_c, signed = cov["choice"], cov["side"], cov["abs"], cov["signed"]
    chose_right = (choice == -1).astype(int)
    go = choice != 0

    def window_features(counts, t0, t1):
        sel = (bc >= t0) & (bc < t1)
        return counts[:, sel, :].mean(axis=1)

    def decode(X, y, n_splits=5):
        y = np.asarray(y)
        classes, n = np.unique(y, return_counts=True)
        if len(classes) < 2 or n.min() < n_splits:
            return np.nan, np.nan
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced"))
        cv = StratifiedKFold(n_splits, shuffle=True, random_state=SEED)
        s = cross_val_score(clf, X, y, cv=cv, scoring="balanced_accuracy")
        return float(s.mean()), float(s.std())

    STIM_WIN, CHOICE_WIN = (0.05, 0.20), (0.0, 0.40)
    levels = np.unique(abs_c)
    lev_pct = levels * 100
    summary = {}

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4))
    axp = axes[0, 0]
    sc = np.unique(signed)
    axp.plot(sc * 100, [np.mean(chose_right[signed == s]) for s in sc], "-o",
             color=OKABE_ITO[4], lw=1.8, path_effects=stroke())
    axp.axhline(0.5, color="k", ls=":", lw=1); axp.axvline(0, color="k", ls=":", lw=1)
    axp.set(xlabel="signed contrast (%)  [+ = right]", ylabel="P(choose right)",
            title="Behaviour: psychometric", ylim=(-0.03, 1.03))

    for ax, r in zip([axes[0, 1], axes[1, 0], axes[1, 1]], REGIONS):
        counts = data["regions"][r]["counts"]
        Xs, Xc = window_features(counts, *STIM_WIN), window_features(counts, *CHOICE_WIN)
        sm, se, cm, ce = [], [], [], []
        for L in levels:
            m = abs_c == L
            a, e = decode(Xs[m & (side != 0)], side[m & (side != 0)]) if L > 0 else (np.nan, np.nan)
            a2, e2 = decode(Xc[m & go], chose_right[m & go])
            sm.append(a); se.append(e); cm.append(a2); ce.append(e2)
        ax.axhline(0.5, color="k", ls=":", lw=1)
        ax.errorbar(lev_pct, sm, yerr=se, fmt="-s", color=OKABE_ITO[4], lw=1.6, capsize=3,
                    path_effects=stroke(), label="stimulus side")
        ax.errorbar(lev_pct, cm, yerr=ce, fmt="-o", color=OKABE_ITO[0], lw=1.6, capsize=3,
                    path_effects=stroke(), label="choice")
        ax.set_xscale("symlog", linthresh=5); ax.set_xticks(lev_pct)
        ax.set_xticklabels([f"{v:g}" for v in lev_pct], fontsize=8)
        ax.set(title=f"{r}  ({counts.shape[2]} units)", xlabel="|contrast| (%)",
               ylabel="balanced accuracy", ylim=(0.30, 1.0))
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        summary[r] = {
            "choice_acc_0pct": None if np.isnan(cm[0]) else round(cm[0], 3),
            "choice_acc_mean": round(float(np.nanmean(cm)), 3),
            "stim_acc_100pct": None if np.isnan(sm[-1]) else round(sm[-1], 3),
        }
    fig.suptitle(f"Decoding stimulus vs choice at different contrast · session {eid8}",
                 y=1.0, fontweight="bold")
    fig.tight_layout()
    savefig(fig, phase, "decoding_vs_contrast", root=ROOT)
    plt.close(fig)
    return summary


# --------------------------------------------------------------------------- HMM figures (from CSVs)


def fig_hmm_from_csv(eid8, phase):
    """Build the stability + forward_fraction figures + numbers from the pre-computed CSVs.

    Reading the frozen CSVs (not re-fitting) keeps every number identical to the recap's, and
    the recap explicitly warns that a committed HMM figure is only evidence if its cell ran in
    the same pass -- these numbers already passed that bar in run_hmm_5sessions.py.
    """
    stab = pd.read_csv(ROOT / "results" / "hmm_5sessions" / "stability_curves.csv")
    read = pd.read_csv(ROOT / "results" / "hmm_5sessions" / "readouts.csv")
    sp = pd.read_csv(ROOT / "results" / "hmm" / "state_phase.csv")
    stab = stab[(stab["align"] == "stimOn") & (stab["eid"] == eid8)]
    read = read[(read["align"] == "stimOn") & (read["eid"] == eid8)]
    sp = sp[sp["eid"] == eid8]

    # --- stability: ARI vs K, one panel per region, canonical K marked ---
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), sharey=True)
    for ax, r in zip(axes, REGIONS):
        s = stab[stab["region"] == r].sort_values("K")
        ax.plot(s["K"], s["ari_mean"], "-o", color=region_color(r), lw=2.0,
                path_effects=stroke(), zorder=3)
        ax.fill_between(s["K"], s["ari_mean"] - s["ari_std"], s["ari_mean"] + s["ari_std"],
                        color=region_color(r), alpha=0.20, lw=0)
        ax.axvline(K_CANONICAL[r], color="0.5", ls=":", lw=1.2)
        ax.set(title=f"{r}  (canonical K={K_CANONICAL[r]})", xlabel="number of states K")
    axes[0].set_ylabel("state stability  (ARI over disjoint fits)")
    fig.suptitle(f"HMM state-stability scan · session {eid8}  (band = ±1 SD over repeats)",
                 y=1.02, fontweight="bold")
    fig.tight_layout()
    savefig(fig, phase, "hmm_stability", root=ROOT)
    plt.close(fig)

    # --- forward_fraction: one bar per region with split-half SD ---
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    ffs = {}
    for i, r in enumerate(REGIONS):
        row = read[read["region"] == r].iloc[0]
        ff, sd = float(row["ff_split_half"]), float(row["ff_split_sd"])
        ffs[r] = (ff, sd)
        ax.bar(i, ff, yerr=sd, width=0.6, color=region_color(r), edgecolor="black",
               linewidth=0.8, capsize=5, error_kw=dict(lw=1.4))
    ax.axhline(0.5, color="0.4", ls=":", lw=1.2)
    ax.text(2.2, 0.505, "chance\n(undirected)", fontsize=8, color="0.4", va="bottom", ha="right")
    ax.set(xticks=range(3), xticklabels=REGIONS, ylabel="forward_fraction (split-half)",
           ylim=(0.45, max(0.85, max(v[0] for v in ffs.values()) + 0.1)),
           title=f"Directedness of the state sequence · session {eid8}")
    fig.tight_layout()
    savefig(fig, phase, "hmm_forward_fraction", root=ROOT)
    plt.close(fig)

    # --- numbers ---
    hmm = {}
    for r in REGIONS:
        rd = read[read["region"] == r].iloc[0]
        spr = sp[sp["region"] == r]
        spr = spr.iloc[0] if len(spr) else None
        hmm[r] = {
            "n_trials": int(rd["n_trials"]), "n_units": int(rd["n_units"]),
            "K_canonical": int(rd["K"]), "best_k_scan": int(rd["best_k_scan"]),
            "ari_at_K": round(float(rd["ari_at_K"]), 3),
            "forward_fraction_split_half": round(float(rd["ff_split_half"]), 3),
            "ff_split_sd": round(float(rd["ff_split_sd"]), 3),
            "switch_rate": round(float(rd["switch_rate"]), 2),
            "baseline_exit_ms": None if pd.isna(rd["baseline_exit_ms"]) else int(rd["baseline_exit_ms"]),
            "state_phase_nmi": None if spr is None else round(float(spr["nmi"]), 4),
            "state_phase_nmi_null": None if spr is None else round(float(spr["nmi_null_mean"]), 4),
            "state_phase_z": None if spr is None else round(float(spr["nmi_z"]), 1),
            "state_phase_beats_null": None if spr is None else bool(spr["beats_null"]),
        }
    return hmm


# --------------------------------------------------------------------------- dim-red (computed)


def compute_dpca(data, cov, eid8, phase):
    """dPCA variance split per region + the decision-minus-stimulus gradient (recap Sec 3)."""
    from scipy.ndimage import gaussian_filter1d
    from src.dynamics.dpca import balanced_conditions, fit_dpca, variance_by_marginalization

    out, panels = {}, {}
    for r in REGIONS:
        try:
            X, trialX, K = balanced_conditions(data["regions"][r]["counts"], cov["stim_right"],
                                               cov["choice"], seed=SEED)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model, _ = fit_dpca(X, trialX, n_components=3)
            var = variance_by_marginalization(model).set_index("marginalization")["pct_of_explained"]
            s, d, t = float(var.get("s", 0)), float(var.get("d", 0)), float(var.get("t", 0))
            out[r] = {"stimulus_pct": round(s, 1), "decision_pct": round(d, 1),
                      "time_pct": round(t, 1), "decision_minus_stimulus": round(d - s, 1),
                      "K_per_cell": int(K)}
            panels[r] = (s, d, t, int(K))
        except Exception as e:  # noqa: BLE001
            out[r] = {"error": f"{type(e).__name__}: {e}"}
            panels[r] = None

    # figure: grouped bars (stim / decision) per region + the gradient line
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(REGIONS)); w = 0.38
    s_vals = [panels[r][0] if panels[r] else np.nan for r in REGIONS]
    d_vals = [panels[r][1] if panels[r] else np.nan for r in REGIONS]
    ax1.bar(x - w / 2, s_vals, w, color=OKABE_ITO[4], edgecolor="black", lw=0.6, label="stimulus")
    ax1.bar(x + w / 2, d_vals, w, color=OKABE_ITO[0], edgecolor="black", lw=0.6, label="decision")
    ax1.set(xticks=x, xticklabels=REGIONS, ylabel="% of demixed variance",
            title="dPCA variance by cause")
    ax1.legend(frameon=False)
    grad = [panels[r][1] - panels[r][0] if panels[r] else np.nan for r in REGIONS]
    ax2.plot(x, grad, "-o", color="black", lw=2.0, path_effects=stroke())
    for xi, g in zip(x, grad):
        if not np.isnan(g):
            ax2.annotate(f"{g:+.1f}", (xi, g), textcoords="offset points", xytext=(0, 8),
                         ha="center", fontsize=10, fontweight="bold")
    ax2.axhline(0, color="0.5", ls=":", lw=1.2)
    ax2.set(xticks=x, xticklabels=REGIONS, ylabel="decision − stimulus  (variance %)",
            title="visual→motor gradient")
    fig.suptitle(f"dPCA — what drives each area's geometry · session {eid8}", y=1.02,
                 fontweight="bold")
    fig.tight_layout()
    savefig(fig, phase, "dpca_variance_split", root=ROOT)
    plt.close(fig)
    return out


def compute_rrr_joint(data, eid8, phase):
    """RRR communication subspace vs within-area control + joint-manifold shares (recap Sec 3)."""
    from src.dynamics.rrr import rrr_cv_curve, saturation_rank, within_area_control
    from src.geometry.embed import population_matrix
    from src.geometry.joint import area_drive, joint_latent, joint_matrix, partial_projections

    gc.collect()
    bin_s = data["config"]["bin_size_s"]
    mats = {r: population_matrix(data["regions"][r]["counts"], None, bin_size_s=bin_s,
                                 smooth_sigma_ms=SMOOTH_MS_GEOM) for r in REGIONS}
    n_trials = data["n_trials"]
    n_bins = data["regions"][REGIONS[0]]["counts"].shape[1]
    groups = np.repeat(np.arange(n_trials), n_bins)
    PAIRS = [("VISp", "SCm"), ("VISp", "MRN"), ("SCm", "MRN"), ("MRN", "SCm")]
    RANKS = list(range(1, 11))

    curves, rrr = {}, {}
    for src, tgt in PAIRS:
        c = rrr_cv_curve(mats[src], mats[tgt], groups, ranks=RANKS)
        curves[(src, tgt)] = c
        rrr[f"{src}->{tgt}"] = {"kind": "cross", "best_r2": round(float(c["mean_r2"].max()), 3),
                                "sat_rank": int(saturation_rank(c))}
    for tgt in REGIONS:
        c = within_area_control(mats[tgt], groups, ranks=RANKS, seed=SEED)
        curves[(tgt, tgt)] = c
        rrr[f"{tgt}(half)->{tgt}"] = {"kind": "within", "best_r2": round(float(c["mean_r2"].max()), 3),
                                      "sat_rank": int(saturation_rank(c))}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, tgt in zip(axes, REGIONS):
        for src, _ in [pr for pr in PAIRS if pr[1] == tgt]:
            c = curves[(src, tgt)]
            ax.errorbar(c["rank"], c["mean_r2"], yerr=c["sem_r2"], marker="o", ms=4, lw=1.6,
                        color=region_color(src), label=f"{src}→{tgt}")
        c = curves[(tgt, tgt)]
        ax.errorbar(c["rank"], c["mean_r2"], yerr=c["sem_r2"], marker="s", ms=4, lw=1.8,
                    color="black", ls="--", label=f"{tgt} half→half")
        ax.set(xlabel="rank", title=f"target: {tgt}")
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("held-out R² (grouped by trial)")
    fig.suptitle(f"Communication subspace — cross-area vs BLACK within-area control · session {eid8}",
                 y=1.02, fontweight="bold")
    fig.tight_layout()
    savefig(fig, phase, "rrr_communication", root=ROOT)
    plt.close(fig)

    Xj, slices = joint_matrix(mats)
    pca_j, Yj = joint_latent(Xj, n_components=10, seed=SEED)
    parts = partial_projections(Xj, pca_j, slices)
    drive = area_drive(parts)
    joint = {r: round(float(drive.set_index("region").loc[r, "var_share"]), 3) for r in REGIONS}
    joint["sum"] = round(sum(joint[r] for r in REGIONS), 3)
    joint["cross_area_covariance"] = round(1 - joint["sum"], 3)
    joint["n_units"] = {r: int(mats[r].shape[1]) for r in REGIONS}
    return {"rrr": rrr, "joint": joint}


# --------------------------------------------------------------------------- ONE-dependent extras


def compute_one_extras(eid8, cov, data, phase, do_gpfa=True):
    """Cross-correlogram, RT-sorted neuron raster, and GPFA timescales — need the raw spikes."""
    import pynapple as nap

    from src.data.access import get_one
    from src.data.build_matrices import region_spike_trains
    from src.data.preprocess import PreprocessConfig, align_times, load_session
    from src.viz.nap_plots import region_crosscorrelogram

    cfg_d = data["config"]
    cfg = PreprocessConfig(align_event=cfg_d["align_event"], pre_s=cfg_d["pre_s"],
                           post_s=cfg_d["post_s"], bin_size_s=cfg_d["bin_size_s"],
                           min_rate_hz=cfg_d["min_rate_hz"], min_rt_s=cfg_d["min_rt_s"],
                           max_rt_s=cfg_d["max_rt_s"], regions=tuple(cfg_d["regions"]),
                           qc=cfg_d["qc"])
    one = get_one(cache_dir=ROOT / "data" / "ONE")
    trials, mask, probe_loads = load_session(one, FULL_EID[eid8], cfg=cfg)
    events = align_times(trials, mask, cfg.align_event)
    assert len(events) == data["n_trials"], f"{len(events)} events vs {data['n_trials']} trials"
    PRE_S, POST_S, BIN_S = cfg.pre_s, cfg.post_s, cfg.bin_size_s
    region_trains = {r: region_spike_trains(probe_loads, r) for r in REGIONS}

    extras = {}

    # --- SCm <-> MRN cross-correlogram ---
    try:
        fig = region_crosscorrelogram(region_trains["SCm"], region_trains["MRN"], "SCm", "MRN",
                                      binsize=0.005, windowsize=0.2)
        fig.suptitle(f"session {eid8}", y=1.02, fontsize=9)
        savefig(fig, phase, "crosscorr_SCm_MRN", root=ROOT)
        plt.close(fig)
        extras["crosscorr"] = "ok"
    except Exception as e:  # noqa: BLE001
        extras["crosscorr"] = f"{type(e).__name__}: {e}"

    # --- RT-sorted single-neuron raster (00 cell 25) ---
    try:
        bc = data["bin_centers"]
        rt = cov["rt"]
        row_of_trial = np.argsort(np.argsort(np.where(np.isfinite(rt), rt, np.inf)))
        rt_curve = np.sort(rt[np.isfinite(rt)])
        ev = nap.Ts(t=events)

        def most_modulated_unit(counts, keep_idx):
            psth = counts.mean(0) / BIN_S
            base = psth[bc < 0].mean(0); peak = psth[bc >= 0].max(0)
            baserate = counts[:, bc < 0, :].mean((0, 1)) / BIN_S
            mod = np.where(baserate < 12, (peak - base) / (base + 2.0), -np.inf)
            return int(mod.argmax())

        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharex=True)
        for ax, r in zip(axes, REGIONS):
            counts = data["regions"][r]["counts"]
            keep_idx = data["regions"][r]["keep_idx"]
            j = most_modulated_unit(counts, keep_idx)
            orig = int(keep_idx[j])
            base_hz = counts[:, bc < 0, j].mean() / BIN_S
            peri = nap.compute_perievent(nap.Ts(t=region_trains[r][orig]), ev, (-PRE_S, POST_S))
            col = region_color(r)
            for trial_i, key in enumerate(peri.keys()):
                tsp = peri[key].t
                if len(tsp):
                    ax.plot(tsp, np.full_like(tsp, row_of_trial[trial_i]), "|", color=col,
                            markersize=1.8, mew=0.5, rasterized=True)
            ax.axvline(0, color="k", ls="--", lw=1)
            ax.plot(rt_curve, np.arange(rt_curve.size), color="black", lw=1.6)
            ax.set(title=f"{r} — unit {orig} ({base_hz:.1f} Hz base)",
                   xlabel="time from stimOn (s)", xlim=(-PRE_S, POST_S))
        axes[0].set_ylabel("trial (sorted by RT)")
        fig.suptitle(f"Single-neuron raster — trials sorted by RT (black = movement onset) · "
                     f"session {eid8}", y=1.02, fontweight="bold")
        fig.tight_layout()
        savefig(fig, phase, "raster_example_neuron", root=ROOT)
        plt.close(fig)
        extras["raster_example_neuron"] = "ok"
    except Exception as e:  # noqa: BLE001
        extras["raster_example_neuron"] = f"{type(e).__name__}: {e}"

    # --- GPFA fitted timescales ---
    if do_gpfa:
        try:
            from src.dynamics.gpfa import fit_gpfa, latent_timescales, trials_to_spiketrains
            taus = {}
            for r in REGIONS:
                st = trials_to_spiketrains(region_trains[r], events, pre_s=PRE_S, post_s=POST_S)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model, _ = fit_gpfa(st, bin_size_s=BIN_S, x_dim=5, em_max_iters=50)
                tau = np.sort(latent_timescales(model, BIN_S))[::-1]
                taus[r] = [round(float(t), 0) for t in tau]
            extras["gpfa_tau_ms"] = taus
            extras["gpfa_slowest_ms"] = {r: max(taus[r]) for r in REGIONS}
        except Exception as e:  # noqa: BLE001
            extras["gpfa_tau_ms"] = f"{type(e).__name__}: {e}"

    return extras


# --------------------------------------------------------------------------- driver


def run_only_one(eid8, do_gpfa=True):
    """Minimal-footprint path: skip the whole core, run ONLY the raw-spike extras and merge them
    into the existing results/session_recap/<eid8>.json. Keeps the process small before the
    ~0.4 GB session load (the core stage grows the process enough to OOM the load otherwise)."""
    set_style()
    phase = f"recap_{eid8}"
    npz = PROC / f"{eid8}__stimOn_pre300_post800_bin40_rate1.npz"
    data = load_processed(npz)
    cov = _covariates(data)
    print(f"[only-one] {eid8}: loading raw session for crosscorr / raster"
          f"{' / GPFA' if do_gpfa else ''} …")
    extras = compute_one_extras(eid8, cov, data, phase, do_gpfa=do_gpfa)
    jpath = OUT / f"{eid8}.json"
    bundle = json.loads(jpath.read_text()) if jpath.exists() else {"eid8": eid8}
    bundle["extras"] = extras
    jpath.write_text(json.dumps(bundle, indent=2))
    print(f"  extras merged -> results/session_recap/{eid8}.json")
    print("  " + json.dumps(extras, indent=2).replace("\n", "\n  "))
    return extras


def run(eid8, do_gpfa=True, do_one=True):
    set_style()
    phase = f"recap_{eid8}"
    npz = PROC / f"{eid8}__stimOn_pre300_post800_bin40_rate1.npz"
    if not npz.exists():
        raise SystemExit(f"missing tensor {npz.name} — build it with scripts/build_processed_all.py")
    data = load_processed(npz)
    cov = _covariates(data)
    print(f"\n{'='*66}\n{eid8}  ({SESSION_LABEL.get(eid8,'')})  "
          f"{data['n_trials']} trials, units "
          f"{ {r: data['regions'][r]['n_units'] for r in REGIONS} }\n{'='*66}")

    bundle = {
        "eid8": eid8, "eid": FULL_EID[eid8], "label": SESSION_LABEL.get(eid8, ""),
        "n_trials": data["n_trials"],
        "n_units": {r: int(data["regions"][r]["n_units"]) for r in REGIONS},
        "window": {"pre_s": data["config"]["pre_s"], "post_s": data["config"]["post_s"],
                   "bin_ms": int(data["config"]["bin_size_s"] * 1000)},
        "figures_dir": f"results/figures/{phase}",
    }

    def _reclaim():
        plt.close("all"); gc.collect()

    t0 = time.time()
    bundle["psth"] = fig_psth_per_area(data, eid8, phase); _reclaim(); print(f"  psth              [{time.time()-t0:.0f}s]")
    fig_macro_raster(data, eid8, phase);                   _reclaim(); print(f"  macro raster      [{time.time()-t0:.0f}s]")
    bundle["decoding"] = fig_decoding(data, cov, eid8, phase); _reclaim(); print(f"  decoding          [{time.time()-t0:.0f}s]")
    bundle["hmm"] = fig_hmm_from_csv(eid8, phase);         _reclaim(); print(f"  hmm (from csv)    [{time.time()-t0:.0f}s]")
    bundle["dpca"] = compute_dpca(data, cov, eid8, phase); _reclaim(); print(f"  dpca              [{time.time()-t0:.0f}s]")
    rj = compute_rrr_joint(data, eid8, phase); _reclaim()
    bundle["rrr"], bundle["joint"] = rj["rrr"], rj["joint"]; print(f"  rrr + joint       [{time.time()-t0:.0f}s]")
    if do_one:
        try:
            bundle["extras"] = compute_one_extras(eid8, cov, data, phase, do_gpfa=do_gpfa)
            print(f"  one extras        [{time.time()-t0:.0f}s]")
        except (MemoryError, Exception) as e:  # noqa: BLE001 — raw-spike stage is best-effort
            bundle["extras"] = {"error": f"{type(e).__name__}: {e}"}
            print(f"  one extras FAILED ({type(e).__name__}: {e}) — core figures still written")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{eid8}.json").write_text(json.dumps(bundle, indent=2))
    print(f"\n  stats  -> results/session_recap/{eid8}.json")
    print(f"  figs   -> results/figures/{phase}/  ({time.time()-t0:.0f}s total)")
    return bundle


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eid", required=True, help="8-char session prefix (e.g. 5c0c560e)")
    ap.add_argument("--no-gpfa", action="store_true", help="skip the slow GPFA fit")
    ap.add_argument("--no-one", action="store_true", help="skip all raw-spike (ONE) figures")
    ap.add_argument("--only-one", action="store_true",
                    help="run ONLY the raw-spike extras (minimal footprint) and merge into the JSON")
    args = ap.parse_args()
    eid8 = args.eid[:8]
    if eid8 not in FULL_EID:
        raise SystemExit(f"unknown eid {eid8}; known: {list(FULL_EID)}")
    if args.only_one:
        run_only_one(eid8, do_gpfa=not args.no_gpfa)
    else:
        run(eid8, do_gpfa=not args.no_gpfa, do_one=not args.no_one)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
