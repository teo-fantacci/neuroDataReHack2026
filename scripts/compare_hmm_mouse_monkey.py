"""Final HMM comparison: mouse first triplet session vs. the monkey session.

Puts the Poisson-HMM read-outs for the two datasets side by side, in ONE deliverable folder:

    results/hmm_mouse_vs_monkey/
        01_directedness_and_switching.png   forward_fraction + switch_rate, mouse vs monkey
        02_state_occupancy.png              P(state | time) trajectories, all areas
        03_transition_flux.png              net transition-flux matrices (directed vs symmetric)
        04_monkey_drift_confound.png        the across-session drift that flattens the monkey HMM
        readouts.csv                        one row per area
        HMM_analysis.md                     the write-up (kept alongside the figures)

Mouse  : IBL session 5c0c560e, {VISp, SCm, MRN}, stimOn-aligned, pre300 40 ms tensors,
         canonical K = {VISp:3, SCm:4, MRN:3}, baseline-biased startprob (the project default).
Monkey : DANDI 000620 ses-2022-08-20 (sub-Elgar), {FEF, DMFC}, DELAY-aligned 100 ms tensors,
         K = {FEF:4, DMFC:3} (stability-scan best K>=3), no startprob bias (notebook 01_HMM_monkey).

Everything is re-fit here from the frozen tensors so the figures are internally consistent; the
numbers reproduce results/hmm/hmm_readouts_5c0c560e.csv and results/hmm/monkey_..._readouts.csv.

    python scripts/compare_hmm_mouse_monkey.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data.monkey import load_monkey_aligned  # noqa: E402
from src.data.preprocess import load_processed  # noqa: E402
from src.dynamics.hmm import (  # noqa: E402
    fit_poisson_hmm, forward_fraction_split_half, net_flux, switch_rate,
)
from src.viz.style import (  # noqa: E402
    DIVERGING, region_color, set_style, state_colors, stroke,
)

OUT = ROOT / "results" / "hmm_mouse_vs_monkey"
SEED = 0
N_ITER = 50

MOUSE_NPZ = ROOT / "data" / "processed" / "5c0c560e__stimOn_pre300_post800_bin40_rate1.npz"
MOUSE_K = {"VISp": 3, "SCm": 4, "MRN": 3}          # canonical, best_K_per_region.md
MONKEY_SESSION = "ses-2022-08-20"
MONKEY_K = {"FEF": 4, "DMFC": 3}                   # stability best K>=3 (notebook 01_HMM_monkey)

# CVD-safe colours: mouse regions keep their project colours; give the monkey areas two
# unused Okabe-Ito hues (purple / green) so nothing collides on a shared axis.
AREA_COLOR = {
    "VISp": region_color("VISp"), "SCm": region_color("SCm"), "MRN": region_color("MRN"),
    "FEF": "#CC79A7", "DMFC": "#009E73",
}


# --------------------------------------------------------------------------- fitting


def fit_area(counts, K, baseline_mask, *, bias):
    """Fit one area and return the result + the split-half (de-circularised) directedness."""
    res = fit_poisson_hmm(counts, K, baseline_mask, n_iter=N_ITER, seed=SEED,
                          bias_startprob=bias)
    sh = forward_fraction_split_half(counts, K, baseline_mask, n_repeats=5, seed=SEED,
                                     n_iter=N_ITER)
    return res, sh


def run_mouse():
    d = load_processed(MOUSE_NPZ)
    bc, bmask = d["bin_centers"], d["baseline_mask"]
    out = {}
    for region in ["VISp", "SCm", "MRN"]:
        counts = d["regions"][region]["counts"]
        res, sh = fit_area(counts, MOUSE_K[region], bmask, bias=True)
        res.bin_centers = bc
        out[region] = {"res": res, "sh": sh, "bin_centers": bc, "align_label": "stimOn",
                       "event0": "stimulus", "n_trials": counts.shape[0],
                       "n_units": counts.shape[2], "K": MOUSE_K[region]}
        print(f"  mouse {region:5s} K={MOUSE_K[region]}  ff={res.forward_fraction:.3f}"
              f"  ff_split={sh['forward_fraction_split_half']:.3f}+/-{sh['std']:.3f}"
              f"  switch={switch_rate(res.states):.2f}")
    return out


def run_monkey():
    mk = load_monkey_aligned(MONKEY_SESSION)
    bc = mk["bin_centers"]
    ev_med = np.nanmedian(mk["events"], axis=0)
    baseline_mask = bc < ev_med[1]                 # fixation (pre-stimulus) = nominal baseline
    out = {"_events_med": ev_med, "_tensors": mk["tensors"], "_bin_centers": bc}
    for area in ["FEF", "DMFC"]:
        counts = mk["tensors"][area]
        res, sh = fit_area(counts, MONKEY_K[area], baseline_mask, bias=False)
        res.bin_centers = bc
        out[area] = {"res": res, "sh": sh, "bin_centers": bc, "align_label": "delay onset",
                     "event0": "delay", "n_trials": counts.shape[0],
                     "n_units": counts.shape[2], "K": MONKEY_K[area]}
        print(f"  monkey {area:5s} K={MONKEY_K[area]}  ff={res.forward_fraction:.3f}"
              f"  ff_split={sh['forward_fraction_split_half']:.3f}+/-{sh['std']:.3f}"
              f"  switch={switch_rate(res.states):.2f}")
    return out


# --------------------------------------------------------------------------- figures

MOUSE_AREAS = ["VISp", "SCm", "MRN"]
MONKEY_AREAS = ["FEF", "DMFC"]


def fig_directedness(mouse, monkey):
    """forward_fraction (directedness) and switch_rate, mouse triplet vs monkey pair."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    groups = [("mouse\n(IBL 5c0c560e)", MOUSE_AREAS, mouse),
              ("monkey\n(000620 ses-2022-08-20)", MONKEY_AREAS, monkey)]

    # ---- (a) forward_fraction ----
    ax = axes[0]
    x = 0
    xticks, xlabels = [], []
    for gname, areas, store in groups:
        for a in areas:
            r = store[a]
            ff = r["res"].forward_fraction
            sh = r["sh"]["forward_fraction_split_half"]
            sd = r["sh"]["std"]
            ax.bar(x, ff, width=0.72, color=AREA_COLOR[a], edgecolor="black", lw=0.8, zorder=2)
            ax.errorbar(x, sh, yerr=sd, fmt="D", ms=6, color="black", mfc="white",
                        capsize=3, zorder=4)
            ax.text(x, max(ff, sh + sd) + 0.014, f"{ff:.2f}", ha="center", va="bottom",
                    fontsize=9)
            xticks.append(x); xlabels.append(a)
            x += 1
        x += 0.7                                    # gap between species
    ax.axhline(0.5, color="0.35", ls=":", lw=1.4)
    # inter-group gap (between MRN at x=2 and FEF at x=3.7) is empty — safe place for the label
    ax.text(2.85, 0.478, "chance (undirected)", fontsize=8, color="0.35",
            va="top", ha="center")
    ax.set(xticks=xticks, ylim=(0.45, 0.86), ylabel="forward_fraction",
           title="(a) Directedness of the state sequence")
    ax.set_xticklabels(xlabels)
    ax.text(0.02, 0.97, "bar = pooled  ·  ◇ = split-half ± SD", transform=ax.transAxes,
            fontsize=8, va="top", color="0.3")
    _species_brackets(ax, groups)

    # ---- (b) switch_rate ----
    ax = axes[1]
    x = 0
    xticks, xlabels = [], []
    for gname, areas, store in groups:
        for a in areas:
            sr = switch_rate(store[a]["res"].states)
            ax.bar(x, sr, width=0.72, color=AREA_COLOR[a], edgecolor="black", lw=0.8, zorder=2)
            ax.text(x, sr + 0.06, f"{sr:.2f}", ha="center", va="bottom", fontsize=9)
            xticks.append(x); xlabels.append(a)
            x += 1
        x += 0.7
    ax.set(xticks=xticks, ylabel="switches per trial",
           title="(b) How often the state changes")
    ax.set_xticklabels(xlabels)
    _species_brackets(ax, groups)

    fig.suptitle("HMM transition structure — mouse first session vs monkey session", y=1.02,
                 fontsize=13)
    fig.tight_layout()
    return fig


def _species_brackets(ax, groups):
    """Draw a light labelled bracket under each species group's ticks."""
    x = 0
    y = ax.get_ylim()[0]
    for gname, areas, _ in groups:
        x0 = x - 0.4
        x1 = x + len(areas) - 1 + 0.4
        ax.annotate("", xy=(x0, y), xytext=(x1, y), xycoords="data",
                    annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color="0.5", lw=1.0))
        ax.text((x0 + x1) / 2, y - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.10, gname,
                ha="center", va="top", fontsize=9, color="0.3")
        x += len(areas) + 0.7


def fig_occupancy(mouse, monkey):
    """P(state | time) for every area — the state sequence itself."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.4))
    order = [("VISp", mouse), ("SCm", mouse), ("MRN", mouse),
             ("FEF", monkey), ("DMFC", monkey)]
    for ax, (a, store) in zip(axes.ravel(), order):
        r = store[a]
        occ = r["res"].occupancy
        bc = r["bin_centers"]
        K = occ.shape[0]
        cols = state_colors(K)
        for s in range(K):
            ax.plot(bc, occ[s], color=cols[s], lw=2.2, path_effects=stroke(), label=f"S{s}")
        ax.axvline(0, color="black", ls="--", lw=1.4)
        ax.set(ylim=(-0.03, 1.05),
               title=f"{a}  (K={K}, {r['n_units']}u, {r['n_trials']}tr)")
        ax.set_xlabel(f"time from {r['event0']} onset (s)")
        ax.legend(fontsize=7, ncol=K, loc="upper right", frameon=False)
    axes[0, 0].set_ylabel("P(state)")
    axes[1, 0].set_ylabel("P(state)")
    axes[1, 2].axis("off")
    # row captions
    fig.text(0.5, 0.99, "MOUSE  (IBL 5c0c560e, stimOn-aligned)", ha="center", va="top",
             fontsize=12, fontweight="bold")
    fig.text(0.5, 0.505, "MONKEY  (000620 ses-2022-08-20, delay-aligned)", ha="center",
             va="top", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.subplots_adjust(hspace=0.55)
    return fig


def fig_flux(mouse, monkey):
    """Net transition-flux matrices: asymmetric (directed) vs symmetric (undirected)."""
    order = [("VISp", mouse), ("SCm", mouse), ("MRN", mouse),
             ("FEF", monkey), ("DMFC", monkey)]
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.7))
    for ax, (a, store) in zip(axes, order):
        res = store[a]["res"]
        flux = net_flux(res.trans_counts)
        K = flux.shape[0]
        vmax = np.abs(flux).max() or 1
        im = ax.imshow(flux, cmap=DIVERGING, vmin=-vmax, vmax=vmax)
        for i in range(K):
            for j in range(K):
                if i != j:
                    ax.text(j, i, f"{flux[i, j]:+d}", ha="center", va="center", fontsize=8)
        ax.set(xticks=range(K), yticks=range(K), xlabel="to state", ylabel="from state",
               title=f"{a}\nfwd frac = {res.forward_fraction:.2f}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Net transition flux  (red = net forward i→j, blue = net backward)  ·  "
                 "asymmetric = directed sequence, symmetric = undirected", y=1.05, fontsize=12)
    fig.tight_layout()
    return fig


def fig_monkey_drift(monkey):
    """The colleague's point: an across-session drift (a strong trial-index PC) flattens the
    monkey HMM — the states latch onto slow rate epochs instead of within-trial phase."""
    tensors = monkey["_tensors"]
    bc = monkey["_bin_centers"]
    bin_s = float(bc[1] - bc[0])
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    # ---- (a) population rate vs trial index: FEF drifts, DMFC is stationary ----
    ax = axes[0]
    for a in ["FEF", "DMFC"]:
        T = tensors[a]                                   # (trials, bins, units) int counts
        pop = T.sum(axis=(1, 2)) / (T.shape[1] * bin_s * T.shape[2])   # sp/s/unit per trial
        tr = np.arange(len(pop))
        rho = np.corrcoef(tr, pop)[0, 1]
        ax.scatter(tr, pop, s=8, color=AREA_COLOR[a], alpha=0.35)
        # smoothed trend
        w = 21
        trend = np.convolve(pop, np.ones(w) / w, mode="valid")
        ax.plot(tr[w // 2: w // 2 + len(trend)], trend, color=AREA_COLOR[a], lw=2.6,
                path_effects=stroke(), label=f"{a}  (r_trial = {rho:+.2f})")
    ax.set(xlabel="trial index (chronological)", ylabel="population rate (sp/s/unit)",
           title="(a) Across-session rate drift")
    ax.legend(fontsize=8, frameon=False)

    # ---- (b) PC1 of the trial×unit mean-rate matrix vs trial index (FEF) ----
    ax = axes[1]
    T = tensors["FEF"]
    Xtr = T.mean(axis=1)                                 # (trials, units) mean count/bin per trial
    Xz = (Xtr - Xtr.mean(0)) / (Xtr.std(0) + 1e-9)
    # PC1 via SVD
    U, S, Vt = np.linalg.svd(Xz - Xz.mean(0), full_matrices=False)
    pc1 = U[:, 0] * S[0]
    var1 = S[0] ** 2 / (S ** 2).sum()
    if np.corrcoef(np.arange(len(pc1)), pc1)[0, 1] < 0:
        pc1 = -pc1                                       # sign so PC1 rises with trial index
    tr = np.arange(len(pc1))
    rho = np.corrcoef(tr, pc1)[0, 1]
    sc = ax.scatter(tr, pc1, c=tr, cmap="cividis", s=14)
    ax.set(xlabel="trial index (chronological)", ylabel="FEF PC1 score",
           title=f"(b) FEF trial-PC1 = the drift\nPC1 var = {var1*100:.0f}%,  r_trial = {rho:+.2f}")
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="trial index")

    # ---- (c) FEF HMM state latches onto trial index, not within-trial time ----
    ax = axes[2]
    res = monkey["FEF"]["res"]
    states = res.states                                  # (trials, bins)
    modal = np.array([np.bincount(row).argmax() for row in states])   # per-trial modal state
    tr = np.arange(len(modal))
    rho = np.corrcoef(tr, modal)[0, 1]
    ax.scatter(tr, modal + np.random.default_rng(0).uniform(-0.12, 0.12, len(modal)),
               c=tr, cmap="cividis", s=14)
    ax.set(xlabel="trial index (chronological)", ylabel="FEF per-trial modal HMM state",
           yticks=range(res.occupancy.shape[0]),
           title=f"(c) FEF states = session epochs\nr(state, trial#) = {rho:+.2f}  "
                 f"(switch = {switch_rate(states):.2f}/trial)")

    fig.suptitle("Why the monkey HMM looks flat: a strong across-session drift confounds the "
                 "within-trial inference (FEF)", y=1.03, fontsize=12.5)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- driver


def readout_table(mouse, monkey) -> pd.DataFrame:
    rows = []
    for species, areas, store in [("mouse", MOUSE_AREAS, mouse),
                                  ("monkey", MONKEY_AREAS, monkey)]:
        for a in areas:
            r = store[a]
            res = r["res"]
            rows.append({
                "species": species, "area": a, "align": r["align_label"],
                "K": r["K"], "n_units": r["n_units"], "n_trials": r["n_trials"],
                "forward_fraction": round(res.forward_fraction, 3),
                "ff_split_half": round(r["sh"]["forward_fraction_split_half"], 3),
                "ff_split_sd": round(r["sh"]["std"], 3),
                "switch_rate": round(switch_rate(res.states), 2),
                "log_likelihood": round(float(res.log_likelihood), 1),
            })
    return pd.DataFrame(rows)


def main() -> int:
    set_style()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("Fitting mouse (5c0c560e) ...")
    mouse = run_mouse()
    print("Fitting monkey (ses-2022-08-20) ...")
    monkey = run_monkey()

    table = readout_table(mouse, monkey)
    table.to_csv(OUT / "readouts.csv", index=False)
    print("\n" + table.to_string(index=False))

    figs = {
        "01_directedness_and_switching": fig_directedness(mouse, monkey),
        "02_state_occupancy": fig_occupancy(mouse, monkey),
        "03_transition_flux": fig_flux(mouse, monkey),
        "04_monkey_drift_confound": fig_monkey_drift(monkey),
    }
    for name, fig in figs.items():
        p = OUT / f"{name}.png"
        fig.savefig(p, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  figure -> {p.relative_to(ROOT)}")

    print(f"\n[{time.time()-t0:.0f}s]  all outputs -> {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
