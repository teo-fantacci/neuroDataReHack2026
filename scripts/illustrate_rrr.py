"""One illustration of the RRR (communication-subspace) result — MOUSE and MONKEY side by side.

    python scripts/illustrate_rrr.py

Reduced-rank regression predicts one area's population from another's, restricted to rank r; the
rank where held-out R^2 saturates is the dimensionality of the channel (Semedo et al. 2019,
Neuron 102:249). The reference that makes it interpretable is the **within-area control** —
predict half of an area's units from its *other* half. Cross-area R^2 that reaches the black
control means the areas share whatever structure is there (common drive, no bottleneck); cross
sitting well *below* the control means a genuinely restricted channel.

The figure contrasts the two datasets on exactly this test:

  * MOUSE  (IBL, session 5c0c560e, VISp/SCm/MRN triplet): the two motor areas **SCm <-> MRN** —
    cross-area prediction RISES TO the within-area control -> **common drive, no restricted
    subspace** (the recap's Sec 3 headline). VISp->SCm is shown faint as the weak/restricted case.
  * MONKEY (DANDI 000620, session ses-2022-08-20, FEF + DMFC): **FEF <-> DMFC** — cross-area
    prediction sits WELL BELOW each area's own control -> **a restricted channel**, and it is
    **asymmetric** (DMFC->FEF > FEF->DMFC).

Both panels use the same method (grouped-by-trial 5-fold CV, ridge RRR, sqrt+smooth+z-scored
counts) so they are directly comparable. Mouse bins 40 ms / smooth 40 ms; monkey bins 100 ms /
smooth 100 ms (each dataset's standing geometry setting).

Writes: results/figures/rrr_illustration/rrr_mouse_monkey.png
        results/figures/rrr_illustration/rrr_summary.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from src.data.preprocess import load_processed  # noqa: E402
from src.dynamics.rrr import rrr_cv_curve, saturation_rank, within_area_control  # noqa: E402
from src.geometry.embed import population_matrix  # noqa: E402
from src.viz.style import OKABE_ITO, region_color, savefig, set_style, stroke  # noqa: E402

SEED = 0
RANKS = list(range(1, 11))
PHASE = "rrr_illustration"

MOUSE_NPZ = ROOT / "data" / "processed" / "5c0c560e__stimOn_pre300_post800_bin40_rate1.npz"
MONKEY_SESSION = "ses-2022-08-20"

# FEF / DMFC are not in the region palette; assign fixed CVD-safe hues (blue / orange, well apart).
FEF_C, DMFC_C = OKABE_ITO[4], OKABE_ITO[0]
CTRL_C = "black"


# --------------------------------------------------------------------------- compute


def rrr_pair(mats: dict[str, np.ndarray], src: str, tgt: str, groups):
    c = rrr_cv_curve(mats[src], mats[tgt], groups, ranks=RANKS)
    return c, round(float(c["mean_r2"].max()), 3), int(saturation_rank(c))


def mouse_curves():
    data = load_processed(MOUSE_NPZ)
    bin_s = data["config"]["bin_size_s"]
    regions = ["VISp", "SCm", "MRN"]
    mats = {r: population_matrix(data["regions"][r]["counts"], None, bin_size_s=bin_s,
                                 smooth_sigma_ms=40.0) for r in regions}
    n_trials = data["n_trials"]
    n_bins = data["regions"]["SCm"]["counts"].shape[1]
    groups = np.repeat(np.arange(n_trials), n_bins)

    out = {"n_trials": n_trials, "n_units": {r: mats[r].shape[1] for r in regions},
           "curves": {}, "summary": []}
    for src, tgt in [("SCm", "MRN"), ("MRN", "SCm"), ("VISp", "SCm")]:
        c, best, sat = rrr_pair(mats, src, tgt, groups)
        out["curves"][f"{src}->{tgt}"] = c
        out["summary"].append({"dataset": "mouse", "pair": f"{src}->{tgt}", "kind": "cross",
                               "best_r2": best, "sat_rank": sat})
    for tgt in ["SCm", "MRN"]:
        c = within_area_control(mats[tgt], groups, ranks=RANKS, seed=SEED)
        out["curves"][f"{tgt} self"] = c
        out["summary"].append({"dataset": "mouse", "pair": f"{tgt} self", "kind": "within",
                               "best_r2": round(float(c["mean_r2"].max()), 3),
                               "sat_rank": int(saturation_rank(c))})
    return out


def monkey_curves():
    from src.data.monkey import load_monkey_tensors
    mk = load_monkey_tensors(MONKEY_SESSION, t0=0.0, t1=4.0, bin_s=0.10)
    bin_s = 0.10
    tensors = mk["tensors"]
    areas = ["FEF", "DMFC"]
    mats = {a: population_matrix(tensors[a], None, bin_size_s=bin_s, smooth_sigma_ms=100.0)
            for a in areas}
    n_trials = mk["n_trials"]
    n_bins = len(mk["bin_centers"])
    groups = np.repeat(np.arange(n_trials), n_bins)

    out = {"n_trials": n_trials, "n_units": {a: mats[a].shape[1] for a in areas},
           "curves": {}, "summary": []}
    for src, tgt in [("DMFC", "FEF"), ("FEF", "DMFC")]:
        c, best, sat = rrr_pair(mats, src, tgt, groups)
        out["curves"][f"{src}->{tgt}"] = c
        out["summary"].append({"dataset": "monkey", "pair": f"{src}->{tgt}", "kind": "cross",
                               "best_r2": best, "sat_rank": sat})
    for tgt in areas:
        c = within_area_control(mats[tgt], groups, ranks=RANKS, seed=SEED)
        out["curves"][f"{tgt} self"] = c
        out["summary"].append({"dataset": "monkey", "pair": f"{tgt} self", "kind": "within",
                               "best_r2": round(float(c["mean_r2"].max()), 3),
                               "sat_rank": int(saturation_rank(c))})
    return out


# --------------------------------------------------------------------------- figure


def _lab(c, name):
    return f"{name}  (R²={c['mean_r2'].max():.2f}, rank {saturation_rank(c)})"


def _plot_curve(ax, c, color, label, ls="-", marker="o", lw=2.0, alpha=1.0):
    ax.errorbar(c["rank"], c["mean_r2"], yerr=c["sem_r2"], marker=marker, ms=4, lw=lw, ls=ls,
                color=color, alpha=alpha, capsize=2.5,
                path_effects=stroke() if ls == "-" else None, label=label)


def make_figure(mouse, monkey):
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.8), sharey=False)
    cm, ck = mouse["curves"], monkey["curves"]

    # ------- MOUSE panel: SCm <-> MRN = symmetric common drive -------
    axm = axes[0]
    _plot_curve(axm, cm["SCm->MRN"], region_color("SCm"), _lab(cm["SCm->MRN"], "SCm → MRN"))
    _plot_curve(axm, cm["MRN->SCm"], region_color("MRN"), _lab(cm["MRN->SCm"], "MRN → SCm"))
    _plot_curve(axm, cm["SCm self"], CTRL_C, _lab(cm["SCm self"], "SCm self-pred. (control)"),
                ls="--", marker="s", lw=1.6)
    _plot_curve(axm, cm["MRN self"], "0.45", _lab(cm["MRN self"], "MRN self-pred. (control)"),
                ls="--", marker="s", lw=1.6)
    _plot_curve(axm, cm["VISp->SCm"], region_color("VISp"),
                _lab(cm["VISp->SCm"], "VISp → SCm (weak channel)"), lw=1.4, alpha=0.7)
    axm.set(xlabel="rank of the communication channel", ylabel="held-out R²  (grouped by trial)",
            ylim=(0, None),
            title=f"MOUSE · IBL · SCm ↔ MRN\n(session 5c0c560e · "
                  f"{mouse['n_units']['SCm']} SCm + {mouse['n_units']['MRN']} MRN units)")
    axm.legend(frameon=False, fontsize=8.3, loc="lower right")
    axm.text(0.03, 0.97,
             "cross-area RISES TO each control,\nBOTH directions → symmetric\ncommon drive "
             "(no bottleneck).\nVISp→SCm is the contrasting\nweak / restricted channel.",
             transform=axm.transAxes, va="top", ha="left", fontsize=9, fontweight="bold",
             color="#0072B2", bbox=dict(boxstyle="round", fc="white", ec="#0072B2", alpha=0.95))

    # ------- MONKEY panel: FEF <-> DMFC = weaker, asymmetric -------
    axk = axes[1]
    _plot_curve(axk, ck["DMFC->FEF"], DMFC_C, _lab(ck["DMFC->FEF"], "DMFC → FEF"))
    _plot_curve(axk, ck["FEF->DMFC"], FEF_C, _lab(ck["FEF->DMFC"], "FEF → DMFC"))
    _plot_curve(axk, ck["FEF self"], CTRL_C, _lab(ck["FEF self"], "FEF self-pred. (control)"),
                ls="--", marker="s", lw=1.6)
    _plot_curve(axk, ck["DMFC self"], "0.45", _lab(ck["DMFC self"], "DMFC self-pred. (control)"),
                ls="--", marker="s", lw=1.6)
    axk.set(xlabel="rank of the communication channel", ylabel="held-out R²  (grouped by trial)",
            ylim=(0, None),
            title=f"MONKEY · DANDI 000620 · FEF ↔ DMFC\n(session {MONKEY_SESSION} · "
                  f"{monkey['n_units']['FEF']} FEF + {monkey['n_units']['DMFC']} DMFC units)")
    axk.legend(frameon=False, fontsize=8.3, loc="upper left")
    axk.text(0.97, 0.02,
             "weaker AND asymmetric:\nDMFC→FEF reaches FEF's control\n(FEF low-dim, shared), but\n"
             "FEF→DMFC stays below DMFC's\n(DMFC high-dim, only partly read out).",
             transform=axk.transAxes, va="bottom", ha="right", fontsize=9, fontweight="bold",
             color="#D55E00", bbox=dict(boxstyle="round", fc="white", ec="#D55E00", alpha=0.95))

    fig.suptitle("Communication subspace (RRR): predict one area from another vs. from its own "
                 "other half — read each coloured cross-area curve against the BLACK control",
                 y=1.0, fontweight="bold", fontsize=12)
    fig.text(0.5, -0.05,
             "Reduced-rank regression predicts a target area's population from a source's, "
             "restricted to rank r (Semedo et al. 2019). The dashed within-area control (predict "
             "half of an area from its other half) is the ceiling any cross-area channel is judged "
             "against. LEFT — mouse motor areas predict each other about as well as themselves, "
             "symmetrically: one shared signal, not a directed relay. RIGHT — monkey frontal areas "
             "are coupled more weakly and directionally (DMFC→FEF > FEF→DMFC). Same method both "
             "datasets: sqrt+smooth+z-scored counts, grouped-by-trial 5-fold CV; bands = ±1 SEM "
             "over folds. Mouse 40 ms bins / 40 ms smoothing; monkey 100 ms / 100 ms.",
             ha="center", va="top", fontsize=8.8, style="italic", wrap=True)
    fig.tight_layout()
    p = savefig(fig, PHASE, "rrr_mouse_monkey", root=ROOT)
    plt.close(fig)
    return p


def main() -> int:
    print("MOUSE RRR (5c0c560e) …")
    mouse = mouse_curves()
    print("MONKEY RRR (ses-2022-08-20) …")
    monkey = monkey_curves()

    rows = mouse["summary"] + monkey["summary"]
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "figures" / PHASE
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "rrr_summary.csv", index=False)

    p = make_figure(mouse, monkey)
    print("\nRRR summary:")
    print(df.to_string(index=False))
    print(f"\nfigure -> {p.relative_to(ROOT)}")
    print(f"table  -> {(out / 'rrr_summary.csv').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
