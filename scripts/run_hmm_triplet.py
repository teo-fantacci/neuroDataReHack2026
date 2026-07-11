"""Fit a Poisson HMM per region on one IBL {MRN, SCm, VISp} session.

    python scripts/run_hmm_triplet.py

Downloads ~440 MB (both probes of the session) on first run, then uses the ONE cache.
For each region it:
  * builds a stimulus-aligned tensor with a real pre-stimulus baseline,
  * fits a Poisson HMM and biases the initial state from the baseline,
  * orders states by time and measures forward/backward transition asymmetry,
  * saves a figure and a numeric summary to results/hmm/.

The scientific question this probes: when states are time-ordered, do the off-diagonal
transitions run predominantly forward (a directed metastable sequence), and is that
consistent across the three areas?
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data.access import get_one, load_trials, load_units, sessions_table  # noqa: E402
from src.data.build_matrices import drop_silent_units, region_tensor_from_probes  # noqa: E402
from src.dynamics.hmm import (  # noqa: E402
    baseline_exit_latency,
    fit_poisson_hmm,
    forward_fraction_null,
    net_flux,
)

SESSION_EID = "5c0c560e-9e1f-45e9-b66e-e4ee7855be84"   # MRN:63, SCm:59, VISp:36, 843 trials
REGIONS = ["VISp", "SCm", "MRN"]                        # ordered sensory -> motor
N_STATES = 5
PRE_S, POST_S, BIN_S = 0.5, 0.8, 0.02                  # 0.5 s baseline before stimulus
SEED = 0
# Shuffle null costs one full HMM fit each -> set via env, default cheap.
#   N_SHUFFLES=0  -> skip (fast); N_SHUFFLES=20 -> publication-ish
N_SHUFFLES = int(os.environ.get("N_SHUFFLES", "10"))


def _add_beryl(clusters):
    from iblatlas.regions import BrainRegions
    br = BrainRegions()
    clusters = clusters.copy()
    clusters["Beryl"] = br.id2acronym(clusters["atlas_id"].values, mapping="Beryl")
    return clusters


def main() -> None:
    out = ROOT / "results" / "hmm"
    out.mkdir(parents=True, exist_ok=True)

    one = get_one()

    bwm = sessions_table(one)
    pids = bwm.loc[bwm.eid == SESSION_EID, "pid"].tolist()
    print(f"session {SESSION_EID}: {len(pids)} probes")

    trials, mask = load_trials(one, SESSION_EID)
    good = trials[mask]
    events = good["stimOn_times"].values
    rt = (good["firstMovement_times"] - good["stimOn_times"]).values
    print(f"usable trials: {len(good)}   median RT: {np.nanmedian(rt):.3f} s")

    probe_loads = []
    for pid in pids:
        spikes, clusters = load_units(one, pid)
        probe_loads.append({"spikes": spikes, "clusters": _add_beryl(clusters)})
        print(f"  probe {pid[:8]}: {clusters.shape[0]} good units")

    summary = []
    for region in REGIONS:
        counts, bin_centers = region_tensor_from_probes(
            probe_loads, region, events, pre_s=PRE_S, post_s=POST_S, bin_size_s=BIN_S
        )
        counts, kept = drop_silent_units(counts, min_total_spikes=1)
        baseline_mask = bin_centers < 0.0
        n_trials, n_bins, n_units = counts.shape
        print(f"\n[{region}] tensor {counts.shape}  ({n_units} units, "
              f"{baseline_mask.sum()} baseline bins)")

        res = fit_poisson_hmm(counts, N_STATES, baseline_mask, seed=SEED, bias_startprob=True)
        res.bin_centers = bin_centers

        if N_SHUFFLES > 0:
            null = forward_fraction_null(counts, N_STATES, baseline_mask,
                                         n_shuffles=N_SHUFFLES, seed=SEED)
            null_mean, null_std = float(null.mean()), float(null.std())
            p_null = float((null >= res.forward_fraction).mean())
        else:
            null_mean = null_std = p_null = float("nan")
        latency = baseline_exit_latency(res.occupancy, bin_centers, baseline_state=0, thresh=0.5)

        print(f"[{region}] converged={res.model.monitor_.converged} "
              f"iters={res.model.monitor_.iter}  LL={res.log_likelihood:,.0f}")
        print(f"[{region}] baseline startprob (time-ordered): "
              f"{np.round(res.startprob_baseline, 2)}")
        print(f"[{region}] forward fraction = {res.forward_fraction:.3f}  "
              f"null={null_mean:.3f}±{null_std:.3f}  p={p_null:.3f}  "
              f"(directed chain vs reversible mixing)")
        print(f"[{region}] baseline-exit latency = {latency*1000:.0f} ms after stimulus")

        _plot_region(region, res, rt, out)
        summary.append({
            "region": region, "n_units": n_units, "n_trials": n_trials,
            "forward_fraction": round(res.forward_fraction, 4),
            "null_mean": None if np.isnan(null_mean) else round(null_mean, 4),
            "p_vs_null": None if np.isnan(p_null) else round(p_null, 4),
            "baseline_exit_ms": None if np.isnan(latency) else round(latency * 1000, 1),
            "converged": bool(res.model.monitor_.converged),
        })

    sdf = pd.DataFrame(summary)
    _cross_area(sdf)
    _write_summary(sdf, out)
    print(f"\nwrote {out/'hmm_triplet_summary.md'} and per-region figures")


def _cross_area(sdf):
    """The inter-areal directionality: order regions by when they leave baseline."""
    lat = sdf.dropna(subset=["baseline_exit_ms"]).sort_values("baseline_exit_ms")
    print("\n=== CROSS-AREA DIRECTIONALITY (baseline-exit order) ===")
    if lat.empty:
        print("  (no region crossed the baseline threshold)")
        return
    arrow = "  ->  ".join(f"{r.region} ({r.baseline_exit_ms:.0f} ms)" for _, r in lat.iterrows())
    print("  " + arrow)
    print("  expectation from anatomy/IBL: VISp (sensory) leads SCm/MRN (motor).")


def _plot_region(region, res, rt, out):
    K = res.occupancy.shape[0]
    bc = res.bin_centers
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    # state raster, trials sorted by RT
    order = np.argsort(rt)
    ax[0].imshow(res.states[order], aspect="auto", interpolation="nearest",
                 extent=[bc[0], bc[-1], len(order), 0], cmap="viridis")
    ax[0].plot(rt[order], np.arange(len(order)) + 0.5, color="r", lw=1, label="first move")
    ax[0].axvline(0, color="w", ls="--", lw=1)
    ax[0].set(xlabel="time from stimOn (s)", ylabel="trial (sorted by RT)",
              title=f"{region}: HMM state (time-ordered), K={K}")
    ax[0].legend(frameon=False, loc="lower right", fontsize=8)

    # occupancy
    for k in range(K):
        ax[1].plot(bc, res.occupancy[k], label=f"state {k}")
    ax[1].axvline(0, color="k", ls="--", lw=1)
    ax[1].axvspan(bc[0], 0, color="0.9", label="baseline")
    ax[1].set(xlabel="time from stimOn (s)", ylabel="P(state)",
              title="occupancy (state 0 = earliest)")
    ax[1].legend(frameon=False, fontsize=7, ncol=2)

    # net transition flux; positive (red) above diagonal = forward
    flux = net_flux(res.trans_counts)
    vmax = np.abs(flux).max() or 1
    im = ax[2].imshow(flux, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax[2].set(xlabel="to state", ylabel="from state",
              title=f"net flux N-N^T  (fwd frac={res.forward_fraction:.2f})",
              xticks=range(K), yticks=range(K))
    for i in range(K):
        for j in range(K):
            if i != j:
                ax[2].text(j, i, f"{flux[i,j]:+d}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax[2], label="net i->j transitions")

    fig.tight_layout()
    fig.savefig(out / f"hmm_{region}.png", dpi=130)
    plt.close(fig)


def _write_summary(df, out):
    lines = [
        "# HMM on IBL triplet {VISp, SCm, MRN} — directionality probe",
        "",
        f"Session `{SESSION_EID}`, K={N_STATES} states, "
        f"{PRE_S*1000:.0f} ms pre-stimulus baseline, {BIN_S*1000:.0f} ms bins.",
        "Initial state biased from the pre-stimulus baseline occupancy.",
        "",
        "**forward_fraction** measures *directedness*: with states time-ordered, the "
        "share of net off-diagonal flux running to later states. Because states are "
        "ordered by their own occupancy time, this does **not** read absolute arrow of "
        "time — it separates a clean directed metastable chain (→1) from reversible "
        "mixing (≈0.5). `null_mean` is the mean over 20 within-trial time-bin shuffles; "
        "`p_vs_null` is the fraction of shuffles matching or exceeding the observed value.",
        "",
        "**baseline_exit_ms**: when the region first leaves its baseline state (P<0.5) "
        "after stimulus onset. The *ordering of this across regions* is the inter-areal "
        "directionality — see the cross-area line below.",
        "",
        df.to_markdown(index=False),
        "",
        _cross_area_md(df),
        "",
        "See `hmm_<region>.png` for the state raster, occupancy, and net-flux matrix.",
        "",
        "> Exploratory. K was fixed at 5, not cross-validated (a Phase 2 job). The absolute "
        "> progression (baseline → later states) is by construction; the non-trivial claims "
        "> are (a) directedness vs the shuffle null, and (b) the cross-area exit ordering.",
    ]
    (out / "hmm_triplet_summary.md").write_text("\n".join(lines), encoding="utf-8")


def _cross_area_md(df):
    lat = df.dropna(subset=["baseline_exit_ms"]).sort_values("baseline_exit_ms")
    if lat.empty:
        return "**Cross-area order:** no region crossed the baseline threshold."
    arrow = " → ".join(f"{r.region} ({r.baseline_exit_ms:.0f} ms)" for _, r in lat.iterrows())
    return f"**Cross-area directionality (baseline-exit order):** {arrow}"


if __name__ == "__main__":
    main()
