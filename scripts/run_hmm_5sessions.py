"""Almost-final HMM run across all 5 {VISp, SCm, MRN} triplet sessions, with state stability.

    python scripts/run_hmm_5sessions.py                 # both alignments, all sessions
    python scripts/run_hmm_5sessions.py --eid 5c0c560e  # one session (quick look)
    python scripts/run_hmm_5sessions.py --align stimOn  # one alignment only

Reads the UNIFIED preprocessed tensors at the current window (pre 0.3 s, 40 ms bins, >= 1 Hz;
`data/processed/<eid>__<align>_pre300_..._bin40_rate1.npz`). It fits one Poisson HMM per
session x region at the canonical K (best_K_per_region.md) AND runs the ARI state-stability scan
over a range of K, so the figures show both *the read-outs at K* and *whether the data support
that K*. Per-session-then-pool (never a pseudo-population — those sessions are not simultaneous;
see docs/worklog/2026-07-14 §5): "K=3 in x/5, forward_fraction 0.8 +/- 0.04" is a replication.

Everything lands in a DEDICATED folder so the almost-final all-sessions results are separate
from the scratch `results/hmm/`:

    results/hmm_5sessions/readouts.csv          one row per align x session x region
    results/hmm_5sessions/stability_curves.csv  one row per align x session x region x K
    results/hmm_5sessions/pooled.csv            mean +/- sd across sessions, per align x region
    results/figures/hmm_5sessions/stability_<align>.png    ARI-vs-K, one line per session
    results/figures/hmm_5sessions/forward_fraction_<align>.png
    results/figures/hmm_5sessions/macro_raster_<align>.png  grouped rate heatmap (the macro raster)

The macro raster: 3 vertical area blocks (VISp / SCm / MRN top-to-bottom); within each block the
5 sessions are stacked, separated by a dark line, each session's units sorted by peak latency;
colour = trial-averaged, peak-normalised firing rate (cividis, CVD-safe). It is the one-glance
"is there a temporal sequence, in every area and every animal?" picture.
"""

from __future__ import annotations

import argparse
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

from src.data.build_matrices import counts_to_rates, smooth_over_time  # noqa: E402
from src.data.preprocess import load_processed, sigma_ms_to_bins  # noqa: E402
from src.dynamics.hmm import (  # noqa: E402
    baseline_exit_latency_robust, fit_poisson_hmm,
    forward_fraction_split_half, stability_scan, switch_rate,
)
from src.viz.style import (  # noqa: E402
    SEQUENTIAL, region_color, savefig, set_style, stroke,
)

PROC = ROOT / "data" / "processed"
OUT = ROOT / "results" / "hmm_5sessions"
FIG_PHASE = "hmm_5sessions"

REGIONS = ["VISp", "SCm", "MRN"]
K_PER_REGION = {"VISp": 3, "SCm": 4, "MRN": 3}    # canonical, best_K_per_region.md
STAB_KS = [2, 3, 4, 5, 6, 7, 8]
N_ITER = 50
STAB_ITER = 25
STAB_REPEATS = 3
STAB_RESTARTS = 2
SEED = 0
SMOOTH_MS = 60.0                                   # display smoothing for the macro raster only


# --------------------------------------------------------------------------- data


def find_sessions(align: str, bin_ms: int, eid: str | None) -> list[Path]:
    """One processed .npz per session at the CURRENT window (pre300), for a fixed alignment."""
    tag = "stimOn" if align.startswith("stim") else "firstMovement"
    hits = sorted(PROC.glob(f"*__{tag}_pre300_*_bin{bin_ms}_*.npz"))
    if eid:
        hits = [p for p in hits if p.name.startswith(eid)]
    seen, out = set(), []
    for p in hits:
        e = p.name.split("__")[0]
        if e not in seen:
            seen.add(e)
            out.append(p)
    return out


# --------------------------------------------------------------------------- per session


def run_one(npz: Path, align: str) -> tuple[list[dict], list[dict], dict]:
    """Fit + stability for one session. Returns (readout rows, stability rows, raster payload)."""
    data = load_processed(npz)
    bin_centers = data["bin_centers"]
    baseline_mask = data["baseline_mask"]
    eid = npz.name.split("__")[0]
    sigma_bins = sigma_ms_to_bins(SMOOTH_MS, data["config"]["bin_size_s"])

    readouts, curves, raster = [], [], {}
    for region in REGIONS:
        if region not in data["regions"]:
            print(f"    {region}: absent, skipping")
            continue
        counts = data["regions"][region]["counts"]
        K = K_PER_REGION[region]

        # --- state-stability scan (the ARI curve) ---
        scan = stability_scan(counts, STAB_KS, n_repeats=STAB_REPEATS, n_iter=STAB_ITER,
                              seed=SEED, n_restarts=STAB_RESTARTS)
        ks = scan["n_states"].to_numpy()
        ari = scan["ari_mean"].to_numpy()
        nontrivial = ks >= 3
        best_k = int(ks[nontrivial][np.argmax(ari[nontrivial])])
        ari_at_K = float(ari[ks == K][0])
        for _, row in scan.iterrows():
            curves.append({"align": align, "eid": eid, "region": region,
                           "K": int(row["n_states"]), "ari_mean": round(float(row["ari_mean"]), 4),
                           "ari_std": round(float(row["ari_std"]), 4),
                           "matched_acc": round(float(row["matched_acc_mean"]), 4)})

        # --- HMM read-outs at the canonical K ---
        res = fit_poisson_hmm(counts, K, baseline_mask, n_iter=N_ITER, seed=SEED,
                              bias_startprob=True)
        sh = forward_fraction_split_half(counts, K, baseline_mask, n_repeats=5, seed=SEED,
                                         n_iter=N_ITER)
        lat = baseline_exit_latency_robust(res.occupancy, bin_centers)

        readouts.append({
            "align": align, "eid": eid, "region": region,
            "n_trials": counts.shape[0], "n_units": counts.shape[2],
            "K": K, "best_k_scan": best_k, "ari_at_K": round(ari_at_K, 3),
            "forward_fraction": round(res.forward_fraction, 3),
            "ff_split_half": round(sh["forward_fraction_split_half"], 3),
            "ff_split_sd": round(sh["std"], 3),
            "switch_rate": round(switch_rate(res.states), 2),
            "baseline_exit_ms": None if np.isnan(lat["latency_s"]) else round(lat["latency_s"] * 1000),
            "log_likelihood": round(float(res.log_likelihood), 1),
        })
        r = readouts[-1]
        print(f"    {region}: K={K} (scan best {best_k}, ARI@K {ari_at_K:.2f})  "
              f"ff={r['forward_fraction']:.2f}  ff_split={r['ff_split_half']:.2f}"
              f"+/-{r['ff_split_sd']:.2f}  switch={r['switch_rate']:.2f}")

        # --- macro-raster payload: trial-averaged, peak-normalised, latency-sorted PSTH ---
        psth = smooth_over_time(counts_to_rates(counts, data["config"]["bin_size_s"]),
                                sigma_bins).mean(axis=0).T          # (units, bins)
        peak = psth.max(axis=1, keepdims=True)
        norm = psth / np.where(peak > 0, peak, 1.0)
        order = np.argsort(np.argmax(norm, axis=1))                # by peak latency
        raster[region] = norm[order]

    return readouts, curves, {"eid": eid, "bins": bin_centers, "blocks": raster}


# --------------------------------------------------------------------------- figures


def fig_stability(curves: pd.DataFrame, align: str):
    """ARI-vs-K, 3 region panels, one line per session (+ mean). Shows whether K is supported."""
    fig, axes = plt.subplots(1, len(REGIONS), figsize=(4.4 * len(REGIONS), 4.0), sharey=True)
    axes = np.atleast_1d(axes)
    eids = sorted(curves["eid"].unique())
    for ax, region in zip(axes, REGIONS):
        sub = curves[curves["region"] == region]
        for j, e in enumerate(eids):
            s = sub[sub["eid"] == e].sort_values("K")
            shade = str(0.15 + 0.6 * j / max(len(eids) - 1, 1))    # light -> dark per session
            ax.plot(s["K"], s["ari_mean"], "-o", color=shade, lw=1.4, ms=4, label=e)
        # region-mean across sessions
        mean = sub.groupby("K")["ari_mean"].mean()
        ax.plot(mean.index, mean.values, color=region_color(region), lw=3,
                path_effects=stroke(), zorder=5, label="mean")
        ax.axvline(K_PER_REGION[region], color="0.5", ls=":", lw=1.2)
        ax.set_title(f"{region}  (canonical K={K_PER_REGION[region]})")
        ax.set_xlabel("number of states K")
    axes[0].set_ylabel("state stability  (ARI)")
    axes[-1].legend(frameon=False, fontsize=7, loc="upper right")
    fig.suptitle(f"HMM state-stability scan — {align}  (darker line = later session; thick = region mean)",
                 y=1.02)
    fig.tight_layout()
    return fig


def fig_forward_fraction(readouts: pd.DataFrame, align: str):
    """Split-half directedness, one point per session, mean bar. The certified result, replicated."""
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    rng = np.random.default_rng(0)
    for region in REGIONS:
        sub = readouts[readouts["region"] == region]
        x = REGIONS.index(region)
        c = region_color(region)
        ax.scatter(np.full(len(sub), x) + rng.uniform(-0.09, 0.09, len(sub)),
                   sub["ff_split_half"], s=48, color=c, edgecolors="black", linewidths=0.6,
                   zorder=3, label=f"{region} (n={len(sub)})")
        m = sub["ff_split_half"].mean()
        ax.plot([x - 0.2, x + 0.2], [m, m], color="black", lw=2.5, zorder=4)
    ax.axhline(0.5, color="0.4", ls=":", lw=1.2)
    ax.text(len(REGIONS) - 0.65, 0.505, "chance (undirected)", fontsize=8, color="0.4", va="bottom")
    ax.set(xticks=range(len(REGIONS)), xticklabels=REGIONS,
           ylabel="forward_fraction (split-half)", ylim=(0.45, 1.0),
           title=f"Directedness across sessions — {align}  (bar = mean)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def fig_macro_raster(rasters: list[dict], align: str):
    """3 vertical area blocks; sessions stacked within each, dark separator line, cividis rate."""
    # order sessions by trial count desc (stable, matches the docs' ordering) via eid appearance
    eids = [r["eid"] for r in rasters]
    bc = rasters[0]["bins"]
    heights = [sum(r["blocks"].get(reg, np.empty((0, 0))).shape[0] for r in rasters)
               for reg in REGIONS]

    fig, axes = plt.subplots(len(REGIONS), 1, figsize=(8.0, 9.5), sharex=True,
                             height_ratios=heights)
    for ax, region in zip(axes, REGIONS):
        stacked, bounds, labels, y = [], [], [], 0
        for r in rasters:
            blk = r["blocks"].get(region)
            if blk is None or blk.shape[0] == 0:
                continue
            stacked.append(blk)
            labels.append((y + blk.shape[0] / 2, r["eid"]))
            y += blk.shape[0]
            bounds.append(y)
        M = np.vstack(stacked)
        im = ax.imshow(M, aspect="auto", cmap=SEQUENTIAL, vmin=0, vmax=1,
                       extent=[bc[0], bc[-1], M.shape[0], 0])
        for b in bounds[:-1]:                                   # dark separator between sessions
            ax.axhline(b, color="black", lw=1.6)
        ax.axvline(0, color="white", lw=1.2)
        ax.set_yticks([yc for yc, _ in labels])                # session eid at each block centre
        ax.set_yticklabels([e for _, e in labels], fontsize=7)
        ax.tick_params(axis="y", length=0)
        ax.set_ylabel(f"{region}  ({M.shape[0]} units)", fontsize=11, fontweight="bold",
                      labelpad=48)
    axes[-1].set_xlabel("time from event (s)")
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="peak-normalised rate")
    fig.suptitle(f"Macro raster — trial-averaged sequence across 5 sessions x 3 areas  ({align})",
                 y=0.995)
    fig.subplots_adjust(left=0.17)
    return fig


# --------------------------------------------------------------------------- driver


def run_alignment(align: str, bin_ms: int, eid: str | None) -> None:
    tag = "stimOn" if align.startswith("stim") else "firstMovement"
    sessions = find_sessions(align, bin_ms, eid)
    if not sessions:
        print(f"  no pre300 tensors for align={tag} bin{bin_ms} "
              f"(build them: scripts/build_processed_all.py)")
        return
    print(f"\n{'='*70}\n{tag}: {len(sessions)} session(s)\n{'='*70}")

    all_read, all_curve, rasters = [], [], []
    for npz in sessions:
        print(f"\n=== {npz.name.split('__')[0]}")
        t0 = time.time()
        rd, cv, ras = run_one(npz, tag)
        all_read.extend(rd); all_curve.extend(cv); rasters.append(ras)
        print(f"  [{time.time()-t0:.0f}s]")

    read = pd.DataFrame(all_read)
    curves = pd.DataFrame(all_curve)
    OUT.mkdir(parents=True, exist_ok=True)
    # append per-alignment (both alignments share the files, distinguished by the `align` column)
    _append_csv(read, OUT / "readouts.csv", tag)
    _append_csv(curves, OUT / "stability_curves.csv", tag)

    pooled = (read.groupby("region")
              .agg(n_sessions=("eid", "nunique"),
                   best_k_mode=("best_k_scan", lambda s: int(s.mode().iloc[0])),
                   ff_split_mean=("ff_split_half", "mean"), ff_split_sd=("ff_split_half", "std"),
                   switch_mean=("switch_rate", "mean"), switch_sd=("switch_rate", "std"),
                   ari_at_K_mean=("ari_at_K", "mean"))
              .reindex(REGIONS).round(3))
    pooled.insert(0, "align", tag)
    _append_csv(pooled.reset_index(), OUT / "pooled.csv", tag)

    set_style()
    for name, fig in [("stability", fig_stability(curves, tag)),
                      ("forward_fraction", fig_forward_fraction(read, tag)),
                      ("macro_raster", fig_macro_raster(rasters, tag))]:
        p = savefig(fig, FIG_PHASE, f"{name}_{tag}", root=ROOT)
        plt.close(fig)
        print(f"  figure -> {p.relative_to(ROOT)}")

    print(f"\nPOOLED ({tag})\n{pooled.to_string()}")


def _append_csv(df: pd.DataFrame, path: Path, tag: str) -> None:
    """Write, replacing any existing rows for this alignment so re-runs stay idempotent."""
    if path.exists():
        old = pd.read_csv(path)
        old = old[old["align"] != tag] if "align" in old.columns else old
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(path, index=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--align", default="both", choices=["stimOn", "firstMovement", "both"])
    ap.add_argument("--bin-ms", type=int, default=40)
    ap.add_argument("--eid", default=None, help="run a single session (8-char prefix)")
    args = ap.parse_args()

    aligns = ["stimOn", "firstMovement"] if args.align == "both" else [args.align]
    for a in aligns:
        run_alignment(a, args.bin_ms, args.eid)
    print(f"\ndata  -> {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
