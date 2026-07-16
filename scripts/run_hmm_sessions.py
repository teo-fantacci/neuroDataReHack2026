"""Run the HMM findings pipeline over every preprocessed session, and pool the statistics.

    python scripts/run_hmm_sessions.py

This is the multi-session version of `notebooks/01_hmm_triplet.ipynb`. It reads the **unified
preprocessed tensors** (`data/processed/<eid>__<align>_...npz`, produced by
`src/data/preprocess.py` / `notebooks/00_preprocessing.ipynb`), so every session is treated
identically and the results are directly comparable.

Why per-session-then-pool, and NOT a pseudo-population
------------------------------------------------------
A state-space model consumes *single-trial simultaneous* population activity. Stitching neurons
from different sessions into one vector destroys exactly that — those neurons were never recorded
together (see docs/worklog/2026-07-14 §5). So we **fit each session separately and pool the
statistics**: "forward_fraction 0.83 +/- 0.04 across n sessions" is a *replication*, a stronger
claim than one session with tighter error bars, and it is the animal-independence argument made
properly.

K per region is taken from the stability analysis (docs/knowledge/best_K_per_region.md); it is
NOT re-selected per session (that would need the slow stability scan on each). Pass
--stability to additionally re-derive K per session.

Read-outs, per session x region (all from src/dynamics/hmm.py):
  forward_fraction         pooled directedness (ordering on the same trials)
  fwd_frac_split_half      directedness with the ordering derived on HELD-OUT trials  <- quote this
  switch_rate              state changes / trial (a FREQUENCY, not a direction)
  baseline_exit_ms         when the population leaves its baseline set (robust; movement-confounded)

Writes:  results/hmm/hmm_sessions.csv        (one row per session x region)
         results/hmm/hmm_sessions_pooled.csv (mean +/- sd across sessions, per region)
         results/figures/phase2_hmm/sessions_forward_fraction.png
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data.preprocess import load_processed  # noqa: E402
from src.dynamics.hmm import (  # noqa: E402
    baseline_exit_latency_robust,
    fit_poisson_hmm,
    forward_fraction_split_half,
    stability_scan,
    switch_rate,
)
from src.viz.style import region_color, savefig, set_style, stroke  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

REGIONS = ["VISp", "SCm", "MRN"]
K_PER_REGION = {"VISp": 3, "SCm": 4, "MRN": 3}   # from docs/knowledge/best_K_per_region.md
N_ITER = 50
SEED = 0
STAB_KS = [2, 3, 4, 5, 6, 7, 8]


def find_sessions(align: str, bin_ms: int) -> list[Path]:
    """One processed .npz per session, for a FIXED alignment and bin size.

    Pinning the bin size matters: `data/processed/` can hold the same session at several bins
    (e.g. bin40 and bin50 while the 40-vs-50 ms question is open). Globbing them all would mix
    timebases and double-count a session in the pool. We therefore match exactly one bin, and
    if a session still has duplicates we keep the first and warn.
    """
    tag = "stimOn" if align.startswith("stim") else "firstMovement"
    hits = sorted(PROC.glob(f"*__{tag}_*_bin{bin_ms}_*.npz"))
    seen, out = {}, []
    for p in hits:
        eid = p.name.split("__")[0]
        if eid in seen:
            print(f"  ! duplicate for {eid} ignored: {p.name} (kept {seen[eid].name})")
            continue
        seen[eid] = p
        out.append(p)
    return out


def k_for(k_map: dict | None, eid: str, region: str, common_k: int | None) -> tuple[int, str]:
    """K to fit at, and where it came from.

    Precedence: --common-k (one K for every session/region, the defensible replication choice)
    -> --k-json per-session best -> --k-json consensus -> the hard-coded default.
    """
    if common_k is not None:
        return common_k, f"common({common_k})"
    if k_map and region in k_map.get("regions", {}):
        reg = k_map["regions"][region]
        ps = reg.get("per_session", {}).get(eid)
        if ps is not None:
            return int(ps["best_k"]), "per-session"
        if "consensus_k" in reg:
            return int(reg["consensus_k"]), "consensus"
    return K_PER_REGION[region], "default"


def run_one(npz: Path, *, do_stability: bool, k_map: dict | None = None, common_k: int | None = None) -> list[dict]:
    data = load_processed(npz)
    cfg = data["config"]
    bin_centers = data["bin_centers"]
    baseline_mask = data["baseline_mask"]
    eid = npz.name.split("__")[0]

    rows = []
    for region in REGIONS:
        if region not in data["regions"]:
            print(f"    {region}: absent in this session, skipping")
            continue
        counts = data["regions"][region]["counts"]

        k, k_src = k_for(k_map, eid, region, common_k)
        if do_stability:   # overrides the map: re-derive K on THIS session, inline
            st = stability_scan(counts, STAB_KS, n_repeats=3, n_iter=25, seed=SEED, n_restarts=2)
            ari = st["ari_mean"].to_numpy()
            ks = st["n_states"].to_numpy()
            nontrivial = ks >= 3
            k = int(ks[nontrivial][np.argmax(ari[nontrivial])])
            k_src = "stability(inline)"

        res = fit_poisson_hmm(counts, k, baseline_mask, n_iter=N_ITER, seed=SEED,
                              bias_startprob=True)
        sh = forward_fraction_split_half(counts, k, baseline_mask, n_repeats=5, seed=SEED,
                                         n_iter=N_ITER)
        lat = baseline_exit_latency_robust(res.occupancy, bin_centers)

        rows.append({
            "eid": eid,
            "align": cfg["align_event"],
            "bin_ms": int(cfg["bin_size_s"] * 1000),
            "region": region,
            "n_trials": counts.shape[0],
            "n_units": counts.shape[2],
            "K": k,
            "K_source": k_src,
            "forward_fraction": round(res.forward_fraction, 3),
            "fwd_frac_split_half": round(sh["forward_fraction_split_half"], 3),
            "fwd_frac_sh_std": round(sh["std"], 3),
            "switch_rate": round(switch_rate(res.states), 2),
            "baseline_exit_ms": None if np.isnan(lat["latency_s"]) else round(lat["latency_s"] * 1000),
        })
        r = rows[-1]
        print(f"    {region}: K={k}  ff={r['forward_fraction']:.2f}  "
              f"ff_split={r['fwd_frac_split_half']:.2f}+/-{r['fwd_frac_sh_std']:.2f}  "
              f"switch={r['switch_rate']:.2f}  exit={r['baseline_exit_ms']}ms")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--align", default="stimOn", choices=["stimOn", "firstMovement"],
                    help="which alignment's processed tensors to use")
    ap.add_argument("--bin-ms", type=int, default=40,
                    help="bin size (ms) to select; must match how the .npz were built (default 40)")
    ap.add_argument("--stability", action="store_true",
                    help="re-derive K per session via the stability scan (slow); overrides --k-json")
    ap.add_argument("--common-k", type=int, default=None,
                    help="fit EVERY session/region at this one K (the common-K replication). "
                         "Overrides --k-json. Use 3 (the defensible common state count).")
    ap.add_argument("--k-json", type=str, default=None,
                    help="k_selection_*.json from run_hmm_kselect.py: fit each session at its OWN "
                         "best K (per-session), falling back to the consensus. Default = the "
                         "hard-coded K_PER_REGION from best_K_per_region.md.")
    args = ap.parse_args()

    set_style()
    k_map = None
    if args.k_json:
        k_map = json.loads(Path(args.k_json).read_text())
        print(f"K from {args.k_json}  (per-session best K; consensus fallback)")

    sessions = find_sessions(args.align, args.bin_ms)
    if not sessions:
        print(f"No processed '{args.align}' tensors at bin{args.bin_ms} in {PROC}.")
        print("Produce them first with scripts/build_processed_all.py.")
        return 1

    print(f"{len(sessions)} session(s) for align={args.align}, bin{args.bin_ms}ms:")
    for s in sessions:
        print(f"  {s.name}")

    all_rows = []
    for npz in sessions:
        print(f"\n=== {npz.name.split('__')[0]}")
        t0 = time.time()
        all_rows.extend(run_one(npz, do_stability=args.stability, k_map=k_map, common_k=args.common_k))
        print(f"  [{time.time()-t0:.0f}s]")

    df = pd.DataFrame(all_rows)
    out = ROOT / "results" / "hmm"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "hmm_sessions.csv", index=False)

    # ---- pool the statistics across sessions, per region ----
    n_sessions = df["eid"].nunique()
    pooled = (
        df.groupby("region")
        .agg(n_sessions=("eid", "nunique"),
             K_mode=("K", lambda s: int(s.mode().iloc[0])),
             forward_fraction_mean=("forward_fraction", "mean"),
             forward_fraction_sd=("forward_fraction", "std"),
             fwd_split_mean=("fwd_frac_split_half", "mean"),
             fwd_split_sd=("fwd_frac_split_half", "std"),
             switch_rate_mean=("switch_rate", "mean"),
             switch_rate_sd=("switch_rate", "std"))
        .reindex(REGIONS)
        .round(3)
    )
    pooled.to_csv(out / "hmm_sessions_pooled.csv")

    # ---- figure: forward_fraction (split-half) per region, one point per session ----
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for region in REGIONS:
        sub = df[df["region"] == region]
        x = list(REGIONS).index(region)
        c = region_color(region)
        ax.scatter(np.full(len(sub), x) + np.random.default_rng(0).uniform(-0.08, 0.08, len(sub)),
                   sub["fwd_frac_split_half"], s=45, color=c, edgecolors="black",
                   linewidths=0.6, zorder=3, label=f"{region} (n={len(sub)})")
        m = sub["fwd_frac_split_half"].mean()
        ax.plot([x - 0.2, x + 0.2], [m, m], color="black", lw=2, zorder=4)
    ax.axhline(0.5, color="0.4", ls=":", lw=1.2)
    ax.text(2.35, 0.505, "chance (undirected)", fontsize=8, color="0.4", va="bottom", ha="right")
    ax.set(xticks=range(len(REGIONS)), xticklabels=REGIONS, ylabel="forward_fraction (split-half)",
           ylim=(0.45, 1.0),
           title=f"Directedness across {n_sessions} session(s) — bar = mean")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    p = savefig(fig, "phase2_hmm", "sessions_forward_fraction", root=ROOT)

    print("\n" + "=" * 70)
    print(f"POOLED across {n_sessions} session(s)")
    print("=" * 70)
    print(pooled.to_string())
    print(f"\nrows   -> {(out / 'hmm_sessions.csv').relative_to(ROOT)}")
    print(f"pooled -> {(out / 'hmm_sessions_pooled.csv').relative_to(ROOT)}")
    print(f"figure -> {p}")
    if n_sessions == 1:
        print("\nNOTE: only one session present. Pooling is trivial until the other triplet")
        print("sessions are preprocessed (the other chat is producing them).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
