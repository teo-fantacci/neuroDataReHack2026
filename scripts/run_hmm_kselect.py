"""Per-session HMM state-count (K) selection, then compare: is there a COMMON K?

    python scripts/run_hmm_kselect.py --bin-ms 40 --align stimOn

Motivation
----------
`run_hmm_sessions.py` takes K per region from ONE session's stability analysis
(`best_K_per_region.md`) and imposes it on all sessions. That can neither *discover* a common
K nor *explain* a divergence. This script instead runs the stability analysis **independently on
every session**, then asks whether the sessions agree.

For each session x region we run the ARI stability scan (fit two independent HMMs on disjoint
trial halves, decode a third held-out set, score the agreement with the Adjusted Rand Index;
see docs/knowledge/ARI_and_state_stability.md). Per session we take the best **non-trivial**
K (K>=3 — K=2 is the guaranteed baseline/evoked split, a documented trap).

Deciding a common K (the maximin rule)
--------------------------------------
A "common K" should be good for *every* session, not just on average. So for each candidate K we
take the **minimum ARI across sessions** (the worst session at that K) and pick the K that
maximises it. We then check whether that K is *within one fold-SD of each session's own peak*:

* if yes for all sessions -> a **defensible common K** (adopt it);
* if not -> the sessions **genuinely disagree**, and the table shows why (usually trial count:
  fewer trials -> flatter/noisier ARI -> lower or unstable best K).

Chaining into the pipeline
--------------------------
Writes `results/hmm/k_selection_<align>_bin<bin>.json` with, per region, each session's best K,
the consensus K, and whether the consensus is defensible. `run_hmm_sessions.py --k-json <that
file>` then fits each session at its **own** best K (falling back to the consensus for any
session not in the file), instead of the hard-coded `K_PER_REGION`.

Also writes results/figures/phase2_hmm/kselect_<align>_bin<bin>.png (ARI-vs-K, one line/session).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data.preprocess import load_processed  # noqa: E402
from src.dynamics.hmm import stability_scan  # noqa: E402
from src.viz.style import savefig, set_style, stroke  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

REGIONS = ["VISp", "SCm", "MRN"]
# Cost knobs are CLI-overridable (see main). Defaults are the "definitive" setting; a faster
# divergence SCREEN uses --repeats 3 --kmax 6.
KS_DEFAULT = [2, 3, 4, 5, 6, 7, 8]
N_REPEATS_DEFAULT = 5
N_RESTARTS = 2
N_ITER = 30
SEED = 0
# distinct markers per session so the plot survives greyscale / colour-vision deficiency
MARKERS = ["o", "s", "^", "D", "v", "P"]


def find_sessions(align: str, bin_ms: int) -> list[Path]:
    tag = "stimOn" if align.startswith("stim") else "firstMovement"
    hits = sorted(PROC.glob(f"*__{tag}_*_bin{bin_ms}_*.npz"))
    seen: dict[str, Path] = {}
    for p in hits:
        seen.setdefault(p.name.split("__")[0], p)
    return list(seen.values())


def scan_session(npz: Path, ks: list[int], n_repeats: int) -> dict:
    """ARI-vs-K per region for one session."""
    data = load_processed(npz)
    eid = npz.name.split("__")[0]
    out = {"eid": eid, "n_trials": int(data["n_trials"]), "regions": {}}
    for region in REGIONS:
        if region not in data["regions"]:
            continue
        counts = data["regions"][region]["counts"]
        t0 = time.time()
        st = stability_scan(counts, ks, n_repeats=n_repeats, n_iter=N_ITER, seed=SEED,
                            n_restarts=N_RESTARTS)
        ks = st["n_states"].to_numpy()
        ari = st["ari_mean"].to_numpy()
        std = st["ari_std"].to_numpy()
        nt = ks >= 3
        best_k = int(ks[nt][np.argmax(ari[nt])])
        out["regions"][region] = {
            "K": ks.tolist(), "ari": ari.tolist(), "ari_std": std.tolist(),
            "n_units": int(counts.shape[2]),
            "best_k": best_k,
            "best_ari": float(ari[list(ks).index(best_k)]),
            "best_std": float(std[list(ks).index(best_k)]),
        }
        print(f"    {region}: best K={best_k}  ARI={out['regions'][region]['best_ari']:.3f}"
              f"  ({counts.shape[2]} units, {out['n_trials']} trials)  [{time.time()-t0:.0f}s]",
              flush=True)
    return out


def consensus_for_region(sessions: list[dict], region: str) -> dict:
    """Maximin common K: the K>=3 that maximises the minimum ARI across sessions."""
    present = [s for s in sessions if region in s["regions"]]
    if not present:
        return {}
    K = np.array(present[0]["regions"][region]["K"])
    nt = K >= 3
    # (n_sessions, n_K) ARI matrix
    A = np.vstack([np.array(s["regions"][region]["ari"]) for s in present])
    min_across = A[:, nt].min(axis=0)                    # worst session at each K>=3
    kk = K[nt]
    cons_k = int(kk[np.argmax(min_across)])

    # is cons_k within one fold-SD of every session's own peak?
    defensible, per_session = True, {}
    for s in present:
        rr = s["regions"][region]
        idx = rr["K"].index(cons_k)
        gap = rr["best_ari"] - rr["ari"][idx]
        within = gap <= rr["best_std"] + rr["ari_std"][idx]
        defensible &= bool(within)
        per_session[s["eid"]] = {
            "best_k": rr["best_k"], "best_ari": round(rr["best_ari"], 3),
            "ari_at_consensus": round(rr["ari"][idx], 3),
            "within_noise_of_peak": bool(within),
            "n_units": rr["n_units"], "n_trials": s["n_trials"],
        }
    agree = len({v["best_k"] for v in per_session.values()}) == 1
    return {
        "consensus_k": cons_k,
        "sessions_agree_on_best_k": agree,
        "consensus_defensible": bool(defensible),
        "per_session": per_session,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--align", default="stimOn", choices=["stimOn", "firstMovement"])
    ap.add_argument("--bin-ms", type=int, default=40)
    ap.add_argument("--repeats", type=int, default=N_REPEATS_DEFAULT,
                    help="A/B/C stability repeats per K (5=definitive, 3=faster screen)")
    ap.add_argument("--kmax", type=int, default=max(KS_DEFAULT),
                    help="largest K to scan (screen with 6; definitive with 8)")
    args = ap.parse_args()
    set_style()

    ks = [k for k in range(2, args.kmax + 1)]
    sessions_p = find_sessions(args.align, args.bin_ms)
    if len(sessions_p) < 2:
        print(f"Need >=2 sessions to compare; found {len(sessions_p)} at "
              f"align={args.align}, bin{args.bin_ms}. (build_processed_all.py produces them.)")
        return 1
    print(f"{len(sessions_p)} sessions, align={args.align}, bin{args.bin_ms}ms, "
          f"K={ks}, {args.repeats} repeats x {N_RESTARTS} restarts\n")

    sessions = []
    for npz in sessions_p:
        print(f"=== {npz.name.split('__')[0]}")
        sessions.append(scan_session(npz, ks, args.repeats))

    # ---- consensus per region ----
    result = {"align": args.align, "bin_ms": args.bin_ms,
              "n_sessions": len(sessions), "regions": {}}
    for region in REGIONS:
        result["regions"][region] = consensus_for_region(sessions, region)

    out = ROOT / "results" / "hmm" / f"k_selection_{args.align}_bin{args.bin_ms}.json"
    out.write_text(json.dumps(result, indent=1))

    # ---- figure: ARI vs K, one line per session, per region ----
    fig, axes = plt.subplots(1, len(REGIONS), figsize=(5 * len(REGIONS), 4.4), sharey=True)
    for ax, region in zip(axes, REGIONS):
        present = [s for s in sessions if region in s["regions"]]
        for si, s in enumerate(present):
            rr = s["regions"][region]
            K = np.array(rr["K"])
            ax.errorbar(K, rr["ari"], yerr=rr["ari_std"], marker=MARKERS[si % len(MARKERS)],
                        ms=5, capsize=2, lw=1.4, path_effects=stroke(),
                        label=f"{s['eid'][:8]} (K*={rr['best_k']}, {s['n_trials']}tr)")
        cons = result["regions"][region]
        if cons:
            ax.axvline(cons["consensus_k"], color="black", ls="--", lw=1.4,
                       label=f"consensus K={cons['consensus_k']}"
                             f"{' ✓' if cons['consensus_defensible'] else ' (weak)'}")
        ax.axvspan(1.6, 2.4, color="0.9", zorder=0)
        ax.axhline(0, color="black", lw=0.8, ls=":")
        ax.set(xlabel="K", title=region, xticks=ks)
        ax.legend(frameon=False, fontsize=7)
    axes[0].set_ylabel("Adjusted Rand Index (0 = chance)")
    fig.suptitle(f"Per-session state-count stability — {args.align}, {args.bin_ms} ms "
                 "(K=2 grey = trivial split)", y=1.0)
    fig.tight_layout()
    p = savefig(fig, "phase2_hmm", f"kselect_{args.align}_bin{args.bin_ms}", root=ROOT)

    # ---- report ----
    print("\n" + "=" * 74)
    print(f"COMMON-K ANALYSIS — {args.align}, {args.bin_ms} ms, {len(sessions)} sessions")
    print("=" * 74)
    for region in REGIONS:
        c = result["regions"][region]
        if not c:
            continue
        bests = {e: v["best_k"] for e, v in c["per_session"].items()}
        verdict = ("COMMON K = %d (defensible)" % c["consensus_k"]) if c["consensus_defensible"] \
            else ("consensus K=%d but NOT within-noise for all -> sessions diverge" % c["consensus_k"])
        print(f"\n{region}:  per-session best K = {bests}")
        print(f"        agree={c['sessions_agree_on_best_k']}  ->  {verdict}")
        for e, v in c["per_session"].items():
            flag = "" if v["within_noise_of_peak"] else "   <-- consensus far from this session's peak"
            print(f"          {e[:8]}: bestK={v['best_k']} (ARI {v['best_ari']}), "
                  f"ARI@consensus={v['ari_at_consensus']}, {v['n_trials']}tr{flag}")
    print(f"\njson   -> {out.relative_to(ROOT)}")
    print(f"figure -> {p}")
    print(f"\nNext: python scripts/run_hmm_sessions.py --bin-ms {args.bin_ms} "
          f"--align {args.align} --k-json {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
