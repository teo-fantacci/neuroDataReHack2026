"""Do the HMM's states track the *phases of the task*? -- IBL brain-wide-map version.

    python scripts/run_hmm_state_phase.py            # all 5 triplet sessions, pool the stats
    python scripts/run_hmm_state_phase.py --eid 5c0c560e

This is the "less steps" cousin of the analysis planned for colleague D.'s monkey working-memory
data (which has fixation / sample / delay / go / response phases). IBL's stimOn-aligned trial
has only three phases we can define from events already stored in the processed tensor:

    phase 0 = pre-stimulus (baseline)      bin time < 0
    phase 1 = post-stim, pre-movement      0 <= t < RT
    phase 2 = post-movement                t >= RT              (RT = firstMovement - stimOn)

We fit the Poisson HMM per region (K from the frozen common-K result), then measure how much
the decoded state aligns with the task phase -- referenced to a **trial-permutation null** that
shuffles each trial's RT across trials. That null is the crux: both the state and the phase are
strong functions of absolute time, so raw mutual information is high for free; only alignment
*above* the null reflects the states tracking the trial-specific movement timing. We also
re-lock the state occupancy onto movement onset to see whether a state boundary follows RT.

Per-session-then-pool (never a pseudo-population -- those sessions are not simultaneous). The
claim is a replication: "state x phase NMI beats its shuffle null in x/5 sessions".

Outputs:
    results/hmm/state_phase.csv                     one row per session x region
    results/hmm/state_phase.json                    same, plus the per-session null summaries
    results/figures/hmm_state_phase/<eid>_<region>.png   P(state|phase) + movement-locked occ
"""

from __future__ import annotations

import argparse
import json
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

from src.data.preprocess import load_processed  # noqa: E402
from src.dynamics.hmm import (  # noqa: E402
    fit_poisson_hmm, state_locked_occupancy, state_phase_alignment, switch_rate,
)
from src.viz.style import SEQUENTIAL, savefig, set_style, state_colors, stroke  # noqa: E402

PROC = ROOT / "data" / "processed"
OUT = ROOT / "results" / "hmm"
FIG_PHASE = "hmm_state_phase"

REGIONS = ["VISp", "SCm", "MRN"]
K_PER_REGION = {"VISp": 3, "SCm": 3, "MRN": 3}   # frozen common-K result (05_hmm_summary)
PHASE_NAMES = ["pre-stim", "stim->move", "post-move"]
SEED, N_ITER, N_SHUFFLES = 0, 50, 200


def session_npz(eid8: str) -> Path:
    return PROC / f"{eid8}__stimOn_pre300_post800_bin40_rate1.npz"


def run_one(eid8: str, make_fig: bool = True) -> list[dict]:
    data = load_processed(session_npz(eid8))
    bin_centers = data["bin_centers"]
    baseline_mask = data["baseline_mask"]
    rt = np.asarray(data["trials"]["rt"], dtype=float)     # s, relative to stimOn (== align 0)
    n_trials = data["n_trials"]

    # landmarks for the phase labels: stimOn is the alignment event (t = 0 on every trial),
    # firstMovement is at t = RT. NaN RT (no detected movement) -> that trial has no phase-2.
    stim0 = np.zeros(n_trials)
    landmarks = np.column_stack([stim0, rt])               # (n_trials, 2)

    rows = []
    for region in REGIONS:
        counts = data["regions"][region]["counts"]
        K = K_PER_REGION[region]
        res = fit_poisson_hmm(counts, K, baseline_mask, n_iter=N_ITER, seed=SEED,
                              bias_startprob=True)
        res.bin_centers = bin_centers

        al = state_phase_alignment(res.states, bin_centers, landmarks, n_states=K,
                                   n_shuffles=N_SHUFFLES, seed=SEED)
        rows.append({
            "eid": eid8, "region": region, "K": K, "n_trials": int(n_trials),
            "forward_fraction": round(res.forward_fraction, 3),
            "switch_rate": round(switch_rate(res.states), 2),
            "nmi": round(al["nmi"], 4),
            "nmi_null_mean": round(al["nmi_null_mean"], 4),
            "nmi_null_std": round(al["nmi_null_std"], 4),
            "nmi_z": round(al["nmi_z"], 2),
            "nmi_p": al["nmi_p"],
            "cramers_v": round(al["cramers_v"], 3),
            "beats_null": bool(al["nmi_p"] < 0.05),
        })
        print(f"  {region}: NMI={al['nmi']:.3f}  null={al['nmi_null_mean']:.3f}"
              f"+/-{al['nmi_null_std']:.3f}  z={al['nmi_z']:+.1f}  p={al['nmi_p']:.3f}"
              f"  V={al['cramers_v']:.2f}")

        if make_fig:
            _figure(eid8, region, res, al, bin_centers, rt)

    return rows


def _figure(eid8, region, res, al, bin_centers, rt):
    K = res.states.max() + 1
    cols = state_colors(K)
    rel_t, occ = state_locked_occupancy(res.states, bin_centers, rt,
                                        n_states=K, win_s=(-0.35, 0.45))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 3.6))

    # left: P(state | phase) -- cividis heatmap, states x phases
    im = axL.imshow(al["p_state_given_phase"], aspect="auto", cmap=SEQUENTIAL,
                    vmin=0, vmax=1)
    axL.set_xticks(range(len(PHASE_NAMES)), PHASE_NAMES)
    axL.set_yticks(range(K), [f"S{i}" for i in range(K)])
    axL.set_ylabel("HMM state")
    axL.set_title(f"{region}  P(state | phase)\n"
                  f"NMI z={al['nmi_z']:+.1f} vs shuffle  (p={al['nmi_p']:.3f})")
    fig.colorbar(im, ax=axL, fraction=0.046, pad=0.04)

    # right: state occupancy re-locked to movement onset (t=0 = firstMovement)
    for s in range(K):
        axR.plot(rel_t, occ[s], color=cols[s], lw=2.0, label=f"S{s}",
                 path_effects=stroke())
    axR.axvline(0.0, color="0.3", ls="--", lw=1)
    axR.set_xlabel("time from movement onset (s)")
    axR.set_ylabel("P(state)")
    axR.set_ylim(-0.02, 1.02)
    axR.set_title(f"{region}  state occupancy locked to movement")
    axR.legend(loc="upper right", fontsize=8, ncol=K)

    fig.suptitle(f"{eid8} -- {region}: do states track task phase?", y=1.03)
    fig.tight_layout()
    path = savefig(fig, FIG_PHASE, f"{eid8}_{region}")
    plt.close(fig)
    print(f"    fig -> {path.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eid", default=None, help="8-char eid prefix; default = all present")
    args = ap.parse_args()

    set_style()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.eid:
        eids = [args.eid[:8]]
    else:
        eids = sorted({p.name[:8] for p in PROC.glob("*__stimOn_pre300_post800_bin40_rate1.npz")})

    t0 = time.time()
    all_rows = []
    for eid8 in eids:
        if not session_npz(eid8).exists():
            print(f"[skip] {eid8}: no stimOn tensor"); continue
        print(f"\n=== {eid8} ===")
        all_rows.extend(run_one(eid8))

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT / "state_phase.csv", index=False)
    (OUT / "state_phase.json").write_text(json.dumps(all_rows, indent=2))

    # pooled replication summary: how often does state x phase alignment beat its null?
    print("\n=== pooled across sessions (replication) ===")
    for region in REGIONS:
        sub = df[df.region == region]
        n_beat = int(sub["beats_null"].sum())
        print(f"  {region}: NMI beats shuffle null in {n_beat}/{len(sub)} sessions; "
              f"z = {sub['nmi_z'].mean():.1f} +/- {sub['nmi_z'].std():.1f}; "
              f"Cramer's V = {sub['cramers_v'].mean():.2f} +/- {sub['cramers_v'].std():.2f}")

    print(f"\nwrote {OUT/'state_phase.csv'}")
    print(f"[{time.time()-t0:.1f}s total]")


if __name__ == "__main__":
    main()
