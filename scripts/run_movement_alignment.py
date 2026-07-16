"""Does an HMM state reflect MOVEMENT rather than the stimulus?

    python scripts/run_movement_alignment.py --bin-ms 40

The test (RT-jitter logic)
--------------------------
For each region we fit the same Poisson HMM twice: once on the **stimOn-aligned** tensor and
once on the **firstMovement-aligned** one. Then, per trial, we find the first time the
population *leaves its baseline set* (`baseline_states`), and look at how that exit time is
distributed **across trials** in each alignment.

The reaction time varies trial to trial (median 171 ms, but a wide spread). So:

* if a state's onset is **locked to the stimulus**, aligning to *movement* smears it by the RT
  jitter  ->  SD(exit | movement)  >  SD(exit | stimulus);
* if it is **locked to movement**, aligning to *movement* removes that jitter
  ->  SD(exit | movement)  <  SD(exit | stimulus).

So the sign of  SD_move - SD_stim  says which event the state tracks, and the RT spread itself
(SD_RT) sets the scale of the effect we could possibly see. This is the standard way to ask
"stimulus- or movement-locked" without assuming the answer.

Caveat baked in: in the movement alignment the 0.4 s "pre" window is mostly *post-stimulus,
pre-movement* activity, not a resting baseline. `baseline_states` still returns the most-occupied
pre-event states; read the movement-aligned "baseline" as "whatever preceded the move", not rest.

Consumes only the unified processed tensors (~1 MB each), so it is memory-safe and needs no
raw-spike download. Requires BOTH alignments to exist at the chosen bin.

Writes:  results/hmm/movement_alignment.csv
         results/figures/phase2_hmm/<eid>_occupancy_stim_vs_move.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data.preprocess import load_processed  # noqa: E402
from src.dynamics.hmm import baseline_states, fit_poisson_hmm  # noqa: E402
from src.viz.style import region_color, savefig, set_style, state_colors, stroke  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

REGIONS = ["VISp", "SCm", "MRN"]
K_PER_REGION = {"VISp": 3, "SCm": 4, "MRN": 3}
N_ITER = 50
SEED = 0


def first_exit_times(res, bin_centers) -> np.ndarray:
    """Per-trial time (s) of the first POST-event bin whose state is not a baseline state.

    NaN for a trial that never leaves baseline in the post-event window.
    """
    bstates = set(baseline_states(res.occupancy, bin_centers).tolist())
    states = res.states                                   # (n_trials, n_bins)
    post = np.where(bin_centers >= 0.0)[0]
    out = np.full(states.shape[0], np.nan)
    for i in range(states.shape[0]):
        for b in post:
            if states[i, b] not in bstates:
                out[i] = bin_centers[b]
                break
    return out


def process_session(eid: str, bin_ms: int) -> list[dict]:
    tag = f"pre400_post800_bin{bin_ms}_rate1"
    stim_p = PROC / f"{eid}__stimOn_{tag}.npz"
    move_p = PROC / f"{eid}__firstMovement_{tag}.npz"
    if not (stim_p.exists() and move_p.exists()):
        print(f"  {eid}: missing an alignment at bin{bin_ms} — skipping")
        return []

    stim = load_processed(stim_p)
    move = load_processed(move_p)
    rt = np.asarray(stim["trials"]["rt"])
    sd_rt_ms = float(np.nanstd(rt) * 1000)

    fig, axes = plt.subplots(2, len(REGIONS), figsize=(5 * len(REGIONS), 7), sharex="col")
    rows = []
    for j, region in enumerate(REGIONS):
        K = K_PER_REGION[region]
        res, exits, levels = {}, {}, {}
        for al, data in (("stimOn", stim), ("firstMovement", move)):
            counts = data["regions"][region]["counts"]
            r = fit_poisson_hmm(counts, K, data["baseline_mask"], n_iter=N_ITER, seed=SEED,
                                bias_startprob=True)
            r.bin_centers = data["bin_centers"]
            res[al] = r
            exits[al] = first_exit_times(r, data["bin_centers"])
            levels[al] = data["bin_centers"]

        sd_stim = float(np.nanstd(exits["stimOn"]) * 1000)
        sd_move = float(np.nanstd(exits["firstMovement"]) * 1000)
        # Which event is the exit tighter to? Lower SD wins.
        locked = "movement" if sd_move < sd_stim else "stimulus"
        rows.append({
            "eid": eid, "bin_ms": bin_ms, "region": region, "K": K,
            "sd_exit_stimOn_ms": round(sd_stim, 1),
            "sd_exit_move_ms": round(sd_move, 1),
            "sd_rt_ms": round(sd_rt_ms, 1),
            "tighter_to": locked,
            "sd_ratio_move_over_stim": round(sd_move / sd_stim, 2) if sd_stim else np.nan,
        })

        # --- occupancy, stimulus-aligned (top) vs movement-aligned (bottom) ---
        for row_i, al in enumerate(("stimOn", "firstMovement")):
            r = res[al]; bc = r.bin_centers
            bstates = set(baseline_states(r.occupancy, bc).tolist())
            cols = state_colors(K)
            for k in range(K):
                lbl = f"state {k}" + ("  (baseline)" if k in bstates else "")
                axes[row_i, j].plot(bc, r.occupancy[k], color=cols[k], label=lbl,
                                    path_effects=stroke())
            axes[row_i, j].axvline(0, color="0.3", ls="--", lw=1.2)
            axes[row_i, j].set_ylim(0, 1)
            if row_i == 0:
                axes[row_i, j].set_title(f"{region}  (K={K})")
            axes[row_i, j].set_ylabel(f"{al}\nP(state)" if j == 0 else "P(state)")
            axes[row_i, j].legend(frameon=False, fontsize=7, ncol=2)
        axes[1, j].set_xlabel("time from alignment event (s)")

    fig.suptitle(f"{eid} — state occupancy: stimulus-aligned (top) vs movement-aligned (bottom)  "
                 f"[bin {bin_ms} ms]", y=1.0)
    fig.tight_layout()
    savefig(fig, "phase2_hmm", f"{eid}_occupancy_stim_vs_move", root=ROOT)
    plt.close(fig)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bin-ms", type=int, default=40)
    args = ap.parse_args()
    set_style()

    tag = f"pre400_post800_bin{args.bin_ms}_rate1"
    eids = sorted({p.name.split("__")[0] for p in PROC.glob(f"*__stimOn_{tag}.npz")})
    if not eids:
        print(f"No stimOn tensors at bin{args.bin_ms} in {PROC}.")
        return 1

    all_rows = []
    for eid in eids:
        print(f"=== {eid}")
        all_rows.extend(process_session(eid, args.bin_ms))
        for r in all_rows:
            if r["eid"] == eid:
                print(f"    {r['region']}: exit SD  stim={r['sd_exit_stimOn_ms']}ms  "
                      f"move={r['sd_exit_move_ms']}ms  (RT spread {r['sd_rt_ms']}ms)  "
                      f"-> tighter to {r['tighter_to'].upper()}")

    if not all_rows:
        print("Nothing processed — need BOTH alignments at this bin.")
        return 1
    df = pd.DataFrame(all_rows)
    out = ROOT / "results" / "hmm" / "movement_alignment.csv"
    df.to_csv(out, index=False)
    print(f"\nsaved -> {out.relative_to(ROOT)}")
    print("\nReading it: 'tighter to MOVEMENT' means the state's onset is more tightly locked to")
    print("the wheel-turn than to the stimulus — i.e. that state reflects MOVEMENT. The RT spread")
    print("column is the scale of jitter that separates the two; a null result would be sd_move ~=")
    print("sqrt(sd_stim^2 + sd_rt^2), i.e. movement alignment strictly worse.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
