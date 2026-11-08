"""Monkey HMM under THREE independent alignments: stimulus, delay, response.

    python scripts/run_monkey_hmm_alignments.py [--session ses-2022-08-20]

Fits a Poisson HMM per area (FEF, DMFC) three times, each time re-zeroing every trial on a different
task event — **stimulus onset** (col 1), **delay onset** (col 2), **response onset** (col 4) — so we
can see whether the state structure re-registers around each anchor. Because this task's timing is
near-deterministic (fixation 0.4 s, then two ~1.0 s intervals), the three alignments are close to a
constant shift of one another; the honest expectation is that the occupancy is the *same shape*
re-referenced, and only the (mildly variable) response/reward end moves. Reported either way.

K is fixed across alignments for a clean comparison (FEF 4, DMFC 3 — the delay-aligned ARI choice);
`bias_startprob=False` (no imposed baseline). Raw counts, 100 ms bins.

Saves:
    results/figures/monkey_hmm/<session>_alignment_comparison.png   3 alignments x 2 areas occupancy
    results/hmm/monkey_<session>_alignments.csv                      read-outs per alignment x area
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data.monkey import load_monkey_aligned, EVENT_NAMES  # noqa: E402
from src.dynamics.hmm import fit_poisson_hmm, switch_rate  # noqa: E402
from src.viz.style import set_style, state_colors, savefig, stroke  # noqa: E402

# alignment name -> (event column, pre s, post s). Each window ~covers the whole task, re-anchored.
ALIGN = {
    "stimulus": (1, 0.4, 3.0),
    "delay":    (2, 1.4, 2.0),
    "response": (4, 2.5, 1.0),
}
K_PER_AREA = {"FEF": 4, "DMFC": 3}     # fixed across alignments (delay-aligned ARI choice)
AREAS = ["FEF", "DMFC"]
PHASE = "monkey_hmm"
SEED = 0


def _phase_lines(ax, ev_med, anchor_col, lo, hi, top=1.03):
    """Draw the phase-onset lines inside [lo, hi] (anchor bold); ``ev_med`` is anchor-relative."""
    for c in range(6):                       # fixation..reward onsets
        t = ev_med[c]
        if not (lo <= t <= hi):
            continue
        anchor = (c == anchor_col)
        ax.axvline(t, color="black" if anchor else "0.6", ls="--",
                   lw=2.2 if anchor else 1.0, zorder=1)
        ax.text(t, top, EVENT_NAMES[c], rotation=90, va="bottom", ha="center",
                fontsize=6.5, color="black" if anchor else "0.45")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="ses-2022-08-20")
    args = ap.parse_args()
    set_style()
    (ROOT / "results" / "hmm").mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(ALIGN), len(AREAS), figsize=(13, 10), squeeze=False)
    rows = []
    for r, (aname, (col, pre, post)) in enumerate(ALIGN.items()):
        mk = load_monkey_aligned(args.session, align_col=col, pre=pre, post=post)
        bc, tensors, events, nT = mk["bin_centers"], mk["tensors"], mk["events"], mk["n_trials"]
        ev_med = np.nanmedian(events, axis=0)          # anchor-relative event medians
        print(f"\n=== align: {aname} (col {col})  kept {nT} trials, {len(bc)} bins "
              f"[{bc[0]:.2f},{bc[-1]:.2f}]s ===")
        for cix, area in enumerate(AREAS):
            K = K_PER_AREA[area]
            baseline_mask = bc < ev_med[1] if col != 1 else bc < 0.0     # pre-stimulus = baseline
            res = fit_poisson_hmm(tensors[area], K, baseline_mask, n_iter=50, seed=SEED,
                                  bias_startprob=False)
            res.bin_centers = bc
            sw = switch_rate(res.states)
            rows.append({"alignment": aname, "area": area, "K": K, "n_trials": nT,
                         "n_units": tensors[area].shape[2],
                         "forward_fraction": round(res.forward_fraction, 3),
                         "switch_rate": round(sw, 2)})
            print(f"  {area}: K={K}  fwd_frac={res.forward_fraction:.3f}  switch={sw:.2f}/trial")

            ax = axes[r][cix]
            cols = state_colors(K)
            for s in range(K):
                ax.plot(bc, res.occupancy[s], color=cols[s], lw=2.0, label=f"S{s}",
                        path_effects=stroke())
            _phase_lines(ax, ev_med, col, bc[0], bc[-1], top=1.03)
            ax.set(ylim=(-0.02, 1.10), xlim=(bc[0], bc[-1]))
            ax.set_xlabel(f"time from {aname.upper()} onset (s)")   # each row = its own alignment
            ax.set_ylabel(f"align: {aname}\nP(state)" if cix == 0 else "P(state)")
            if r == 0:
                ax.set_title(f"{area}", fontsize=12)
            ax.legend(fontsize=7, ncol=K, loc="lower right")

    fig.suptitle(f"{args.session}: HMM occupancy under 3 independent alignments "
                 f"(bold dashed = the anchor event)", y=0.995, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    figp = savefig(fig, PHASE, f"{args.session}_alignment_comparison", root=ROOT)
    plt.close(fig)

    df = pd.DataFrame(rows)
    csvp = ROOT / "results" / "hmm" / f"monkey_{args.session}_alignments.csv"
    df.to_csv(csvp, index=False)
    print("\n=== read-outs (alignment x area) ===")
    print(df.to_string(index=False))
    print(f"\nfigure -> {figp.relative_to(ROOT)}")
    print(f"table  -> {csvp.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
