"""Population PSTH per area, split by stimulus side (left vs right), for one IBL session.

For the project's working session (5c0c560e) it plots, for each of {VISp, SCm, MRN}, the
population firing rate **averaged across neurons and trials** as a function of time from stimulus
onset, with a separate line for **left-stimulus** and **right-stimulus** trials (± 1 SEM across
trials). Stimulus side is read from the trial covariates frozen in the processed tensor:

    left-stim  : contrastLeft  > 0   (433 trials)
    right-stim : contrastRight > 0   (325 trials)
    excluded   : 0 %-contrast trials, which have no stimulus side (85 trials)

Raw 40 ms binned rates, no smoothing (the population + trial average is already smooth). Colours
are the Okabe-Ito blue/orange pair — a CVD-safe (red-green-safe) contrast, NOT red-vs-green.

    python scripts/psth_by_stim_side.py                 # default session 5c0c560e
    python scripts/psth_by_stim_side.py --eid <8char>   # another built session
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data.preprocess import load_processed  # noqa: E402
from src.viz.style import savefig, set_style, stroke  # noqa: E402

PROC = ROOT / "data" / "processed"
REGIONS = ["VISp", "SCm", "MRN"]
# CVD-safe categorical pair for the two conditions (blue vs orange; safe under red-green CVD)
COND_COLOR = {"left": "#0072B2", "right": "#E69F00"}


def population_psth(counts: np.ndarray, bin_s: float, mask: np.ndarray):
    """Mean ± SEM population rate (sp/s) over time for the trials in ``mask``.

    counts : (trials, bins, units) int. Rate = counts / bin_s. The population rate of one trial
    is the mean over units; we then average that over the selected trials and return its SEM.
    """
    rate = counts[mask] / bin_s                       # (n, bins, units) sp/s
    pop = rate.mean(axis=2)                            # (n, bins) population rate per trial
    n = pop.shape[0]
    mean = pop.mean(axis=0)
    sem = pop.std(axis=0, ddof=1) / np.sqrt(n)
    return mean, sem, n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eid", default="5c0c560e", help="session prefix (default 5c0c560e)")
    ap.add_argument("--align", default="stimOn")
    args = ap.parse_args()

    hits = sorted(PROC.glob(f"{args.eid}*__{args.align}_pre300_*_bin40_*.npz"))
    if not hits:
        raise SystemExit(f"no pre300 40 ms {args.align} tensor for {args.eid} in {PROC}")
    npz = hits[0]
    d = load_processed(npz)
    bc = d["bin_centers"]
    bin_s = d["config"]["bin_size_s"]

    # Drop any partial edge bin whose right edge falls outside the [-pre, +post] window: the
    # final bin here is half-width (centre 0.80 spans [0.78,0.82] but the window ends at 0.80),
    # so it carries ~half the spikes and would show a spurious cliff at the end of the PSTH.
    post_s = float(d["config"].get("post_s", bc[-1]))
    valid = bc + bin_s / 2.0 <= post_s + 1e-9
    n_trim = int((~valid).sum())
    bc = bc[valid]
    for r in d["regions"]:
        d["regions"][r]["counts"] = d["regions"][r]["counts"][:, valid, :]
    if n_trim:
        print(f"trimmed {n_trim} partial edge bin(s); plotting {len(bc)} full 40 ms bins "
              f"up to {bc[-1]:.2f} s")

    cL = np.nan_to_num(np.asarray(d["trials"]["contrastLeft"]))
    cR = np.nan_to_num(np.asarray(d["trials"]["contrastRight"]))
    masks = {"left": cL > 0, "right": cR > 0}
    n_zero = int(((cL == 0) & (cR == 0)).sum())
    print(f"session {args.eid}: left {int(masks['left'].sum())}  right "
          f"{int(masks['right'].sum())}  (excluded {n_zero} zero-contrast trials)")

    set_style()
    fig, axes = plt.subplots(1, len(REGIONS), figsize=(4.6 * len(REGIONS), 4.3))
    axes = np.atleast_1d(axes)
    for ax, region in zip(axes, REGIONS):
        if region not in d["regions"]:
            ax.set_title(f"{region} (absent)"); ax.axis("off"); continue
        counts = d["regions"][region]["counts"]
        n_units = counts.shape[2]
        for cond in ("left", "right"):
            mean, sem, n = population_psth(counts, bin_s, masks[cond])
            c = COND_COLOR[cond]
            ax.fill_between(bc, mean - sem, mean + sem, color=c, alpha=0.22, lw=0)
            ax.plot(bc, mean, color=c, lw=2.4, path_effects=stroke(),
                    label=f"{cond}-stim (n={n})")
        ax.axvline(0, color="black", ls="--", lw=1.3)
        ax.set_title(f"{region}  ({n_units} units)")
        ax.set_xlabel("time from stimulus onset (s)")
        ax.set_ylabel("population rate (sp/s)")
        ax.legend(frameon=False, fontsize=9, loc="upper left")

    fig.suptitle(f"Population PSTH by stimulus side — IBL session {args.eid} "
                 f"(mean ± SEM over trials; averaged over neurons)  ·  40 ms bins",
                 y=1.03, fontsize=12)
    fig.tight_layout()

    phase = f"recap_{args.eid}"
    p = savefig(fig, phase, "psth_by_stim_side", root=ROOT)
    plt.close(fig)
    print(f"figure -> {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
