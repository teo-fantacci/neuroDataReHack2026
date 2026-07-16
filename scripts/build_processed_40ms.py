"""Rebuild the frozen tensors at 40 ms (the unified bin), both alignments. Working session.

Supersedes the 50 ms tensors written by notebooks/00_preprocessing.ipynb. 40 ms is now the
project-wide bin: it passes the rSLDS ground truth in every region AND preserves SCm's HMM
state-stability (which degraded at 50 ms). See docs/worklog/2026-07-14_preprocessing-unification.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.access import get_one                                   # noqa: E402
from src.data.preprocess import (                                     # noqa: E402
    PreprocessConfig, WORKING_EID, align_times, load_session,
    preprocess_session, save_processed, usable_mask,
)

OUT = ROOT / "data" / "processed"


def main() -> None:
    one = get_one(cache_dir=ROOT / "data" / "ONE")
    trials, mask, probe_loads = load_session(one, WORKING_EID)

    for align in ("stimOn_times", "firstMovement_times"):
        cfg = PreprocessConfig(align_event=align, pre_s=0.4, post_s=0.8,
                               bin_size_s=0.04, min_rate_hz=1.0)
        ev = align_times(trials, mask, align)
        keep = usable_mask(trials, mask, align)
        print(f"\n=== {align}  ({len(ev)} trials) ===")

        data = preprocess_session(probe_loads, ev, cfg)

        tt = trials[mask][keep]
        data["trials"] = {
            "choice": tt["choice"].to_numpy(),
            "feedbackType": tt["feedbackType"].to_numpy(),
            "contrastLeft": tt["contrastLeft"].to_numpy(),
            "contrastRight": tt["contrastRight"].to_numpy(),
            "rt": (tt["firstMovement_times"] - tt["stimOn_times"]).to_numpy(),
        }
        path = OUT / f"{WORKING_EID[:8]}__{cfg.tag()}.npz"
        save_processed(data, path)
        print(f"  -> {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
