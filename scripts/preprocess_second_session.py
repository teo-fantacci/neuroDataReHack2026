"""Download the second triplet session (111c1762) and freeze it through the SAME
preprocessing pipeline as the working session. Mirrors notebooks/01_preprocessing.ipynb
cell 23, so the two sessions are guaranteed to be processed identically.

Run once. Downloads ~hundreds of MB of spike sorting on first call (cached in data/ONE/).
"""
import sys, time, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.access import get_one
from src.data.preprocess import (
    PreprocessConfig, WORKING_EID, load_session, align_times, usable_mask,
    preprocess_session, save_processed, trial_timing_table, window_occupancy,
    unit_yield_table,
)

EID = "111c1762-7908-47e0-9f40-2f2ee55b6505"   # clean VISp, full population; user pick 2026-07-14

one = get_one(cache_dir=ROOT / "data" / "ONE")

t0 = time.time()
print(f"loading session {EID[:8]} (downloads spikes on first run)...")
trials, mask, probe_loads = load_session(one, EID)
print(f"[{time.time()-t0:.0f}s]  {int(mask.sum())} usable trials, {len(probe_loads)} probes")

# --- diagnostics that decide whether the frozen window means the same thing here ---------
print("\nTRIAL TIMING (does the 0.4/0.8 window mean the same thing as in 5c0c560e?)")
print(trial_timing_table(trials, mask).to_string(index=False))
print("\nWINDOW OCCUPANCY at pre=0.4 post=0.8")
print(window_occupancy(trials, mask, post_s=0.8, pre_s=0.4).to_string(index=False))

print("\nUNIT YIELD vs rate threshold (is the 1 Hz cut still ~free here?)")
ev0 = align_times(trials, mask, "stimOn_times")
print(unit_yield_table(probe_loads, ev0, PreprocessConfig()).to_string(index=False))

# --- freeze both alignments, exactly as notebook 01 does --------------------------------
OUT = ROOT / "data" / "processed"
written = []
for align in ("stimOn_times", "firstMovement_times"):
    cfg = PreprocessConfig(align_event=align)   # defaults ARE the frozen config
    ev = align_times(trials, mask, align)
    keep = usable_mask(trials, mask, align)
    print(f"\n=== {align}  ({len(ev)} trials; {int((~keep).sum())} dropped for NaN event) ===")

    data = preprocess_session(probe_loads, ev, cfg)

    tt = trials[mask][keep]
    data["trials"] = {
        "choice": tt["choice"].to_numpy(),
        "feedbackType": tt["feedbackType"].to_numpy(),
        "contrastLeft": tt["contrastLeft"].to_numpy(),
        "contrastRight": tt["contrastRight"].to_numpy(),
        "rt": (tt["firstMovement_times"] - tt["stimOn_times"]).to_numpy(),
    }
    path = OUT / f"{EID[:8]}__{cfg.tag()}.npz"
    save_processed(data, path)
    written.append(path)
    print(f"  -> {path.relative_to(ROOT)}")

print(f"\nDONE [{time.time()-t0:.0f}s]. Frozen tensors for {EID[:8]}:")
for p in written:
    print("   ", p.relative_to(ROOT))
