"""Download + preprocess ALL five {VISp, SCm, MRN} triplet sessions, at 40 ms, both alignments.

    python scripts/build_processed_all.py                 # all 5, bin 40 ms, both alignments
    python scripts/build_processed_all.py --eids 90c61c38 # just some (8-char prefixes ok)
    python scripts/build_processed_all.py --bin-ms 50     # a different bin

Generalises `build_processed_40ms.py` (working session only) to the whole replication set. Uses
the SAME unified pipeline (`src/data/preprocess.py`) so every session is frozen identically and
the tensors are directly comparable — the config is serialised inside each `.npz`.

The five sessions (from docs/knowledge/the_five_triplet_sessions.md):
    5c0c560e  steinmetz   843 usable trials   (already built)
    90c61c38  zador       656
    111c1762  churchland  364   (was built at 50 ms; re-built here at 40 ms)
    90d1e82c  danlab      353
    8a3a0197  churchland  290

Memory: each session loads ~16 M raw spikes (~0.5 GB) while binning. We process ONE session at
a time and release the spike tables before the next, so peak RAM is ~one session, not five.
Existing `.npz` are skipped unless --force.

Writes:  data/processed/<eid8>__stimOn_pre300_post800_bin40_rate1.npz   (+ firstMovement)
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.access import get_one  # noqa: E402
from src.data.preprocess import (  # noqa: E402
    PreprocessConfig,
    align_times,
    load_session,
    preprocess_session,
    save_processed,
    usable_mask,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"

# The five triplet sessions (full EIDs), in the doc's order.
TRIPLET_EIDS = [
    "5c0c560e-9e1f-45e9-b66e-e4ee7855be84",   # steinmetz  843
    "90c61c38-b9fd-4cc3-9795-29160d2f8e55",   # zador      656
    "111c1762-7908-47e0-9f40-2f2ee55b6505",   # churchland 364
    "90d1e82c-c96f-496c-ad4e-ee3f02067f25",   # danlab     353
    "8a3a0197-b40a-449f-be55-c00b23253bbf",   # churchland 290
]
ALIGNMENTS = ("stimOn_times", "firstMovement_times")


def free_gb() -> float:
    try:
        import psutil
        return round(psutil.virtual_memory().available / 1e9, 2)
    except Exception:
        return float("nan")


def build_one(one, eid: str, bin_ms: int, *, force: bool) -> None:
    tags = {a: PreprocessConfig(align_event=a, pre_s=0.3, post_s=0.8,
                                bin_size_s=bin_ms / 1000.0, min_rate_hz=1.0).tag()
            for a in ALIGNMENTS}
    paths = {a: OUT / f"{eid[:8]}__{tags[a]}.npz" for a in ALIGNMENTS}
    if not force and all(p.exists() for p in paths.values()):
        print(f"  both alignments already present — skipping (use --force to rebuild)")
        return

    t0 = time.time()
    trials, mask, probe_loads = load_session(one, eid)   # DOWNLOAD (cached) + load spikes
    print(f"  loaded: {int(mask.sum())} usable trials, "
          f"{sum(len(pl['clusters']) for pl in probe_loads)} good units  "
          f"[{time.time()-t0:.0f}s, {free_gb()} GB free]")

    for align in ALIGNMENTS:
        cfg = PreprocessConfig(align_event=align, pre_s=0.3, post_s=0.8,
                               bin_size_s=bin_ms / 1000.0, min_rate_hz=1.0)
        ev = align_times(trials, mask, align)
        keep = usable_mask(trials, mask, align)
        data = preprocess_session(probe_loads, ev, cfg, verbose=False)

        tt = trials[mask][keep]
        data["trials"] = {
            "choice": tt["choice"].to_numpy(),
            "feedbackType": tt["feedbackType"].to_numpy(),
            "contrastLeft": tt["contrastLeft"].to_numpy(),
            "contrastRight": tt["contrastRight"].to_numpy(),
            "rt": (tt["firstMovement_times"] - tt["stimOn_times"]).to_numpy(),
        }
        save_processed(data, paths[align])
        shapes = {r: data["regions"][r]["counts"].shape for r in data["regions"]}
        print(f"    {align:20s} -> {paths[align].name}   {shapes}")

    del probe_loads, trials, mask     # release ~0.5 GB of spikes before the next session
    gc.collect()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eids", nargs="*", default=None,
                    help="8-char (or full) EID prefixes; default = all five triplet sessions")
    ap.add_argument("--bin-ms", type=int, default=40)
    ap.add_argument("--force", action="store_true", help="rebuild even if the .npz exists")
    args = ap.parse_args()

    eids = TRIPLET_EIDS
    if args.eids:
        eids = [e for e in TRIPLET_EIDS if any(e.startswith(p) for p in args.eids)]
        if not eids:
            print(f"No triplet EID matches {args.eids}. Known: {[e[:8] for e in TRIPLET_EIDS]}")
            return 1

    OUT.mkdir(parents=True, exist_ok=True)
    one = get_one(cache_dir=ROOT / "data" / "ONE")
    print(f"building {len(eids)} session(s) at bin{args.bin_ms}ms, both alignments  "
          f"[{free_gb()} GB free]\n")

    for i, eid in enumerate(eids, 1):
        print(f"=== [{i}/{len(eids)}] {eid[:8]}")
        try:
            build_one(one, eid, args.bin_ms, force=args.force)
        except Exception as e:   # noqa: BLE001 — one bad session must not sink the batch
            print(f"  !! FAILED: {type(e).__name__}: {e}")
        print()

    print("done. Processed tensors now in data/processed/:")
    for p in sorted(OUT.glob(f"*_bin{args.bin_ms}_*.npz")):
        print(f"  {p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
