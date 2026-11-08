"""Build a session's tensors from its DANDI NWB file and compare them with the frozen ONE tensors.

    python scripts/compare_nwb_vs_one.py                      # working session 5c0c560e
    python scripts/compare_nwb_vs_one.py --eid 111c1762-7908-47e0-9f40-2f2ee55b6505
    python scripts/compare_nwb_vs_one.py --one-alf-dir data/ONE/<lab>/Subjects/<subj>/<date>/001/alf

The NWB route (``src/data/load_nwb.py``) feeds the same ``preprocess_session`` with the same
configuration the frozen files were built with (``scripts/build_processed_all.py``: pre 0.3 s,
post 0.8 s, 40 ms bins, units >= 1 Hz, stimOn and firstMovement alignment). Any difference is
therefore a difference in the *data* the two routes serve, not in our code.

Reports, per alignment and region: usable trials, units, whether the unit selection matches, and
how many count entries / trials differ. With ``--one-alf-dir`` it also diffs the two trials tables
event by event. Writes ``results/nwb_vs_one/<eid8>.json``. Streams ~100-300 MB; a few minutes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.load_nwb import load_session_nwb  # noqa: E402
from src.data.preprocess import (  # noqa: E402
    WORKING_EID, PreprocessConfig, align_times, load_processed, preprocess_session, usable_mask,
)

ALIGNMENTS = ["stimOn_times", "firstMovement_times"]
EVENTS = ["stimOn_times", "firstMovement_times", "feedback_times", "goCue_times",
          "probabilityLeft", "choice", "feedbackType", "contrastLeft", "contrastRight"]


def compare_tensors(frozen: dict, built: dict, rows=None) -> dict:
    """Entry-by-entry comparison of the frozen (ONE) and NWB-built tensors.

    ``rows=(rows_frozen, rows_nwb)`` pairs trials by their original trial index, because the two
    routes can keep different trials. Units are paired by their original position in the region
    (``keep_idx``), so a unit kept by only one route is counted, not mis-paired.
    """
    out = {"n_trials_frozen": frozen["n_trials"], "n_trials_nwb": built["n_trials"], "regions": {}}
    for region, fr in frozen["regions"].items():
        b = built["regions"][region]
        row = {"units_frozen": int(fr["counts"].shape[2]), "units_nwb": int(b["counts"].shape[2]),
               "same_units": bool(np.array_equal(fr["keep_idx"], b["keep_idx"]))}
        common, ia, ib = np.intersect1d(fr["keep_idx"], b["keep_idx"], return_indices=True)
        row["units_compared"] = int(common.size)
        A, B = fr["counts"][:, :, ia], b["counts"][:, :, ib]
        if rows is not None:
            A, B = A[rows[0]], B[rows[1]]
            row["trials_compared"] = int(len(rows[0]))
        if A.shape == B.shape:
            diff = A != B
            row.update({
                "identical": bool(not diff.any()),
                "entries_differing": int(diff.sum()),
                "fraction_entries_differing": float(diff.mean()),
                "trials_differing": int(diff.any(axis=(1, 2)).sum()),
                "max_abs_count_diff": int(np.abs(A.astype(int) - B).max()) if diff.any() else 0,
            })
        else:
            row["identical"] = False
        out["regions"][region] = row
    return out


def matched_rows(one_trials, one_mask, nwb_trials, nwb_mask, align):
    """Rows of the frozen and NWB tensors holding the same original trial, plus the unmatched ones."""
    ia = np.flatnonzero(one_mask.to_numpy() & np.isfinite(one_trials[align].to_numpy(dtype=float)))
    ib = np.flatnonzero(nwb_mask.to_numpy() & np.isfinite(nwb_trials[align].to_numpy(dtype=float)))
    common = np.intersect1d(ia, ib)
    return (np.searchsorted(ia, common), np.searchsorted(ib, common),
            np.setdiff1d(ia, ib).tolist(), np.setdiff1d(ib, ia).tolist())


def compare_trials(one_alf_dir: Path, nwb_trials) -> dict:
    import pandas as pd

    one = pd.read_parquet(one_alf_dir / "_ibl_trials.table.pqt")
    if len(one) != len(nwb_trials):
        return {"n_one": len(one), "n_nwb": len(nwb_trials)}
    rows = {}
    for c in EVENTS:
        a, b = one[c].to_numpy(dtype=float), nwb_trials[c].to_numpy(dtype=float)
        both = np.isfinite(a) & np.isfinite(b)
        d = np.abs(a - b)
        rows[c] = {"nan_pattern_equal": bool(np.array_equal(np.isfinite(a), np.isfinite(b))),
                   "n_differing": int((d[both] > 0).sum()),
                   "n_differing_over_1ms": int((d[both] > 1e-3).sum()),
                   "max_abs_diff": float(d[both].max()) if both.any() else 0.0}
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--eid", default=WORKING_EID)
    ap.add_argument("--cache-dir", default=str(ROOT / "data" / "nwb_cache"))
    ap.add_argument("--one-alf-dir", default=None, help="ONE cache 'alf' folder of the session")
    args = ap.parse_args()

    base = PreprocessConfig(pre_s=0.3, post_s=0.8, bin_size_s=0.04, min_rate_hz=1.0)
    t0 = time.time()
    trials, mask, loads = load_session_nwb(args.eid, cfg=base, regions=list(base.regions),
                                           cache_dir=args.cache_dir)
    print(f"NWB loaded: {len(trials)} trials, {int(mask.sum())} pass the paper mask, "
          f"probes {[l['probe_name'] for l in loads]}  [{time.time() - t0:.0f}s]")

    report = {"eid": args.eid, "mask_passing_nwb": int(mask.sum()), "alignments": {}}
    one_trials = one_mask = None
    if args.one_alf_dir:
        # The ONE trials table from the local cache, masked by the same brainwidemap function.
        import pandas as pd
        from types import SimpleNamespace
        from brainwidemap import load_trials_and_mask
        raw = pd.read_parquet(Path(args.one_alf_dir) / "_ibl_trials.table.pqt")
        one_trials, one_mask = load_trials_and_mask(None, args.eid, min_rt=base.min_rt_s,
                                                    max_rt=base.max_rt_s,
                                                    sess_loader=SimpleNamespace(trials=raw))
        report["mask_passing_one"] = int(one_mask.sum())

    for align in ALIGNMENTS:
        cfg = PreprocessConfig(align_event=align, pre_s=0.3, post_s=0.8, bin_size_s=0.04,
                               min_rate_hz=1.0)
        frozen_path = ROOT / "data" / "processed" / f"{args.eid[:8]}__{cfg.tag()}.npz"
        frozen = load_processed(frozen_path)
        ev = align_times(trials, mask, align)
        built = preprocess_session(loads, ev, cfg, verbose=False)
        rows = None
        if one_trials is not None:
            ra, rb, only_one, only_nwb = matched_rows(one_trials, one_mask, trials, mask, align)
            rows = (ra, rb)
        res = compare_tensors(frozen, built, rows)
        if rows is not None:
            res["trials_only_in_one"], res["trials_only_in_nwb"] = only_one, only_nwb
        tt = trials[mask][usable_mask(trials, mask, align)]
        ra_, rb_ = rows if rows is not None else (slice(None), slice(None))
        if rows is not None or len(tt) == frozen["n_trials"]:
            res["covariates_identical"] = {
                k: bool(np.array_equal(np.asarray(frozen["trials"][k], float)[ra_],
                                       tt[k].to_numpy(dtype=float)[rb_], equal_nan=True))
                for k in ["choice", "feedbackType", "contrastLeft", "contrastRight"]}
        report["alignments"][align] = res
        print(f"\n{align}: trials frozen {res['n_trials_frozen']} / NWB {res['n_trials_nwb']}"
              + (f"; {len(rows[0])} matched, only ONE {res['trials_only_in_one']}, "
                 f"only NWB {res['trials_only_in_nwb']}" if rows is not None else ""))
        for region, r in res["regions"].items():
            print(f"  {region:5s} units {r['units_frozen']}/{r['units_nwb']} "
                  f"compared={r['units_compared']}  "
                  f"identical={r['identical']}  "
                  f"entries differing={r.get('entries_differing', 'n/a')} "
                  f"({100 * r.get('fraction_entries_differing', float('nan')):.4f} %)  "
                  f"trials differing={r.get('trials_differing', 'n/a')}")

    if args.one_alf_dir:
        from src.data.load_nwb import open_session, trials_table
        with open_session(args.eid, cache_dir=args.cache_dir) as s:
            report["trials_table_vs_one"] = compare_trials(Path(args.one_alf_dir), trials_table(s.nwbfile))
        print("\ntrials table, NWB vs ONE:")
        for c, r in report["trials_table_vs_one"].items():
            print(f"  {c:20s} {r}")

    out = ROOT / "results" / "nwb_vs_one" / f"{args.eid[:8]}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {out.relative_to(ROOT)}  [{time.time() - t0:.0f}s total]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
