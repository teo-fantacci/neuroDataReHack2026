"""Phase 1 — choose the brain regions we will actually analyse.

Everything here is **metadata only**: it reads the cached BWM aggregate tables and
downloads no neural data.

The governing constraint is that single-trial methods (HMM, rSLDS, geometry, directed
flow) need *simultaneously recorded* populations. So the unit of selection is the
**session** (``eid``), not the region: a region is only usable if it was recorded at the
same time as the other regions we want to compare it against.

Key functions
-------------
``session_trial_counts``   usable trials per session (the paper's ``bwm_include`` flag)
``cooccurrence_sweep``     how region count varies with the units-per-region threshold
``best_sessions``          ranked sessions meeting unit + trial criteria
``region_set_frequency``   which region combinations recur across sessions
``top_unconstrained_sessions``  best simultaneous coverage over *all* Beryl regions
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


def session_trial_counts(one) -> pd.DataFrame:
    """Usable trials per session, using the paper's own ``bwm_include`` flag.

    Reads the cached aggregate trials table -- no per-session download.

    Returns columns ``['eid', 'n_trials_total', 'n_trials_usable']``.
    """
    from brainwidemap.bwm_loading import download_aggregate_tables

    path = download_aggregate_tables(one, type="trials")
    trials = pd.read_parquet(path)
    out = trials.groupby("eid").agg(
        n_trials_total=("bwm_include", "size"),
        n_trials_usable=("bwm_include", "sum"),
    )
    return out.reset_index().astype({"n_trials_usable": int})


def _region_units_per_session(unit_df: pd.DataFrame, regions: Sequence[str] | None) -> pd.DataFrame:
    """Good units per (eid, region), pooling a session's probes (they are simultaneous)."""
    df = unit_df if regions is None else unit_df[unit_df["Beryl"].isin(regions)]
    return (
        df.groupby(["eid", "Beryl"], observed=True)
        .size()
        .reset_index(name="n_units")
    )


def cooccurrence_sweep(
    unit_df: pd.DataFrame,
    regions: Sequence[str],
    thresholds: Iterable[int] = (10, 15, 20, 25, 30, 40, 50),
    max_regions: int = 6,
) -> pd.DataFrame:
    """How many sessions yield >= k target regions, as a function of the unit threshold?

    The threshold is the crux of Phase 1: lower it and more regions "co-occur", but each
    population gets too small to fit a population model. This makes that trade-off
    explicit rather than leaving it to a magic number in a config file.

    Returns a (threshold x k) table of session counts.
    """
    per = _region_units_per_session(unit_df, regions)
    rows = []
    for thr in thresholds:
        kept = per[per["n_units"] >= thr]
        sizes = kept.groupby("eid", observed=True).size()
        row = {"min_units": thr}
        for k in range(2, max_regions + 1):
            row[f">={k}_regions"] = int((sizes >= k).sum())
        row["max_regions_any_session"] = int(sizes.max()) if len(sizes) else 0
        rows.append(row)
    return pd.DataFrame(rows)


def best_sessions(
    unit_df: pd.DataFrame,
    regions: Sequence[str],
    trial_counts: pd.DataFrame | None = None,
    min_units_per_region: int = 20,
    min_regions_per_session: int = 2,
    min_trials: int = 0,
) -> pd.DataFrame:
    """Rank sessions by simultaneous coverage of the target regions.

    Returns one row per qualifying session with the regions present, their unit counts,
    and (if ``trial_counts`` given) the usable trial count.
    """
    per = _region_units_per_session(unit_df, regions)
    per = per[per["n_units"] >= min_units_per_region]

    rows = []
    for eid, grp in per.groupby("eid", observed=True):
        if len(grp) < min_regions_per_session:
            continue
        grp = grp.sort_values("n_units", ascending=False)
        rows.append(
            {
                "eid": eid,
                "n_regions": len(grp),
                "total_units": int(grp["n_units"].sum()),
                "min_region_units": int(grp["n_units"].min()),
                "regions": list(grp["Beryl"]),
                "units": list(grp["n_units"].astype(int)),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    if trial_counts is not None:
        out = out.merge(trial_counts, on="eid", how="left")
        out = out[out["n_trials_usable"].fillna(0) >= min_trials]

    sort_cols = ["n_regions", "min_region_units", "total_units"]
    return out.sort_values(sort_cols, ascending=False).reset_index(drop=True)


def region_set_frequency(
    unit_df: pd.DataFrame,
    regions: Sequence[str],
    set_size: int = 3,
    min_units_per_region: int = 20,
) -> pd.DataFrame:
    """Which specific region combinations recur across sessions?

    A combination recorded in several sessions lets us treat sessions as replicates,
    which matters far more for statistics than one session with slightly more units.
    """
    per = _region_units_per_session(unit_df, regions)
    per = per[per["n_units"] >= min_units_per_region]

    counter: Counter = Counter()
    for _, grp in per.groupby("eid", observed=True):
        present = sorted(grp["Beryl"])
        if len(present) < set_size:
            continue
        for combo in combinations(present, set_size):
            counter[combo] += 1

    if not counter:
        return pd.DataFrame(columns=["regions", "n_sessions"])

    return (
        pd.DataFrame(
            [{"regions": " + ".join(c), "n_sessions": n} for c, n in counter.items()]
        )
        .sort_values("n_sessions", ascending=False)
        .reset_index(drop=True)
    )


def top_unconstrained_sessions(
    unit_df: pd.DataFrame,
    min_units_per_region: int = 20,
    top_n: int = 10,
) -> pd.DataFrame:
    """Best simultaneous coverage across **all** Beryl regions, ignoring our axis.

    This answers a question the project plan never asked: is the ~3-region ceiling a
    property of *our chosen axis*, or of the IBL recording geometry itself? If some
    session yields 6 well-sampled regions -- just not the ones we picked -- then the
    axis is the problem, not the dataset, and the SHIELD pivot is not forced.
    """
    per = _region_units_per_session(unit_df, regions=None)
    per = per[per["n_units"] >= min_units_per_region]

    rows = []
    for eid, grp in per.groupby("eid", observed=True):
        grp = grp.sort_values("n_units", ascending=False)
        rows.append(
            {
                "eid": eid,
                "n_regions": len(grp),
                "total_units": int(grp["n_units"].sum()),
                "regions": list(grp["Beryl"]),
                "units": list(grp["n_units"].astype(int)),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["n_regions", "total_units"], ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
