"""Access layer for the IBL Brain-Wide Map (DANDI:000409, via ONE-api / openalyx).

Three layers are involved, and it helps to keep them straight:

  * **Alyx** is IBL's experiment-metadata database (a Django REST API). It records
    *what* was recorded -- subjects, sessions, probe insertions, which datasets
    exist -- but not the neural data itself.
  * **openalyx** is the public, read-only Alyx mirror. ONE-api ships its URL and a
    shared account (``intbrainlab``) as *defaults*, so no personal account or
    registration is needed. Verified 2026-07-08.
  * **ONE-api** is the Python client. It queries openalyx for metadata, then fetches
    the actual arrays over HTTP from the Flatiron data server and caches them on disk.

Everything in this module is metadata-only and cheap **except** :func:`load_units`,
which downloads spike sorting for one probe insertion.

References
----------
IBL et al. 2025, Nature 645:177-191 (the Brain-Wide Map).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

_log = logging.getLogger(__name__)

# --- Public openalyx. These are ONE-api's own defaults (one/params.py::default). ---
PUBLIC_ALYX_URL = "https://openalyx.internationalbrainlab.org"
PUBLIC_ALYX_PASSWORD = "international"

# --- Reproducibility pins -------------------------------------------------------
# `bwm_units` asserts a hash of the unit UUIDs against this data freeze, i.e. it
# guarantees we analyse exactly the units the paper analysed.
PAPER_FREEZE = "2023_12_bwm_release"

# The freeze above is only reproducible if the *aggregate tables* come from the
# matching tag. The installed brainwidemap defaults to "2026_Q2_IBL_et_al_BWM",
# which (a) has no md5 entry for the 'trials' table -> KeyError, and (b) yields a
# unit set that fails the paper's UUID-hash assertion. Pinning to 2024_Q2 fixes both.
# See docs/worklog/2026-07-08_phase0-setup.md for the full trace.
AGGREGATE_TAG = "2024_Q2_IBL_et_al_BWM"

# Expected hash of the frozen unit set, as asserted inside brainwidemap.bwm_units.
# Recorded here so a future upstream change that silently breaks the pin is visible.
PAPER_UNIT_UUID_HASH = "d16d0b38d392b18c0ce8b615ec89d60d7c901df2eeb3432986b62130af28ef01"

_TAG_PATCH_APPLIED = False


def _pin_aggregate_tag(tag: str = AGGREGATE_TAG) -> None:
    """Force ``brainwidemap`` to fetch aggregate tables from ``tag``.

    Works around an upstream bug: ``download_aggregate_tables`` defaults to the
    2026_Q2 tag, whose ``md5_expected`` dict lacks a ``'trials'`` key, so the
    integrity check raises ``KeyError`` *after* a successful download. Pinning to
    the 2024_Q2 tag both avoids the KeyError and restores agreement with
    ``PAPER_FREEZE``.

    Idempotent. Called automatically by :func:`get_one`.
    """
    global _TAG_PATCH_APPLIED
    if _TAG_PATCH_APPLIED:
        return

    import brainwidemap
    import brainwidemap.bwm_loading as bl

    original = bl.download_aggregate_tables

    def _pinned(one, target_path=None, type="clusters", tag=None, overwrite=False):
        if tag is not None and tag != _pinned.tag:
            _log.debug("Overriding requested aggregate tag %s -> %s", tag, _pinned.tag)
        return original(
            one, target_path=target_path, type=type, tag=_pinned.tag, overwrite=overwrite
        )

    _pinned.tag = tag
    # bwm_units resolves this name as a module global at call time, so patching
    # bwm_loading is what actually matters; the package-level alias is for callers.
    bl.download_aggregate_tables = _pinned
    brainwidemap.download_aggregate_tables = _pinned
    _TAG_PATCH_APPLIED = True
    _log.info("Pinned brainwidemap aggregate tables to tag=%s", tag)


def get_one(cache_dir: str | Path | None = None, mode: str = "remote"):
    """Return an authenticated ONE client against the public openalyx mirror.

    No personal account is required. Also applies :func:`_pin_aggregate_tag`.

    Parameters
    ----------
    cache_dir
        Where downloaded data is cached. ``None`` uses ONE's default
        (``~/Downloads/ONE``). Point this at a large disk if space is tight.
    mode
        ``'remote'`` queries openalyx live; ``'local'`` uses only the on-disk cache.
    """
    from one.api import ONE

    _pin_aggregate_tag()
    kwargs = {}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    return ONE(
        base_url=PUBLIC_ALYX_URL,
        password=PUBLIC_ALYX_PASSWORD,
        silent=True,
        mode=mode,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Metadata tables (cheap; the aggregate tables total ~100 MB, cached once)
# ---------------------------------------------------------------------------


def sessions_table(one) -> pd.DataFrame:
    """Released BWM sessions and probe insertions.

    Returns a DataFrame with columns
    ``['pid', 'eid', 'probe_name', 'session_number', 'date', 'subject', 'lab']``.
    Expect 699 insertions across 459 sessions for ``PAPER_FREEZE``.
    """
    from brainwidemap import bwm_query

    return bwm_query(one, freeze=PAPER_FREEZE)


def units_table(one) -> pd.DataFrame:
    """Every analysis-grade unit in the BWM, with its Beryl region.

    Applies the paper's own unit/session criteria and asserts the unit-UUID hash,
    so this is *exactly* the population the paper analysed: 62,990 units across
    210 Beryl regions and 454 sessions.

    Key columns: ``Beryl`` (region acronym), ``eid``, ``pid``, ``cluster_id``,
    ``firing_rate``, ``x``/``y``/``z`` (CCF coordinates).
    """
    from brainwidemap import bwm_units

    _pin_aggregate_tag()
    return bwm_units(one, freeze=PAPER_FREEZE)


def units_per_region(unit_df: pd.DataFrame) -> pd.DataFrame:
    """Good-unit yield per (session, probe, region).

    This is the table Phase 1 prunes on. Yields are *per insertion*, because
    single-trial methods need a simultaneously recorded population.
    """
    out = (
        unit_df.groupby(["eid", "pid", "Beryl"], observed=True)
        .size()
        .reset_index(name="n_units")
        .sort_values("n_units", ascending=False)
    )
    return out.reset_index(drop=True)


def region_cooccurrence(
    unit_df: pd.DataFrame,
    regions: Sequence[str],
    min_units_per_region: int = 20,
    min_regions_per_session: int = 2,
) -> pd.DataFrame:
    """Sessions in which >= ``min_regions_per_session`` target regions co-occur.

    "Co-occur" means recorded in the *same session* (``eid``), hence
    simultaneously. Units may come from either probe of that session.

    Returns one row per qualifying session, with the regions found and their
    unit counts, sorted by (n_regions, total_units) descending.
    """
    per_region = (
        unit_df[unit_df["Beryl"].isin(regions)]
        .groupby(["eid", "Beryl"], observed=True)
        .size()
        .reset_index(name="n_units")
    )
    per_region = per_region[per_region["n_units"] >= min_units_per_region]

    rows = []
    for eid, grp in per_region.groupby("eid", observed=True):
        if len(grp) < min_regions_per_session:
            continue
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

    if not rows:
        return pd.DataFrame(columns=["eid", "n_regions", "total_units", "regions", "units"])

    return (
        pd.DataFrame(rows)
        .sort_values(["n_regions", "total_units"], ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Heavy loaders (these hit the Flatiron data server)
# ---------------------------------------------------------------------------


def load_units(one, pid: str, qc: float = 1.0):
    """Download spike trains + cluster info for the good units of one insertion.

    .. warning:: This downloads data (order of ~100 MB-1 GB per insertion).

    Returns
    -------
    spikes : dict
        Keys ``['times', 'clusters', 'amps', 'depths']``, already filtered to good
        units. ``spikes['clusters']`` indexes directly into ``clusters`` rows.
    clusters : pandas.DataFrame
        One row per good unit; ``clusters['acronym']`` is its region.
    """
    from brainwidemap import load_good_units

    return load_good_units(one, pid, qc=qc)


def load_trials(one, eid: str, min_rt: float = 0.08, max_rt: float = 2.0):
    """Trials table plus the paper's inclusion mask for one session.

    Returns
    -------
    trials : pandas.DataFrame
        Includes ``stimOn_times``, ``firstMovement_times``, ``feedback_times``,
        ``choice``, ``feedbackType``, ``contrastLeft``/``contrastRight``,
        ``probabilityLeft``.
    mask : pandas.Series
        Boolean; ``True`` for trials passing the paper's criteria (reaction time in
        range, no NaN in the key events). Always apply it before analysis.
    """
    from brainwidemap import load_trials_and_mask

    return load_trials_and_mask(one, eid, min_rt=min_rt, max_rt=max_rt)


def dataset_size_bytes(one, pid: str) -> pd.DataFrame:
    """Metadata-only: how big is the spike sorting for this insertion?

    Queries Alyx for dataset file sizes **without downloading anything**. Useful for
    budgeting disk/bandwidth before committing to a download.
    """
    eid, pname = one.pid2eid(pid)
    dsets = one.alyx.rest(
        "datasets",
        "list",
        session=str(eid),
        no_cache=True,
    )
    rows = [
        {
            "name": d["name"],
            "collection": d.get("collection"),
            "size_bytes": d.get("file_size") or 0,
        }
        for d in dsets
    ]
    df = pd.DataFrame(rows)
    if not df.empty and pname:
        df = df[df["collection"].fillna("").str.contains(str(pname), regex=False)]
    return df.reset_index(drop=True)
