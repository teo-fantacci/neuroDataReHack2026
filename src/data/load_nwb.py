"""Load an IBL Brain-Wide Map session from its NWB file on DANDI:000409.

An alternative to the ONE route in ``src/data/access.py``. :func:`load_session_nwb` returns the
same three objects as ``src.data.preprocess.load_session`` -- ``(trials, mask, probe_loads)`` -- so
``preprocess_session`` and everything downstream run unchanged on either source.

Streaming follows the pattern in the NWB/DANDI documentation and in CatalystNeuro's IBL-to-NWB
README: ``DandiAPIClient`` finds the asset, ``remfile`` streams it over HTTP, ``h5py`` opens it,
and ``pynwb.NWBHDF5IO(..., load_namespaces=True)`` reads it with the IBL extensions. Only the
tables and the spike-time slices that are needed are fetched.

How the NWB content maps onto what the ONE route returns (all checked on session ``5c0c560e``):

* **Units.** The NWB units table holds every sorted cluster, with IBL's ``cluster_uuid``. Its
  UUIDs are identical to the paper's cluster table, and spike times are bit-identical to ONE's.
  "Good" units are ``ibl_quality_score >= qc`` (ONE: ``label >= qc``), kept in cluster-id order.
* **Regions.** A unit's Beryl region is the ``beryl_location`` of its ``max_electrode`` in the
  ``ElectrodesIBLBregma`` localization table.
* **Trials.** NeuroConv column names are renamed to IBL's ALF names (``TRIAL_COLUMNS``); choice,
  feedback and contrast are re-encoded the way IBL encodes them. The inclusion mask is computed
  by ``brainwidemap.load_trials_and_mask`` itself, given this table.

Known difference: the NWB trials table is **not** identical to the one ONE serves for the paper
freeze (one stimulus-onset time, every movement-onset time by ~0.4 ms, one feedback time on
``5c0c560e``). Tensors built from NWB therefore differ slightly from the frozen ones; see
``scripts/compare_nwb_vs_one.py`` and ``docs/knowledge/nwb_vs_one_comparison.md``. The ONE route
remains the one behind every result in this repository.

Requires ``dandi``, ``pynwb``, ``remfile`` and ``h5py`` (all in ``requirements.txt``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

DANDISET_ID = "000409"

#: NeuroConv (NWB) trial column -> IBL ALF trial column.
TRIAL_COLUMNS = {
    "start_time": "intervals_0",
    "stop_time": "intervals_1",
    "gabor_stimulus_onset_time": "stimOn_times",
    "gabor_stimulus_offset_time": "stimOff_times",
    "auditory_cue_time": "goCue_times",
    "wheel_movement_onset_time": "firstMovement_times",
    "choice_registration_time": "response_times",
    "feedback_time": "feedback_times",
    "probability_left": "probabilityLeft",
    "reward_volume_uL": "rewardVolume",
}

#: IBL convention: choice = +1 for a clockwise wheel turn, -1 counter-clockwise, 0 no-go.
#: Verified trial by trial against ONE on session 5c0c560e.
CHOICE_CODES = {"clockwise": 1.0, "counter_clockwise": -1.0}


def _as_str(values) -> np.ndarray:
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in values])


def session_url(eid: str, *, dandiset_id: str = DANDISET_ID, version: str = "draft") -> str:
    """Streaming URL of the session's processed NWB file (``desc-processed``) on DANDI."""
    from dandi.dandiapi import DandiAPIClient

    with DandiAPIClient() as client:
        dandiset = client.get_dandiset(dandiset_id, version)
        assets = list(dandiset.get_assets_by_glob(f"*{eid}*desc-processed*.nwb"))
        if len(assets) != 1:
            raise LookupError(f"expected one processed NWB asset for {eid}, found {len(assets)}")
        return assets[0].get_content_url(follow_redirects=1, strip_query=False)


@dataclass
class NWBSession:
    """An open, streamed NWB file. Use as a context manager so the HTTP handles get closed."""

    nwbfile: object
    _handles: list = field(default_factory=list, repr=False)

    def close(self) -> None:
        for h in reversed(self._handles):
            try:
                h.close()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def open_session(eid: str | None = None, *, url: str | None = None,
                 cache_dir: str | Path | None = None) -> NWBSession:
    """Stream a session's NWB file. Give ``eid`` (looked up on DANDI) or a direct ``url``.

    ``cache_dir`` enables remfile's disk cache, so re-reading the same byte ranges (a second run of
    a notebook) does not hit the network again.
    """
    import h5py
    import remfile
    from pynwb import NWBHDF5IO

    if url is None:
        if eid is None:
            raise ValueError("give either eid or url")
        url = session_url(eid)
    kwargs = {}
    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        kwargs["disk_cache"] = remfile.DiskCache(str(cache_dir))
    rf = remfile.File(url, **kwargs)
    h5 = h5py.File(rf, "r")
    io = NWBHDF5IO(file=h5, load_namespaces=True)
    return NWBSession(nwbfile=io.read(), _handles=[rf, h5, io])


# --------------------------------------------------------------------------- trials

def trials_from_columns(columns: dict) -> pd.DataFrame:
    """NWB trial columns (name -> array) -> an IBL-style trials DataFrame.

    Kept separate from file access so the mapping can be unit-tested without a download.
    """
    out = {alf: np.asarray(columns[nwb], dtype=float)
           for nwb, alf in TRIAL_COLUMNS.items() if nwb in columns}

    choice = _as_str(columns["mouse_wheel_choice"])
    out["choice"] = np.array([CHOICE_CODES.get(c, 0.0) for c in choice])

    out["feedbackType"] = np.where(np.asarray(columns["is_mouse_rewarded"], dtype=bool), 1.0, -1.0)

    side = _as_str(columns["gabor_stimulus_side"])
    contrast = np.asarray(columns["gabor_stimulus_contrast"], dtype=float) / 100.0   # % -> fraction
    out["contrastLeft"] = np.where(side == "left", contrast, np.nan)
    out["contrastRight"] = np.where(side == "right", contrast, np.nan)
    return pd.DataFrame(out)


def trials_table(nwbfile) -> pd.DataFrame:
    """The session's trials as an IBL-style DataFrame (ALF column names and encodings)."""
    tr = nwbfile.trials
    return trials_from_columns({c: tr[c].data[:] for c in tr.colnames})


def trials_and_mask(nwbfile, *, min_rt: float = 0.08, max_rt: float = 2.0):
    """Trials plus the Brain-Wide-Map paper's inclusion mask, computed by ``brainwidemap`` itself.

    ``load_trials_and_mask`` accepts a pre-loaded session object; handing it this table means the
    paper's exact criteria (reaction-time bounds, NaN exclusions, truncation to the
    performance-passing block) are applied without re-implementing them.
    """
    from brainwidemap import load_trials_and_mask

    stand_in = SimpleNamespace(trials=trials_table(nwbfile))
    return load_trials_and_mask(None, nwbfile.session_id, min_rt=min_rt, max_rt=max_rt,
                                sess_loader=stand_in)


# --------------------------------------------------------------------------- units

def _beryl_by_electrode(nwbfile) -> np.ndarray:
    loc = nwbfile.lab_meta_data["localization"]
    table = loc.anatomical_coordinates_tables["ElectrodesIBLBregma"]
    entity = np.asarray(table["localized_entity"].data[:], dtype=np.int64)
    beryl = _as_str(table["beryl_location"].data[:])
    out = np.full(len(nwbfile.electrodes), "", dtype=object)
    out[entity] = beryl
    return out


def good_units(nwbfile, probe_name: str, *, qc: float = 1.0,
               regions: list[str] | None = None):
    """Good units of one probe -> ``(spikes, clusters)``, shaped like ``brainwidemap.load_good_units``.

    ``spikes`` has ``times`` (sorted ascending, as ``bin_spikes_by_event`` requires) and
    ``clusters`` (row index into ``clusters``). ``clusters`` has ``cluster_uuid``, ``Beryl``,
    ``firing_rate`` and ``unit_name``, one row per good unit in cluster-id order.

    ``regions`` restricts which units' spike times are streamed (saves time); the returned
    ``clusters`` table then holds only those units, so ``keep_idx`` refers to that subset.
    """
    units = nwbfile.units
    probe = _as_str(units["probe_name"].data[:])
    score = np.asarray(units["ibl_quality_score"].data[:], dtype=float)
    max_el = np.asarray(units["max_electrode"].data[:], dtype=np.int64)
    beryl = _beryl_by_electrode(nwbfile)[max_el]

    rows = np.flatnonzero((np.char.lower(probe.astype(str)) == probe_name.lower())
                          & (score >= qc - 1e-9))
    if regions is not None:
        rows = rows[np.isin(beryl[rows], list(regions))]

    clusters = pd.DataFrame({
        "unit_name": _as_str(units["unit_name"].data[:])[rows] if "unit_name" in units.colnames else rows,
        "cluster_uuid": _as_str(units["cluster_uuid"].data[:])[rows],
        "Beryl": beryl[rows],
        "firing_rate": np.asarray(units["firing_rate"].data[:], dtype=float)[rows],
        "nwb_row": rows,
    })

    # Ragged column: in pynwb, units["spike_times"] returns the *index* (a VectorIndex), not the
    # times. The times are the index's target; read only each unit's slice of it.
    vector_index = units["spike_times_index"]
    index = np.asarray(vector_index.data[:], dtype=np.int64)
    data = vector_index.target.data
    times, labels = [], []
    for i, r in enumerate(rows):
        lo = 0 if r == 0 else int(index[r - 1])
        t = np.asarray(data[lo:int(index[r])], dtype=float)
        times.append(t)
        labels.append(np.full(t.size, i, dtype=np.int64))
    times = np.concatenate(times) if times else np.zeros(0)
    labels = np.concatenate(labels) if labels else np.zeros(0, dtype=np.int64)
    order = np.argsort(times, kind="stable")
    return {"times": times[order], "clusters": labels[order]}, clusters


def probe_names(nwbfile) -> list[str]:
    """Probe names present in the units table, sorted (``['Probe00', 'Probe01']``)."""
    return sorted(set(_as_str(nwbfile.units["probe_name"].data[:])))


def load_session_nwb(eid: str | None = None, *, url: str | None = None, cfg=None,
                     regions: list[str] | None = None, cache_dir: str | Path | None = None):
    """``(trials, mask, probe_loads)`` from DANDI, the NWB twin of ``preprocess.load_session``.

    Pass the result straight to ``src.data.preprocess.preprocess_session``. Probes are returned in
    name order. ``regions=None`` streams every good unit; pass ``cfg.regions`` to fetch only those.
    """
    from src.data.preprocess import PreprocessConfig

    cfg = cfg or PreprocessConfig()
    with open_session(eid, url=url, cache_dir=cache_dir) as s:
        trials, mask = trials_and_mask(s.nwbfile, min_rt=cfg.min_rt_s, max_rt=cfg.max_rt_s)
        loads = []
        for p in probe_names(s.nwbfile):
            spikes, clusters = good_units(s.nwbfile, p, qc=cfg.qc, regions=regions)
            loads.append({"spikes": spikes, "clusters": clusters, "probe_name": p})
    return trials, mask, loads
