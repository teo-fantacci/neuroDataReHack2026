"""Load the macaque working-memory dataset (DANDI 000620) into the project tensor shape.

The per-animal pickle (e.g. ``notebooks/data/000620_sub-Elgar_hmm.pkl.xz``) is
``dict[session] -> {"FEF": arr, "DMFC": arr, "trial_times": (n_trials, 7)}``, where each area
``arr`` is an object ndarray ``(n_trials, n_neurons)`` whose cells are 1-D vectors of **spike
times (s) from trial start**. FEF and DMFC are recorded *simultaneously* within a session
(identical trial counts). This turns one session's areas into raw **integer** count tensors
``(trials, bins, neurons)`` — the shape every model in the project consumes — and returns the
task-event times alongside.

``trial_times`` holds **7 monotonic event times per trial** (relative to trial start). Per the
task's **6 phases** (fixation, stimulus, delay, cue, response, reward), the 7 columns are the phase
**onsets** plus the trial end:

    col 0 = fixation onset (= trial start, 0)        col 4 = response onset (~2.70 s)
    col 1 = stimulus onset (~0.40 s)                 col 5 = reward onset   (~3.15 s)
    col 2 = DELAY onset    (~1.42 s)                 col 6 = reward/trial end (~3.52 s)
    col 3 = cue onset      (~2.45 s)

The stimulus and delay phases are each ~1.0 s; only the response/reward epoch varies trial-to-trial.

**Alignment.** The default here is to align every trial to its **delay onset** (``col 2``) via
:func:`load_monkey_aligned`, so the working-memory delay period sits at ``[0, ~+1.0] s`` with
fixation+stimulus at negative times — the WM state is most visible that way. A delay-aligned window
``[-1.4, +2.2] s`` stays inside every trial, so ~all trials are kept (unlike a trial-start
``[0, 4] s`` window, which dropped the ~10 % shortest trials).

The earlier ``000620_hmm.pkl.xz`` (no ``trial_times``) is superseded by the per-animal files.
"""

from __future__ import annotations

import lzma
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[2] / "notebooks" / "data"
DEFAULT_PKL = DATA_DIR / "000620_sub-Elgar_hmm.pkl.xz"
AREAS = ("FEF", "DMFC")

# The task's 6 phases, in order.
PHASE_NAMES = ["fixation", "stimulus", "delay", "cue", "response", "reward"]
# The 7 columns of `trial_times` = phase onsets (col 6 = reward/trial end).
EVENT_NAMES = ["fixation", "stimulus", "delay", "cue", "response", "reward", "end"]
DELAY_COL = 2               # column giving delay onset — the default alignment event


def animal_pkl(animal: str) -> Path:
    """Path to a given animal's pickle, e.g. ``animal_pkl("sub-Elgar")``."""
    return DATA_DIR / f"000620_{animal}_hmm.pkl.xz"


def load_pickle(pkl_path: str | Path = DEFAULT_PKL) -> dict:
    """Decompress + unpickle the WHOLE dataset: ``dict[session][area] -> object array``.

    This holds all 26 sessions in memory at once (~1-2 GB peak while dill runs) — on a RAM-tight
    machine it can raise ``MemoryError``. Prefer :func:`load_session_dict`, which caches one session
    to a small file so notebooks never re-unpickle the whole blob.
    """
    import dill as pickle

    with lzma.open(pkl_path, "rb") as f:
        return pickle.load(f)


def session_cache_path(session: str, pkl_path: str | Path = DEFAULT_PKL) -> Path:
    """Where the small per-session cache for ``session`` lives (next to the source pickle)."""
    p = Path(pkl_path)
    return p.parent / "_session_cache" / f"{p.stem}__{session}.pkl.xz"


def load_session_dict(session: str, *, pkl_path: str | Path = DEFAULT_PKL,
                      data: dict | None = None) -> dict:
    """One session's dict (``{"FEF","DMFC","trial_times"}``), via a small per-session cache.

    Order of preference: an already-loaded ``data`` dict → the per-session cache file → a one-off
    full unpickle (which then *writes* the cache for next time). This keeps notebook memory tiny:
    only the first ever call pays the full-pickle cost; afterwards each session loads from a few-MB
    file. Delete ``notebooks/data/_session_cache/`` to rebuild.
    """
    import dill as pickle

    if data is not None:
        return data[session]

    cache = session_cache_path(session, pkl_path)
    if cache.exists():
        with lzma.open(cache, "rb") as f:
            return pickle.load(f)

    sess = load_pickle(pkl_path)[session]           # one-off full load
    cache.parent.mkdir(parents=True, exist_ok=True)
    with lzma.open(cache, "wb", preset=1) as f:     # preset 1 = fast; the session is small
        pickle.dump(sess, f)
    return sess


def usable_sessions(data: dict, areas=AREAS) -> list[str]:
    """Sessions with at least one trial in every requested area (12 of 26 are empty)."""
    out = []
    for sk, sess in data.items():
        if all(np.asarray(sess.get(a)).ndim == 2 and np.asarray(sess[a]).shape[0] > 0
               for a in areas):
            out.append(sk)
    return sorted(out)


def area_tensor(area_arr, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(n_trials, n_neurons) of spike-time vectors -> (int counts (n_tr, n_bins, n_nrn), last/trial)."""
    area_arr = np.asarray(area_arr)
    n_tr, n_nrn = area_arr.shape
    counts = np.zeros((n_tr, len(edges) - 1, n_nrn), dtype=np.int32)
    last = np.zeros(n_tr)
    for i in range(n_tr):
        for n in range(n_nrn):
            el = area_arr[i, n]
            if el is not None and len(el):
                counts[i, :, n], _ = np.histogram(el, bins=edges)
                last[i] = max(last[i], el.max())
    return counts, last


def load_monkey_tensors(
    session: str,
    *,
    pkl_path: str | Path = DEFAULT_PKL,
    areas=AREAS,
    t0: float = 0.0,
    t1: float = 4.0,
    bin_s: float = 0.10,
    data: dict | None = None,
) -> dict:
    """One session's areas as raw count tensors on a shared, duration-filtered trial set.

    Returns a dict::

        {"tensors": {area: (n_kept, n_bins, n_units) int},
         "bin_centers": (n_bins,), "keep": (n_trials_orig,) bool,
         "n_trials": int, "session": str, "window": (t0, t1, bin_s)}

    ``keep`` is shared across areas (a trial is kept iff its last spike in *either* area reaches
    ``t1``), so the areas stay trial-aligned for any later joint analysis.
    """
    sess = load_session_dict(session, pkl_path=pkl_path, data=data)
    edges = np.arange(t0, t1 + 1e-9, bin_s)
    bin_centers = edges[:-1] + bin_s / 2

    raw, last = {}, None
    for a in areas:
        raw[a], la = area_tensor(sess[a], edges)
        last = la if last is None else np.maximum(last, la)
    keep = last >= t1
    tensors = {a: raw[a][keep] for a in areas}

    # task-event times per KEPT trial, (n_kept, 7), if present (superseded pickle lacks them)
    events = None
    if "trial_times" in sess:
        events = np.asarray(sess["trial_times"], dtype=float)[keep]

    return {
        "tensors": tensors, "bin_centers": bin_centers, "keep": keep,
        "n_trials": int(keep.sum()), "session": session, "window": (t0, t1, bin_s),
        "events": events, "event_names": EVENT_NAMES,
    }


def area_tensor_aligned(area_arr, align_t: np.ndarray, edges: np.ndarray):
    """Like :func:`area_tensor` but bin each trial's spikes relative to ``align_t[trial]``.

    Returns ``(counts, last_rel)`` where ``last_rel[i]`` is the last spike time of trial ``i``
    *relative to its alignment* — used to drop trials whose window is not fully covered by spikes.
    """
    area_arr = np.asarray(area_arr)
    n_tr, n_nrn = area_arr.shape
    counts = np.zeros((n_tr, len(edges) - 1, n_nrn), dtype=np.int32)
    last_rel = np.full(n_tr, -np.inf)
    for i in range(n_tr):
        for n in range(n_nrn):
            el = area_arr[i, n]
            if el is not None and len(el):
                counts[i, :, n], _ = np.histogram(el - align_t[i], bins=edges)
                last_rel[i] = max(last_rel[i], el.max() - align_t[i])
    return counts, last_rel


def load_monkey_aligned(
    session: str,
    *,
    align_col: int = DELAY_COL,
    pre: float = 1.4,
    post: float = 2.2,
    bin_s: float = 0.10,
    areas=AREAS,
    pkl_path: str | Path = DEFAULT_PKL,
    data: dict | None = None,
) -> dict:
    """One session as **event-aligned** count tensors: each trial re-zeroed at ``trial_times[:, align_col]``.

    Default ``align_col = DELAY_COL`` (delay onset) with window ``[-pre, +post]`` s, so the delay /
    working-memory period sits at ``[0, ~+1.0] s`` and fixation+stimulus at negative times. Because
    the events are near-fixed in time, this is close to a constant shift, but it is done per trial so
    the (variable) response/reward events line up too.

    Returns a dict like :func:`load_monkey_tensors` plus:
      ``events``      (n_kept, 7) event times **relative to the alignment** (delay onset = 0).
      ``align_col``   which event was used as t = 0.
      ``phase_names`` the 6 task-phase names.
    ``bin_centers`` are relative to the alignment (0 = the aligned event).

    The **binned result** (dense int tensors + events) is cached to a small `.npz` keyed by all the
    window parameters, and reloaded with ``allow_pickle=False``. That matters on RAM-tight machines:
    it skips re-unpickling the session's ~40 k spike-time arrays every run (that unpickle is the
    memory spike), so a notebook reload costs only a few MB. Delete `_session_cache/` to rebuild.
    """
    def _pack(tensors, bin_centers, keep, events_rel):
        return {
            "tensors": tensors, "bin_centers": bin_centers, "keep": keep,
            "n_trials": int(keep.sum()), "session": session,
            "window": (-pre, post, bin_s), "align_col": align_col,
            "events": events_rel, "event_names": EVENT_NAMES, "phase_names": PHASE_NAMES,
        }

    cache = session_cache_path(session, pkl_path).with_name(
        f"{Path(pkl_path).stem}__{session}__aligned_c{align_col}"
        f"_pre{int(round(pre*100))}_post{int(round(post*100))}_bin{int(round(bin_s*1000))}"
        f"_{'-'.join(areas)}.npz")
    if data is None and cache.exists():
        z = np.load(cache, allow_pickle=False)
        return _pack({a: z[f"{a}__counts"] for a in areas},
                     z["bin_centers"], z["keep"], z["events"])

    sess = load_session_dict(session, pkl_path=pkl_path, data=data)
    tt = np.asarray(sess["trial_times"], dtype=float)
    align_t = tt[:, align_col]

    edges = np.arange(-pre, post + 1e-9, bin_s)
    bin_centers = edges[:-1] + bin_s / 2

    raw, last_rel = {}, None
    for a in areas:
        raw[a], lr = area_tensor_aligned(sess[a], align_t, edges)
        last_rel = lr if last_rel is None else np.maximum(last_rel, lr)
    keep = last_rel >= (post - 1e-9)          # window's far edge must be covered by spikes
    tensors = {a: raw[a][keep] for a in areas}
    events_rel = tt[keep] - align_t[keep][:, None]     # all 7 events, relative to the alignment

    if data is None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, bin_centers=bin_centers, keep=keep, events=events_rel,
                            **{f"{a}__counts": tensors[a] for a in areas})
    return _pack(tensors, bin_centers, keep, events_rel)
