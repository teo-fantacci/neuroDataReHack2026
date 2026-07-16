"""Run Linderman's `ssm` rSLDS from the working env, and return our own RSLDSResult.

`ssm` cannot live in `neurodatarehack` (it needs numpy 1.x; we are on 2.3.5, and pynapple /
numba / pandas 3.0 would be dragged down with a downgrade). So we keep it in a separate
`ndrh_ssm` env and talk to it over a file, by calling that env's interpreter **directly**
(not ``conda run`` — see ``ssm_python`` for why that fails on Windows). Build the env once with
``scripts/install_ssm_env.sh``.

This whole module is **optional**: if the env is not built, ``ssm_env_available()`` returns
False and callers fall back to the exact-EM backend in ``rslds.py``. Nothing here is needed to
run the notebooks end-to-end.

The point of this wrapper: the **same** ``RSLDSResult`` comes back, so every downstream
figure and statistic (``fixed_points``, ``eigenvalues``, ``classify``, ``flow_field``,
``order_consistency``, ...) is written once and runs against both backends. That makes the
comparison between the two an honest apples-to-apples one, which is the whole reason for
having two.

What genuinely differs between the backends
-------------------------------------------
* ``rslds.fit_rslds``  — latent = PCA on Anscombe counts, **fixed**; Gaussian; exact EM.
  Fast (seconds). Its weakness is real: on sparse IBL units the PCA latent is largely noise,
  which pushes |lambda| towards 1 and makes every regime look like an integrator.
* ``rslds_ssm.fit_rslds_ssm`` — latent **inferred jointly** with the dynamics under a true
  **Poisson** likelihood, by Laplace-EM. Slow (minutes) and only a lower bound on the
  likelihood, but it is the published, tested method and it does not inherit the PCA
  weakness.

If the two agree on the attractor structure and the sequence, that is a much stronger claim
than either alone. If they disagree, believe `ssm` and work out why the PCA latent failed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from src.dynamics.rslds import RSLDSResult, _order_by_time

SSM_ENV = "ndrh_ssm"


def ssm_python(env: str = SSM_ENV) -> Path | None:
    """Absolute path to the ssm env's interpreter, or None if that env is not built.

    We invoke that python.exe **directly** rather than going through ``conda run``. On
    Windows ``conda`` is a .bat in ``condabin`` and is not on a subprocess's PATH, so
    ``conda run`` raises WinError 2 from inside a notebook kernel — and ``conda run`` also
    refuses commands whose arguments contain newlines. Calling the interpreter directly
    sidesteps both, needs no shell, and works identically on Linux.
    """
    # sys.prefix is <base>/envs/neurodatarehack -> siblings live in <base>/envs/
    candidates = [Path(sys.prefix).parent / env]
    if "CONDA_EXE" in os.environ:                       # <base>/Scripts/conda.exe
        candidates.append(Path(os.environ["CONDA_EXE"]).parents[1] / "envs" / env)

    for prefix in candidates:
        for exe in (prefix / "python.exe", prefix / "bin" / "python"):
            if exe.exists():
                return exe
    return None


def ssm_env_available(env: str = SSM_ENV) -> bool:
    """Is the ssm env built AND does `import ssm` work in it? Lets a notebook degrade."""
    exe = ssm_python(env)
    if exe is None:
        return False
    try:
        out = subprocess.run([str(exe), "-c", "import ssm"], capture_output=True, timeout=180)
        return out.returncode == 0
    except Exception:
        return False


def fit_rslds_ssm(
    counts: np.ndarray,
    n_states: int,
    *,
    n_latent: int = 3,
    n_iter: int = 50,
    seed: int = 0,
    env: str = SSM_ENV,
    root: Path | None = None,
    verbose: bool = True,
) -> RSLDSResult:
    """Fit a Poisson rSLDS with `ssm` (out of process) on **raw counts**.

    Note that this takes *counts*, not latents: `ssm` infers the latent itself. Do not
    smooth them — the Poisson emission model wants integers, same as the HMM.
    """
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    script = root / "scripts" / "run_rslds_ssm.py"
    if not script.exists():
        raise FileNotFoundError(f"missing bridge script: {script}")

    exe = ssm_python(env)
    if exe is None:
        raise RuntimeError(
            f"conda env `{env}` not found. Build it once with scripts/install_ssm_env.sh "
            "(no admin rights needed — it uses conda's mingw toolchain)."
        )

    counts = np.ascontiguousarray(counts).astype(np.int32)
    tmp = Path(tempfile.mkdtemp(prefix="rslds_ssm_"))
    fin, fout = tmp / "tensor.npz", tmp / "fit.npz"
    np.savez_compressed(fin, counts=counts)

    cmd = [
        str(exe), str(script),
        "--in", str(fin), "--out", str(fout),
        "--n-states", str(n_states), "--n-latent", str(n_latent),
        "--n-iter", str(n_iter), "--seed", str(seed),
    ]
    proc = subprocess.run(cmd, capture_output=not verbose, text=True)
    if proc.returncode != 0 or not fout.exists():
        err = (proc.stderr or "")[-2000:] if not verbose else "(see output above)"
        raise RuntimeError(
            f"ssm rSLDS failed (exit {proc.returncode}). Is the `{env}` env built? "
            f"Run scripts/install_ssm_env.sh.\n{err}"
        )

    z = np.load(fout, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    if verbose and not meta["converged"]:
        print(
            f"  /!\\ ssm Laplace-EM may NOT have converged (tail ELBO drift "
            f"{meta['elbo_tail_drift']:.1e}). Raise --n-iter before believing the fit.",
            file=sys.stderr,
        )

    x, states = z["x"], z["states"]
    A, b = z["A"], z["b"]
    K, D = A.shape[0], A.shape[1]
    n_trials, T = states.shape

    # RSLDSResult wants a full posterior; ssm hands back the Viterbi path. A one-hot gamma is
    # the honest stand-in — it is what `_order_by_time` and the occupancy plots consume, and
    # nothing downstream treats it as a soft posterior.
    gamma = np.zeros((n_trials, T, K))
    gamma[np.arange(n_trials)[:, None], np.arange(T)[None, :], states] = 1.0

    res = RSLDSResult(
        x=x, states=states, gamma=gamma,
        A=A, b=b,
        Q=np.tile(np.eye(D), (K, 1, 1)),      # ssm's diagonal_gaussian noise; unused downstream
        log_pi=np.zeros((K, K)),              # recurrent_only: transitions live in R
        R=z["Rs"], pi0=np.full(K, 1.0 / K),
        log_likelihood=float(z["elbos"][-1]), # an ELBO, not an exact LL -- see below
        ll_trace=list(map(float, z["elbos"])),
        recurrent=True,
    )
    res.backend = "ssm"
    res.emission_C = z["C"]                   # (units, D): which neurons define the latent
    res.emission_d = z["d"]
    res.meta = meta
    return _order_by_time(res)
