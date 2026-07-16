#!/usr/bin/env bash
# Build the `ndrh_ssm` environment: Linderman's `ssm` (the real rSLDS), isolated from the
# working `neurodatarehack` env.
#
# WHY A SEPARATE ENV
# ------------------
# `ssm` is pinned to the numpy 1.x era. Installing it into `neurodatarehack` would drag
# numpy 2.3.5 back to 1.x, and pynapple / numba / pandas 3.0 would go down with it. The
# 2026-07-12 worklog made "additive installs only" a rule; this respects it. The notebook
# stays in `neurodatarehack` and shells out to this env (see scripts/run_rslds_ssm.py).
#
# WHY THIS SCRIPT EXISTS AT ALL (Windows)
# ---------------------------------------
# `pip install git+https://github.com/lindermanlab/ssm` fails on Windows, twice over:
#
#   1. Its legacy setup.py does `import numpy` at build time without declaring numpy as a
#      build requirement, so under PEP-517 build isolation metadata generation dies with
#      `ModuleNotFoundError: No module named 'numpy'`.  -> `--no-build-isolation`.
#   2. It then needs a C++ compiler for `ssm/cstats.pyx`, and a stock Windows box has none:
#      `error: Microsoft Visual C++ 14.0 or greater is required`.
#
# The obvious fix is to install ~6 GB of MSVC Build Tools (admin). This script avoids that
# entirely by using conda's mingw-w64 toolchain, which needs no admin rights. Two extra
# hurdles, both handled below:
#
#   a. conda's gcc 5.3 is old enough that `INTPTR_MAX` is not visible to C++ without
#      `__STDC_LIMIT_MACROS`, so pyport.h fails to compile. -> patch extra_compile_args.
#   b. conda ships no mingw import library for python311.dll, so every `Py*` symbol comes
#      back `undefined reference` at link time. -> generate libpython311.a with gendef +
#      dlltool.
#
# On Linux (i.e. at Janelia) none of this is needed: `pip install --no-build-isolation
# git+...` just works, because gcc is there. This script is the Windows workaround.
#
# Verified end-to-end 2026-07-13: builds, imports, and fits a Poisson rSLDS with
# `transitions="recurrent_only"` + Laplace-EM on synthetic data.
set -euo pipefail

ENV_NAME=ndrh_ssm
PYVER=3.11
SRC="${TMPDIR:-/tmp}/ssm_src"

source "$(conda info --base)/etc/profile.d/conda.sh"

conda create -n "$ENV_NAME" python="$PYVER" -y
# ssm is a numpy-1.x package; pinning it here is the whole point of the separate env.
conda run -n "$ENV_NAME" pip install "numpy<2" cython setuptools wheel
conda run -n "$ENV_NAME" pip install tqdm autograd numba scikit-learn matplotlib seaborn

ENV_PREFIX="$(conda info --base)/envs/$ENV_NAME"

if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
    echo ">>> Windows: installing mingw toolchain (no admin required)"
    conda install -n "$ENV_NAME" -c conda-forge m2w64-toolchain libpython -y
    MINGW_BIN="$ENV_PREFIX/Library/mingw-w64/bin"

    echo ">>> generating the python${PYVER//./} import library mingw needs"
    (
        cd "$ENV_PREFIX"
        "$MINGW_BIN/gendef.exe" python311.dll
        "$MINGW_BIN/dlltool.exe" --dllname python311.dll --def python311.def \
                                 --output-lib libpython311.a
        mv libpython311.a libs/
        rm -f python311.def
    )
    EXTRA_BUILD=(--compiler=mingw32)
else
    EXTRA_BUILD=()
fi

rm -rf "$SRC"
git clone --depth 1 https://github.com/lindermanlab/ssm "$SRC"
cd "$SRC"

if [[ ${#EXTRA_BUILD[@]} -gt 0 ]]; then
    echo ">>> patching setup.py: gcc 5.3 needs __STDC_LIMIT_MACROS to see INTPTR_MAX"
    conda run -n "$ENV_NAME" python - <<'PY'
import io
s = io.open("setup.py", encoding="utf-8").read()
old = 'extra_compile_args=["-fopenmp"] if USE_OPENMP else [],'
new = ('extra_compile_args=(["-fopenmp"] if USE_OPENMP else []) + '
       '["-D__STDC_LIMIT_MACROS", "-D__STDC_CONSTANT_MACROS", "-std=gnu++11"],')
assert old in s, "setup.py changed upstream; re-check the patch"
io.open("setup.py", "w", encoding="utf-8").write(s.replace(old, new))
print("patched setup.py")
PY
    printf '[build]\ncompiler=mingw32\n' > setup.cfg
    conda run -n "$ENV_NAME" python setup.py build_ext --inplace "${EXTRA_BUILD[@]}"
fi

conda run -n "$ENV_NAME" pip install --no-build-isolation .
conda run -n "$ENV_NAME" python -c "import ssm; from ssm import SLDS; print('ssm OK:', ssm.__file__)"

echo
echo "Done. The notebook (running in neurodatarehack) reaches this env via:"
echo "    conda run -n $ENV_NAME python scripts/run_rslds_ssm.py --in <tensor.npz> --out <fit.npz>"
