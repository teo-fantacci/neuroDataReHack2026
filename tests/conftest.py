"""Shared test setup.

Every test runs on small synthetic data: nothing here downloads from IBL/ONE or DANDI, so the
suite runs offline and in CI. Tests that need minutes, or the optional `ndrh_ssm` conda env,
are marked ``@pytest.mark.slow`` and skipped unless you run ``pytest -m slow``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_script(name: str):
    """Import ``scripts/<name>.py`` as a module without running its ``__main__`` block."""
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"scripts_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def rng():
    return __import__("numpy").random.default_rng(0)


@pytest.fixture
def ground_truth_script():
    """``scripts/validate_rslds_ground_truth.py`` as a module (its simulator, unchanged)."""
    return load_script("validate_rslds_ground_truth")
