"""
Root pytest configuration — lets `python3 -m pytest -q` run every prototype's
tests from the repository root.

Each prototype is a flat directory of top-level modules (`search.py`,
`demo.py`, ...), and two prototypes share some module names.  Before a test
file is imported, and before each test runs, we make that file's prototype
directory the first entry on sys.path and park any modules that were loaded
from a *different* prototype directory, so `import search` always resolves to
the right prototype.  Running pytest from inside a single prototype directory
is unaffected.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import pytest

ROOT = Path(__file__).resolve().parent
PROTO_DIRS = sorted(p for p in ROOT.glob("prototype_*") if p.is_dir())

# Modules parked while another prototype is active: {proto_dir: {name: module}}
_parked: Dict[Path, Dict[str, object]] = {p: {} for p in PROTO_DIRS}
_active: Optional[Path] = None


def _proto_for(path: Path) -> Optional[Path]:
    path = Path(path).resolve()
    for proto in PROTO_DIRS:
        if proto == path or proto in path.parents:
            return proto
    return None


def _module_proto(mod) -> Optional[Path]:
    f = getattr(mod, "__file__", None)
    if not f:
        return None
    parent = Path(f).resolve().parent
    return parent if parent in _parked else None


def _activate(proto: Optional[Path]) -> None:
    global _active
    if proto is None or proto == _active:
        return
    for name, mod in list(sys.modules.items()):
        owner = _module_proto(mod)
        if owner is not None and owner != proto:
            _parked[owner][name] = sys.modules.pop(name)
    sys.modules.update(_parked[proto])
    _parked[proto] = {}
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() not in _parked]
    sys.path.insert(0, str(proto))
    _active = proto


@pytest.hookimpl(tryfirst=True)
def pytest_pycollect_makemodule(module_path, parent):
    _activate(_proto_for(module_path))


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    _activate(_proto_for(item.path))
