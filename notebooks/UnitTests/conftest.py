# notebooks/UnitTests/conftest.py
from pathlib import Path
import importlib.util
import os
import sys
import subprocess
import shutil
import pytest

ROOT = Path(__file__).parent                 # .../notebooks/UnitTests
NOTEBOOKS = ROOT.parent                      # .../notebooks
PROJECT_ROOT = NOTEBOOKS.parent              # repo root

TEST_OUTPUT = ROOT / "TestOutput"
TEST_OUTPUT.mkdir(parents=True, exist_ok=True)

@pytest.fixture(scope="session")
def test_output_dir():
    return TEST_OUTPUT

def _import_module_by_path(mod_name: str, file_path: Path):
    if not file_path.exists():
        raise FileNotFoundError(f"Cannot find module file at: {file_path}")
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module

def _export_ipynb_to_py(ipynb: Path, py_out: Path) -> None:
    """Export a .ipynb to .py using nbconvert (CLI)."""
    # Prefer nbconvert CLI so we don’t add heavy runtime deps to tests
    if not shutil.which("jupyter"):
        raise FileNotFoundError(
            f"'jupyter' not found in PATH; cannot export notebook {ipynb}"
        )
    py_out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "jupyter", "nbconvert", "--to", "script",
        str(ipynb),
        "--output", py_out.name  # writes to py_out.parent by default (current dir of ipynb)
    ]
    subprocess.run(cmd, check=True, cwd=str(ipynb.parent))

def _find_or_create_module_file() -> Path:
    # Allow override
    env_path = os.environ.get("MOD_PATH")
    if env_path:
        p = Path(env_path).expanduser().resolve()
        if p.exists():
            return p
        raise FileNotFoundError(f"MOD_PATH points to a non-existent file: {p}")

    preferred_py = "CodeAttemptFNAL_SeqVer.py"
    preferred_ipynb = "CodeAttemptFNAL_SeqVer.ipynb"

    # Look for .py in common places
    candidates = [
        NOTEBOOKS / preferred_py,
        PROJECT_ROOT / preferred_py,
        PROJECT_ROOT / "src" / preferred_py,
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()

    # If .py not found, but .ipynb exists in notebooks, export it
    ipynb_path = NOTEBOOKS / preferred_ipynb
    if ipynb_path.exists():
        py_target = NOTEBOOKS / preferred_py
        _export_ipynb_to_py(ipynb_path, py_target)
        if py_target.exists():
            return py_target.resolve()

    # Last chance: search upwards for either file
    for base in [NOTEBOOKS, PROJECT_ROOT]:
        hits_py = list(base.rglob(preferred_py))
        if hits_py:
            return hits_py[0].resolve()
        hits_nb = list(base.rglob(preferred_ipynb))
        if hits_nb:
            # export next to the notebook
            py_target = hits_nb[0].with_suffix(".py")
            _export_ipynb_to_py(hits_nb[0], py_target)
            if py_target.exists():
                return py_target.resolve()

    raise FileNotFoundError(
        "Could not locate CodeAttemptFNAL_SeqVer.py or .ipynb.\n"
        "Set MOD_PATH=/full/path/to/your_file.py to override."
    )

@pytest.fixture(scope="session")
def mod():
    file_path = _find_or_create_module_file()
    parent = file_path.parent
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))
    return _import_module_by_path("CodeAttemptFNAL_SeqVer", file_path)
