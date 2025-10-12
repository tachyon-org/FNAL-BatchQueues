from pathlib import Path
import pandas as pd
import pytest

def test_load_dataframe_selects_present_columns(mod, tmp_path: Path, monkeypatch):
    # Skip gracefully if pyarrow isn't installed (needed by pandas.to_parquet)
    pytest.importorskip("pyarrow")

    # Arrange: create a parquet with a subset of columns your loader can handle
    df = pd.DataFrame({
        "User": ["alice", "bob"],
        "CumulativeSlotTime": [10, 20],
        "JobsubClientIpAddress": ["10.0.0.1", "10.0.0.2"],
        "Cmd": ["echo hi", "echo bye"],
        "Environment": ["{}", "{}"],
        # Intentionally omit some optional columns like RequestMemory, etc.
    })

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(data_dir / "part.parquet")  # requires pyarrow

    # Act
    out = mod.load_dataframe(str(data_dir))

    # Assert: present columns should survive
    assert "User" in out.columns
    assert "Cmd" in out.columns
    assert "Environment" in out.columns
    assert len(out) == 2
