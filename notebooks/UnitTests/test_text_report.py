from pathlib import Path
import pandas as pd
import re

def test_write_cmd_env_report_basic_creation(test_output_dir: Path, mod):
    df = pd.DataFrame({
        mod.USER_COL: ["alice"],
        "Cmd": ["python -c 'print(123)'"],
        "Environment": ['{"OMP_NUM_THREADS": "2"}'],
        "MATCH_EXP_JOB_Site": ["EXAMPLE"],
    })
    out_path = test_output_dir / "cmd_env_report_basic.txt"
    p = mod.write_cmd_env_report(df, out_path, human_wrap=80, include_meta=True)
    assert p.exists()
    txt = p.read_text(encoding="utf-8")
    assert "Cmd:" in txt and "Environment:" in txt and "alice" in txt
