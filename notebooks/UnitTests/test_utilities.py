# notebooks/UnitTests/test_utilities.py
from pathlib import Path
import json
import pandas as pd
import pytest

# ---- GarbleTokenMapper internals -------------------------------------------

def test_extract_trailing_int_edges(mod):
    f = mod.GarbleTokenMapper._extract_trailing_int
    assert f("UR123") == 123
    assert f("abc0009") == 9
    assert f("no-digits") is None
    assert f("") is None
    assert f(None) is None

# ---- dump_json / to_jagged_array ------------------------------------------

def test_dump_json_and_to_jagged_array(tmp_path: Path, mod):
    # to_jagged_array with empty dict -> empty list
    assert mod.to_jagged_array({}) == []

    # dump_json writes a valid, pretty JSON file
    payload = {"ok": True, "n": 3}
    out = tmp_path / "payload.json"
    mod.dump_json(payload, out)
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data == payload

# ---- load_dataframe error path --------------------------------------------

def test_load_dataframe_missing_dir_raises(mod):
    with pytest.raises(FileNotFoundError):
        mod.load_dataframe("__this_dir_should_not_exist__")

# ---- _s and _parse_env edge cases -----------------------------------------

def test__s_and_parse_env_variants(mod):
    # _s should handle NaN/None/strings
    assert mod._s(None) == ""
    assert mod._s(pd.NA) == ""
    assert mod._s("x") == "x"

    # JSON dict
    pairs = mod._parse_env('{"A":"1","B":"2"}')
    assert ("A", "1") in pairs and ("B", "2") in pairs

    # JSON list of mixed items (dict + KEY=VAL + other)
    pairs2 = mod._parse_env('[{"K":"V"},"X=Y",123]')
    # We accept at least these two interpretations; exact order not guaranteed
    assert ("K", "V") in pairs2
    assert any(k == "X" and v == "Y" for k, v in pairs2) or ("ITEM", "123") in pairs2

    # KEY=VAL string separated by commas
    pairs3 = mod._parse_env("PATH=/usr/bin,OMP_NUM_THREADS=8")
    assert ("OMP_NUM_THREADS", "8") in pairs3

    # Fallback to raw string (no recognizable pairs)
    pairs4 = mod._parse_env("gibberish-without-equals")
    assert pairs4 and pairs4[0][0] in ("ENV", "gibberish-without-equals")  # tolerant

# ---- failed_users_payload: missing cols branch -----------------------------

def test_failed_users_payload_missing_required_cols(mod):
    # Missing NumJobStarts / NumJobCompletions -> returns empty with note
    df = pd.DataFrame({mod.USER_COL: ["alice", "bob"]})
    payload = mod.failed_users_payload(df, user_mapper=mod.GarbleTokenMapper(prefix="UR"))
    assert payload["failed_users"] == []
    assert payload["meta"]["distinct_failed_users"] == 0
    assert "note" in payload["meta"]

# ---- site_users_payload: case-insensitive + token toggle -------------------

def test_site_users_payload_case_and_token_toggle(mod):
    df = pd.DataFrame({
        mod.USER_COL: ["Alice", "2bb4d3bkyc4cy6b5b7hkxm", "bob"],
        "MATCH_EXP_JOB_Site": ["example", "EXAMPLE", "ExAmPlE"],
    })
    # No token filtering, case-insensitive site match
    p_all = mod.site_users_payload(df, site="EXAMPLE", case_insensitive=True, filter_tokens=False)
    assert set(p_all["users_at_site"]) == {"Alice", "2bb4d3bkyc4cy6b5b7hkxm", "bob"}

    # With token filtering
    p_filtered = mod.site_users_payload(df, site="EXAMPLE", case_insensitive=True, filter_tokens=True)
    assert "2bb4d3bkyc4cy6b5b7hkxm" not in p_filtered["users_at_site"]
    assert {"Alice", "bob"}.issubset(set(p_filtered["users_at_site"]))

# ---- write_cmd_env_report: missing meta col path ---------------------------

def test_write_cmd_env_report_missing_meta_cols_still_writes(mod, test_output_dir: Path, capsys):
    # Intentionally omit some meta columns; should warn then proceed
    df = pd.DataFrame({
        mod.USER_COL: ["zoe"],
        "Cmd": ["echo ok"],
        "Environment": ["A=1"],
    })
    out = test_output_dir / "cmd_env_report_missing_meta.txt"
    p = mod.write_cmd_env_report(df, out, include_meta=True)
    assert p.exists()
    txt = p.read_text(encoding="utf-8")
    assert "Cmd:" in txt and "Environment:" in txt and "zoe" in txt
