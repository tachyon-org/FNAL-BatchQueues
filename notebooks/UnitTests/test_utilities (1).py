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

# ---- site_jobs_payload: case-insensitive & structure checks ----------------

def test_site_jobs_payload_case_insensitive_and_content(mod):
    df = pd.DataFrame({
        mod.USER_COL: ["Alice", "bob", "carol"],
        "RequestMemory": [2000, 1000, 500],
        "CumulativeSlotTime": [727, 11, 3],
        mod.IP_COL: ["111.111.111.111", "10.0.0.1", "8.8.8.8"],
        "MATCH_EXP_JOB_Site": ["example", "EXAMPLE", "OTHER"],
        "DAG_NodesFailed": [None, 0, None],
        "NumJobCompletions": ["1", "1", "0"],
        "NumJobStarts": [1, 1, 0],
        "Cmd": ["/path/a.sh", "/path/b.sh", "/path/c.sh"],
        "Environment": ["ENV=alice", "ENV=bob", "ENV=carol"],
    })

    payload = mod.site_jobs_payload(df, site="EXAMPLE", case_insensitive=True)
    # meta
    assert payload["meta"]["site"] == "EXAMPLE"
    assert payload["meta"]["total_jobs_at_site"] == 2
    # columns_included should match df_site columns (same as df here)
    assert set(payload["meta"]["columns_included"]) == set(df.columns)

    jobs = payload["jobs_at_site"]
    assert isinstance(jobs, list) and len(jobs) == 2
    # ensure it returned dict records with expected keys
    for rec in jobs:
        assert set(df.columns).issubset(rec.keys())

    # spot-check contents for both matching rows
    sites = {rec["MATCH_EXP_JOB_Site"] for rec in jobs}
    assert sites == {"example", "EXAMPLE"}
    users = {rec[mod.USER_COL] for rec in jobs}
    assert users == {"Alice", "bob"}

def test_site_jobs_payload_case_sensitive(mod):
    df = pd.DataFrame({
        mod.USER_COL: ["Alice", "bob"],
        "MATCH_EXP_JOB_Site": ["example", "EXAMPLE"],
        "Cmd": ["a", "b"],
        "Environment": ["A=1", "B=2"],
    })

    # Case-sensitive: only exact case "EXAMPLE" should match → 1 job
    payload_cs = mod.site_jobs_payload(df, site="EXAMPLE", case_insensitive=False)
    assert payload_cs["meta"]["total_jobs_at_site"] == 1
    assert payload_cs["jobs_at_site"][0][mod.USER_COL] == "bob"

    # Case-insensitive: both should match → 2 jobs
    payload_ci = mod.site_jobs_payload(df, site="EXAMPLE", case_insensitive=True)
    assert payload_ci["meta"]["total_jobs_at_site"] == 2
    assert {r[mod.USER_COL] for r in payload_ci["jobs_at_site"]} == {"Alice", "bob"}

def test_site_jobs_payload_missing_site_col(mod):
    df = pd.DataFrame({
        mod.USER_COL: ["zoe"],
        "Cmd": ["echo ok"],
        "Environment": ["X=1"],
        # Intentionally omit MATCH_EXP_JOB_Site
    })
    payload = mod.site_jobs_payload(df, site="EXAMPLE", site_col="MATCH_EXP_JOB_Site")
    assert payload["jobs_at_site"] == []
    assert payload["meta"]["total_jobs_at_site"] == 0
    assert "Missing column" in payload["meta"]["note"]
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
