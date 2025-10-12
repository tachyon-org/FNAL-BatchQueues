import pandas as pd

def _sample_df(mod):
    return pd.DataFrame({
        mod.USER_COL: ["alice", "bob", "alice", ""],
        mod.IP_COL:   ["192.168.1.1", "10.0.0.1", "invalid", "8.8.8.8"],
        "MATCH_EXP_JOB_Site": ["EXAMPLE", "OTHER", "EXAMPLE", "EXAMPLE"],
        mod.NUM_STARTS_COL: [1, 1, 1, 0],
        mod.NUM_COMPLETIONS_COL: [1, 0, 1, 0],
        "Cmd": ["echo hi", "run job", "sleep 1", "noop"],
        "Environment": ['{"OMP_NUM_THREADS":"4"}', 'PATH=/usr/bin', '', 'KEY=VAL'],
    })

def test_build_obfuscations_and_jagged(mod):
    df = _sample_df(mod)
    users_dict, ips_dict, user_mapper, ip_mapper = mod.build_obfuscations(df)
    assert len(users_dict) >= 3
    assert len(ips_dict) >= 3
    uj = mod.to_jagged_array(users_dict)
    ij = mod.to_jagged_array(ips_dict)
    assert all(len(row) == 3 for row in uj)
    assert all(len(row) == 3 for row in ij)

def test_make_output_and_summary_payloads(mod):
    df = _sample_df(mod)
    users_dict, ips_dict, *_ = mod.build_obfuscations(df)
    out_json = mod.make_output_json(df, users_dict, ips_dict)
    assert '"total_rows"' in out_json

    summary = mod.make_summary_payload(df, users_dict, ips_dict)
    assert "users" in summary and "ips" in summary and "user_ip_correlations" in summary
    assert summary["meta"]["total_rows"] == len(df)

def test_failed_users_payload_counts(mod):
    df = _sample_df(mod)
    users_dict, ips_dict, user_mapper, _ = mod.build_obfuscations(df)
    payload = mod.failed_users_payload(df, user_mapper=user_mapper)
    assert "failed_users" in payload
    assert isinstance(payload["failed_users"], list)

def test_site_users_payload_with_filtering(mod):
    df = _sample_df(mod)
    # Add a token-looking user at EXAMPLE
    df.loc[len(df)] = ["2bb4d3bkyc4cy6b5b7hkxm", "1.1.1.1", "EXAMPLE", 1, 1, "cmd", "ENV=1"]
    payload = mod.site_users_payload(df, site="EXAMPLE", filter_tokens=True)
    assert "users_at_site" in payload
    assert "alice" in payload["users_at_site"]
    assert "2bb4d3bkyc4cy6b5b7hkxm" not in payload["users_at_site"]
