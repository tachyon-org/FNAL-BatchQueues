from pathlib import Path
import pandas as pd

def test_is_valid_user_basic(mod):
    assert mod.is_valid_user("alice")
    assert mod.is_valid_user("  bob  ")
    assert not mod.is_valid_user("")
    assert not mod.is_valid_user(None)
    assert not mod.is_valid_user(pd.NA)

def test_is_valid_ipv4_variants(mod):
    assert mod.is_valid_ipv4("0.0.0.0")
    assert mod.is_valid_ipv4("192.168.1.1")
    assert mod.is_valid_ipv4("255.255.255.255")
    assert not mod.is_valid_ipv4("256.0.0.1")
    assert not mod.is_valid_ipv4("1.2.3")
    assert not mod.is_valid_ipv4("abc.def.ghi.jkl")
    assert not mod.is_valid_ipv4(None)
