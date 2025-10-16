from pathlib import Path
import json

def test_mapper_sequences_and_dupes(mod, tmp_path: Path):
    m = mod.GarbleTokenMapper(prefix="UR", start=1)
    t1 = m.add("alice")
    t2 = m.add("bob")
    t1b = m.add("alice")  # duplicate increments count, same token
    assert t1 == "UR1"
    assert t2 == "UR2"
    assert t1b == "UR1"
    assert m._by_orig["alice"].count == 2

def test_mapper_export_import(mod, tmp_path: Path):
    m = mod.GarbleTokenMapper(prefix="IP", start=5)
    m.add("10.0.0.1")
    m.add("10.0.0.2")
    out = tmp_path / "map.json"
    m.export_to_json(out)

    m2 = mod.GarbleTokenMapper(prefix="IGNORED", start=1)
    m2.load_from_json(out)
    assert m2.prefix == "IP"
    assert m2._by_orig["10.0.0.1"].token == "IP5"
    assert m2._by_orig["10.0.0.2"].token == "IP6"
