import hashlib
import yaml
from scripts.freeze_artifacts import freeze

def test_freeze_hashes_file_and_writes_provenance(tmp_path):
    blob = tmp_path / "model.gguf"; blob.write_bytes(b"weights")
    reg = tmp_path / "registry.yaml"
    reg.write_text(yaml.safe_dump({"models": {"m1": {
        "local_path": str(blob),
        "provenance": {"source_repo": "x/y", "download_date": "TO-FREEZE",
                       "sha256": "TO-FREEZE", "v1_continuity": True}}}}))
    assert freeze(reg) == {"m1": "frozen"}
    d = yaml.safe_load(reg.read_text())
    assert d["models"]["m1"]["provenance"]["sha256"] == hashlib.sha256(b"weights").hexdigest()
    assert d["models"]["m1"]["provenance"]["download_date"] != "TO-FREEZE"
    assert freeze(reg) == {"m1": "already-frozen"}
