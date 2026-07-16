"""P0 artifact freeze: SHA256 + download_date (TESTPLAN 3.5). Later mismatch = recorded finding, never auto-refetch."""
import hashlib
import sys
from datetime import date
from pathlib import Path

import yaml


def _sha256(path: Path, chunk=1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def freeze(registry_path: Path | str, models: list[str] | None = None) -> dict:
    registry_path = Path(registry_path)
    data = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    out = {}
    for name, m in data["models"].items():
        if models and name not in models:
            continue
        prov = m.get("provenance", {})
        if prov.get("sha256") not in (None, "TO-FREEZE"):
            out[name] = "already-frozen"
            continue
        p = Path(str(m.get("local_path", "")))
        if not p.is_file():
            out[name] = "missing-file"
            continue
        prov["sha256"] = _sha256(p)
        prov["download_date"] = str(date.today())
        m["provenance"] = prov
        out[name] = "frozen"
    registry_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return out


if __name__ == "__main__":
    result = freeze(Path(__file__).resolve().parents[1] / "config" / "registry.yaml",
                    models=sys.argv[1:] or None)
    for k, v in sorted(result.items()):
        print(f"{k}: {v}")
    sys.exit(1 if "missing-file" in result.values() else 0)
