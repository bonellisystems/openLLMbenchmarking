import yaml
from pathlib import Path
def test_axis_cal_pairs_present_and_pinned():
    for axis in ("axis5", "axis8"):
        d = yaml.safe_load(Path(f"grading/calibration/b2/{axis}.yaml").read_text(encoding="utf-8"))
        assert isinstance(d["strong"], str) and d["strong"].strip()
        assert isinstance(d["weak"], str) and d["weak"].strip()
        assert set(d["author"]) >= {"model", "prompt_sha", "params"}
