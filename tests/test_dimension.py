import pytest
from llmtest.judging.dimension import Dim, resolve_dims, cal_ref

def test_b1_resolves_to_unit():
    assert resolve_dims(1, "b1.cybersecurity-01", None) == [Dim("unit", "cybersecurity")]

def test_b2_resolves_one_dim_per_judged_axis():
    assert resolve_dims(2, "b2.error-recovery-01", [1, 5]) == [Dim("axis", "axis5")]
    assert resolve_dims(2, "b2.faith-01", [5, 8]) == [Dim("axis", "axis5"), Dim("axis", "axis8")]

def test_b2_with_no_judged_axis_yields_nothing():
    assert resolve_dims(2, "b2.selection-01", [1, 2]) == []

def test_b1_bad_task_id_raises():
    with pytest.raises(ValueError):
        resolve_dims(1, "b2.error-recovery-01", None)

def test_b2_without_axes_raises():
    with pytest.raises(ValueError):
        resolve_dims(2, "b2.error-recovery-01", None)

def test_cal_ref_paths():
    assert cal_ref(Dim("axis", "axis5")).as_posix().endswith("grading/calibration/b2/axis5.yaml")

def test_b1_malformed_unit_missing_number_raises():
    # exercises _unit_from_b1's "no hyphen" ValueError (a b1.-prefixed id with no -NN),
    # which the existing b2.* test never reaches (it hits the outer prefix guard)
    with pytest.raises(ValueError):
        resolve_dims(1, "b1.cybersecurity", None)

def test_cal_ref_rejects_unit_dim():
    with pytest.raises(ValueError):
        cal_ref(Dim("unit", "cybersecurity"))
