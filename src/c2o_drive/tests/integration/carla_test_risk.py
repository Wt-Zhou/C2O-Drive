from __future__ import annotations
from c2o_drive.algorithms.c2osr.risk import union_risk


def test_union_risk_bounds() -> None:
    p = union_risk([0.1, 0.2, 0.3])
    assert 0.0 <= p <= 1.0
    assert p >= max(0.1, 0.2, 0.3)
