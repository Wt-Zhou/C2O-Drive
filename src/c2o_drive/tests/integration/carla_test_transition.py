from __future__ import annotations
from c2o_drive.algorithms.c2osr.transition import sample_dirichlet


def test_sample_dirichlet_sum_to_one() -> None:
    z = [0.2, 0.3, 0.5]
    t = sample_dirichlet(z, concentration=10.0)
    assert abs(sum(t) - 1.0) < 1e-6
