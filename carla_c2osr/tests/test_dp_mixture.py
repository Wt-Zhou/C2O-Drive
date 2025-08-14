from __future__ import annotations
from carla_c2osr.agents.c2osr.dp_mixture import DPMixtureConfig, DirichletProcessMixture


def test_dp_update_and_sample() -> None:
    cfg = DPMixtureConfig(alpha=0.5, eta=1.0, add_thresh=0.15, grid_num_cells=16)
    dp = DirichletProcessMixture(cfg)
    for i in range(5):
        dp.update_with_observation(cell_index=i % 4)
    zs = dp.sample_z_vectors(4)
    assert len(zs) == 4
    assert all(abs(sum(z) - 1.0) < 1e-6 for z in zs)
