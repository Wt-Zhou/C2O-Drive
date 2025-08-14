from __future__ import annotations
from carla_c2osr.env.types import EgoState
from carla_c2osr.agents.c2osr.lattice import generate_lattice_controls


def test_generate_lattice_len() -> None:
    ego = EgoState(position_m=(0.0, 0.0), velocity_mps=(0.0, 0.0), heading_rad=0.0)
    trajs = generate_lattice_controls(ego, horizon=5)
    assert len(trajs) >= 3
    assert all(len(t.controls) == 5 for t in trajs)
