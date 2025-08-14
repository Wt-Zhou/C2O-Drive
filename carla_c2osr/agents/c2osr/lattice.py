from __future__ import annotations
from typing import List
from carla_c2osr.env.types import EgoState, Trajectory, EgoControl


def generate_lattice_controls(ego: EgoState, horizon: int) -> List[Trajectory]:
    """生成占位的 lattice 轨迹：直行三条（左/中/右轻微转向）。"""
    controls_sets = [
        [EgoControl(throttle=0.4, steer=-0.1, brake=0.0) for _ in range(horizon)],
        [EgoControl(throttle=0.4, steer=0.0, brake=0.0) for _ in range(horizon)],
        [EgoControl(throttle=0.4, steer=0.1, brake=0.0) for _ in range(horizon)],
    ]
    trajectories: List[Trajectory] = []
    for controls in controls_sets:
        states = [ego for _ in controls]
        trajectories.append(Trajectory(states=states, controls=controls))
    return trajectories
