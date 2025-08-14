from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple


Vector2 = Tuple[float, float]


@dataclass
class AgentState:
    """环境中其他交通参与者状态。

    Attributes:
        agent_id: 参与者唯一标识。
        position_m: (x, y) 位置，单位米。
        velocity_mps: (vx, vy) 速度，单位米/秒。
    """
    agent_id: str
    position_m: Vector2
    velocity_mps: Vector2


@dataclass
class EgoState:
    """自车状态。"""
    position_m: Vector2
    velocity_mps: Vector2
    heading_rad: float


@dataclass
class EgoControl:
    """自车控制。值域限制在 [0,1] 或 [-1,1]，具体由下游 wrapper 处理。"""
    throttle: float
    steer: float
    brake: float


@dataclass
class WorldState:
    """世界状态快照。"""
    time_s: float
    ego: EgoState
    agents: List[AgentState]


@dataclass
class Trajectory:
    """自车规划轨迹（占位）。"""
    states: List[EgoState]
    controls: List[EgoControl]


@dataclass
class OccupancyGrid:
    """占据概率栅格（局部）。"""
    width_cells: int
    height_cells: int
    cell_size_m: float
    origin_m: Vector2
    data: List[float]  # 行优先扁平化，范围 [0,1]
