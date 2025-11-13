from __future__ import annotations
from dataclasses import dataclass
from carla_c2osr.agents.base import BaseAgent
from carla_c2osr.env.types import WorldState, EgoControl


@dataclass
class DQNConfig:
    """DQN配置参数"""
    seed: int = 0
    learning_rate: float = 1e-4
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: int = 10000


class DQNAgent(BaseAgent):
    """DQN基线算法（占位实现，待完善）"""

    def __init__(self, cfg: DQNConfig) -> None:
        self.cfg = cfg

    def reset(self) -> None:  # type: ignore[override]
        pass

    def update(self, world: WorldState) -> None:  # type: ignore[override]
        pass

    def act(self, world: WorldState) -> EgoControl:  # type: ignore[override]
        return EgoControl(throttle=0.0, steer=0.0, brake=0.0)
