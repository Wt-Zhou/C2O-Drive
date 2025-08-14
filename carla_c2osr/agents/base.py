from __future__ import annotations
from typing import Protocol
from carla_c2osr.env.types import WorldState, EgoControl


class BaseAgent(Protocol):
    """智能体基础接口。"""

    def reset(self) -> None:
        """重置内部状态。"""
        ...

    def update(self, world: WorldState) -> None:
        """基于新观测更新内部模型（如 DP 后验）。"""
        ...

    def act(self, world: WorldState) -> EgoControl:
        """给出下一步控制动作。"""
        ...
