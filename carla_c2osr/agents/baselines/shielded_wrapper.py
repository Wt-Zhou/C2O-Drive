from __future__ import annotations
from dataclasses import dataclass
from carla_c2osr.agents.base import BaseAgent
from carla_c2osr.env.types import WorldState, EgoControl


@dataclass
class ShieldConfig:
    epsilon: float = 0.1


class ShieldedWrapper(BaseAgent):
    def __init__(self, inner: BaseAgent, cfg: ShieldConfig) -> None:
        self.inner = inner
        self.cfg = cfg

    def reset(self) -> None:  # type: ignore[override]
        self.inner.reset()

    def update(self, world: WorldState) -> None:  # type: ignore[override]
        self.inner.update(world)

    def act(self, world: WorldState) -> EgoControl:  # type: ignore[override]
        ctrl = self.inner.act(world)
        # 占位：不做修改
        return ctrl
