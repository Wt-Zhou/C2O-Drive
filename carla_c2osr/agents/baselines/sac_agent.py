from __future__ import annotations
from dataclasses import dataclass
from carla_c2osr.agents.base import BaseAgent
from carla_c2osr.env.types import WorldState, EgoControl


@dataclass
class SACConfig:
    seed: int = 0


class SACAgent(BaseAgent):
    def __init__(self, cfg: SACConfig) -> None:
        self.cfg = cfg

    def reset(self) -> None:  # type: ignore[override]
        pass

    def update(self, world: WorldState) -> None:  # type: ignore[override]
        pass

    def act(self, world: WorldState) -> EgoControl:  # type: ignore[override]
        return EgoControl(throttle=0.0, steer=0.0, brake=0.0)
