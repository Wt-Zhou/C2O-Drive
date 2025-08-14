from __future__ import annotations

class ControlWrapper:
    """控制信号封装占位。"""

    def apply(self, throttle: float, steer: float, brake: float) -> None:
        raise NotImplementedError
