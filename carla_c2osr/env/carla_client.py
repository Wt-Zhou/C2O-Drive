from __future__ import annotations
from dataclasses import dataclass


@dataclass
class CarlaClient:
    """CARLA 客户端占位。实际实现应在此封装连接、同步步进、传感器注册等。"""
    host: str = "localhost"
    port: int = 2000

    def connect(self) -> None:
        raise NotImplementedError("CARLA not installed. This is a placeholder.")
