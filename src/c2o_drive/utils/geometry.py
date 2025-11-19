from __future__ import annotations
from typing import Tuple
import math


def normalize_angle(angle_rad: float) -> float:
    return (angle_rad + math.pi) % (2 * math.pi) - math.pi


def distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
