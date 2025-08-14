from __future__ import annotations
from typing import List
import math


def union_risk(probabilities: List[float]) -> float:
    """并集界: 1 - ∏(1-p)。"""
    prod = 1.0
    for p in probabilities:
        prod *= (1.0 - max(0.0, min(1.0, p)))
    return 1.0 - prod


def independent_risk(probabilities: List[float]) -> float:
    """独立近似下的同一公式，与 union_risk 等价占位。"""
    return union_risk(probabilities)
