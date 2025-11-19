from __future__ import annotations
from typing import List

def success_rate(success_flags: List[bool]) -> float:
    return float(sum(1 for s in success_flags if s)) / max(1, len(success_flags))
