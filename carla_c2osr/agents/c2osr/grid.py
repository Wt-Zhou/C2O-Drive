from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple


@dataclass
class GridSpec:
    size_m: float
    cell_m: float
    macro: bool

    @property
    def num_cells(self) -> int:
        side = int(self.size_m / self.cell_m)
        return side * side


class GridMapper:
    """将世界坐标映射到局部栅格索引（简化占位）。"""

    def __init__(self, spec: GridSpec) -> None:
        self.spec = spec
        self.side = int(spec.size_m / spec.cell_m)

    def world_to_cell(self, position_m: Tuple[float, float]) -> int:
        # 简化：以 (0,0) 为中心窗口
        half = self.spec.size_m / 2.0
        x, y = position_m
        ix = int((x + half) / self.spec.cell_m)
        iy = int((y + half) / self.spec.cell_m)
        ix = max(0, min(self.side - 1, ix))
        iy = max(0, min(self.side - 1, iy))
        return iy * self.side + ix
