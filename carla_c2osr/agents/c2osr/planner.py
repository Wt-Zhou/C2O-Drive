from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

from carla_c2osr.env.types import WorldState, EgoState, Trajectory
from .risk import union_risk


@dataclass
class PlannerConfig:
    horizon: int
    samples: int
    risk_mode: str
    epsilon: float
    gamma: float


class C2OSRPlanner:
    """占位规划器：根据样本化占据概率，对轨迹进行最坏样本风险评估。"""

    def __init__(self, cfg: PlannerConfig) -> None:
        self.cfg = cfg

    def _trajectory_collision_prob(self, traj: Trajectory, q_layers: List[List[float]]) -> float:
        # 占位：使用每层平均占据概率与并集界作为风险
        per_step = []
        for _ in traj.states:
            if not q_layers:
                per_step.append(0.0)
            else:
                avg = float(np.mean(q_layers[0]))
                per_step.append(avg)
        return union_risk(per_step)

    def plan(self, world: WorldState, trajectories: List[Trajectory], samples_q: List[List[List[float]]]) -> Tuple[Trajectory, Dict]:
        # samples_q: [S][layer][cell_prob]
        best_traj = trajectories[0]
        best_score = -1e9
        info: Dict = {}
        for traj in trajectories:
            worst = 1.0
            for s_q in samples_q:
                risk = self._trajectory_collision_prob(traj, s_q)
                worst = min(worst, 1.0 - risk)  # 下界 = 1 - 风险
            if worst > best_score:
                best_score = worst
                best_traj = traj
        info["q_lower_bound"] = best_score
        return best_traj, info
