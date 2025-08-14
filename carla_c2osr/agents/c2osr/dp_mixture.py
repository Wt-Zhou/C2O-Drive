from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import numpy as np


@dataclass
class DPMixtureConfig:
    alpha: float
    eta: float
    add_thresh: float
    grid_num_cells: int


@dataclass
class DirichletProcessMixture:
    """简化的 DP 混合，占位实现：
    - 维护若干原子 z^{(k)}（在栅格上的概率向量）与其计数 n_k；
    - update_with_observation: 对观测 cell i 计算责任，软更新原子或开新原子；
    - sample_z_vectors: 从现有原子按权重采样 z（简化后验预测）。
    TODO: 使用严格的 CRP 后验与基测度 G0 采样。
    """

    cfg: DPMixtureConfig
    atoms: List[np.ndarray] = field(default_factory=list)
    counts: List[float] = field(default_factory=list)

    def _g0_prior(self) -> np.ndarray:
        vec = np.ones(self.cfg.grid_num_cells, dtype=float)
        vec /= vec.sum()
        return vec

    def update_with_observation(self, cell_index: int) -> None:
        if not self.atoms:
            z_new = self._g0_prior()
            self.atoms.append(z_new)
            self.counts.append(0.0)
        # 责任计算（简化）：r_k ∝ n_k * z_k[i]; r_new ∝ alpha * G0[i]
        g0 = self._g0_prior()
        scores = [n * z[cell_index] for z, n in zip(self.atoms, self.counts)]
        score_new = self.cfg.alpha * g0[cell_index]
        total = float(sum(scores) + score_new)
        if total <= 0:
            responsibilities = [0.0 for _ in scores]
            r_new = 1.0
        else:
            responsibilities = [s / total for s in scores]
            r_new = score_new / total
        # 软更新现有原子
        for k, z in enumerate(self.atoms):
            r_k = responsibilities[k] if k < len(responsibilities) else 0.0
            if r_k > 0.0:
                z *= (1.0 - self.cfg.eta * r_k)
                z[cell_index] += self.cfg.eta * r_k
                z /= z.sum() + 1e-12
                self.counts[k] += r_k
        # 可能新建原子
        if r_new >= self.cfg.add_thresh:
            z_new = self._g0_prior()
            z_new *= (1.0 - self.cfg.eta)
            z_new[cell_index] += self.cfg.eta
            z_new /= z_new.sum() + 1e-12
            self.atoms.append(z_new)
            self.counts.append(r_new)

    def sample_z_vectors(self, num_samples: int) -> List[List[float]]:
        if not self.atoms:
            base = self._g0_prior()
            return [base.tolist() for _ in range(num_samples)]
        weights = np.array([max(1e-8, c) for c in self.counts], dtype=float)
        weights = weights / weights.sum()
        idx = np.random.choice(len(self.atoms), size=num_samples, p=weights)
        return [self.atoms[i].tolist() for i in idx]

    def get_atom_vectors(self) -> List[List[float]]:
        return [z.tolist() for z in self.atoms]
