from __future__ import annotations
from typing import List
import numpy as np


def sample_dirichlet(prob_vector: List[float], concentration: float) -> List[float]:
    """对给定概率向量 z 采样 T ~ Dirichlet(c * z)。"""
    z = np.asarray(prob_vector, dtype=float)
    z = np.clip(z, 1e-12, 1.0)
    z = z / z.sum()
    alpha = concentration * z
    sample = np.random.dirichlet(alpha)
    return sample.tolist()
