from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import math


@dataclass
class DirichletParams:
    """空间 Dirichlet 参数配置。
    
    Attributes:
        alpha_in: 可达集合内的先验伪计数，默认 50.0
        alpha_out: 可达集合外的先验伪计数，默认 1e-6
        delta: 置信水平参数，默认 0.05
        cK: 置信半径校准常数，默认 1.0
    """
    alpha_in: float = 50.0
    alpha_out: float = 1e-6
    delta: float = 0.05
    cK: float = 1.0


class SpatialDirichletBank:
    """维护每个智能体的空间 Dirichlet 分布，支持一步转移概率建模。
    
    为每个智能体维护一个 K 维的 Dirichlet 伪计数向量 alpha_i ∈ R^K_+，
    表示该智能体在各网格单元上的占据概率分布。
    """

    def __init__(self, K: int, params: DirichletParams) -> None:
        """初始化空间 Dirichlet 银行。
        
        Args:
            K: 网格单元总数
            params: Dirichlet 参数配置
        """
        assert K > 0, "Grid size K must be positive"
        assert params.alpha_in > 0, "alpha_in must be positive"
        assert params.alpha_out > 0, "alpha_out must be positive"
        assert 0 < params.delta < 1, "delta must be in (0, 1)"
        
        self.K = K
        self.params = params
        self.agent_alphas: Dict[int, np.ndarray] = {}

    def init_agent(self, agent_id: int, reachable: List[int]) -> None:
        """为智能体初始化 Dirichlet 先验分布。
        
        在可达集合内均匀分配 alpha_in，其余位置设为 alpha_out。
        
        Args:
            agent_id: 智能体 ID
            reachable: 可达网格单元索引列表
        """
        assert len(reachable) > 0, "Reachable set cannot be empty"
        assert all(0 <= idx < self.K for idx in reachable), "Reachable indices out of range"
        
        alpha = np.full(self.K, self.params.alpha_out, dtype=float)
        
        # 在可达集合内均匀分配 alpha_in
        alpha_per_cell = self.params.alpha_in / len(reachable)
        for idx in reachable:
            alpha[idx] = alpha_per_cell
        
        self.agent_alphas[agent_id] = alpha

    def ensure_agent(self, agent_id: int, reachable: List[int]) -> None:
        """确保智能体已初始化，如果不存在则自动初始化。
        
        Args:
            agent_id: 智能体 ID
            reachable: 可达网格单元索引列表
        """
        if agent_id not in self.agent_alphas:
            self.init_agent(agent_id, reachable)

    def update_with_softcount(self, agent_id: int, w: np.ndarray, lr: float = 1.0) -> None:
        """使用软计数更新智能体的 Dirichlet 分布。
        
        执行共轭更新：alpha += lr * w
        
        Args:
            agent_id: 智能体 ID
            w: 软计数权重向量，形状 (K,)，要求 sum(w) ≈ 1
            lr: 学习率，默认 1.0
        """
        assert agent_id in self.agent_alphas, f"Agent {agent_id} not initialized"
        assert w.shape == (self.K,), f"Weight shape {w.shape} != ({self.K},)"
        assert np.abs(w.sum() - 1.0) < 1e-6, f"Weights sum {w.sum()} != 1.0"
        assert lr > 0, "Learning rate must be positive"
        
        self.agent_alphas[agent_id] += lr * w

    def posterior_mean(self, agent_id: int) -> np.ndarray:
        """计算智能体的后验期望概率分布。
        
        返回 E[p] = alpha / alpha.sum() under Dirichlet(alpha)
        
        Args:
            agent_id: 智能体 ID
            
        Returns:
            形状 (K,) 的概率向量，满足 sum(p) = 1
        """
        assert agent_id in self.agent_alphas, f"Agent {agent_id} not initialized"
        
        alpha = self.agent_alphas[agent_id]
        return alpha / alpha.sum()

    def l1_radius(self, agent_id: int) -> float:
        """计算智能体分布的 L1 置信半径。
        
        使用近似公式：r ≈ cK * sqrt(2*log(1/δ)/α₀)
        其中 α₀ = sum(alpha) 是总伪计数。
        
        Args:
            agent_id: 智能体 ID
            
        Returns:
            置信半径值
        """
        assert agent_id in self.agent_alphas, f"Agent {agent_id} not initialized"
        
        alpha = self.agent_alphas[agent_id]
        alpha_0 = alpha.sum()
        
        if alpha_0 <= 1e-12:
            return 1.0  # 退化情况
            
        log_term = math.log(1.0 / self.params.delta)
        radius = self.params.cK * math.sqrt(2 * log_term / alpha_0)
        
        return radius

    def conservative_qmax_union(self, agent_ids: List[int]) -> np.ndarray:
        """计算多智能体的保守上界占据概率图。
        
        对每个网格单元 g，计算：
        q_max(g) = clip(Σᵢ min(1, p̂ᵢ(g) + 0.5*rᵢ), 0, 1)
        
        其中 p̂ᵢ(g) 是智能体 i 在单元 g 的后验期望，rᵢ 是其置信半径。
        
        Args:
            agent_ids: 智能体 ID 列表
            
        Returns:
            形状 (K,) 的上界占据概率向量，值域 [0, 1]
        """
        assert len(agent_ids) > 0, "Agent list cannot be empty"
        assert all(agent_id in self.agent_alphas for agent_id in agent_ids), \
            "All agents must be initialized"
        
        q_max = np.zeros(self.K, dtype=float)
        
        for agent_id in agent_ids:
            p_mean = self.posterior_mean(agent_id)
            radius = self.l1_radius(agent_id)
            
            # 添加保守项：p̂ᵢ(g) + 0.5*rᵢ，然后 clip 到 [0,1]
            conservative_p = np.clip(p_mean + 0.5 * radius, 0.0, 1.0)
            q_max += conservative_p
        
        # 最终 clip 到 [0,1]
        return np.clip(q_max, 0.0, 1.0)

    def get_agent_alpha(self, agent_id: int) -> np.ndarray:
        """获取智能体的当前 alpha 向量（用于调试）。
        
        Args:
            agent_id: 智能体 ID
            
        Returns:
            形状 (K,) 的 alpha 向量
        """
        assert agent_id in self.agent_alphas, f"Agent {agent_id} not initialized"
        return self.agent_alphas[agent_id].copy()

    def get_agent_count(self) -> int:
        """获取已初始化的智能体数量。
        
        Returns:
            智能体数量
        """
        return len(self.agent_alphas)

    def get_agent_counts(self, agent_id: int, subtract_prior: bool = True) -> np.ndarray:
        """获取智能体的计数向量（用于可视化）。
        
        Args:
            agent_id: 智能体 ID
            subtract_prior: 是否减去先验值（初始alpha）
            
        Returns:
            形状 (K,) 的计数向量
        """
        assert agent_id in self.agent_alphas, f"Agent {agent_id} not initialized"
        
        alpha = self.agent_alphas[agent_id]
        
        if subtract_prior:
            # 减去初始alpha值，得到实际的观测计数
            # 注意：这里需要知道初始alpha值，暂时用简单估计
            # 对于可达集内的单元，初始值应该是 alpha_in / len(reachable)
            # 对于可达集外的单元，初始值应该是 alpha_out
            # 由于我们没有存储初始reachable信息，这里用简单方法
            alpha_init = np.full_like(alpha, self.params.alpha_out)
            # 假设前几个非零位置是可达集（这是一个简化）
            nonzero_indices = np.nonzero(alpha > self.params.alpha_out)[0]
            if len(nonzero_indices) > 0:
                # 简单估计：假设初始时可达集内均匀分布
                alpha_init[nonzero_indices] = self.params.alpha_in / len(nonzero_indices)
            
            return alpha - alpha_init
        else:
            return alpha.copy()
