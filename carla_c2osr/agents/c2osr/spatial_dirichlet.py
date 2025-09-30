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


class MultiTimestepSpatialDirichletBank:
    """维护每个智能体在多个时间步的空间Dirichlet分布。
    
    为每个智能体在每个时间步维护一个独立的K维Dirichlet伪计数向量，
    支持多时间步转移概率建模。
    """

    def __init__(self, K: int, params: DirichletParams, horizon: int = 3) -> None:
        """初始化多时间步空间Dirichlet银行。
        
        Args:
            K: 网格单元总数
            params: Dirichlet参数配置
            horizon: 预测时间步数
        """
        assert K > 0, "Grid size K must be positive"
        assert horizon > 0, "Horizon must be positive"
        
        self.K = K
        self.params = params
        self.horizon = horizon
        
        # 每个智能体在每个时间步的alpha参数
        # agent_alphas[agent_id][timestep] = alpha_vector
        self.agent_alphas: Dict[int, Dict[int, np.ndarray]] = {}

    def init_agent(self, agent_id: int, reachable_sets: Dict[int, List[int]]) -> None:
        """为智能体在所有时间步初始化Dirichlet先验分布。
        
        Args:
            agent_id: 智能体ID
            reachable_sets: {timestep: [reachable_cell_indices]}
        """
        self.agent_alphas[agent_id] = {}
        
        for timestep in range(1, self.horizon + 1):
            alpha = np.full(self.K, self.params.alpha_out, dtype=float)
            
            if timestep in reachable_sets:
                reachable = reachable_sets[timestep]
                if len(reachable) > 0:
                    alpha_in_per_cell = self.params.alpha_in / len(reachable)
                    for cell_idx in reachable:
                        if 0 <= cell_idx < self.K:
                            alpha[cell_idx] = alpha_in_per_cell
            
            self.agent_alphas[agent_id][timestep] = alpha

    def update_with_softcount(self, agent_id: int, timestep: int, w: np.ndarray, lr: float = 1.0) -> None:
        """使用软计数更新指定时间步的Dirichlet参数。
        
        Args:
            agent_id: 智能体ID
            timestep: 时间步
            w: 软计数向量 (K维)
            lr: 学习率
        """
        if agent_id not in self.agent_alphas:
            raise ValueError(f"Agent {agent_id} not initialized")
        
        if timestep not in self.agent_alphas[agent_id]:
            raise ValueError(f"Timestep {timestep} not initialized for agent {agent_id}")
        
        # 更新alpha参数：alpha_new = alpha_old + lr * w
        self.agent_alphas[agent_id][timestep] += lr * w

    def get_agent_alpha(self, agent_id: int, timestep: int) -> np.ndarray:
        """获取智能体在指定时间步的alpha参数。"""
        if agent_id not in self.agent_alphas:
            raise ValueError(f"Agent {agent_id} not initialized")
        
        if timestep not in self.agent_alphas[agent_id]:
            raise ValueError(f"Timestep {timestep} not initialized for agent {agent_id}")
        
        return self.agent_alphas[agent_id][timestep].copy()

    def posterior_mean(self, agent_id: int, timestep: int) -> np.ndarray:
        """计算智能体在指定时间步的后验均值概率。"""
        alpha = self.get_agent_alpha(agent_id, timestep)
        return alpha / alpha.sum()

    def sample_trajectory(self, agent_id: int) -> Dict[int, np.ndarray]:
        """从智能体的多时间步Dirichlet分布中采样一条完整轨迹。
        
        Returns:
            {timestep: probability_vector} 每个时间步的概率分布
        """
        if agent_id not in self.agent_alphas:
            raise ValueError(f"Agent {agent_id} not initialized")
        
        trajectory = {}
        for timestep in range(1, self.horizon + 1):
            if timestep in self.agent_alphas[agent_id]:
                alpha = self.agent_alphas[agent_id][timestep]
                # 从Dirichlet分布采样
                prob_vector = np.random.dirichlet(alpha)
                trajectory[timestep] = prob_vector
        
        return trajectory

    def l1_radius(self, agent_id: int, timestep: int) -> float:
        """计算智能体在指定时间步的L1置信半径。"""
        alpha = self.get_agent_alpha(agent_id, timestep)
        alpha_sum = alpha.sum()
        
        if alpha_sum <= 0:
            return float('inf')
        
        # 计算L1置信半径
        term1 = math.sqrt(math.log(2.0 / self.params.delta) / (2 * alpha_sum))
        term2 = math.log(2.0 / self.params.delta) / (3 * alpha_sum)
        
        return self.params.cK * (term1 + term2)


class OptimizedMultiTimestepSpatialDirichletBank:
    """终极优化版本：维度仅等于可达集大小的多时间步空间狄利克雷银行
    
    核心优化：
    1. 每个时间步的Dirichlet分布维度只等于该时间步的可达集大小
    2. 直接在可达集上操作，无需后处理
    3. 支持高效的期望计算，完全消除采样
    """

    def __init__(self, K: int, params: DirichletParams, horizon: int = 8) -> None:
        """初始化优化的多时间步空间Dirichlet银行。
        
        Args:
            K: 网格单元总数（用于兼容性，实际维度会动态调整）
            params: Dirichlet参数配置
            horizon: 时间范围
        """
        self.K = K
        self.params = params
        self.horizon = horizon
        
        # 存储每个agent在每个时间步的alpha参数和可达集
        # agent_alphas[agent_id][timestep] = np.array of size len(reachable_set)
        # agent_reachable_sets[agent_id][timestep] = List[int] 可达集的cell indices
        self.agent_alphas: Dict[int, Dict[int, np.ndarray]] = {}
        self.agent_reachable_sets: Dict[int, Dict[int, List[int]]] = {}

    def init_agent(self, agent_id: int, reachable_sets: Dict[int, List[int]]) -> None:
        """为智能体初始化优化的Dirichlet先验分布。
        
        Args:
            agent_id: 智能体ID
            reachable_sets: {timestep: [reachable_cell_indices]} 每个时间步的可达集
        """
        self.agent_alphas[agent_id] = {}
        self.agent_reachable_sets[agent_id] = {}
        
        # 计算均匀分配的alpha_in值
        for timestep, reachable in reachable_sets.items():
            if len(reachable) == 0:
                continue
                
            # 存储可达集
            self.agent_reachable_sets[agent_id][timestep] = reachable.copy()
            
            # 初始化alpha：维度只等于可达集大小，每个位置都是alpha_in_per_cell
            alpha_in_per_cell = self.params.alpha_in / len(reachable)
            self.agent_alphas[agent_id][timestep] = np.full(len(reachable), alpha_in_per_cell)

    def update_with_softcount(self, agent_id: int, timestep: int, 
                            historical_cells: List[int], lr: float = 1.0) -> None:
        """使用历史数据更新优化的Dirichlet分布。
        
        Args:
            agent_id: 智能体ID
            timestep: 时间步
            historical_cells: 历史观测的cell indices
            lr: 学习率
        """
        if agent_id not in self.agent_alphas:
            raise ValueError(f"Agent {agent_id} not initialized")
        
        if timestep not in self.agent_alphas[agent_id]:
            raise ValueError(f"Timestep {timestep} not initialized for agent {agent_id}")
        
        reachable_cells = self.agent_reachable_sets[agent_id][timestep]
        alpha = self.agent_alphas[agent_id][timestep]
        
        # 构建软计数：只对可达集内的历史数据计数
        soft_count = np.zeros(len(reachable_cells))
        for cell in historical_cells:
            if cell in reachable_cells:
                idx = reachable_cells.index(cell)  # 找到在可达集中的索引
                soft_count[idx] += lr
        
        # 更新alpha参数
        self.agent_alphas[agent_id][timestep] += soft_count

    def sample_transition_distributions(self, agent_id: int, n_samples: int = 20) -> Dict[int, List[np.ndarray]]:
        """采样多个transition分布组合。
        
        Returns:
            {timestep: [prob_vector_1, prob_vector_2, ...]} 每个样本的概率分布
        """
        if agent_id not in self.agent_alphas:
            raise ValueError(f"Agent {agent_id} not initialized")
        
        distributions = {}
        for timestep in self.agent_alphas[agent_id]:
            alpha = self.agent_alphas[agent_id][timestep]
            samples = []
            for _ in range(n_samples):
                # 直接在可达集维度上采样
                prob_vector = np.random.dirichlet(alpha)
                samples.append(prob_vector)
            distributions[timestep] = samples
        
        return distributions

    def get_reachable_sets(self, agent_id: int) -> Dict[int, List[int]]:
        """获取智能体的可达集。"""
        if agent_id not in self.agent_reachable_sets:
            raise ValueError(f"Agent {agent_id} not initialized")
        return self.agent_reachable_sets[agent_id].copy()

    def posterior_mean(self, agent_id: int, timestep: int) -> np.ndarray:
        """计算智能体在指定时间步的后验均值概率（在完整K维空间中）。"""
        if agent_id not in self.agent_alphas:
            raise ValueError(f"Agent {agent_id} not initialized")
        
        if timestep not in self.agent_alphas[agent_id]:
            raise ValueError(f"Timestep {timestep} not initialized for agent {agent_id}")
        
        # 获取可达集上的后验均值
        alpha = self.agent_alphas[agent_id][timestep]
        reachable_cells = self.agent_reachable_sets[agent_id][timestep]
        prob_reachable = alpha / alpha.sum()
        
        # 映射到完整的K维空间
        full_prob = np.zeros(self.K)
        for i, cell in enumerate(reachable_cells):
            full_prob[cell] = prob_reachable[i]
        
        return full_prob

    def get_agent_alpha(self, agent_id: int, timestep: int) -> np.ndarray:
        """获取智能体在指定时间步的alpha参数（兼容性方法）。"""
        if agent_id not in self.agent_alphas:
            raise ValueError(f"Agent {agent_id} not initialized")
        
        if timestep not in self.agent_alphas[agent_id]:
            raise ValueError(f"Timestep {timestep} not initialized for agent {agent_id}")
        
        return self.agent_alphas[agent_id][timestep].copy()
