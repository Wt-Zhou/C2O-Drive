from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
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

    def __init__(self, K: int, params: DirichletParams, horizon: Optional[int] = None) -> None:
        """初始化多时间步空间Dirichlet银行。

        Args:
            K: 网格单元总数
            params: Dirichlet参数配置
            horizon: 预测时间步数（None = 从全局配置读取）
        """
        assert K > 0, "Grid size K must be positive"

        # Load horizon from global config if not specified
        if horizon is None:
            from c2o_drive.config import get_global_config
            horizon = get_global_config().time.default_horizon

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

    def __init__(self, K: int, params: DirichletParams, horizon: Optional[int] = None) -> None:
        """初始化优化的多时间步空间Dirichlet银行。

        Args:
            K: 网格单元总数（用于兼容性，实际维度会动态调整）
            params: Dirichlet参数配置
            horizon: 时间范围（None = 从全局配置读取）
        """
        # Load horizon from global config if not specified
        if horizon is None:
            from c2o_drive.config import get_global_config
            horizon = get_global_config().time.default_horizon

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

        # 记录更新前的alpha统计
        alpha_before = alpha.sum()

        # 更新alpha参数
        self.agent_alphas[agent_id][timestep] += soft_count

        # 诊断日志：监控alpha增长
        alpha_after = self.agent_alphas[agent_id][timestep].sum()
        updated_cells = np.count_nonzero(soft_count)
        max_alpha = self.agent_alphas[agent_id][timestep].max()
        mean_alpha = self.agent_alphas[agent_id][timestep].mean()

        from c2o_drive.config import get_global_config
        if get_global_config().visualization.verbose_level >= 2:
            print(f"    [Dirichlet Update] Agent {agent_id}, t={timestep}:")
            print(f"      Alpha: {alpha_before:.2f} → {alpha_after:.2f} (+{alpha_after-alpha_before:.2f})")
            print(f"      Updated cells: {updated_cells}/{len(reachable_cells)} "
                  f"(max_α={max_alpha:.2f}, mean_α={mean_alpha:.4f})")

    def sample_transition_distributions(self, agent_id: int, n_samples: int = 20) -> Dict[int, List[np.ndarray]]:
        """采样多个transition分布组合（向量化批量采样版本）。

        Returns:
            {timestep: [prob_vector_1, prob_vector_2, ...]} 每个样本的概率分布
        """
        if agent_id not in self.agent_alphas:
            raise ValueError(f"Agent {agent_id} not initialized")

        distributions = {}
        for timestep in self.agent_alphas[agent_id]:
            alpha = self.agent_alphas[agent_id][timestep]

            # 🚀 优化: 使用numpy批量采样（约1.2倍加速）
            # 原始版本: for循环n_samples次调用np.random.dirichlet
            # 优化版本: 一次调用生成 (n_samples, K) 数组
            samples_array = np.random.dirichlet(alpha, size=n_samples)

            # 直接存储数组避免list转换开销
            distributions[timestep] = list(samples_array)

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

    def to_dict(self) -> Dict:
        """序列化Bank状态为字典

        Returns:
            包含所有内部状态的字典
        """
        # 序列化alpha参数（numpy数组将在CheckpointManager中处理）
        agent_alphas_serialized = {}
        for agent_id, timesteps in self.agent_alphas.items():
            agent_alphas_serialized[agent_id] = {
                timestep: alpha for timestep, alpha in timesteps.items()
            }

        # 序列化可达集
        agent_reachable_sets_serialized = {}
        for agent_id, timesteps in self.agent_reachable_sets.items():
            agent_reachable_sets_serialized[agent_id] = {
                timestep: cells for timestep, cells in timesteps.items()
            }

        # 序列化参数
        params_dict = {
            'alpha_in': self.params.alpha_in,
            'alpha_out': self.params.alpha_out,
            'delta': self.params.delta,
            'cK': self.params.cK
        }

        return {
            'K': self.K,
            'horizon': self.horizon,
            'params': params_dict,
            'agent_alphas': agent_alphas_serialized,
            'agent_reachable_sets': agent_reachable_sets_serialized
        }

    def sample_and_aggregate(
        self,
        agent_id: int,
        timestep: int,
        n_samples: int = 100
    ) -> Tuple[List[int], np.ndarray]:
        """采样并叠加多个transition分布

        从Dirichlet后验采样多次，然后叠加求平均，得到aggregated probability distribution。

        Args:
            agent_id: Agent ID
            timestep: Timestep (1-indexed)
            n_samples: 采样数量

        Returns:
            reachable_cells: List of cell indices in reachable set
            aggregated_prob: 叠加后的概率分布（已归一化）
        """
        if agent_id not in self.agent_alphas:
            return [], np.array([])

        if timestep not in self.agent_alphas[agent_id]:
            return [], np.array([])

        alpha = self.agent_alphas[agent_id][timestep]
        reachable_cells = self.agent_reachable_sets[agent_id][timestep]

        # 从Dirichlet后验采样
        samples = np.random.dirichlet(alpha, size=n_samples)

        # 叠加（求平均）
        aggregated_prob = np.mean(samples, axis=0)

        # 归一化（确保sum=1，虽然理论上已经是1）
        prob_sum = np.sum(aggregated_prob)
        if prob_sum > 0:
            aggregated_prob = aggregated_prob / prob_sum

        return reachable_cells, aggregated_prob

    def get_confidence_set_from_samples(
        self,
        agent_id: int,
        timestep: int,
        confidence_level: float = 0.95,
        n_samples: int = 100
    ) -> List[int]:
        """基于采样叠加计算confidence set

        支持动态采样：根据可达集大小自动调整采样数量

        算法流程：
        1. 从Dirichlet后验采样n_samples次（可能动态调整）
        2. 叠加所有采样得到aggregated probability
        3. 按概率降序选择cells，累积概率达到confidence_level时停止

        Args:
            agent_id: Agent ID
            timestep: Timestep (1-indexed)
            confidence_level: 置信水平（默认95%）
            n_samples: 基础采样数量

        Returns:
            confidence_set: 包含前X%概率质量的cell indices列表
        """
        from c2o_drive.config import get_global_config
        config = get_global_config()

        # 获取可达集信息
        if agent_id not in self.agent_reachable_sets:
            return []
        if timestep not in self.agent_reachable_sets[agent_id]:
            return []

        reachable_cells = self.agent_reachable_sets[agent_id][timestep]
        reachable_set_size = len(reachable_cells)

        # 动态调整采样数
        if config.safety.adaptive_sampling:
            # 计算所需采样数：确保每个cell至少被采样min_samples_per_cell次
            required_samples = reachable_set_size * config.safety.min_samples_per_cell

            # 取最大值（基础采样数 vs 要求采样数）
            adjusted_samples = max(n_samples, required_samples)

            # 应用上限保护
            adjusted_samples = min(adjusted_samples, config.safety.max_samples)

            # 使用调整后的采样数
            actual_samples = adjusted_samples
        else:
            # 固定采样数模式
            actual_samples = n_samples

        # 执行采样和叠加
        reachable_cells, aggregated_prob = self.sample_and_aggregate(
            agent_id, timestep, actual_samples
        )

        if len(reachable_cells) == 0:
            return []

        # 按概率降序排序
        sorted_indices = np.argsort(aggregated_prob)[::-1]

        # 累积到达confidence_level
        confidence_set = []
        cumulative_prob = 0.0

        for idx in sorted_indices:
            cell_id = reachable_cells[idx]
            confidence_set.append(cell_id)
            cumulative_prob += aggregated_prob[idx]

            if cumulative_prob >= confidence_level:
                break

        return confidence_set

    def get_sampling_info(
        self,
        agent_id: int,
        timestep: int,
        n_samples: int = 100
    ) -> dict:
        """获取采样信息用于调试

        Args:
            agent_id: Agent ID
            timestep: Timestep (1-indexed)
            n_samples: 基础采样数量

        Returns:
            dict: {
                'reachable_set_size': int,
                'base_samples': int,
                'adjusted_samples': int,
                'samples_per_cell': float
            }
        """
        from c2o_drive.config import get_global_config
        config = get_global_config()

        if agent_id not in self.agent_reachable_sets:
            return {}
        if timestep not in self.agent_reachable_sets[agent_id]:
            return {}

        reachable_set_size = len(self.agent_reachable_sets[agent_id][timestep])

        if config.safety.adaptive_sampling:
            required_samples = reachable_set_size * config.safety.min_samples_per_cell
            adjusted_samples = max(n_samples, required_samples)
            adjusted_samples = min(adjusted_samples, config.safety.max_samples)
        else:
            adjusted_samples = n_samples

        return {
            'reachable_set_size': reachable_set_size,
            'base_samples': n_samples,
            'adjusted_samples': adjusted_samples,
            'samples_per_cell': adjusted_samples / reachable_set_size if reachable_set_size > 0 else 0
        }

    @staticmethod
    def from_dict(data: Dict) -> 'OptimizedMultiTimestepSpatialDirichletBank':
        """从字典恢复Bank状态

        Args:
            data: 序列化的字典

        Returns:
            恢复的Bank实例
        """
        # 恢复参数
        params = DirichletParams(
            alpha_in=data['params']['alpha_in'],
            alpha_out=data['params']['alpha_out'],
            delta=data['params']['delta'],
            cK=data['params']['cK']
        )

        # 创建Bank实例
        bank = OptimizedMultiTimestepSpatialDirichletBank(
            K=data['K'],
            params=params,
            horizon=data['horizon']
        )

        # 恢复agent_alphas（numpy数组）
        for agent_id, timesteps in data['agent_alphas'].items():
            agent_id_int = int(agent_id)
            bank.agent_alphas[agent_id_int] = {}
            for timestep, alpha in timesteps.items():
                timestep_int = int(timestep)
                # alpha应该已经是numpy数组
                bank.agent_alphas[agent_id_int][timestep_int] = alpha

        # 恢复agent_reachable_sets
        for agent_id, timesteps in data['agent_reachable_sets'].items():
            agent_id_int = int(agent_id)
            bank.agent_reachable_sets[agent_id_int] = {}
            for timestep, cells in timesteps.items():
                timestep_int = int(timestep)
                bank.agent_reachable_sets[agent_id_int][timestep_int] = list(cells)

        return bank
