"""
Q值评估模块

提供基于Dirichlet分布的Q值评估和风险评估功能。
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional, Union
from carla_c2osr.env.types import EgoState, AgentState
from carla_c2osr.agents.c2osr.grid import GridMapper
from carla_c2osr.agents.c2osr.spatial_dirichlet import SpatialDirichletBank
from carla_c2osr.evaluation.rewards import RewardCalculator, DistanceBasedCollisionDetector


class QEvaluator:
    """Q值评估器"""
    
    def __init__(self,
                 reward_calculator: Optional[RewardCalculator] = None,
                 collision_detector: Optional[DistanceBasedCollisionDetector] = None,
                 default_n_samples: int = 10):
        """
        Args:
            reward_calculator: 奖励计算器
            collision_detector: 碰撞检测器
            default_n_samples: 默认采样数量
        """
        self.reward_calculator = reward_calculator or RewardCalculator()
        self.collision_detector = collision_detector or DistanceBasedCollisionDetector()
        self.default_n_samples = default_n_samples
    
    def evaluate_q_values(self,
                         bank: SpatialDirichletBank,
                         agent_id: int,
                         reachable: List[int],
                         ego_state: EgoState,
                         ego_next_state: EgoState,
                         agent_state: AgentState,
                         grid: GridMapper,
                         ego_trajectory_cells: List[int],
                         n_samples: Optional[int] = None,
                         rng: Optional[np.random.Generator] = None,
                         verbose: bool = False,
                         reference_path: Optional[List[Union[Tuple[float, float], np.ndarray]]] = None) -> List[float]:
        """评估智能体在当前状态下的Q值（reward期望）。

        Args:
            bank: Dirichlet Bank
            agent_id: 智能体ID
            reachable: 可达集
            ego_state: 自车当前状态
            ego_next_state: 自车下一状态
            agent_state: 智能体当前状态
            grid: 网格映射器
            ego_trajectory_cells: 自车轨迹格子ID列表
            n_samples: 采样数量
            rng: 随机数生成器
            verbose: 是否打印详细信息
            reference_path: 参考中心线路径，用于计算偏移惩罚

        Returns:
            n个采样对应的reward值列表
        """
        if rng is None:
            rng = np.random.default_rng()
        
        if n_samples is None:
            n_samples = self.default_n_samples
        
        # 获取当前alpha
        alpha = bank.get_agent_alpha(agent_id)
        
        # 从Dirichlet分布采样n次
        sampled_probs = rng.dirichlet(alpha, size=n_samples)
        
        # 早期检查：自车轨迹是否与agent可达集重叠
        overlap_exists = bool(set(ego_trajectory_cells) & set(reachable))

        if verbose:
            print(f"    Agent {agent_id} 可达集大小: {len(reachable)}, 自车轨迹格子: {ego_trajectory_cells}")
            if overlap_exists:
                overlap_cells = set(ego_trajectory_cells) & set(reachable)
                print(f"    检测到重叠格子: {list(overlap_cells)}")
            else:
                print(f"    无重叠 - 跳过碰撞检测")

        rewards = []
        gamma = self.reward_calculator.gamma

        for i, probs in enumerate(sampled_probs):
            # 提取可达集上的概率分布
            reachable_probs = probs[reachable]

            # 计算时序碰撞概率（考虑折扣因子）
            if overlap_exists:
                collision_prob, collision_count, collision_details = \
                    self.collision_detector.calculate_temporal_collision_probability(
                        reachable_probs, reachable, ego_trajectory_cells, gamma
                    )
            else:
                # 无重叠，跳过碰撞检测
                collision_prob = 0.0
                collision_count = 0
                collision_details = []

            # 创建智能体下一状态（简化，使用当前状态）
            agent_next_state = agent_state

            # 使用reward计算器计算reward（基于概率的碰撞惩罚 + 中心线偏移）
            reward = self.reward_calculator.calculate_reward(
                ego_state=ego_state,
                ego_next_state=ego_next_state,
                agent_state=agent_state,
                agent_next_state=agent_next_state,
                collision_probability=collision_prob,
                reference_path=reference_path
            )

            rewards.append(reward)

            if verbose:
                print(f"      采样{i+1}: 加权碰撞概率{collision_prob:.3f}, 碰撞格子数{collision_count}, "
                      f"总reward{reward:.2f}")
                if collision_details:
                    print(f"        碰撞详情: {collision_details[:3]}...")  # 只显示前3个
        
        return rewards
    
    def sample_agent_transitions(self, 
                                bank: SpatialDirichletBank, 
                                agent_id: int, 
                                reachable: List[int], 
                                n_samples: Optional[int] = None,
                                rng: Optional[np.random.Generator] = None) -> List[int]:
        """从Dirichlet分布采样智能体的可能转移。
        
        Args:
            bank: Dirichlet Bank
            agent_id: 智能体ID
            reachable: 可达集
            n_samples: 采样数量
            rng: 随机数生成器
            
        Returns:
            采样的转移单元ID列表
        """
        if rng is None:
            rng = np.random.default_rng()
        
        if n_samples is None:
            n_samples = self.default_n_samples
        
        # 获取当前alpha
        alpha = bank.get_agent_alpha(agent_id)
        
        # 从Dirichlet分布采样
        sampled_probs = rng.dirichlet(alpha, size=n_samples)
        
        # 对每个采样，选择概率最高的可达单元
        transitions = []
        for probs in sampled_probs:
            # 只在可达集内选择
            reachable_probs = probs[reachable]
            max_idx = np.argmax(reachable_probs)
            transitions.append(reachable[max_idx])
        
        return transitions

