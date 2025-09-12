"""
Q值评估模块

提供基于Dirichlet分布的Q值评估和风险评估功能。
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional
from carla_c2osr.env.types import EgoState, AgentState
from carla_c2osr.agents.c2osr.grid import GridMapper
from carla_c2osr.agents.c2osr.spatial_dirichlet import SpatialDirichletBank
from carla_c2osr.evaluation.rewards import RewardCalculator, CollisionDetector


class QEvaluator:
    """Q值评估器"""
    
    def __init__(self, 
                 reward_calculator: Optional[RewardCalculator] = None,
                 collision_detector: Optional[CollisionDetector] = None,
                 default_n_samples: int = 10):
        """
        Args:
            reward_calculator: 奖励计算器
            collision_detector: 碰撞检测器
            default_n_samples: 默认采样数量
        """
        self.reward_calculator = reward_calculator or RewardCalculator()
        self.collision_detector = collision_detector or CollisionDetector()
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
                         verbose: bool = False) -> List[float]:
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
        
        # 检查自车未来轨迹是否与agent可达集重叠
        overlap_cells = set(ego_trajectory_cells) & set(reachable)
        has_overlap = len(overlap_cells) > 0
        
        if verbose:
            print(f"    Agent {agent_id} 可达集大小: {len(reachable)}, 自车轨迹格子: {ego_trajectory_cells}")
            print(f"    重叠格子: {list(overlap_cells) if has_overlap else '无重叠'}")
        
        rewards = []
        for i, probs in enumerate(sampled_probs):
            # 提取可达集上的概率分布
            reachable_probs = probs[reachable]
            
            # 计算碰撞概率
            collision_prob, collision_count = self.collision_detector.calculate_collision_probability(
                reachable_probs, reachable, overlap_cells
            )
            
            # 判断是否发生碰撞
            collision = collision_prob > self.collision_detector.collision_threshold
            
            # 创建智能体下一状态（简化，只更新位置）
            # 这里我们简化处理，使用当前状态作为下一状态
            agent_next_state = agent_state
            
            # 使用reward计算器计算reward
            reward = self.reward_calculator.calculate_reward(
                ego_state=ego_state,
                ego_next_state=ego_next_state,
                agent_state=agent_state,
                agent_next_state=agent_next_state,
                collision=collision
            )
            
            rewards.append(reward)
            
            if verbose:
                print(f"      采样{i+1}: 碰撞概率{collision_prob:.3f}, 碰撞格子数{collision_count}, "
                      f"碰撞{collision}, 总reward{reward:.2f}")
        
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

