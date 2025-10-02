"""
奖励计算和碰撞检测模块

提供智能体交互的奖励计算和碰撞检测功能。
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, List, Optional, Union
from carla_c2osr.env.types import EgoState, AgentState


def calculate_distance_to_path(point: Union[Tuple[float, float], np.ndarray],
                               path: List[Union[Tuple[float, float], np.ndarray]]) -> float:
    """计算点到路径的最小距离（垂直距离）。

    Args:
        point: 查询点 (x, y)
        path: 参考路径点列表 [(x1, y1), (x2, y2), ...]

    Returns:
        点到路径的最小距离
    """
    if not path:
        return 0.0

    point = np.array(point)
    min_distance = float('inf')

    for i in range(len(path) - 1):
        p1 = np.array(path[i])
        p2 = np.array(path[i + 1])

        # 计算点到线段的距离
        segment_vec = p2 - p1
        point_vec = point - p1

        # 线段长度
        segment_length_sq = np.dot(segment_vec, segment_vec)

        if segment_length_sq < 1e-6:  # 线段退化为点
            distance = np.linalg.norm(point - p1)
        else:
            # 投影参数 t
            t = np.dot(point_vec, segment_vec) / segment_length_sq
            t = np.clip(t, 0.0, 1.0)  # 限制在线段范围内

            # 最近点
            closest_point = p1 + t * segment_vec
            distance = np.linalg.norm(point - closest_point)

        min_distance = min(min_distance, distance)

    return min_distance


class RewardCalculator:
    """奖励计算器"""
    
    def __init__(self,
                 collision_penalty: float = -100.0,
                 speed_reward_weight: float = 1.0,
                 acceleration_penalty_weight: float = 0.1,
                 target_speed: float = 5.0,
                 safe_distance: float = 3.0,
                 distance_penalty_weight: float = 2.0,
                 gamma: float = 0.95,
                 centerline_offset_penalty_weight: float = 1.0):
        """
        Args:
            collision_penalty: 碰撞惩罚值（最大值）
            speed_reward_weight: 速度奖励权重
            acceleration_penalty_weight: 加速度惩罚权重
            target_speed: 目标速度
            safe_distance: 安全距离
            distance_penalty_weight: 距离惩罚权重
            gamma: 时序折扣因子
            centerline_offset_penalty_weight: 中心线偏移惩罚权重
        """
        self.collision_penalty = collision_penalty
        self.speed_reward_weight = speed_reward_weight
        self.acceleration_penalty_weight = acceleration_penalty_weight
        self.target_speed = target_speed
        self.safe_distance = safe_distance
        self.distance_penalty_weight = distance_penalty_weight
        self.gamma = gamma
        self.centerline_offset_penalty_weight = centerline_offset_penalty_weight
    
    def calculate_reward(self,
                        ego_state: EgoState,
                        ego_next_state: EgoState,
                        agent_state: AgentState,
                        agent_next_state: AgentState,
                        collision_probability: float = 0.0,
                        reference_path: Optional[List[Union[Tuple[float, float], np.ndarray]]] = None) -> float:
        """计算reward值。

        Args:
            ego_state: 自车当前状态
            ego_next_state: 自车下一状态
            agent_state: 智能体当前状态
            agent_next_state: 智能体下一状态
            collision_probability: 碰撞概率 [0, 1]
            reference_path: 参考中心线路径，用于计算偏移惩罚

        Returns:
            reward值
        """
        reward = 0.0

        # 基于概率的碰撞惩罚
        if collision_probability > 0:
            reward += self.collision_penalty * collision_probability

        # 速度奖励（鼓励保持合理速度）
        ego_speed = np.linalg.norm(ego_state.velocity_mps)
        speed_reward = -abs(ego_speed - self.target_speed) * self.speed_reward_weight
        reward += speed_reward

        # 加速度惩罚（鼓励平滑驾驶）
        ego_accel = np.linalg.norm(np.array(ego_next_state.velocity_mps) - np.array(ego_state.velocity_mps))
        accel_penalty = -ego_accel * self.acceleration_penalty_weight
        reward += accel_penalty

        # 距离奖励（与智能体保持安全距离）
        ego_pos = np.array(ego_state.position_m)
        agent_pos = np.array(agent_state.position_m)
        distance = np.linalg.norm(ego_pos - agent_pos)
        if distance < self.safe_distance:
            distance_penalty = -(self.safe_distance - distance) * self.distance_penalty_weight
            reward += distance_penalty

        # 中心线偏移惩罚（鼓励沿中心线行驶）
        if reference_path is not None and len(reference_path) > 0:
            centerline_offset = calculate_distance_to_path(ego_next_state.position_m, reference_path)
            offset_penalty = -centerline_offset * self.centerline_offset_penalty_weight
            reward += offset_penalty

        return reward


class DistanceBasedCollisionDetector:
    """基于距离和概率的碰撞检测器"""
    
    def __init__(self, collision_threshold: float = 0.1):
        """
        Args:
            collision_threshold: 碰撞检测阈值
        """
        self.collision_threshold = collision_threshold
    
    def check_collision(self, 
                       agent_cell: int, 
                       ego_trajectory_cells: List[int], 
                       agent_probability: float) -> bool:
        """检查智能体和自车是否发生碰撞。
        
        Args:
            agent_cell: 智能体到达的格子ID
            ego_trajectory_cells: 自车轨迹的格子ID列表
            agent_probability: 智能体到达该格子的概率
            
        Returns:
            是否发生碰撞
        """
        # 如果智能体概率大于阈值且自车也到达该格子，则发生碰撞
        return agent_probability > self.collision_threshold and agent_cell in ego_trajectory_cells
    
    def calculate_collision_probability(self,
                                      reachable_probs: np.ndarray,
                                      reachable: List[int],
                                      overlap_cells: set) -> Tuple[float, int]:
        """计算碰撞概率。

        Args:
            reachable_probs: 可达集上的概率分布
            reachable: 可达集
            overlap_cells: 重叠格子集合

        Returns:
            (碰撞概率, 碰撞格子数)
        """
        collision_prob = 0.0
        collision_count = 0

        if overlap_cells:
            # 计算重叠格子上的总概率
            for cell_idx, cell_id in enumerate(reachable):
                if cell_id in overlap_cells:
                    cell_prob = reachable_probs[cell_idx]
                    if cell_prob > self.collision_threshold:
                        collision_prob += cell_prob
                        collision_count += 1

        return collision_prob, collision_count

    def calculate_temporal_collision_probability(self,
                                                reachable_probs: np.ndarray,
                                                reachable: List[int],
                                                ego_trajectory_cells: List[int],
                                                gamma: float = 0.95) -> Tuple[float, int, List[Tuple[int, int, float]]]:
        """按时序计算碰撞概率，考虑折扣因子。

        Args:
            reachable_probs: 可达集上的概率分布
            reachable: 可达集
            ego_trajectory_cells: 自车轨迹格子列表（按时序排列）
            gamma: 时序折扣因子

        Returns:
            (加权碰撞概率, 碰撞格子数, 碰撞详情列表[(时间步, 格子ID, 概率)])
        """
        collision_prob = 0.0
        collision_count = 0
        collision_details = []

        # 建立可达集的索引映射
        reachable_dict = {cell_id: idx for idx, cell_id in enumerate(reachable)}

        # 按时序遍历自车轨迹
        for time_step, ego_cell in enumerate(ego_trajectory_cells):
            if ego_cell in reachable_dict:
                cell_idx = reachable_dict[ego_cell]
                cell_prob = reachable_probs[cell_idx]

                if cell_prob > self.collision_threshold:
                    # 应用时序折扣
                    discounted_prob = cell_prob * (gamma ** time_step)
                    collision_prob += discounted_prob
                    collision_count += 1
                    collision_details.append((time_step, ego_cell, cell_prob))

        return collision_prob, collision_count, collision_details

