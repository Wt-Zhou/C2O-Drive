"""Reward function implementations for driving tasks.

This module provides modular reward components that can be combined
to create custom reward functions.
"""

from __future__ import annotations
from typing import Dict, Any
import numpy as np

from c2o_drive.core.environment import RewardFunction
from c2o_drive.environments.carla.types import WorldState, EgoState, AgentState


class SafetyReward(RewardFunction):
    """Reward for maintaining safe distances from obstacles."""

    def __init__(self,
                 collision_penalty: float = -100.0,
                 critical_distance: float = 2.0,
                 distance_weight: float = 1.0,
                 near_miss_weight: float = 0.3):
        """
        Args:
            collision_penalty: Large penalty for collisions
            critical_distance: Critical safe distance threshold (meters)
            distance_weight: Weight for critical distance penalty
            near_miss_weight: Weight for near-miss penalty
        """
        self.collision_penalty = collision_penalty
        self.critical_distance = critical_distance
        self.distance_weight = distance_weight
        self.near_miss_weight = near_miss_weight

    def compute(self, state: WorldState, action: Any,
                next_state: WorldState, info: Dict[str, Any]) -> float:
        """Compute safety reward based on distances to other agents.

        使用CARLA的OBB碰撞检测和精确距离计算（考虑车辆朝向和尺寸）。
        """
        # Check for collision
        if info.get('collision', False):
            return self.collision_penalty

        # 优先使用CARLA提供的near_miss检测和距离（基于OBB）
        if 'near_miss' in info and 'min_distance_to_agents' in info:
            near_miss = info['near_miss']
            min_dist = info['min_distance_to_agents']

            if min_dist == float('inf'):
                return 0.0  # No agents

            # 如果触发near-miss（基于扩大OBB的精确检测）
            if near_miss:
                # Near-miss惩罚（距离越近惩罚越大）
                # 假设near_miss在2-4m范围内，给线性惩罚
                from c2o_drive.config import get_global_config
                near_miss_threshold = get_global_config().safety.near_miss_threshold_m
                # 将距离映射到[0, near_miss_threshold]范围
                normalized_dist = min(max(min_dist, 0), near_miss_threshold)
                return -self.near_miss_weight * (near_miss_threshold - normalized_dist)
            else:
                # 安全区域
                return 0.1 * self.distance_weight

        # Fallback: 如果没有CARLA的OBB检测，使用简单的中心点距离
        # （用于非CARLA环境，如scenario replay）
        from c2o_drive.config import get_global_config
        near_miss_threshold = get_global_config().safety.near_miss_threshold_m

        # Compute minimum distance to all agents
        ego_pos = np.array(next_state.ego.position_m)
        min_dist = float('inf')

        for agent in next_state.agents:
            agent_pos = np.array(agent.position_m)
            dist = np.linalg.norm(ego_pos - agent_pos)
            min_dist = min(min_dist, dist)

        # Reward for maintaining safe distance (分级惩罚)
        if min_dist == float('inf'):
            return 0.0  # No agents

        if min_dist < self.critical_distance:
            # 严重惩罚：距离 < critical_distance（默认2米）
            return -self.distance_weight * (self.critical_distance - min_dist)
        elif min_dist < near_miss_threshold:
            # 轻度惩罚：critical_distance ≤ 距离 < near_miss_threshold（从global_config读取，默认4米）
            return -self.near_miss_weight * (near_miss_threshold - min_dist)
        else:
            # 安全奖励：距离 ≥ near_miss_threshold
            return 0.1 * self.distance_weight


class ComfortReward(RewardFunction):
    """Reward for smooth, comfortable driving (low jerk, reasonable acceleration)."""

    def __init__(self,
                 jerk_penalty_weight: float = 0.3,
                 accel_penalty_weight: float = 0.15):
        """
        Args:
            jerk_penalty_weight: Weight for jerk penalty
            accel_penalty_weight: Weight for acceleration penalty
        """
        self.jerk_penalty_weight = jerk_penalty_weight
        self.accel_penalty_weight = accel_penalty_weight

    def compute(self, state: WorldState, action: Any,
                next_state: WorldState, info: Dict[str, Any]) -> float:
        """Compute comfort reward based on acceleration and jerk."""
        # Get acceleration from info
        accel = info.get('acceleration', 0.0)
        jerk = info.get('jerk', 0.0)

        # Penalize high acceleration and jerk
        accel_penalty = -self.accel_penalty_weight * abs(accel)
        jerk_penalty = -self.jerk_penalty_weight * abs(jerk)

        return accel_penalty + jerk_penalty


class EfficiencyReward(RewardFunction):
    """Reward for efficient progress toward goal."""

    def __init__(self,
                 speed_target: float = 5.0,
                 speed_weight: float = 0.1,  # 降低到0.1，避免每步累加过多
                 progress_weight: float = 0.3):
        """
        Args:
            speed_target: Target speed in m/s
            speed_weight: Weight for speed reward
            progress_weight: Weight for forward progress
        """
        self.speed_target = speed_target
        self.speed_weight = speed_weight
        self.progress_weight = progress_weight

    def compute(self, state: WorldState, action: Any,
                next_state: WorldState, info: Dict[str, Any]) -> float:
        """Compute efficiency reward based on speed and progress."""
        # Speed reward (closer to target is better)
        ego_vel = np.array(next_state.ego.velocity_mps)
        speed = np.linalg.norm(ego_vel)
        speed_reward = -self.speed_weight * abs(speed - self.speed_target)

        # Progress reward: 降低权重，避免每步累加过多
        # 原来progress_weight=2.0导致50步累加100，远超碰撞惩罚-100
        # 现在降低到0.1，50步累加5.0，更合理
        if 'forward_progress' in info:
            progress = info['forward_progress']
        else:
            # Fallback: 使用x轴前进（适用于朝东的场景）
            progress = next_state.ego.position_m[0] - state.ego.position_m[0]
        progress_reward = self.progress_weight * progress

        return speed_reward + progress_reward


class CenterlineReward(RewardFunction):
    """Reward for staying close to reference path centerline."""

    def __init__(self,
                 max_deviation: float = 2.0,
                 weight: float = 1.0):
        """
        Args:
            max_deviation: Maximum acceptable deviation in meters
            weight: Penalty weight
        """
        self.max_deviation = max_deviation
        self.weight = weight

    def compute(self, state: WorldState, action: Any,
                next_state: WorldState, info: Dict[str, Any]) -> float:
        """Compute centerline deviation penalty."""
        # 优先使用预计算的横向偏离（已根据初始朝向正确计算）
        if 'lateral_deviation' in info:
            deviation = info['lateral_deviation']
        else:
            # Fallback: 使用y轴偏离（适用于沿x轴移动的场景）
            reference_y = info.get('reference_y', 0.0)
            ego_y = next_state.ego.position_m[1]
            deviation = abs(ego_y - reference_y)

        if deviation > self.max_deviation:
            return -self.weight * (deviation - self.max_deviation)
        return 0.0


class TimeReward(RewardFunction):
    """Small penalty for each timestep to encourage task completion."""

    def __init__(self, time_penalty: float = -0.1):
        """
        Args:
            time_penalty: Penalty per timestep
        """
        self.time_penalty = time_penalty

    def compute(self, state: WorldState, action: Any,
                next_state: WorldState, info: Dict[str, Any]) -> float:
        """Return constant time penalty."""
        return self.time_penalty


def create_default_reward() -> RewardFunction:
    """Create default composite reward function with balanced weights.

    Returns:
        Composite reward function suitable for general driving tasks
    """
    from c2o_drive.core.environment import CompositeRewardFunction

    components = [
        (SafetyReward(collision_penalty=-100.0), 1.0),
        (ComfortReward(), 0.5),
        (EfficiencyReward(speed_target=5.0), 1.0),
        (CenterlineReward(), 0.3),
        (TimeReward(), 1.0),
    ]

    return CompositeRewardFunction(components)
