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
                 min_distance: float = 2.0,
                 distance_weight: float = 1.0):
        """
        Args:
            collision_penalty: Large penalty for collisions
            min_distance: Minimum safe distance in meters
            distance_weight: Weight for distance-based reward
        """
        self.collision_penalty = collision_penalty
        self.min_distance = min_distance
        self.distance_weight = distance_weight

    def compute(self, state: WorldState, action: Any,
                next_state: WorldState, info: Dict[str, Any]) -> float:
        """Compute safety reward based on distances to other agents."""
        # Check for collision
        if info.get('collision', False):
            return self.collision_penalty

        # Compute minimum distance to all agents
        ego_pos = np.array(next_state.ego.position_m)
        min_dist = float('inf')

        for agent in next_state.agents:
            agent_pos = np.array(agent.position_m)
            dist = np.linalg.norm(ego_pos - agent_pos)
            min_dist = min(min_dist, dist)

        # Reward for maintaining safe distance
        if min_dist == float('inf'):
            return 0.0  # No agents

        if min_dist < self.min_distance:
            # Penalty for being too close
            return -self.distance_weight * (self.min_distance - min_dist)
        else:
            # Small positive reward for safe distance
            return 0.1 * self.distance_weight


class ComfortReward(RewardFunction):
    """Reward for smooth, comfortable driving (low jerk, reasonable acceleration)."""

    def __init__(self,
                 jerk_penalty_weight: float = 1.0,
                 accel_penalty_weight: float = 0.5):
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
                 speed_weight: float = 1.0,
                 progress_weight: float = 2.0):
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

        # Progress reward (forward movement along x-axis)
        progress = next_state.ego.position_m[0] - state.ego.position_m[0]
        progress_reward = self.progress_weight * max(0, progress)

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
        # Get reference path from info
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
