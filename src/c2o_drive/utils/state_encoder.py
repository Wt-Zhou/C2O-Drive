"""Unified state encoder for all algorithms.

This module provides a standardized way to encode WorldState into feature vectors
that can be used by all RL algorithms (PPO, SAC, DQN, etc.).
"""

from __future__ import annotations
from typing import Optional
import numpy as np

from c2o_drive.core.types import WorldState


class UnifiedStateEncoder:
    """Unified state encoder that converts WorldState to fixed-size feature vector.

    All algorithms should use this encoder to ensure consistent state representation.
    This allows fair comparison and parameter sharing across different algorithms.

    Feature Structure (128 dimensions):
    - Ego features (7): [pos_x, pos_y, speed, sin(yaw), cos(yaw), accel_x, accel_y]
    - Goal features (3): [rel_x, rel_y, distance]
    - Agent features (10 agents × 6): [rel_x, rel_y, vx, vy, sin(heading), cos(heading)]
    - Padding to 128 dimensions

    Attributes:
        STATE_DIM: Global constant for state dimension (128)
        max_agents: Maximum number of agents to encode
        position_scale: Normalization scale for positions (meters)
        velocity_scale: Normalization scale for velocities (m/s)
    """

    STATE_DIM: int = 128  # Global constant

    def __init__(
        self,
        max_agents: int = 10,
        position_scale: float = 100.0,
        velocity_scale: float = 30.0,
    ):
        """Initialize unified state encoder.

        Args:
            max_agents: Maximum number of agents to encode
            position_scale: Scale for normalizing positions
            velocity_scale: Scale for normalizing velocities
        """
        self.max_agents = max_agents
        self.position_scale = position_scale
        self.velocity_scale = velocity_scale

    def encode(self, world_state: WorldState) -> np.ndarray:
        """Encode WorldState into fixed-size feature vector.

        Args:
            world_state: Current world state

        Returns:
            Feature vector of shape (STATE_DIM,) with dtype float32
        """
        features = []

        # 1. Ego vehicle features (7 dimensions)
        ego = world_state.ego
        ego_pos = np.array(ego.position_m)
        ego_vel = np.array(ego.velocity_mps)
        ego_speed = np.linalg.norm(ego_vel)

        # Extract acceleration if available, otherwise use zero
        if hasattr(ego, 'acceleration_mps2') and ego.acceleration_mps2 is not None:
            ego_accel = np.array(ego.acceleration_mps2)
        else:
            ego_accel = np.array([0.0, 0.0])

        features.extend([
            ego_pos[0] / self.position_scale,  # Normalized x position
            ego_pos[1] / self.position_scale,  # Normalized y position
            ego_speed / self.velocity_scale,   # Normalized speed
            np.sin(ego.yaw_rad),                # Heading sine
            np.cos(ego.yaw_rad),                # Heading cosine
            ego_accel[0] / 10.0,                # Normalized x acceleration
            ego_accel[1] / 10.0,                # Normalized y acceleration
        ])

        # 2. Goal features (3 dimensions)
        if hasattr(world_state, 'goal') and world_state.goal is not None:
            goal = world_state.goal
            goal_pos = np.array(goal.position_m)
            rel_x = (goal_pos[0] - ego_pos[0]) / self.position_scale
            rel_y = (goal_pos[1] - ego_pos[1]) / self.position_scale
            distance = np.linalg.norm(goal_pos - ego_pos) / self.position_scale
            features.extend([rel_x, rel_y, distance])
        else:
            # No goal available
            features.extend([0.0, 0.0, 0.0])

        # 3. Agent features (max_agents × 6 dimensions)
        agents = world_state.agents[:self.max_agents]
        for agent in agents:
            agent_pos = np.array(agent.position_m)
            agent_vel = np.array(agent.velocity_mps)

            # Relative position
            rel_x = (agent_pos[0] - ego_pos[0]) / self.position_scale
            rel_y = (agent_pos[1] - ego_pos[1]) / self.position_scale

            # Velocity components
            vx = agent_vel[0] / self.velocity_scale
            vy = agent_vel[1] / self.velocity_scale

            # Heading (sin/cos encoding)
            heading_rad = agent.heading_rad
            sin_heading = np.sin(heading_rad)
            cos_heading = np.cos(heading_rad)

            features.extend([rel_x, rel_y, vx, vy, sin_heading, cos_heading])

        # Pad remaining agent slots with zeros
        num_encoded_agents = len(agents)
        num_missing_agents = self.max_agents - num_encoded_agents
        features.extend([0.0] * (num_missing_agents * 6))

        # 4. Pad to STATE_DIM if needed
        while len(features) < self.STATE_DIM:
            features.append(0.0)

        # Convert to numpy array and truncate to STATE_DIM
        feature_array = np.array(features[:self.STATE_DIM], dtype=np.float32)

        return feature_array

    def encode_batch(self, world_states: list[WorldState]) -> np.ndarray:
        """Encode a batch of world states.

        Args:
            world_states: List of world states

        Returns:
            Feature array of shape (batch_size, STATE_DIM) with dtype float32
        """
        return np.stack([self.encode(state) for state in world_states], axis=0)

    @property
    def feature_info(self) -> dict:
        """Get information about feature dimensions and structure.

        Returns:
            Dictionary describing feature structure
        """
        ego_dim = 7
        goal_dim = 3
        agent_dim = self.max_agents * 6
        total_used = ego_dim + goal_dim + agent_dim
        padding_dim = self.STATE_DIM - total_used

        return {
            'total_dim': self.STATE_DIM,
            'ego_dim': ego_dim,
            'goal_dim': goal_dim,
            'agent_dim': agent_dim,
            'padding_dim': padding_dim,
            'max_agents': self.max_agents,
            'feature_ranges': {
                'ego': (0, ego_dim),
                'goal': (ego_dim, ego_dim + goal_dim),
                'agents': (ego_dim + goal_dim, ego_dim + goal_dim + agent_dim),
                'padding': (total_used, self.STATE_DIM),
            }
        }


# Global singleton instance for convenience
_global_encoder: Optional[UnifiedStateEncoder] = None


def get_global_encoder() -> UnifiedStateEncoder:
    """Get the global state encoder singleton.

    Returns:
        Global UnifiedStateEncoder instance
    """
    global _global_encoder
    if _global_encoder is None:
        _global_encoder = UnifiedStateEncoder()
    return _global_encoder


def set_global_encoder(encoder: UnifiedStateEncoder) -> None:
    """Set the global state encoder singleton.

    Args:
        encoder: UnifiedStateEncoder instance to set as global
    """
    global _global_encoder
    _global_encoder = encoder


__all__ = [
    'UnifiedStateEncoder',
    'get_global_encoder',
    'set_global_encoder',
]
