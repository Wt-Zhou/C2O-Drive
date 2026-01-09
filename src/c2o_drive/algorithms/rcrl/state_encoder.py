"""State encoder module for RCRL algorithm.

This module converts WorldState observations into fixed-dimension
feature vectors suitable for neural network processing.
"""

from typing import List, Optional, Tuple
import numpy as np

from c2o_drive.core.types import WorldState, EgoState, AgentState
from c2o_drive.algorithms.rcrl.config import StateEncoderConfig


class StateEncoder:
    """Encodes WorldState into fixed-dimension feature vectors.

    The encoder creates a structured representation containing:
    - Ego vehicle features (position, velocity, heading)
    - Goal features (from reference path)
    - Agent features (relative positions, velocities, headings)

    All features are normalized to facilitate neural network training.
    """

    def __init__(self, config: StateEncoderConfig):
        """Initialize state encoder.

        Args:
            config: State encoder configuration
        """
        self.config = config
        self.state_dim = config.state_dim
        self.max_agents = config.max_agents
        self.vmax = config.normalization_vmax
        self.dmax = config.normalization_dmax

    def encode(
        self,
        world_state: WorldState,
        reference_path: Optional[List[Tuple[float, float]]] = None,
    ) -> np.ndarray:
        """Encode WorldState into feature vector.

        Feature structure:
        - Ego features (7): [vx, vy, |v|, sin(yaw), cos(yaw), x, y]
        - Goal features (3): [rel_x, rel_y, distance]
        - Agent features (max_agents × 6 each):
            [rel_x, rel_y, rel_vx, rel_vy, sin(heading), cos(heading)]
        - Zero padding to state_dim

        All positions and velocities are normalized.

        Args:
            world_state: Current world state
            reference_path: Optional reference path for goal extraction

        Returns:
            Feature vector of shape (state_dim,)
        """
        features = []

        # Encode ego features
        ego_features = self._encode_ego(world_state.ego)
        features.extend(ego_features)

        # Encode goal features
        if reference_path and len(reference_path) > 0:
            goal_features = self._encode_goal(world_state.ego, reference_path)
        else:
            goal_features = [0.0, 0.0, 0.0]
        features.extend(goal_features)

        # Encode agent features (up to max_agents)
        agent_features = self._encode_agents(world_state.ego, world_state.agents)
        features.extend(agent_features)

        # Pad or truncate to state_dim
        if len(features) < self.state_dim:
            features.extend([0.0] * (self.state_dim - len(features)))
        else:
            features = features[: self.state_dim]

        return np.array(features, dtype=np.float32)

    def _encode_ego(self, ego: EgoState) -> List[float]:
        """Encode ego vehicle state.

        Args:
            ego: Ego vehicle state

        Returns:
            List of 7 normalized ego features
        """
        vx = ego.velocity_mps[0] / self.vmax
        vy = ego.velocity_mps[1] / self.vmax
        v_mag = np.linalg.norm(ego.velocity_mps) / self.vmax
        sin_yaw = np.sin(ego.yaw_rad)
        cos_yaw = np.cos(ego.yaw_rad)
        x = ego.position_m[0] / self.dmax
        y = ego.position_m[1] / self.dmax

        return [vx, vy, v_mag, sin_yaw, cos_yaw, x, y]

    def _encode_goal(
        self, ego: EgoState, reference_path: List[Tuple[float, float]]
    ) -> List[float]:
        """Encode goal features from reference path.

        Finds the closest point on the reference path ahead of the ego vehicle.

        Args:
            ego: Ego vehicle state
            reference_path: List of waypoints defining the reference path

        Returns:
            List of 3 normalized goal features [rel_x, rel_y, distance]
        """
        ego_pos = np.array(ego.position_m)
        ego_yaw = ego.yaw_rad

        # Find closest ahead waypoint
        min_distance = float("inf")
        closest_point = reference_path[0]

        for waypoint in reference_path:
            wp_vec = np.array(waypoint) - ego_pos

            # Check if waypoint is ahead (dot product with heading vector)
            heading_vec = np.array([np.cos(ego_yaw), np.sin(ego_yaw)])
            if np.dot(wp_vec, heading_vec) > 0:
                distance = np.linalg.norm(wp_vec)
                if distance < min_distance:
                    min_distance = distance
                    closest_point = waypoint

        # Compute relative position
        rel_pos = np.array(closest_point) - ego_pos

        # Transform to ego frame
        cos_yaw = np.cos(ego_yaw)
        sin_yaw = np.sin(ego_yaw)
        rel_x_ego = rel_pos[0] * cos_yaw + rel_pos[1] * sin_yaw
        rel_y_ego = -rel_pos[0] * sin_yaw + rel_pos[1] * cos_yaw

        # Normalize
        rel_x_norm = rel_x_ego / self.dmax
        rel_y_norm = rel_y_ego / self.dmax
        distance_norm = min_distance / self.dmax

        return [rel_x_norm, rel_y_norm, distance_norm]

    def _encode_agents(
        self, ego: EgoState, agents: List[AgentState]
    ) -> List[float]:
        """Encode agent features.

        Args:
            ego: Ego vehicle state
            agents: List of agent states

        Returns:
            List of agent features (6 * max_agents)
        """
        features = []
        ego_pos = np.array(ego.position_m)
        ego_vel = np.array(ego.velocity_mps)

        for i in range(self.max_agents):
            if i < len(agents):
                agent = agents[i]

                # Relative position (world frame)
                rel_pos = np.array(agent.position_m) - ego_pos
                rel_x_norm = rel_pos[0] / self.dmax
                rel_y_norm = rel_pos[1] / self.dmax

                # Relative velocity
                rel_vel = np.array(agent.velocity_mps) - ego_vel
                rel_vx_norm = rel_vel[0] / self.vmax
                rel_vy_norm = rel_vel[1] / self.vmax

                # Agent heading (trigonometric encoding)
                sin_heading = np.sin(agent.heading_rad)
                cos_heading = np.cos(agent.heading_rad)

                features.extend(
                    [
                        rel_x_norm,
                        rel_y_norm,
                        rel_vx_norm,
                        rel_vy_norm,
                        sin_heading,
                        cos_heading,
                    ]
                )
            else:
                # Zero padding for missing agents
                features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        return features

    def get_feature_dim(self) -> int:
        """Get the dimension of encoded features.

        Returns:
            Feature dimension
        """
        return self.state_dim


__all__ = ["StateEncoder"]
