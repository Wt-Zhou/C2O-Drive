"""WorldState Encoder for Rainbow DQN

Encodes WorldState into fixed-dimension feature vectors.
Aligned with PPO's hand-crafted feature extraction for consistency.
"""

from typing import List
import numpy as np
import torch
import torch.nn as nn

from c2o_drive.core.types import WorldState


class WorldStateEncoder(nn.Module):
    """Encodes WorldState into fixed-dimension feature vector.

    Uses the same hand-crafted feature structure as PPO:
    - Ego: normalized position, speed, heading (cos/sin)
    - Goal (if available): relative position
    - Nearest agents: relative position, speed, heading (cos/sin)
    - Zero-padding to reach feature_dim
    """

    def __init__(self, config, max_agents: int = 10):
        """Initialize WorldState encoder.

        Args:
            config: Configuration with state_feature_dim attribute
            max_agents: Maximum number of agents to include (aligned with PPO)
        """
        super().__init__()

        self.config = config
        self.max_agents = max_agents
        self.state_feature_dim = config.network.state_feature_dim

    def forward(self, world_state_batch: List[WorldState]) -> torch.Tensor:
        """Encode batch of WorldState observations.

        Args:
            world_state_batch: List of WorldState objects

        Returns:
            Encoded features of shape (batch_size, state_feature_dim)
        """
        device = torch.device(self.config.device if hasattr(self.config, "device") else "cpu")
        features_batch = []

        for ws in world_state_batch:
            features = []

            # Ego features
            ego = ws.ego
            ego_speed = np.linalg.norm(ego.velocity_mps)
            features.extend([
                ego.position_m[0] / 100.0,
                ego.position_m[1] / 100.0,
                ego_speed / 30.0,
                np.cos(ego.yaw_rad),
                np.sin(ego.yaw_rad),
            ])

            # Goal features (if available)
            if hasattr(ws, 'goal') and ws.goal is not None:
                rel_x = (ws.goal.position_m[0] - ego.position_m[0]) / 100.0
                rel_y = (ws.goal.position_m[1] - ego.position_m[1]) / 100.0
                features.extend([rel_x, rel_y])
            else:
                features.extend([0.0, 0.0])

            # Agent features (nearest N agents)
            for agent in ws.agents[:self.max_agents]:
                rel_x = (agent.position_m[0] - ego.position_m[0]) / 100.0
                rel_y = (agent.position_m[1] - ego.position_m[1]) / 100.0
                agent_speed = np.linalg.norm(agent.velocity_mps)
                features.extend([
                    rel_x,
                    rel_y,
                    agent_speed / 30.0,
                    np.cos(agent.heading_rad),
                    np.sin(agent.heading_rad),
                ])

            # Pad with zeros to reach state_feature_dim
            if len(features) < self.state_feature_dim:
                features.extend([0.0] * (self.state_feature_dim - len(features)))

            features_batch.append(features[:self.state_feature_dim])

        return torch.tensor(features_batch, dtype=torch.float32, device=device)
