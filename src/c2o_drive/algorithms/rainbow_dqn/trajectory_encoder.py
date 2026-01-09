"""WorldState Encoder for Rainbow DQN

Encodes variable-length WorldState observations into fixed-dimension feature vectors
suitable for neural network processing. Uses Self-Attention to handle variable numbers
of agents.
"""

from typing import List
import torch
import torch.nn as nn

from c2o_drive.core.types import WorldState


class WorldStateEncoder(nn.Module):
    """Encodes WorldState into fixed-dimension feature vector.

    Handles the challenge of variable-length agent lists by:
    1. Encoding each agent independently
    2. Using multi-head self-attention to aggregate agent features
    3. Combining ego and aggregated agent features

    Architecture:
        Ego State (5D) → MLP → Ego Features (128D)
        Agent States (Nx6D) → MLP → Agent Features (Nx128D)
        Agent Features → Self-Attention → Aggregated Features (128D)
        [Ego Features; Aggregated Features] → MLP → Output (feature_dim)

    Attributes:
        config: Configuration containing state_feature_dim
        max_agents: Maximum number of agents to process (for padding)
    """

    def __init__(self, config, max_agents: int = 20):
        """Initialize WorldState encoder.

        Args:
            config: Configuration with state_feature_dim attribute
            max_agents: Maximum number of agents (for padding/masking)
        """
        super().__init__()

        self.config = config
        self.max_agents = max_agents
        self.state_feature_dim = config.network.state_feature_dim

        # Ego encoder: position(2) + velocity(2) + yaw(1) = 5D
        self.ego_encoder = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )

        # Agent encoder: position(2) + velocity(2) + heading(1) + type(1) = 6D
        self.agent_encoder = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )

        # Multi-head attention for agent aggregation
        self.attention = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=4,
            batch_first=True,
            dropout=0.1
        )

        # Fusion layer: ego(128) + aggregated_agents(128) = 256D
        self.fusion = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.state_feature_dim)
        )

    def forward(self, world_state_batch: List[WorldState]) -> torch.Tensor:
        """Encode batch of WorldState observations.

        Args:
            world_state_batch: List of WorldState objects

        Returns:
            Encoded features of shape (batch_size, state_feature_dim)
        """
        batch_size = len(world_state_batch)
        device = next(self.parameters()).device

        # === Encode Ego States ===
        ego_features = []
        for ws in world_state_batch:
            # Extract ego features: [pos_x, pos_y, vel_x, vel_y, yaw]
            ego_feat = torch.tensor([
                ws.ego.position_m[0],
                ws.ego.position_m[1],
                ws.ego.velocity_mps[0],
                ws.ego.velocity_mps[1],
                ws.ego.yaw_rad
            ], dtype=torch.float32, device=device)
            ego_features.append(self.ego_encoder(ego_feat))

        ego_features = torch.stack(ego_features)  # (batch, 128)

        # === Encode Agent States ===
        # Handle variable number of agents with padding and masking
        agent_features = torch.zeros(batch_size, self.max_agents, 128, device=device)
        agent_mask = torch.ones(batch_size, self.max_agents, dtype=torch.bool, device=device)

        for i, ws in enumerate(world_state_batch):
            for j, agent in enumerate(ws.agents[:self.max_agents]):  # Truncate if too many
                # Extract agent features: [pos_x, pos_y, vel_x, vel_y, heading, type]
                agent_type_encoding = self._encode_agent_type(agent.agent_type)

                agent_feat = torch.tensor([
                    agent.position_m[0],
                    agent.position_m[1],
                    agent.velocity_mps[0],
                    agent.velocity_mps[1],
                    agent.heading_rad,
                    agent_type_encoding
                ], dtype=torch.float32, device=device)

                agent_features[i, j] = self.agent_encoder(agent_feat)
                agent_mask[i, j] = False  # False = not masked (valid)

        # === Aggregate Agents with Self-Attention ===
        # Use ego as query, agents as key/value
        ego_query = ego_features.unsqueeze(1)  # (batch, 1, 128)

        if agent_features.size(1) > 0:
            agent_aggregated, _ = self.attention(
                query=ego_query,
                key=agent_features,
                value=agent_features,
                key_padding_mask=agent_mask
            )
            agent_aggregated = agent_aggregated.squeeze(1)  # (batch, 128)
        else:
            # No agents: use zeros
            agent_aggregated = torch.zeros_like(ego_features)

        # === Fusion ===
        combined = torch.cat([ego_features, agent_aggregated], dim=1)  # (batch, 256)
        state_features = self.fusion(combined)  # (batch, state_feature_dim)

        return state_features

    def _encode_agent_type(self, agent_type) -> float:
        """Encode agent type as a scalar value.

        Args:
            agent_type: AgentType enum value

        Returns:
            Scalar encoding (0.0 for vehicle, 0.5 for pedestrian, 1.0 for bicycle)
        """
        # Simple encoding: map agent types to [0, 1]
        type_map = {
            'vehicle': 0.0,
            'pedestrian': 0.5,
            'bicycle': 1.0
        }

        # Handle both string and enum types
        if hasattr(agent_type, 'value'):
            type_str = agent_type.value
        else:
            type_str = str(agent_type).lower()

        return type_map.get(type_str, 0.0)
