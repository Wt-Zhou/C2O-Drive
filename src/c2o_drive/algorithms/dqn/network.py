"""Neural network architectures for DQN."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class QNetwork(nn.Module):
    """Deep Q-Network architecture.

    A fully connected neural network that approximates Q-values
    for state-action pairs.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
    ):
        """Initialize Q-Network.

        Args:
            state_dim: Dimension of input state
            action_dim: Number of discrete actions
            hidden_dims: Dimensions of hidden layers
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Build network layers
        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, action_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            state: Input state tensor of shape (batch_size, state_dim)

        Returns:
            Q-values for all actions, shape (batch_size, action_dim)
        """
        return self.network(state)


class DuelingQNetwork(nn.Module):
    """Dueling DQN architecture.

    Separates state value and action advantage computation
    for improved learning stability.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
    ):
        """Initialize Dueling Q-Network.

        Args:
            state_dim: Dimension of input state
            action_dim: Number of discrete actions
            hidden_dims: Dimensions of hidden layers
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Shared feature extractor
        self.feature_layers = []
        prev_dim = state_dim

        for i, hidden_dim in enumerate(hidden_dims[:-1]):
            self.feature_layers.append(nn.Linear(prev_dim, hidden_dim))
            self.feature_layers.append(nn.ReLU())
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*self.feature_layers)

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], 1)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], action_dim)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            state: Input state tensor of shape (batch_size, state_dim)

        Returns:
            Q-values for all actions, shape (batch_size, action_dim)
        """
        features = self.feature_extractor(state)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # Combine value and advantage (Wang et al. 2016)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values