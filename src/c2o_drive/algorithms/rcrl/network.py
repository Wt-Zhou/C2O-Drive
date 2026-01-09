"""Neural network architectures for RCRL algorithm.

This module implements Dueling DQN and related network architectures
for value-based reinforcement learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from c2o_drive.algorithms.rcrl.config import NetworkConfig


class DuelingDQN(nn.Module):
    """Dueling DQN architecture.

    Separates state value V(s) and advantage A(s,a) streams,
    then combines them as: Q(s,a) = V(s) + (A(s,a) - mean(A(s,·)))

    This architecture often learns more stable value functions
    compared to standard DQN.
    """

    def __init__(self, config: NetworkConfig, n_actions: int):
        """Initialize Dueling DQN.

        Args:
            config: Network configuration
            n_actions: Number of discrete actions
        """
        super().__init__()
        self.config = config
        self.n_actions = n_actions

        # Shared feature extraction layers
        self.feature = self._build_feature_network(
            config.state_dim, config.hidden_dims, config.activation, config.dropout_rate
        )

        # Value stream: V(s)
        feature_dim = config.hidden_dims[-1]
        self.value_stream = nn.Sequential(
            nn.Linear(feature_dim, 128), self._get_activation(config.activation), nn.Linear(128, 1)
        )

        # Advantage stream: A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(feature_dim, 128),
            self._get_activation(config.activation),
            nn.Linear(128, n_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            state: State tensor [batch, state_dim]

        Returns:
            Q-values tensor [batch, n_actions]
        """
        # Extract features
        features = self.feature(state)  # [batch, feature_dim]

        # Compute value and advantages
        value = self.value_stream(features)  # [batch, 1]
        advantages = self.advantage_stream(features)  # [batch, n_actions]

        # Combine using dueling formula
        # Q(s,a) = V(s) + (A(s,a) - mean_a(A(s,a)))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))

        return q_values

    def _build_feature_network(
        self, input_dim: int, hidden_dims: List[int], activation: str, dropout_rate: float
    ) -> nn.Module:
        """Build shared feature extraction network.

        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            activation: Activation function name
            dropout_rate: Dropout rate (0 means no dropout)

        Returns:
            Feature extraction module
        """
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self._get_activation(activation))

            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

            prev_dim = hidden_dim

        return nn.Sequential(*layers)

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function module.

        Args:
            activation: Activation function name

        Returns:
            Activation module
        """
        if activation == "relu":
            return nn.ReLU()
        elif activation == "tanh":
            return nn.Tanh()
        elif activation == "elu":
            return nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")


class StandardDQN(nn.Module):
    """Standard DQN architecture (non-dueling).

    Provided as an alternative to Dueling DQN.
    """

    def __init__(self, config: NetworkConfig, n_actions: int):
        """Initialize standard DQN.

        Args:
            config: Network configuration
            n_actions: Number of discrete actions
        """
        super().__init__()
        self.config = config
        self.n_actions = n_actions

        # Build fully connected network
        layers = []
        prev_dim = config.state_dim

        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self._get_activation(config.activation))

            if config.dropout_rate > 0:
                layers.append(nn.Dropout(config.dropout_rate))

            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, n_actions))

        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            state: State tensor [batch, state_dim]

        Returns:
            Q-values tensor [batch, n_actions]
        """
        return self.network(state)

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function module.

        Args:
            activation: Activation function name

        Returns:
            Activation module
        """
        if activation == "relu":
            return nn.ReLU()
        elif activation == "tanh":
            return nn.Tanh()
        elif activation == "elu":
            return nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")


__all__ = ["DuelingDQN", "StandardDQN"]
