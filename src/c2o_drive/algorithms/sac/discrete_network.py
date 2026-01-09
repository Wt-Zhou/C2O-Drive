"""Neural network architectures for discrete-action SAC.

This module provides network architectures adapted for SAC with discrete action
spaces using Categorical policies instead of Gaussian policies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def build_mlp(input_dim: int, hidden_dims: Tuple[int, ...]) -> nn.Module:
    """Build a multi-layer perceptron with ReLU activations.

    Args:
        input_dim: Input feature dimension
        hidden_dims: Tuple of hidden layer dimensions

    Returns:
        Sequential MLP module

    Example:
        >>> mlp = build_mlp(128, (256, 256))
        >>> output = mlp(torch.randn(32, 128))  # (batch=32, input_dim=128)
        >>> print(output.shape)  # torch.Size([32, 256])
    """
    layers = []
    prev_dim = input_dim

    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(nn.ReLU())
        prev_dim = hidden_dim

    return nn.Sequential(*layers)


class CategoricalActor(nn.Module):
    """Categorical policy network for discrete-action SAC.

    This network outputs a probability distribution over discrete actions
    using a Categorical distribution (softmax over logits).

    Key differences from continuous SAC Actor:
    - Outputs discrete action probabilities instead of Gaussian parameters
    - Uses softmax activation instead of tanh
    - No reparameterization trick needed for discrete actions

    Architecture:
        State (state_dim) → MLP (hidden_dims) → Logits (action_dim) → Softmax → Probs

    Attributes:
        state_dim: Input state dimension
        action_dim: Number of discrete actions
        shared_net: Shared feature extraction network
        logits_head: Output layer for action logits
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
    ):
        """Initialize Categorical Actor network.

        Args:
            state_dim: Dimension of input state features
            action_dim: Number of discrete actions (dynamically computed from lattice)
            hidden_dims: Dimensions of hidden layers
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Shared feature extraction network
        self.shared_net = build_mlp(state_dim, hidden_dims)

        # Output layer: logits for each discrete action
        self.logits_head = nn.Linear(hidden_dims[-1], action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass to get action probabilities.

        Args:
            state: Input state tensor of shape (batch_size, state_dim)
                   or (state_dim,) for single state

        Returns:
            Action probabilities of shape (batch_size, action_dim)
            or (action_dim,) for single state
        """
        # Handle single state input (no batch dimension)
        squeeze_output = False
        if state.dim() == 1:
            state = state.unsqueeze(0)  # Add batch dimension
            squeeze_output = True

        # Extract features
        features = self.shared_net(state)

        # Compute logits and convert to probabilities
        logits = self.logits_head(features)
        probs = F.softmax(logits, dim=-1)

        # Remove batch dimension if input was single state
        if squeeze_output:
            probs = probs.squeeze(0)

        return probs

    def get_logits(self, state: torch.Tensor) -> torch.Tensor:
        """Get raw logits before softmax (useful for entropy calculation).

        Args:
            state: Input state tensor

        Returns:
            Raw logits before softmax
        """
        squeeze_output = False
        if state.dim() == 1:
            state = state.unsqueeze(0)
            squeeze_output = True

        features = self.shared_net(state)
        logits = self.logits_head(features)

        if squeeze_output:
            logits = logits.squeeze(0)

        return logits


class DiscreteQCritic(nn.Module):
    """Q-network for discrete actions in SAC.

    This network outputs Q-values for all discrete actions simultaneously,
    allowing efficient computation of Q(s, a) for all actions without
    needing to evaluate each action separately.

    Architecture:
        State (state_dim) → MLP (hidden_dims) → Q-values (action_dim)

    Attributes:
        state_dim: Input state dimension
        action_dim: Number of discrete actions
        q_net: Q-network that outputs Q-value for each action
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
    ):
        """Initialize Discrete Q-Critic network.

        Args:
            state_dim: Dimension of input state features
            action_dim: Number of discrete actions
            hidden_dims: Dimensions of hidden layers
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Build Q-network
        layers = []
        layers.extend(list(build_mlp(state_dim, hidden_dims).children()))
        layers.append(nn.Linear(hidden_dims[-1], action_dim))
        self.q_net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass to get Q-values for all actions.

        Args:
            state: Input state tensor of shape (batch_size, state_dim)
                   or (state_dim,) for single state

        Returns:
            Q-values for all actions of shape (batch_size, action_dim)
            or (action_dim,) for single state
        """
        # Handle single state input
        squeeze_output = False
        if state.dim() == 1:
            state = state.unsqueeze(0)
            squeeze_output = True

        # Compute Q-values for all actions
        q_values = self.q_net(state)

        # Remove batch dimension if input was single state
        if squeeze_output:
            q_values = q_values.squeeze(0)

        return q_values

    def get_q_value(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Get Q-value for specific action(s).

        Args:
            state: State tensor of shape (batch_size, state_dim)
            action: Action indices of shape (batch_size,) or (batch_size, 1)

        Returns:
            Q-values for specified actions of shape (batch_size,)
        """
        q_values = self.forward(state)  # (batch_size, action_dim)

        # Handle action shape
        if action.dim() == 2 and action.size(1) == 1:
            action = action.squeeze(1)  # (batch_size, 1) → (batch_size,)

        # Gather Q-values for specific actions
        q_selected = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

        return q_selected


__all__ = [
    'build_mlp',
    'CategoricalActor',
    'DiscreteQCritic',
]
