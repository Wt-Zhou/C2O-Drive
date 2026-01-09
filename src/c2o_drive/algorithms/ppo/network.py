"""Neural network architectures for PPO algorithm.

This module provides the Actor-Critic network for PPO.
"""

from typing import Tuple
import torch
import torch.nn as nn


class ActorCriticNetwork(nn.Module):
    """Actor-Critic network for PPO.

    Uses shared feature extraction layers, then splits into:
    - Actor head: outputs action logits (for discrete actions)
    - Critic head: outputs state value

    Attributes:
        shared_net: Shared feature extraction layers
        actor_head: Actor output layer
        critic_head: Critic output layer
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256)
    ):
        """Initialize Actor-Critic network.

        Args:
            state_dim: Dimension of state features
            action_dim: Number of discrete actions
            hidden_dims: Tuple of hidden layer dimensions
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Shared feature extraction layers
        layers = []
        in_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
            ])
            in_dim = hidden_dim
        self.shared_net = nn.Sequential(*layers)

        # Actor head (outputs action logits)
        self.actor_head = nn.Linear(in_dim, action_dim)

        # Critic head (outputs state value)
        self.critic_head = nn.Linear(in_dim, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights using orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

        # Special initialization for actor and critic heads
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network.

        Args:
            state: State tensor of shape (batch_size, state_dim)
                   or (state_dim,) for single state

        Returns:
            logits: Action logits of shape (batch_size, action_dim) or (action_dim,)
            value: State value of shape (batch_size, 1) or (1,)
        """
        # Handle single state input
        single_input = state.dim() == 1
        if single_input:
            state = state.unsqueeze(0)  # Add batch dimension

        # Shared feature extraction
        features = self.shared_net(state)

        # Actor output (logits)
        logits = self.actor_head(features)

        # Critic output (state value)
        value = self.critic_head(features)

        # Remove batch dimension if single input
        if single_input:
            logits = logits.squeeze(0)
            value = value.squeeze(0)

        return logits, value


__all__ = ['ActorCriticNetwork']
