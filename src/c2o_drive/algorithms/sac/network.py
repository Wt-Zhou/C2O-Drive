"""Neural network architectures for SAC."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Tuple, Optional

LOG_STD_MIN = -20
LOG_STD_MAX = 2


class Actor(nn.Module):
    """Stochastic policy network for SAC.

    Outputs mean and log standard deviation for a Gaussian policy.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        max_action: float = 1.0,
    ):
        """Initialize Actor network.

        Args:
            state_dim: Dimension of input state
            action_dim: Dimension of action space
            hidden_dims: Dimensions of hidden layers
            max_action: Maximum absolute value for actions
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

        # Build network layers
        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        self.trunk = nn.Sequential(*layers)

        # Output layers for mean and log_std
        self.mean_head = nn.Linear(prev_dim, action_dim)
        self.log_std_head = nn.Linear(prev_dim, action_dim)

    def forward(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network.

        Args:
            state: Input state tensor of shape (batch_size, state_dim)

        Returns:
            Tuple of (mean, log_std) for the policy distribution
        """
        features = self.trunk(state)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)

        return mean, log_std

    def sample(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from the policy.

        Args:
            state: Input state tensor
            deterministic: If True, return deterministic action (mean)

        Returns:
            Tuple of (action, log_prob)
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()

        if deterministic:
            action = torch.tanh(mean) * self.max_action
            log_prob = torch.zeros_like(action[:, 0])
            return action, log_prob

        # Sample from Gaussian
        normal = Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        action = torch.tanh(x_t) * self.max_action

        # Compute log probability with correction for tanh squashing
        log_prob = normal.log_prob(x_t)
        # Enforcing action bound through tanh squashing
        log_prob -= torch.log(self.max_action * (1 - action.pow(2) / self.max_action**2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=False)

        return action, log_prob


class Critic(nn.Module):
    """Q-function network for SAC.

    Implements two Q-networks to mitigate positive bias.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
    ):
        """Initialize Critic network.

        Args:
            state_dim: Dimension of input state
            action_dim: Dimension of action space
            hidden_dims: Dimensions of hidden layers
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Q1 network
        self.q1_layers = []
        prev_dim = state_dim + action_dim

        for hidden_dim in hidden_dims:
            self.q1_layers.append(nn.Linear(prev_dim, hidden_dim))
            self.q1_layers.append(nn.ReLU())
            prev_dim = hidden_dim

        self.q1_layers.append(nn.Linear(prev_dim, 1))
        self.q1 = nn.Sequential(*self.q1_layers)

        # Q2 network
        self.q2_layers = []
        prev_dim = state_dim + action_dim

        for hidden_dim in hidden_dims:
            self.q2_layers.append(nn.Linear(prev_dim, hidden_dim))
            self.q2_layers.append(nn.ReLU())
            prev_dim = hidden_dim

        self.q2_layers.append(nn.Linear(prev_dim, 1))
        self.q2 = nn.Sequential(*self.q2_layers)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through both Q-networks.

        Args:
            state: Input state tensor of shape (batch_size, state_dim)
            action: Action tensor of shape (batch_size, action_dim)

        Returns:
            Tuple of (Q1_value, Q2_value)
        """
        sa = torch.cat([state, action], dim=1)
        q1 = self.q1(sa)
        q2 = self.q2(sa)
        return q1, q2

    def q1_forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through Q1 network only.

        Args:
            state: Input state tensor
            action: Action tensor

        Returns:
            Q1 value
        """
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa)