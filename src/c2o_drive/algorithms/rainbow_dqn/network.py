"""Rainbow Network Architecture

Combines three Rainbow components in the network structure:
1. Dueling Network Architecture - separate value and advantage streams
2. Distributional RL (C51) - output probability distributions over returns
3. Noisy Nets - parameter-space noise for exploration

Reference:
    Hessel et al. "Rainbow: Combining Improvements in Deep Reinforcement Learning" (2017)
"""

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from c2o_drive.core.types import WorldState
from c2o_drive.algorithms.rainbow_dqn.config import RainbowDQNConfig
from c2o_drive.algorithms.rainbow_dqn.noisy_linear import NoisyLinear
from c2o_drive.algorithms.rainbow_dqn.trajectory_encoder import WorldStateEncoder


class RainbowNetwork(nn.Module):
    """Rainbow DQN Network with Dueling, C51, and Noisy Nets.

    Architecture:
        WorldState → Encoder → Features
                                ↓
                        ┌───────┴────────┐
                        ↓                ↓
                   Value Stream    Advantage Stream
                   (Noisy Layers)  (Noisy Layers)
                        ↓                ↓
                   V(s) dist.      A(s,a) dist.
                        └───────┬────────┘
                                ↓
                        Q(s,a) = V(s) + (A(s,a) - mean(A(s,·)))
                                ↓
                        (batch, actions, atoms)

    Attributes:
        config: Rainbow DQN configuration
        num_actions: Number of discrete actions (trajectories)
        num_atoms: Number of atoms for C51 distribution
        v_min: Minimum value for distribution support
        v_max: Maximum value for distribution support
        support: Atom values for distribution
    """

    def __init__(self, config: RainbowDQNConfig):
        """Initialize Rainbow network.

        Args:
            config: Rainbow DQN configuration
        """
        super().__init__()

        self.config = config
        self.num_actions = config.network.num_actions
        self.num_atoms = config.network.num_atoms
        self.v_min = config.network.v_min
        self.v_max = config.network.v_max

        # Support: atom values for C51 distribution
        self.register_buffer(
            'support',
            torch.linspace(self.v_min, self.v_max, self.num_atoms)
        )

        # WorldState encoder
        self.state_encoder = WorldStateEncoder(config)

        feature_dim = config.network.state_feature_dim

        # === Dueling Architecture ===

        # Value stream: V(s) as distribution over atoms
        self.value_stream = nn.Sequential(
            NoisyLinear(feature_dim, 512, sigma_init=config.network.noisy_sigma),
            nn.ReLU(),
            NoisyLinear(512, self.num_atoms, sigma_init=config.network.noisy_sigma)
        )

        # Advantage stream: A(s,a) as distribution over atoms
        # Output: num_actions * num_atoms
        self.advantage_stream = nn.Sequential(
            NoisyLinear(feature_dim, 512, sigma_init=config.network.noisy_sigma),
            nn.ReLU(),
            NoisyLinear(512, self.num_actions * self.num_atoms, sigma_init=config.network.noisy_sigma)
        )

    def forward(self, world_states: List[WorldState]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through Rainbow network.

        Args:
            world_states: List of WorldState observations

        Returns:
            Tuple of (q_dist, q_values):
            - q_dist: Q-value distributions, shape (batch, num_actions, num_atoms)
            - q_values: Expected Q-values, shape (batch, num_actions)
        """
        batch_size = len(world_states)

        # === Encode WorldStates ===
        features = self.state_encoder(world_states)  # (batch, feature_dim)

        # === Dueling Streams ===

        # Value stream: V(s) distribution
        value = self.value_stream(features)  # (batch, num_atoms)
        value = value.view(batch_size, 1, self.num_atoms)  # (batch, 1, num_atoms)

        # Advantage stream: A(s,a) distribution
        advantage = self.advantage_stream(features)  # (batch, num_actions * num_atoms)
        advantage = advantage.view(batch_size, self.num_actions, self.num_atoms)  # (batch, num_actions, num_atoms)

        # === Dueling Aggregation ===
        # Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a))
        q_logits = value + (advantage - advantage.mean(dim=1, keepdim=True))

        # === C51: Convert to probability distribution ===
        q_dist = F.softmax(q_logits, dim=2)  # (batch, num_actions, num_atoms)

        # === Compute expected Q-values ===
        # Q(s,a) = E_z[z * p(z)]
        q_values = (q_dist * self.support).sum(dim=2)  # (batch, num_actions)

        return q_dist, q_values

    def reset_noise(self):
        """Reset noise in all Noisy Linear layers.

        Should be called before each forward pass during training to
        sample new noise values for exploration.
        """
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    def get_action(self, world_state: WorldState, deterministic: bool = False) -> int:
        """Select action for a single WorldState.

        Args:
            world_state: Current observation
            deterministic: If True, select argmax action; if False, reset noise first

        Returns:
            Action index (trajectory ID)
        """
        if not deterministic:
            self.reset_noise()

        with torch.no_grad():
            _, q_values = self.forward([world_state])
            action = q_values.argmax(dim=1).item()

        return action
