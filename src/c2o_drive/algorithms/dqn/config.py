"""Configuration for DQN algorithm."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DQNConfig:
    """Configuration for Deep Q-Network algorithm.

    Attributes:
        state_dim: Dimension of the state space
        action_dim: Dimension of the action space
        learning_rate: Learning rate for the optimizer
        gamma: Discount factor for future rewards
        epsilon_start: Initial exploration rate
        epsilon_end: Final exploration rate
        epsilon_decay: Number of steps for epsilon decay
        batch_size: Batch size for training
        buffer_size: Size of the replay buffer
        target_update_freq: Frequency of target network updates
        hidden_dims: Hidden layer dimensions for the Q-network
        device: Device to run the model on ('cuda' or 'cpu')
    """

    # State and action dimensions
    state_dim: int = 128
    action_dim: int = 9  # 3x3 discrete actions (steer x throttle/brake)

    # Learning parameters
    learning_rate: float = 1e-4
    gamma: float = 0.99

    # Exploration parameters
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: int = 10000

    # Training parameters
    batch_size: int = 32
    buffer_size: int = 100000
    target_update_freq: int = 1000

    # Network architecture
    hidden_dims: tuple = (256, 256)

    # Device
    device: str = "cpu"

    def __post_init__(self):
        """Validate configuration."""
        assert self.state_dim > 0, "State dimension must be positive"
        assert self.action_dim > 0, "Action dimension must be positive"
        assert 0 <= self.gamma <= 1, "Gamma must be in [0, 1]"
        assert self.epsilon_start >= self.epsilon_end, "Epsilon start must be >= epsilon end"