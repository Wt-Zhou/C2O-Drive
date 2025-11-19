"""Configuration for SAC algorithm."""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class SACConfig:
    """Configuration for Soft Actor-Critic algorithm.

    Attributes:
        state_dim: Dimension of the state space
        action_dim: Dimension of the action space
        max_action: Maximum absolute value for actions
        learning_rate: Learning rate for all optimizers
        actor_lr: Learning rate for actor (if different from learning_rate)
        critic_lr: Learning rate for critics (if different from learning_rate)
        alpha_lr: Learning rate for entropy coefficient
        gamma: Discount factor for future rewards
        tau: Soft update coefficient for target networks
        batch_size: Batch size for training
        buffer_size: Size of the replay buffer
        hidden_dims: Hidden layer dimensions for networks
        initial_alpha: Initial value for entropy coefficient
        target_entropy: Target entropy for automatic tuning (None for auto)
        reward_scale: Scale factor for rewards
        device: Device to run the model on ('cuda' or 'cpu')
    """

    # State and action dimensions
    state_dim: int = 128
    action_dim: int = 2  # Continuous: steering and throttle/brake

    # Action bounds
    max_action: float = 1.0

    # Learning parameters
    learning_rate: float = 3e-4
    actor_lr: Optional[float] = None
    critic_lr: Optional[float] = None
    alpha_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005

    # Training parameters
    batch_size: int = 256
    buffer_size: int = 1000000

    # Network architecture
    hidden_dims: Tuple[int, ...] = (256, 256)

    # Entropy regularization
    initial_alpha: float = 0.2
    target_entropy: Optional[float] = None  # If None, uses -action_dim

    # Reward scaling
    reward_scale: float = 1.0

    # Device
    device: str = "cpu"

    def __post_init__(self):
        """Validate and set default values."""
        assert self.state_dim > 0, "State dimension must be positive"
        assert self.action_dim > 0, "Action dimension must be positive"
        assert 0 <= self.gamma <= 1, "Gamma must be in [0, 1]"
        assert 0 < self.tau <= 1, "Tau must be in (0, 1]"

        # Set default learning rates if not specified
        if self.actor_lr is None:
            self.actor_lr = self.learning_rate
        if self.critic_lr is None:
            self.critic_lr = self.learning_rate

        # Set default target entropy
        if self.target_entropy is None:
            self.target_entropy = -float(self.action_dim)