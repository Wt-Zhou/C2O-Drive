"""Configuration class for PPO algorithm.

This module provides configuration dataclasses for the PPO
(Proximal Policy Optimization) algorithm.
"""

from dataclasses import dataclass, field
from typing import Tuple

from c2o_drive.algorithms.base import PlannerConfig
from c2o_drive.algorithms.c2osr.config import LatticePlannerConfig


@dataclass
class PPOConfig(PlannerConfig):
    """PPO algorithm configuration.

    Attributes:
        lattice: Lattice planner configuration (shared with C2OSR)
        state_dim: State feature dimension
        hidden_dims: Hidden layer dimensions for network
        learning_rate: Learning rate for optimizer
        gamma: Discount factor
        gae_lambda: Lambda for GAE (Generalized Advantage Estimation)
        clip_epsilon: Clipping parameter for PPO objective
        clip_grad_norm: Gradient clipping norm
        value_loss_coef: Coefficient for value loss
        entropy_coef: Coefficient for entropy bonus
        n_epochs: Number of epochs for PPO update
        batch_size: Mini-batch size for training
        buffer_size: Rollout buffer capacity
        use_gae: Whether to use GAE
        normalize_advantage: Whether to normalize advantages
        device: Device for PyTorch (cpu/cuda)
    """
    # ========== Lattice Configuration (shared with C2OSR) ==========
    lattice: LatticePlannerConfig = field(default_factory=LatticePlannerConfig)

    # ========== Network Structure ==========
    state_dim: int = 128  # Feature dimension
    hidden_dims: Tuple[int, ...] = (256, 256)

    # ========== PPO Hyperparameters ==========
    horizon: int = 10  # Planning horizon (synchronized with lattice)
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # Clipping parameters
    clip_epsilon: float = 0.2
    clip_grad_norm: float = 0.5

    # Loss function weights
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01

    # Training parameters
    n_epochs: int = 10  # PPO update epochs
    batch_size: int = 64
    buffer_size: int = 2048  # Rollout buffer capacity

    # ========== Other ==========
    use_gae: bool = True  # Use Generalized Advantage Estimation
    normalize_advantage: bool = True
    device: str = "cpu"

    @property
    def action_dim(self) -> int:
        """Dynamically compute action dimension from lattice config.

        Returns:
            Number of discrete actions = len(lateral_offsets) * len(speed_variations)
        """
        return len(self.lattice.lateral_offsets) * len(self.lattice.speed_variations)

    def __post_init__(self):
        """Synchronize horizon to lattice config (same as C2OSR pattern)."""
        self.lattice.horizon = self.horizon

    @classmethod
    def from_global_config(cls) -> 'PPOConfig':
        """Create PPOConfig from GlobalConfig.

        This ensures all algorithms use the same lattice parameters
        as the single source of truth.

        Returns:
            PPOConfig instance with parameters from GlobalConfig
        """
        from c2o_drive.config import get_global_config
        gc = get_global_config()

        return cls(
            lattice=LatticePlannerConfig(
                lateral_offsets=gc.lattice.lateral_offsets,
                speed_variations=gc.lattice.speed_variations,
                dt=gc.lattice.dt,
                horizon=gc.lattice.horizon,
            ),
            horizon=gc.lattice.horizon,
        )


__all__ = ['PPOConfig']
