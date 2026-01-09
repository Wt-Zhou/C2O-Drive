"""Configuration class for SAC algorithm (Planner mode, discrete actions).

This module provides configuration dataclasses for the SAC
(Soft Actor-Critic) algorithm adapted for discrete action spaces with
lattice-based trajectory planning.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional

from c2o_drive.algorithms.base import PlannerConfig
from c2o_drive.algorithms.c2osr.config import LatticePlannerConfig


@dataclass
class SACConfig(PlannerConfig):
    """SAC algorithm configuration (Planner mode, discrete actions).

    This configuration adapts SAC for discrete lattice-based action spaces
    while preserving SAC's core features: double Q-networks, entropy
    regularization, and automatic temperature adjustment.

    Key differences from continuous SAC:
    - Uses Categorical policy instead of Gaussian
    - Discrete Q-networks instead of continuous critics
    - Action space defined by lattice parameters (dynamically computed)

    Attributes:
        lattice: Lattice planner configuration (shared with C2OSR/PPO)
        state_dim: State feature dimension (unified across all algorithms)
        hidden_dims: Hidden layer dimensions for actor and critic networks
        horizon: Planning horizon (synchronized with lattice)
        learning_rate: Learning rate for all optimizers
        actor_lr: Learning rate for actor (if different from learning_rate)
        critic_lr: Learning rate for critics (if different from learning_rate)
        alpha_lr: Learning rate for entropy coefficient
        gamma: Discount factor
        tau: Soft update coefficient for target networks
        initial_alpha: Initial value for entropy coefficient
        target_entropy: Target entropy for automatic tuning (None for auto)
        batch_size: Mini-batch size for training
        buffer_size: Replay buffer capacity
        update_interval: Update frequency (steps between updates)
        device: Device for PyTorch (cpu/cuda)
    """
    # ========== Lattice Configuration (shared with C2OSR/PPO) ==========
    lattice: LatticePlannerConfig = field(default_factory=LatticePlannerConfig)

    # ========== Network Structure ==========
    state_dim: int = 128  # Unified state feature dimension
    hidden_dims: Tuple[int, ...] = (256, 256)

    # ========== SAC Hyperparameters ==========
    horizon: int = 10  # Planning horizon (synchronized with lattice)
    learning_rate: float = 3e-4
    actor_lr: Optional[float] = None  # If None, uses learning_rate
    critic_lr: Optional[float] = None  # If None, uses learning_rate
    alpha_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005  # Soft update coefficient for target networks

    # Entropy regularization
    initial_alpha: float = 0.2
    target_entropy: Optional[float] = None  # If None, auto-set to -action_dim * 0.5

    # ========== Training Parameters ==========
    batch_size: int = 256
    buffer_size: int = 100000  # Smaller than continuous SAC (discrete needs fewer samples)
    update_interval: int = 1  # Update every N steps

    # ========== Device ==========
    device: str = "cpu"

    @property
    def action_dim(self) -> int:
        """Dynamically compute action dimension from lattice config.

        This ensures action_dim automatically adapts when lattice parameters change.
        Same pattern as C2OSR and PPO.

        Returns:
            Number of discrete actions = len(lateral_offsets) * len(speed_variations)

        Examples:
            - lateral_offsets=[-3,-2,0,2,3], speed_variations=[4.0] → action_dim=5
            - lateral_offsets=[-3,-2,0,2,3], speed_variations=[4,6,8] → action_dim=15
        """
        return len(self.lattice.lateral_offsets) * len(self.lattice.speed_variations)

    def __post_init__(self):
        """Validate and set default values."""
        # Synchronize horizon to lattice (same as C2OSR/PPO pattern)
        self.lattice.horizon = self.horizon

        # Set default learning rates if not specified
        if self.actor_lr is None:
            self.actor_lr = self.learning_rate
        if self.critic_lr is None:
            self.critic_lr = self.learning_rate

        # Auto-set target entropy for discrete actions
        # Use -action_dim * 0.5 as default (encourages exploration but not too much)
        if self.target_entropy is None:
            self.target_entropy = -self.action_dim * 0.5

        # Validation
        assert self.state_dim > 0, "State dimension must be positive"
        assert 0 <= self.gamma <= 1, "Gamma must be in [0, 1]"
        assert 0 < self.tau <= 1, "Tau must be in (0, 1]"
        assert self.learning_rate > 0, "Learning rate must be positive"
        assert self.batch_size > 0, "Batch size must be positive"
        assert self.buffer_size >= self.batch_size, "Buffer size must be >= batch size"

    @classmethod
    def from_global_config(cls) -> 'SACConfig':
        """Create SACConfig from GlobalConfig.

        This ensures all algorithms use the same lattice and reward parameters
        as the single source of truth.

        Returns:
            SACConfig instance with parameters from GlobalConfig

        Example:
            ```python
            from c2o_drive.config import get_global_config
            from c2o_drive.algorithms.sac import SACConfig

            # Modify global config
            gc = get_global_config()
            gc.lattice.speed_variations = [4.0, 6.0, 8.0]  # Change from 1 to 3 speeds

            # Create SAC config (automatically gets action_dim=15)
            config = SACConfig.from_global_config()
            assert config.action_dim == 15  # 5 lateral × 3 speeds
            ```
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


__all__ = ['SACConfig']
