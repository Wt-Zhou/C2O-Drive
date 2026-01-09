"""Configuration dataclasses for RCRL algorithm.

This module defines the hierarchical configuration structure following
the C2OSR pattern with single source of truth for shared parameters.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Literal

from c2o_drive.algorithms.base import PlannerConfig


@dataclass
class ReachabilityConfig:
    """Configuration for reachability set computation.

    Attributes:
        horizon: Planning horizon in timesteps (synced from parent)
        dt: Time step duration in seconds (synced from parent)
        use_ego_reachability: Whether to compute ego reachable sets
        use_agent_reachability: Whether to compute agent reachable sets
        ego_expand_radius: Radius in cells to expand ego trajectory for control error
        grid_cell_size: Grid cell size in meters (for GridMapper)
        grid_size: Total grid size in meters (for GridMapper)
    """

    horizon: int = 10
    dt: float = 0.1
    use_ego_reachability: bool = True
    use_agent_reachability: bool = True
    ego_expand_radius: int = 1
    grid_cell_size: float = 0.5
    grid_size: float = 100.0


@dataclass
class ConstraintConfig:
    """Configuration for safety constraint enforcement.

    Attributes:
        mode: Constraint mode - "hard" (filter unsafe) or "soft" (penalty)
        soft_penalty_weight: Weight for safety penalty in soft mode
        collision_threshold: Distance threshold for collision detection (meters)
        use_probabilistic: Whether to use probabilistic collision checking
    """

    mode: Literal["hard", "soft"] = "soft"
    soft_penalty_weight: float = 100.0
    collision_threshold: float = 0.1
    use_probabilistic: bool = False


@dataclass
class NetworkConfig:
    """Configuration for DQN neural network architecture.

    Attributes:
        state_dim: Dimension of state feature vector
        hidden_dims: List of hidden layer dimensions
        use_dueling: Whether to use Dueling DQN architecture
        use_noisy: Whether to use Noisy Networks for exploration
        activation: Activation function name ("relu", "tanh", "elu")
        dropout_rate: Dropout rate (0 means no dropout)
    """

    state_dim: int = 128
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    use_dueling: bool = True
    use_noisy: bool = False
    activation: str = "relu"
    dropout_rate: float = 0.0


@dataclass
class TrainingConfig:
    """Configuration for DQN training process.

    Attributes:
        gamma: Discount factor for future rewards (synced from parent)
        learning_rate: Learning rate for optimizer
        batch_size: Batch size for training
        buffer_capacity: Maximum capacity of replay buffer
        min_buffer_size: Minimum buffer size before training starts
        target_update_freq: Frequency (in steps) to update target network
        train_freq: Frequency (in steps) to perform training update
        epsilon_start: Initial epsilon for epsilon-greedy exploration
        epsilon_end: Final epsilon after decay
        epsilon_decay_steps: Number of steps for epsilon decay
        double_dqn: Whether to use Double DQN algorithm
        grad_clip_norm: Maximum gradient norm for clipping (0 means no clipping)
    """

    gamma: float = 0.99
    learning_rate: float = 3e-4
    batch_size: int = 64
    buffer_capacity: int = 50000
    min_buffer_size: int = 1000
    target_update_freq: int = 100
    train_freq: int = 1
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 10000
    double_dqn: bool = True
    grad_clip_norm: float = 10.0


@dataclass
class LatticePlannerConfig:
    """Configuration for lattice trajectory generation.

    Attributes:
        horizon: Planning horizon in timesteps (synced from parent)
        dt: Time step duration in seconds (synced from parent)
        lateral_offsets: Lateral offset options in meters
        target_speeds: Target speed options in m/s
        include_emergency_brake: Whether to include emergency brake action

    Note:
        num_trajectories is dynamically computed from lateral_offsets × target_speeds
    """

    horizon: int = 10
    dt: float = 0.1
    lateral_offsets: List[float] = field(default_factory=lambda: [-3.0, -2.0, 0.0, 2.0, 3.0])
    target_speeds: List[float] = field(default_factory=lambda: [4.0])
    include_emergency_brake: bool = False

    @property
    def num_trajectories(self) -> int:
        """Dynamically compute number of trajectories.

        Returns:
            len(lateral_offsets) × len(target_speeds)
        """
        return len(self.lateral_offsets) * len(self.target_speeds)


@dataclass
class StateEncoderConfig:
    """Configuration for state encoding.

    Attributes:
        state_dim: Target dimension for encoded state
        max_agents: Maximum number of agents to encode
        normalization_vmax: Maximum speed for velocity normalization (m/s)
        normalization_dmax: Maximum distance for position normalization (m)
    """

    state_dim: int = 128
    max_agents: int = 10
    normalization_vmax: float = 20.0
    normalization_dmax: float = 100.0


@dataclass
class RCRLPlannerConfig(PlannerConfig):
    """Main configuration for RCRL planner algorithm.

    This configuration follows the C2OSR pattern with hierarchical structure
    and single source of truth for shared parameters (horizon, dt, gamma).

    Attributes:
        horizon: Planning horizon in timesteps (single source of truth)
        dt: Time step duration in seconds (single source of truth)
        gamma: Discount factor for rewards (single source of truth)
        device: Device for neural network ("cpu", "cuda", "cuda:0", etc.)
        use_reference_path: Whether to use reference path for planning
        max_episode_steps: Maximum steps per episode
        reachability: Reachability computation configuration
        constraint: Safety constraint configuration
        network: Neural network architecture configuration
        training: DQN training configuration
        lattice: Lattice planner configuration
        state_encoder: State encoding configuration
    """

    # Single source of truth parameters
    horizon: int = 10
    dt: float = 0.1
    gamma: float = 0.99

    # Device configuration
    device: str = "cpu"

    # Planning configuration
    use_reference_path: bool = True
    max_episode_steps: int = 500

    # Sub-configurations
    reachability: ReachabilityConfig = field(default_factory=ReachabilityConfig)
    constraint: ConstraintConfig = field(default_factory=ConstraintConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    lattice: LatticePlannerConfig = field(default_factory=LatticePlannerConfig)
    state_encoder: StateEncoderConfig = field(default_factory=StateEncoderConfig)

    def __post_init__(self):
        """Synchronize shared parameters to all sub-configurations.

        This ensures single source of truth for horizon, dt, and gamma.
        Following the C2OSR pattern.
        """
        # Synchronize horizon
        self.reachability.horizon = self.horizon
        self.lattice.horizon = self.horizon

        # Synchronize dt
        self.reachability.dt = self.dt
        self.lattice.dt = self.dt

        # Synchronize gamma
        self.training.gamma = self.gamma

        # Ensure state_dim consistency
        self.network.state_dim = self.state_encoder.state_dim

    @classmethod
    def from_global_config(cls) -> "RCRLPlannerConfig":
        """Create configuration from global config.

        Returns:
            RCRLPlannerConfig instance with parameters from global config
        """
        from c2o_drive.config.global_config import get_global_config

        gc = get_global_config()

        return cls(
            horizon=gc.lattice.horizon,
            dt=gc.lattice.dt,
            gamma=gc.reward.gamma,
            lattice=LatticePlannerConfig(
                lateral_offsets=gc.lattice.lateral_offsets,
                target_speeds=gc.lattice.speed_variations,
                horizon=gc.lattice.horizon,
                dt=gc.lattice.dt,
            ),
        )

    def get_n_actions(self) -> int:
        """Calculate total number of actions.

        Returns:
            Total number of discrete actions
        """
        n_lattice = len(self.lattice.lateral_offsets) * len(self.lattice.target_speeds)
        if self.lattice.include_emergency_brake:
            return n_lattice + 1
        return n_lattice


__all__ = [
    "ReachabilityConfig",
    "ConstraintConfig",
    "NetworkConfig",
    "TrainingConfig",
    "LatticePlannerConfig",
    "StateEncoderConfig",
    "RCRLPlannerConfig",
]
