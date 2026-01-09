"""Rainbow DQN Configuration Classes

Hierarchical configuration system following C2OSR's design pattern.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from c2o_drive.algorithms.base import PlannerConfig


@dataclass
class RainbowNetworkConfig:
    """Network architecture configuration for Rainbow DQN.

    Attributes:
        state_feature_dim: Dimension of encoded state features
        num_atoms: Number of atoms for C51 distributional RL
        v_min: Minimum value for value distribution
        v_max: Maximum value for value distribution
        noisy_sigma: Initial noise parameter for Noisy Nets
        num_actions: Number of possible actions (set automatically)
    """
    state_feature_dim: int = 256
    num_atoms: int = 51
    v_min: float = -100.0
    v_max: float = 100.0
    noisy_sigma: float = 0.5
    num_actions: int = 15  # Will be set to lattice.num_trajectories


@dataclass
class ReplayBufferConfig:
    """Prioritized Experience Replay configuration.

    Attributes:
        capacity: Maximum buffer size
        alpha: Prioritization exponent (0=uniform, 1=full prioritization)
        beta_start: Initial importance sampling weight
        beta_frames: Number of frames to anneal beta to 1.0
    """
    capacity: int = 100000
    alpha: float = 0.6
    beta_start: float = 0.4
    beta_frames: int = 100000


@dataclass
class TrainingConfig:
    """Training hyperparameters configuration.

    Attributes:
        batch_size: Minibatch size for updates
        learning_rate: Adam learning rate
        gamma: Discount factor
        n_step: Number of steps for n-step returns
        target_update_freq: Frequency (in updates) to sync target network
        gradient_clip: Maximum gradient norm for clipping
        warmup_steps: Number of steps before training starts
    """
    batch_size: int = 32
    learning_rate: float = 6.25e-5
    gamma: float = 0.99
    n_step: int = 3
    target_update_freq: int = 1000
    gradient_clip: float = 10.0
    warmup_steps: int = 1000


@dataclass
class LatticePlannerConfig:
    """Lattice trajectory planner configuration (reused from C2OSR).

    Attributes:
        lateral_offsets: List of lateral offset values (meters)
        speed_variations: List of target speed values (m/s)
        horizon: Planning horizon (number of time steps)
        dt: Time step duration (seconds)

    Note:
        num_trajectories is dynamically computed from lateral_offsets × speed_variations
    """
    lateral_offsets: List[float] = field(default_factory=lambda: [-3.0, -2.0, 0.0, 2.0, 3.0])
    speed_variations: List[float] = field(default_factory=lambda: [4.0, 6.0, 8.0])
    horizon: int = 10
    dt: float = 1.0

    @property
    def num_trajectories(self) -> int:
        """Dynamically compute number of trajectories.

        Returns:
            len(lateral_offsets) × len(speed_variations)
        """
        return len(self.lateral_offsets) * len(self.speed_variations)


@dataclass
class GridConfig:
    """Grid discretization configuration (reused from C2OSR).

    Attributes:
        grid_size_m: Grid cell size in meters
        bounds_x: (min, max) x-coordinates for grid
        bounds_y: (min, max) y-coordinates for grid
    """
    grid_size_m: float = 0.5
    bounds_x: Tuple[float, float] = (-50.0, 50.0)
    bounds_y: Tuple[float, float] = (-50.0, 50.0)


@dataclass
class RainbowDQNConfig(PlannerConfig):
    """Complete Rainbow DQN configuration.

    This is the main configuration class that aggregates all sub-configurations.
    Follows C2OSR's pattern of hierarchical configuration with automatic parameter
    synchronization in __post_init__.

    Attributes:
        network: Network architecture configuration
        replay: Replay buffer configuration
        training: Training hyperparameters
        lattice: Lattice planner configuration
        grid: Grid discretization configuration
        device: Device for computation ('cuda' or 'cpu')
        seed: Random seed for reproducibility
    """
    # Sub-configurations
    network: RainbowNetworkConfig = field(default_factory=RainbowNetworkConfig)
    replay: ReplayBufferConfig = field(default_factory=ReplayBufferConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    lattice: LatticePlannerConfig = field(default_factory=LatticePlannerConfig)
    grid: GridConfig = field(default_factory=GridConfig)

    # Device and seed
    device: str = "cuda"
    seed: int = 42

    def __post_init__(self):
        """Synchronize parameters across sub-configurations.

        This ensures consistency between top-level parameters and
        sub-configuration parameters, following C2OSR's pattern.
        """
        # Sync learning rate and gamma
        self.training.learning_rate = self.learning_rate
        self.training.gamma = self.gamma

        # Sync action space size (number of trajectories)
        self.network.num_actions = self.lattice.num_trajectories

    @classmethod
    def from_global_config(cls) -> 'RainbowDQNConfig':
        """Create RainbowDQNConfig from global configuration.

        Returns:
            RainbowDQNConfig instance with parameters from global config
        """
        from c2o_drive.config import get_global_config

        gc = get_global_config()

        return cls(
            lattice=LatticePlannerConfig(
                lateral_offsets=gc.lattice.lateral_offsets if hasattr(gc, 'lattice') else [-3.0, -2.0, 0.0, 2.0, 3.0],
                speed_variations=gc.lattice.speed_variations if hasattr(gc, 'lattice') else [4.0, 6.0, 8.0],
                horizon=gc.time.default_horizon if hasattr(gc, 'time') else 10,
                dt=gc.time.dt if hasattr(gc, 'time') else 1.0,
            ),
            grid=GridConfig(
                grid_size_m=gc.grid.cell_size_m if hasattr(gc, 'grid') else 0.5,
                bounds_x=(gc.grid.x_min, gc.grid.x_max) if hasattr(gc, 'grid') else (-50.0, 50.0),
                bounds_y=(gc.grid.y_min, gc.grid.y_max) if hasattr(gc, 'grid') else (-50.0, 50.0),
            ),
            gamma=gc.c2osr.gamma if hasattr(gc, 'c2osr') else 0.99,
            seed=gc.seed if hasattr(gc, 'seed') else 42,
        )
