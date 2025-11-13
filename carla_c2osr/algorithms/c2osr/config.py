"""Configuration classes for C2OSR algorithm.

This module provides configuration dataclasses for the C2OSR
(Dirichlet-based planning) algorithm.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from carla_c2osr.algorithms.base import PlannerConfig, EvaluatorConfig


@dataclass
class GridConfig:
    """Grid discretization configuration.

    Attributes:
        grid_size_m: Size of each grid cell in meters
        bounds_x: X-axis bounds (min, max) in meters
        bounds_y: Y-axis bounds (min, max) in meters
    """
    grid_size_m: float = 0.5  # Match old configuration (global_config.py:47)
    bounds_x: tuple[float, float] = (-50.0, 50.0)
    bounds_y: tuple[float, float] = (-50.0, 50.0)


@dataclass
class DirichletConfig:
    """Dirichlet distribution configuration.

    Attributes:
        alpha_in: Prior strength for reachable cells
        alpha_out: Prior strength for non-reachable cells
        learning_rate: Learning rate for updating with historical data
        use_multistep: Whether to use multi-timestep Dirichlet bank
        use_optimized: Whether to use optimized implementation
    """
    alpha_in: float = 50.0
    alpha_out: float = 1e-6
    learning_rate: float = 1.0
    use_multistep: bool = True
    use_optimized: bool = True


@dataclass
class LatticePlannerConfig:
    """Lattice trajectory planner configuration.

    Attributes:
        lateral_offsets: List of lateral offset samples in meters
        speed_variations: List of target speed samples in m/s
        num_trajectories: Target number of trajectories to generate
        horizon: Planning horizon in timesteps (None = read from global config)
        dt: Time step in seconds
    """
    lateral_offsets: List[float] = field(default_factory=lambda: [-3.0, -2.0, 0.0, 2.0, 3.0])
    speed_variations: List[float] = field(default_factory=lambda: [4.0])
    num_trajectories: int = 25
    horizon: Optional[int] = None  # Read from global config if not specified
    dt: float = 1.0

    def __post_init__(self):
        """Load horizon from global config if not explicitly set."""
        if self.horizon is None:
            from carla_c2osr.config import get_global_config
            self.horizon = get_global_config().time.default_horizon


@dataclass
class QValueConfig:
    """Q-value calculation configuration.

    Attributes:
        horizon: Prediction horizon in timesteps (None = read from global config)
        n_samples: Number of samples for Dirichlet sampling
        selection_percentile: Percentile for Q-value selection (0=min, 1=max)
        gamma: Discount factor for future rewards
    """
    horizon: Optional[int] = None  # Read from global config if not specified
    n_samples: int = 50
    selection_percentile: float = 0.05  # 5th percentile
    gamma: float = 0.9

    def __post_init__(self):
        """Load horizon from global config if not explicitly set."""
        if self.horizon is None:
            from carla_c2osr.config import get_global_config
            self.horizon = get_global_config().time.default_horizon


@dataclass
class RewardWeightsConfig:
    """Reward function weight configuration.

    Attributes:
        collision_penalty: Penalty for collision
        collision_threshold: Collision detection threshold
        collision_check_cell_radius: Cell pruning radius for collision check
        comfort_weight: Weight for comfort reward
        efficiency_weight: Weight for efficiency reward
        safety_weight: Weight for safety reward
        max_accel_penalty: Maximum acceleration penalty
        max_jerk_penalty: Maximum jerk penalty
        acceleration_penalty_weight: Weight for acceleration penalty
        jerk_penalty_weight: Weight for jerk penalty
        max_comfortable_accel: Maximum comfortable acceleration (m/s²)
        speed_reward_weight: Weight for speed reward
        target_speed: Target speed (m/s)
        progress_reward_weight: Weight for progress reward
        safe_distance: Safe distance (m)
        distance_penalty_weight: Weight for distance penalty
        centerline_offset_penalty_weight: Weight for centerline offset penalty
    """
    # Collision parameters
    collision_penalty: float = -50.0
    collision_threshold: float = 0.01
    collision_check_cell_radius: int = 6

    # Weight parameters
    comfort_weight: float = 1.0
    efficiency_weight: float = 1.0
    safety_weight: float = 10.0

    # Comfort parameters
    max_accel_penalty: float = -1.0
    max_jerk_penalty: float = -2.0
    acceleration_penalty_weight: float = 0.1
    jerk_penalty_weight: float = 0.05
    max_comfortable_accel: float = 1.0

    # Efficiency parameters
    speed_reward_weight: float = 1.0
    target_speed: float = 5.0
    progress_reward_weight: float = 1.0

    # Safety distance parameters
    safe_distance: float = 3.0
    distance_penalty_weight: float = 0.0

    # Centerline offset parameters
    centerline_offset_penalty_weight: float = 2.0


@dataclass
class C2OSRPlannerConfig(PlannerConfig):
    """Complete configuration for C2OSR planner.

    Attributes:
        grid: Grid discretization config
        dirichlet: Dirichlet distribution config
        lattice: Lattice planner config
        q_value: Q-value calculation config
        reward_weights: Reward function weights
        buffer_capacity: Maximum capacity of trajectory buffer
        min_buffer_size: Minimum buffer size before using Dirichlet
    """
    # Subconfigs
    grid: GridConfig = field(default_factory=GridConfig)
    dirichlet: DirichletConfig = field(default_factory=DirichletConfig)
    lattice: LatticePlannerConfig = field(default_factory=LatticePlannerConfig)
    q_value: QValueConfig = field(default_factory=QValueConfig)
    reward_weights: RewardWeightsConfig = field(default_factory=RewardWeightsConfig)

    # Buffer settings
    buffer_capacity: int = 10000
    min_buffer_size: int = 10
    trajectory_storage_multiplier: int = 1  # 轨迹存储倍数（数据增强，默认10）

    # ===== 统一的Horizon配置（唯一真实来源）=====
    horizon: int = 10  # 🎯 修改这里即可统一所有模块的horizon
    # 所有子配置（lattice, q_value）以及组件（DirichletBank, Buffer）都将使用这个值
    # ============================================

    # Base planner settings (inherited from PlannerConfig)
    learning_rate: float = 1.0  # Used for Dirichlet learning
    gamma: float = 0.9  # Discount factor

    def __post_init__(self):
        """Sync related parameters after initialization."""
        # 🎯 强制同步统一的horizon到所有子配置（单一真实来源）
        self.lattice.horizon = self.horizon
        self.q_value.horizon = self.horizon

        # Sync other parameters
        self.q_value.gamma = self.gamma
        self.dirichlet.learning_rate = self.learning_rate

    @classmethod
    def from_global_config(cls) -> 'C2OSRPlannerConfig':
        """Create config from global configuration.

        Returns:
            C2OSRPlannerConfig initialized from global config
        """
        try:
            from carla_c2osr.config import get_global_config

            gc = get_global_config()

            return cls(
                # 🎯 统一的horizon从全局配置读取
                horizon=gc.time.default_horizon,

                # Grid config
                grid=GridConfig(
                    grid_size_m=gc.grid.cell_size_m,
                    bounds_x=(gc.grid.x_min, gc.grid.x_max),
                    bounds_y=(gc.grid.y_min, gc.grid.y_max),
                ),
                # Dirichlet config
                dirichlet=DirichletConfig(
                    alpha_in=gc.dirichlet.alpha_in,
                    alpha_out=gc.dirichlet.alpha_out,
                    learning_rate=gc.dirichlet.learning_rate,
                ),
                # Lattice config (horizon将在__post_init__中自动同步)
                lattice=LatticePlannerConfig(
                    lateral_offsets=gc.lattice.lateral_offsets,
                    speed_variations=gc.lattice.speed_variations,
                    num_trajectories=gc.lattice.num_trajectories,
                    # 不传horizon，由__post_init__自动同步
                    dt=gc.time.dt,
                ),
                # Q-value config (horizon将在__post_init__中自动同步)
                q_value=QValueConfig(
                    # 不传horizon，由__post_init__自动同步
                    n_samples=gc.sampling.q_value_samples,
                    selection_percentile=gc.c2osr.q_selection_percentile,
                    gamma=gc.c2osr.gamma,
                ),
                # Reward weights
                reward_weights=RewardWeightsConfig(
                    collision_penalty=gc.reward.collision_penalty,
                    collision_threshold=gc.reward.collision_threshold,
                    collision_check_cell_radius=gc.reward.collision_check_cell_radius,
                    comfort_weight=gc.reward.comfort_weight,
                    efficiency_weight=gc.reward.efficiency_weight,
                    safety_weight=gc.reward.safety_weight,
                    max_accel_penalty=gc.reward.max_accel_penalty,
                    max_jerk_penalty=gc.reward.max_jerk_penalty,
                    acceleration_penalty_weight=gc.reward.acceleration_penalty_weight,
                    jerk_penalty_weight=gc.reward.jerk_penalty_weight,
                    max_comfortable_accel=gc.reward.max_comfortable_accel,
                    speed_reward_weight=gc.reward.speed_reward_weight,
                    target_speed=gc.reward.target_speed,
                    progress_reward_weight=gc.reward.progress_reward_weight,
                    safe_distance=gc.reward.safe_distance,
                    distance_penalty_weight=gc.reward.distance_penalty_weight,
                    centerline_offset_penalty_weight=gc.reward.centerline_offset_penalty_weight,
                ),
                # Buffer settings
                buffer_capacity=10000,
                min_buffer_size=10,
                trajectory_storage_multiplier=gc.matching.trajectory_storage_multiplier,
                # Base settings
                learning_rate=gc.dirichlet.learning_rate,
                gamma=gc.c2osr.gamma,
                seed=0,
            )
        except ImportError:
            # If global config not available, use defaults
            return cls()


@dataclass
class C2OSREvaluatorConfig(EvaluatorConfig):
    """Configuration for C2OSR trajectory evaluator.

    Attributes:
        grid: Grid discretization config
        dirichlet: Dirichlet distribution config
        q_value: Q-value calculation config
        reward_weights: Reward function weights
    """
    grid: GridConfig = field(default_factory=GridConfig)
    dirichlet: DirichletConfig = field(default_factory=DirichletConfig)
    q_value: QValueConfig = field(default_factory=QValueConfig)
    reward_weights: RewardWeightsConfig = field(default_factory=RewardWeightsConfig)

    @classmethod
    def from_planner_config(cls, planner_config: C2OSRPlannerConfig) -> 'C2OSREvaluatorConfig':
        """Create evaluator config from planner config.

        Args:
            planner_config: C2OSR planner configuration

        Returns:
            C2OSREvaluatorConfig with matching parameters
        """
        return cls(
            grid=planner_config.grid,
            dirichlet=planner_config.dirichlet,
            q_value=planner_config.q_value,
            reward_weights=planner_config.reward_weights,
            batch_size=32,
            use_cache=True,
            seed=planner_config.seed,
        )


__all__ = [
    'GridConfig',
    'DirichletConfig',
    'LatticePlannerConfig',
    'QValueConfig',
    'RewardWeightsConfig',
    'C2OSRPlannerConfig',
    'C2OSREvaluatorConfig',
]
