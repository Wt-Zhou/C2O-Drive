"""Internal C2OSR components.

This module provides clean imports for all existing C2OSR code
without modifying the original implementation. It acts as an
adapter layer for the algorithm interface.
"""

# Grid and spatial modules
from carla_c2osr.agents.c2osr.grid import (
    GridSpec,
    GridMapper,
)

from carla_c2osr.agents.c2osr.spatial_dirichlet import (
    DirichletParams,
    SpatialDirichletBank,
    MultiTimestepSpatialDirichletBank,
    OptimizedMultiTimestepSpatialDirichletBank,
)

from carla_c2osr.agents.c2osr.trajectory_buffer import (
    TrajectoryBuffer,
    AgentTrajectoryData,
    ScenarioState,
)

from carla_c2osr.agents.c2osr.risk import (
    compose_union_singlelayer,
)

# Evaluation modules
from carla_c2osr.evaluation.q_value_calculator import (
    QValueCalculator,
    QValueConfig as OriginalQValueConfig,
    RewardCalculator,
)

from carla_c2osr.evaluation.collision_detector import (
    ShapeBasedCollisionDetector,
)

from carla_c2osr.evaluation.q_evaluator import (
    QEvaluator,
)

# Utility modules
from carla_c2osr.utils.lattice_planner import (
    LatticePlanner,
    LatticeTrajectory,
    QuinticPolynomial,
)

# Environment modules
from carla_c2osr.env.types import (
    AgentState,
    EgoState,
    WorldState,
    AgentType,
    EgoControl,
)

from carla_c2osr.env.scenario_manager import (
    ScenarioManager,
)


__all__ = [
    # Grid
    'GridSpec',
    'GridMapper',
    # Dirichlet
    'DirichletParams',
    'SpatialDirichletBank',
    'MultiTimestepSpatialDirichletBank',
    'OptimizedMultiTimestepSpatialDirichletBank',
    # Buffer
    'TrajectoryBuffer',
    'AgentTrajectoryData',
    'ScenarioState',
    # Risk
    'compose_union_singlelayer',
    # Q-value
    'QValueCalculator',
    'OriginalQValueConfig',
    'RewardCalculator',
    # Collision
    'ShapeBasedCollisionDetector',
    # Evaluator
    'QEvaluator',
    # Lattice
    'LatticePlanner',
    'LatticeTrajectory',
    'QuinticPolynomial',
    # Types
    'AgentState',
    'EgoState',
    'WorldState',
    'AgentType',
    'EgoControl',
    # Scenario
    'ScenarioManager',
]
