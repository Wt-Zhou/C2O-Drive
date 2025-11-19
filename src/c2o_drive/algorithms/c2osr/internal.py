"""Internal C2OSR components.

This module provides clean imports for all existing C2OSR code
without modifying the original implementation. It acts as an
adapter layer for the algorithm interface.
"""

# Grid and spatial modules
from c2o_drive.algorithms.c2osr.grid_mapper import (
    GridSpec,
    GridMapper,
)

from c2o_drive.algorithms.c2osr.dirichlet import (
    DirichletParams,
    SpatialDirichletBank,
    MultiTimestepSpatialDirichletBank,
    OptimizedMultiTimestepSpatialDirichletBank,
)

from c2o_drive.algorithms.c2osr.trajectory_buffer import (
    TrajectoryBuffer,
    AgentTrajectoryData,
    ScenarioState,
)

from c2o_drive.algorithms.c2osr.risk import (
    compose_union_singlelayer,
)

# Evaluation modules
from c2o_drive.algorithms.c2osr.q_value import (
    QValueCalculator,
    QValueConfig as OriginalQValueConfig,
    RewardCalculator,
)

from c2o_drive.utils.collision import (
    ShapeBasedCollisionDetector,
)

from c2o_drive.algorithms.c2osr.q_evaluator import QEvaluator

# Utility modules
from c2o_drive.utils.lattice_planner import (
    LatticePlanner,
    LatticeTrajectory,
    QuinticPolynomial,
)

# Environment modules
from c2o_drive.core.types import (
    AgentState,
    EgoState,
    WorldState,
    AgentType,
    EgoControl,
)

from c2o_drive.environments.virtual.scenario_manager import (
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
