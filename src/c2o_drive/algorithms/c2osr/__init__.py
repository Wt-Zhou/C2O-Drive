"""C2OSR (Closed-loop Online Safety Reasoning) Algorithm.

This module implements the C2OSR algorithm for safe autonomous driving
using Dirichlet process models and Q-value based trajectory selection.
"""

from c2o_drive.algorithms.c2osr.planner import C2OSRPlanner
from c2o_drive.algorithms.c2osr.config import C2OSRPlannerConfig
from c2o_drive.algorithms.c2osr.q_value import QValueCalculator
from c2o_drive.algorithms.c2osr.q_evaluator import QEvaluator
from c2o_drive.algorithms.c2osr.rewards import (
    RewardCalculator,
    DistanceBasedCollisionDetector,
)
from c2o_drive.algorithms.c2osr.dirichlet import (
    DirichletParams,
    SpatialDirichletBank,
    MultiTimestepSpatialDirichletBank,
    OptimizedMultiTimestepSpatialDirichletBank,
)
from c2o_drive.algorithms.c2osr.trajectory_buffer import TrajectoryBuffer
from c2o_drive.algorithms.c2osr.grid_mapper import GridMapper, GridSpec

__all__ = [
    # Main components
    "C2OSRPlanner",
    "C2OSRPlannerConfig",
    "QValueCalculator",
    "QEvaluator",
    # Dirichlet components
    "DirichletParams",
    "SpatialDirichletBank",
    "MultiTimestepSpatialDirichletBank",
    "OptimizedMultiTimestepSpatialDirichletBank",
    # Supporting components
    "TrajectoryBuffer",
    "GridMapper",
    "GridSpec",
    "RewardCalculator",
    "DistanceBasedCollisionDetector",
]
