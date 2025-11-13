"""C2OSR algorithm implementation.

This module provides the C2OSR (Dirichlet-based) planning algorithm
adapted to work with the standard planner and evaluator interfaces.
"""

from carla_c2osr.algorithms.c2osr.config import (
    GridConfig,
    DirichletConfig,
    LatticePlannerConfig,
    QValueConfig,
    RewardWeightsConfig,
    C2OSRPlannerConfig,
    C2OSREvaluatorConfig,
)
from carla_c2osr.algorithms.c2osr.planner import C2OSRPlanner
from carla_c2osr.algorithms.c2osr.evaluator import C2OSREvaluator
from carla_c2osr.algorithms.c2osr.factory import (
    create_c2osr_planner,
    create_c2osr_evaluator,
    create_c2osr_planner_evaluator_pair,
)

__all__ = [
    # Configuration
    'GridConfig',
    'DirichletConfig',
    'LatticePlannerConfig',
    'QValueConfig',
    'RewardWeightsConfig',
    'C2OSRPlannerConfig',
    'C2OSREvaluatorConfig',
    # Core classes
    'C2OSRPlanner',
    'C2OSREvaluator',
    # Factory functions
    'create_c2osr_planner',
    'create_c2osr_evaluator',
    'create_c2osr_planner_evaluator_pair',
]
