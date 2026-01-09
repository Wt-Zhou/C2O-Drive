"""Reachability-Constrained Reinforcement Learning (RCRL) Algorithm.

This module implements a safety-aware RL baseline that integrates
reachability analysis with deep Q-learning for autonomous driving.

Key Features:
- Forward reachable set computation for ego and agents
- Configurable hard/soft safety constraints
- Online learning with DQN
- Strict adherence to C2OSR architectural patterns
"""

from c2o_drive.algorithms.rcrl.config import (
    RCRLPlannerConfig,
    ReachabilityConfig,
    ConstraintConfig,
    NetworkConfig,
    TrainingConfig,
)
from c2o_drive.algorithms.rcrl.planner import RCRLPlanner
from c2o_drive.algorithms.rcrl.factory import (
    create_rcrl_planner,
    create_hard_constraint_planner,
    create_soft_constraint_planner,
)

__all__ = [
    "RCRLPlanner",
    "RCRLPlannerConfig",
    "ReachabilityConfig",
    "ConstraintConfig",
    "NetworkConfig",
    "TrainingConfig",
    "create_rcrl_planner",
    "create_hard_constraint_planner",
    "create_soft_constraint_planner",
]

__version__ = "0.1.0"
