"""Core interfaces for the C2O-Drive framework.

This module provides the fundamental abstractions that enable
algorithm and environment modularity:

- Environment: Gym-style driving environment interface
- Planner: Unified interface for all planning algorithms
- Evaluator: Trajectory evaluation interface
- StateSpace: State discretization and representation

All concrete implementations should implement these interfaces
to be compatible with the unified training framework.
"""

# Environment interfaces
from carla_c2osr.core.environment import (
    DrivingEnvironment,
    RewardFunction,
    CompositeRewardFunction,
    StepResult,
    Space,
    Box,
    Discrete,
)

# Planner interfaces
from carla_c2osr.core.planner import (
    BasePlanner,
    EpisodicPlanner,
    Transition,
    UpdateMetrics,
    PlannerFactory,
    register_planner,
    create_planner,
    list_planners,
)

# Evaluator interfaces
from carla_c2osr.core.evaluator import (
    TrajectoryEvaluator,
    HeuristicEvaluator,
    LearnedEvaluator,
    EvaluationContext,
    EvaluationResult,
)

# State space interfaces
from carla_c2osr.core.state_space import (
    StateSpaceDiscretizer,
    ReachabilityComputer,
    GridBasedDiscretizer,
    FeatureBasedDiscretizer,
    PathCoordinateDiscretizer,
    DiscreteState,
)

__all__ = [
    # Environment
    "DrivingEnvironment",
    "RewardFunction",
    "CompositeRewardFunction",
    "StepResult",
    "Space",
    "Box",
    "Discrete",
    # Planner
    "BasePlanner",
    "EpisodicPlanner",
    "Transition",
    "UpdateMetrics",
    "PlannerFactory",
    "register_planner",
    "create_planner",
    "list_planners",
    # Evaluator
    "TrajectoryEvaluator",
    "HeuristicEvaluator",
    "LearnedEvaluator",
    "EvaluationContext",
    "EvaluationResult",
    # State space
    "StateSpaceDiscretizer",
    "ReachabilityComputer",
    "GridBasedDiscretizer",
    "FeatureBasedDiscretizer",
    "PathCoordinateDiscretizer",
    "DiscreteState",
]
