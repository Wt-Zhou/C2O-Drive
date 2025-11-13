"""Base classes for algorithm implementations.

This module provides base classes that extend core interfaces
for specific algorithm implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar

from carla_c2osr.core.planner import BasePlanner, EpisodicPlanner, Transition, UpdateMetrics
from carla_c2osr.core.evaluator import TrajectoryEvaluator


ObsType = TypeVar('ObsType')
ActType = TypeVar('ActType')
StateType = TypeVar('StateType')
TrajectoryType = TypeVar('TrajectoryType')


@dataclass
class AlgorithmConfig:
    """Base configuration for all algorithms.

    Attributes:
        seed: Random seed for reproducibility
        verbose: Whether to print debug information
        name: Algorithm name for identification
    """
    seed: int = 0
    verbose: bool = False
    name: str = "BaseAlgorithm"


@dataclass
class PlannerConfig(AlgorithmConfig):
    """Base configuration for planner algorithms.

    Attributes:
        learning_rate: Learning rate for updates
        gamma: Discount factor for future rewards
        update_frequency: How often to update the model
    """
    learning_rate: float = 1e-4
    gamma: float = 0.99
    update_frequency: int = 1


@dataclass
class EvaluatorConfig(AlgorithmConfig):
    """Base configuration for trajectory evaluators.

    Attributes:
        batch_size: Number of trajectories to evaluate in batch
        use_cache: Whether to cache evaluation results
    """
    batch_size: int = 32
    use_cache: bool = True


class Algorithm(ABC):
    """Base class for all algorithms.

    Provides common functionality like configuration management,
    random seeding, and statistics tracking.
    """

    def __init__(self, config: AlgorithmConfig):
        """Initialize algorithm with configuration.

        Args:
            config: Algorithm configuration
        """
        self.config = config
        self._step_count = 0
        self._episode_count = 0
        self._stats: Dict[str, List[float]] = {}

        # Set random seed if specified
        if config.seed > 0:
            self._set_seed(config.seed)

    def _set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility.

        Args:
            seed: Random seed value
        """
        import random
        import numpy as np

        random.seed(seed)
        np.random.seed(seed)

        # Try to set torch seed if available
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass

    def _log_stat(self, key: str, value: float) -> None:
        """Log a statistic for tracking.

        Args:
            key: Statistic name
            value: Statistic value
        """
        if key not in self._stats:
            self._stats[key] = []
        self._stats[key].append(value)

    def get_stats(self) -> Dict[str, List[float]]:
        """Get all logged statistics.

        Returns:
            Dictionary of statistic name to list of values
        """
        return self._stats.copy()

    def clear_stats(self) -> None:
        """Clear all logged statistics."""
        self._stats.clear()

    @abstractmethod
    def save(self, path: str) -> None:
        """Save algorithm state to disk.

        Args:
            path: Path to save the algorithm state
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load algorithm state from disk.

        Args:
            path: Path to load the algorithm state from
        """
        pass


class BaseAlgorithmPlanner(Algorithm, BasePlanner[ObsType, ActType], Generic[ObsType, ActType]):
    """Base class for planner algorithms.

    Combines Algorithm and BasePlanner interfaces.
    """

    def __init__(self, config: PlannerConfig):
        """Initialize planner algorithm.

        Args:
            config: Planner configuration
        """
        Algorithm.__init__(self, config)
        self.planner_config = config

    @abstractmethod
    def select_action(
        self,
        observation: ObsType,
        deterministic: bool = False,
        **kwargs
    ) -> ActType:
        """Select action based on observation.

        Args:
            observation: Current observation
            deterministic: Whether to use deterministic policy
            **kwargs: Additional algorithm-specific parameters

        Returns:
            Selected action
        """
        pass

    @abstractmethod
    def update(self, transition: Transition[ObsType, ActType]) -> UpdateMetrics:
        """Update algorithm with a transition.

        Args:
            transition: State transition to learn from

        Returns:
            Metrics from the update
        """
        pass

    def reset(self) -> None:
        """Reset planner state for new episode."""
        self._episode_count += 1


class EpisodicAlgorithmPlanner(
    BaseAlgorithmPlanner[ObsType, ActType],
    EpisodicPlanner[ObsType, ActType],
    Generic[ObsType, ActType]
):
    """Base class for episodic planner algorithms.

    Extends BaseAlgorithmPlanner with trajectory planning capability.
    """

    @abstractmethod
    def plan_trajectory(
        self,
        observation: ObsType,
        horizon: int,
        **kwargs
    ) -> List[ActType]:
        """Plan a trajectory of actions.

        Args:
            observation: Current observation
            horizon: Planning horizon (number of steps)
            **kwargs: Additional algorithm-specific parameters

        Returns:
            Planned trajectory as list of actions
        """
        pass


class BaseAlgorithmEvaluator(
    Algorithm,
    TrajectoryEvaluator[StateType, TrajectoryType],
    Generic[StateType, TrajectoryType]
):
    """Base class for trajectory evaluator algorithms.

    Combines Algorithm and TrajectoryEvaluator interfaces.
    """

    def __init__(self, config: EvaluatorConfig):
        """Initialize evaluator algorithm.

        Args:
            config: Evaluator configuration
        """
        Algorithm.__init__(self, config)
        self.evaluator_config = config
        self._cache: Dict[int, Any] = {} if config.use_cache else None

    @abstractmethod
    def evaluate(
        self,
        trajectory: TrajectoryType,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate a single trajectory.

        Args:
            trajectory: Trajectory to evaluate
            context: Evaluation context information

        Returns:
            Evaluation results
        """
        pass

    def evaluate_batch(
        self,
        trajectories: List[TrajectoryType],
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Evaluate multiple trajectories in batch.

        Args:
            trajectories: List of trajectories to evaluate
            context: Evaluation context information

        Returns:
            List of evaluation results
        """
        # Default implementation: evaluate one by one
        # Subclasses can override for true batch processing
        return [self.evaluate(traj, context) for traj in trajectories]

    def clear_cache(self) -> None:
        """Clear evaluation cache if enabled."""
        if self._cache is not None:
            self._cache.clear()


__all__ = [
    'AlgorithmConfig',
    'PlannerConfig',
    'EvaluatorConfig',
    'Algorithm',
    'BaseAlgorithmPlanner',
    'EpisodicAlgorithmPlanner',
    'BaseAlgorithmEvaluator',
]
