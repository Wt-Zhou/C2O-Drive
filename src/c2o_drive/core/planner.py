"""Unified planner interface for all decision-making algorithms.

This module defines the base interface that all planning/control algorithms
must implement, including:
- C2OSR (Dirichlet-based planning)
- DQN (Deep Q-Network)
- SAC (Soft Actor-Critic)
- Other RL or classical planning algorithms
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, TypeVar, Generic, Optional, List
from dataclasses import dataclass
from pathlib import Path


ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


@dataclass
class Transition(Generic[ObsType, ActType]):
    """A single transition in the MDP.

    Attributes:
        state: Current state/observation
        action: Action taken
        reward: Reward received
        next_state: Next state/observation
        terminated: Whether episode terminated
        truncated: Whether episode was truncated
        info: Additional information
    """
    state: ObsType
    action: ActType
    reward: float
    next_state: ObsType
    terminated: bool
    truncated: bool = False
    info: Dict[str, Any] | None = None


@dataclass
class UpdateMetrics:
    """Metrics returned after a learning update.

    Attributes:
        loss: Training loss (if applicable)
        q_value: Average Q-value (if applicable)
        policy_entropy: Policy entropy (for stochastic policies)
        custom: Additional algorithm-specific metrics
    """
    loss: float | None = None
    q_value: float | None = None
    policy_entropy: float | None = None
    custom: Dict[str, Any] | None = None


class BasePlanner(ABC, Generic[ObsType, ActType]):
    """Base interface for all planning/decision-making algorithms.

    All algorithms (C2OSR, DQN, SAC, etc.) must implement this interface
    to be compatible with the unified training framework.

    Type Parameters:
        ObsType: Type of observations (e.g., np.ndarray, WorldState)
        ActType: Type of actions (e.g., np.ndarray, Trajectory, EgoControl)
    """

    @abstractmethod
    def select_action(self, observation: ObsType,
                      deterministic: bool = False,
                      **kwargs) -> ActType:
        """Select an action given the current observation.

        Args:
            observation: Current observation from environment
            deterministic: Whether to use deterministic policy (for evaluation)
            **kwargs: Additional algorithm-specific parameters

        Returns:
            action: Action to execute in the environment
        """
        pass

    @abstractmethod
    def update(self, transition: Transition[ObsType, ActType]) -> UpdateMetrics:
        """Update the planner based on a new transition.

        This method performs one step of learning/adaptation.
        For model-free RL, this typically updates the neural network.
        For C2OSR, this updates the Dirichlet posterior.

        Args:
            transition: A transition from the environment

        Returns:
            metrics: Learning metrics (loss, Q-value, etc.)
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the planner's internal state for a new episode.

        This is called at the beginning of each episode.
        For stateless algorithms, this may be a no-op.
        For stateful algorithms (e.g., RNN-based), this resets hidden states.
        """
        pass

    def save_checkpoint(self, path: str | Path) -> None:
        """Save planner state to disk.

        Args:
            path: Path to save checkpoint

        Raises:
            NotImplementedError: If checkpoint saving is not implemented
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement checkpointing")

    def load_checkpoint(self, path: str | Path) -> None:
        """Load planner state from disk.

        Args:
            path: Path to load checkpoint from

        Raises:
            NotImplementedError: If checkpoint loading is not implemented
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement checkpointing")

    def set_training_mode(self, mode: bool) -> None:
        """Set training or evaluation mode.

        Args:
            mode: True for training mode, False for evaluation mode

        This affects behavior like exploration noise, batch norm, dropout, etc.
        """
        pass


class EpisodicPlanner(BasePlanner[ObsType, ActType]):
    """Base class for planners that plan over an entire episode.

    Some algorithms (like trajectory optimization) plan over a full horizon
    rather than selecting single actions. This class provides a more suitable
    interface for such algorithms.
    """

    @abstractmethod
    def plan_trajectory(self, observation: ObsType,
                        horizon: int,
                        **kwargs) -> List[ActType]:
        """Plan a trajectory of actions over a horizon.

        Args:
            observation: Current observation
            horizon: Planning horizon (number of steps)
            **kwargs: Algorithm-specific parameters

        Returns:
            trajectory: List of actions for the entire horizon
        """
        pass

    def select_action(self, observation: ObsType,
                      deterministic: bool = False,
                      **kwargs) -> ActType:
        """Select action by planning and returning the first action."""
        horizon = kwargs.get('horizon', 10)
        trajectory = self.plan_trajectory(observation, horizon, **kwargs)
        return trajectory[0] if trajectory else None


class PlannerFactory:
    """Factory for creating planners by name.

    Example:
        factory = PlannerFactory()
        factory.register('dqn', DQNPlanner)
        planner = factory.create('dqn', env=env, config=config)
    """

    def __init__(self):
        self._registry: Dict[str, type] = {}

    def register(self, name: str, planner_class: type[BasePlanner]) -> None:
        """Register a planner class.

        Args:
            name: Name to register under (e.g., 'dqn', 'sac', 'c2osr')
            planner_class: The planner class to register
        """
        self._registry[name] = planner_class

    def create(self, name: str, **kwargs) -> BasePlanner:
        """Create a planner instance by name.

        Args:
            name: Registered planner name
            **kwargs: Arguments to pass to planner constructor

        Returns:
            planner: Instantiated planner

        Raises:
            ValueError: If planner name is not registered
        """
        if name not in self._registry:
            raise ValueError(f"Unknown planner: {name}. "
                           f"Available: {list(self._registry.keys())}")
        return self._registry[name](**kwargs)

    def list_available(self) -> List[str]:
        """List all registered planner names."""
        return list(self._registry.keys())


# Global factory instance
_factory = PlannerFactory()


def register_planner(name: str, planner_class: type[BasePlanner]) -> None:
    """Register a planner globally."""
    _factory.register(name, planner_class)


def create_planner(name: str, **kwargs) -> BasePlanner:
    """Create a planner from the global registry."""
    return _factory.create(name, **kwargs)


def list_planners() -> List[str]:
    """List all globally registered planners."""
    return _factory.list_available()
