"""Driving environment interface following Gymnasium standard.

This module defines the base interface for all driving simulation environments.
It follows the Gymnasium (formerly OpenAI Gym) API for compatibility with
existing RL algorithms and tools.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, TypeVar, Generic
from dataclasses import dataclass
import numpy as np


# Type variables for flexibility
ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


@dataclass
class StepResult(Generic[ObsType]):
    """Result of a single environment step.

    Attributes:
        observation: The observation after taking the action
        reward: The reward received
        terminated: Whether the episode ended (e.g., collision, goal reached)
        truncated: Whether the episode was truncated (e.g., time limit)
        info: Additional information dictionary
    """
    observation: ObsType
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any]


class Space(ABC):
    """Abstract base class for action/observation spaces."""

    @abstractmethod
    def sample(self) -> Any:
        """Randomly sample an element from this space."""
        pass

    @abstractmethod
    def contains(self, x: Any) -> bool:
        """Check if x is a valid member of this space."""
        pass


class Box(Space):
    """Continuous space defined by bounds.

    Example: Box(low=-1.0, high=1.0, shape=(2,)) for 2D continuous control
    """

    def __init__(self, low: float | np.ndarray, high: float | np.ndarray,
                 shape: Tuple[int, ...] | None = None, dtype: type = np.float32):
        if isinstance(low, (int, float)):
            if shape is None:
                raise ValueError("shape must be provided if low is scalar")
            self.low = np.full(shape, low, dtype=dtype)
            self.high = np.full(shape, high, dtype=dtype)
        else:
            self.low = np.array(low, dtype=dtype)
            self.high = np.array(high, dtype=dtype)
        self.shape = self.low.shape
        self.dtype = dtype

    def sample(self) -> np.ndarray:
        return np.random.uniform(self.low, self.high, self.shape).astype(self.dtype)

    def contains(self, x: Any) -> bool:
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        return bool(np.all(x >= self.low) and np.all(x <= self.high))


class Discrete(Space):
    """Discrete space with n possible values {0, 1, ..., n-1}.

    Example: Discrete(5) for choosing from 5 discrete actions
    """

    def __init__(self, n: int):
        self.n = n

    def sample(self) -> int:
        return int(np.random.randint(self.n))

    def contains(self, x: Any) -> bool:
        return isinstance(x, (int, np.integer)) and 0 <= x < self.n


class DrivingEnvironment(ABC, Generic[ObsType, ActType]):
    """Base class for all driving simulation environments.

    This interface follows the Gymnasium standard:
    - reset() initializes a new episode
    - step() advances the simulation by one timestep
    - Supports both continuous and discrete action spaces

    Type Parameters:
        ObsType: Type of observations (e.g., np.ndarray, Dict, WorldState)
        ActType: Type of actions (e.g., np.ndarray, Trajectory, EgoControl)
    """

    @abstractmethod
    def reset(self, seed: int | None = None, options: Dict[str, Any] | None = None) -> Tuple[ObsType, Dict[str, Any]]:
        """Reset the environment to an initial state.

        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset (e.g., specific scenario)

        Returns:
            observation: Initial observation
            info: Additional information dictionary
        """
        pass

    @abstractmethod
    def step(self, action: ActType) -> StepResult[ObsType]:
        """Execute one timestep of the environment.

        Args:
            action: Action to execute

        Returns:
            StepResult containing (observation, reward, terminated, truncated, info)
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Clean up resources (e.g., close CARLA connection)."""
        pass

    @property
    @abstractmethod
    def observation_space(self) -> Space:
        """The observation space specification."""
        pass

    @property
    @abstractmethod
    def action_space(self) -> Space:
        """The action space specification."""
        pass

    def render(self) -> None:
        """Render the environment (optional, for visualization)."""
        pass

    def seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        np.random.seed(seed)


class RewardFunction(ABC):
    """Abstract base class for reward functions.

    Allows modular composition of different reward components.
    """

    @abstractmethod
    def compute(self, state: Any, action: Any, next_state: Any, info: Dict[str, Any]) -> float:
        """Compute reward for a transition.

        Args:
            state: Current state
            action: Action taken
            next_state: Next state after action
            info: Additional information

        Returns:
            reward: Scalar reward value
        """
        pass


class CompositeRewardFunction(RewardFunction):
    """Composes multiple reward components with weights.

    Example:
        reward = CompositeRewardFunction([
            (comfort_reward, 1.0),
            (efficiency_reward, 2.0),
            (safety_reward, 10.0)
        ])
    """

    def __init__(self, components: list[Tuple[RewardFunction, float]]):
        """
        Args:
            components: List of (reward_function, weight) tuples
        """
        self.components = components

    def compute(self, state: Any, action: Any, next_state: Any, info: Dict[str, Any]) -> float:
        total_reward = 0.0
        for reward_fn, weight in self.components:
            total_reward += weight * reward_fn.compute(state, action, next_state, info)
        return total_reward
