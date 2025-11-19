"""Trajectory evaluator interface.

This module defines interfaces for evaluating trajectory quality,
which is a key component in trajectory-based planning algorithms.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, TypeVar, Generic, Tuple
from dataclasses import dataclass


StateType = TypeVar("StateType")
TrajectoryType = TypeVar("TrajectoryType")


@dataclass
class EvaluationContext:
    """Context information for trajectory evaluation.

    Attributes:
        current_state: Current state of the environment
        reference_path: Reference path/route to follow (optional)
        other_agents: Information about other agents (optional)
        horizon: Planning horizon
        dt: Timestep duration
        custom: Additional context information
    """
    current_state: Any
    reference_path: Any | None = None
    other_agents: List[Any] | None = None
    horizon: int = 10
    dt: float = 0.1
    custom: Dict[str, Any] | None = None


@dataclass
class EvaluationResult:
    """Result of trajectory evaluation.

    Attributes:
        q_value: Quality/Q-value of the trajectory
        reward_breakdown: Breakdown of reward components
        collision_probability: Estimated collision probability
        safety_score: Safety score
        comfort_score: Comfort score (low jerk, smooth)
        efficiency_score: Efficiency score (speed, progress)
        is_valid: Whether trajectory is valid (kinematically feasible, etc.)
        info: Additional evaluation information
    """
    q_value: float
    reward_breakdown: Dict[str, float] | None = None
    collision_probability: float | None = None
    safety_score: float | None = None
    comfort_score: float | None = None
    efficiency_score: float | None = None
    is_valid: bool = True
    info: Dict[str, Any] | None = None


class TrajectoryEvaluator(ABC, Generic[StateType, TrajectoryType]):
    """Base interface for trajectory evaluation.

    Trajectory evaluators assign quality scores to candidate trajectories.
    Different algorithms use different evaluation methods:
    - C2OSR: Dirichlet-based Q-value with uncertainty modeling
    - DQN/SAC: Neural network Q-function
    - Classical: Weighted sum of heuristic costs
    """

    @abstractmethod
    def evaluate(self,
                 trajectory: TrajectoryType,
                 context: EvaluationContext) -> EvaluationResult:
        """Evaluate a single trajectory.

        Args:
            trajectory: The trajectory to evaluate
            context: Evaluation context (state, reference path, etc.)

        Returns:
            result: Evaluation result with Q-value and metrics
        """
        pass

    def evaluate_batch(self,
                       trajectories: List[TrajectoryType],
                       context: EvaluationContext) -> List[EvaluationResult]:
        """Evaluate a batch of trajectories.

        Default implementation calls evaluate() for each trajectory.
        Subclasses can override for more efficient batch evaluation.

        Args:
            trajectories: List of trajectories to evaluate
            context: Shared evaluation context

        Returns:
            results: List of evaluation results
        """
        return [self.evaluate(traj, context) for traj in trajectories]

    def select_best(self,
                    trajectories: List[TrajectoryType],
                    context: EvaluationContext,
                    criterion: str = "q_value") -> Tuple[TrajectoryType, EvaluationResult]:
        """Evaluate trajectories and select the best one.

        Args:
            trajectories: List of candidate trajectories
            context: Evaluation context
            criterion: Selection criterion ("q_value", "safety", "comfort", etc.)

        Returns:
            best_trajectory: The best trajectory
            best_result: Evaluation result of the best trajectory
        """
        results = self.evaluate_batch(trajectories, context)

        # Filter out invalid trajectories
        valid_pairs = [(traj, res) for traj, res in zip(trajectories, results) if res.is_valid]

        if not valid_pairs:
            raise ValueError("No valid trajectories found")

        # Select based on criterion
        if criterion == "q_value":
            best_traj, best_res = max(valid_pairs, key=lambda pair: pair[1].q_value)
        elif criterion == "safety":
            if any(res.safety_score is None for _, res in valid_pairs):
                raise ValueError("Safety score not available for all trajectories")
            best_traj, best_res = max(valid_pairs, key=lambda pair: pair[1].safety_score)
        elif criterion == "comfort":
            if any(res.comfort_score is None for _, res in valid_pairs):
                raise ValueError("Comfort score not available for all trajectories")
            best_traj, best_res = max(valid_pairs, key=lambda pair: pair[1].comfort_score)
        else:
            raise ValueError(f"Unknown criterion: {criterion}")

        return best_traj, best_res


class HeuristicEvaluator(TrajectoryEvaluator[StateType, TrajectoryType]):
    """Simple heuristic-based evaluator.

    Evaluates trajectories using hand-crafted cost functions without learning.
    Useful as a baseline or for debugging.
    """

    def __init__(self,
                 safety_weight: float = 1.0,
                 comfort_weight: float = 1.0,
                 efficiency_weight: float = 1.0):
        """
        Args:
            safety_weight: Weight for safety cost
            comfort_weight: Weight for comfort cost
            efficiency_weight: Weight for efficiency cost
        """
        self.safety_weight = safety_weight
        self.comfort_weight = comfort_weight
        self.efficiency_weight = efficiency_weight

    @abstractmethod
    def compute_safety_cost(self, trajectory: TrajectoryType, context: EvaluationContext) -> float:
        """Compute safety cost (lower is better)."""
        pass

    @abstractmethod
    def compute_comfort_cost(self, trajectory: TrajectoryType, context: EvaluationContext) -> float:
        """Compute comfort cost (lower is better)."""
        pass

    @abstractmethod
    def compute_efficiency_cost(self, trajectory: TrajectoryType, context: EvaluationContext) -> float:
        """Compute efficiency cost (lower is better)."""
        pass

    def evaluate(self,
                 trajectory: TrajectoryType,
                 context: EvaluationContext) -> EvaluationResult:
        """Evaluate by combining weighted costs."""
        safety_cost = self.compute_safety_cost(trajectory, context)
        comfort_cost = self.compute_comfort_cost(trajectory, context)
        efficiency_cost = self.compute_efficiency_cost(trajectory, context)

        # Combine costs (negative because lower cost is better)
        q_value = -(self.safety_weight * safety_cost +
                   self.comfort_weight * comfort_cost +
                   self.efficiency_weight * efficiency_cost)

        return EvaluationResult(
            q_value=q_value,
            reward_breakdown={
                'safety': -safety_cost,
                'comfort': -comfort_cost,
                'efficiency': -efficiency_cost
            },
            safety_score=-safety_cost,
            comfort_score=-comfort_cost,
            efficiency_score=-efficiency_cost,
            is_valid=True
        )


class LearnedEvaluator(TrajectoryEvaluator[StateType, TrajectoryType]):
    """Base class for learned trajectory evaluators.

    Uses learned models (e.g., neural networks) to evaluate trajectories.
    Subclasses include DQN evaluator, SAC critic, etc.
    """

    @abstractmethod
    def update_from_experience(self,
                              trajectory: TrajectoryType,
                              actual_return: float,
                              context: EvaluationContext) -> Dict[str, float]:
        """Update the evaluator from experience.

        Args:
            trajectory: The trajectory that was executed
            actual_return: The actual cumulative reward received
            context: Context when trajectory was evaluated

        Returns:
            metrics: Training metrics (loss, etc.)
        """
        pass
