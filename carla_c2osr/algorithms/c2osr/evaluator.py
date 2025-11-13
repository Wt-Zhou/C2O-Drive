"""C2OSR Evaluator implementation.

This module provides the C2OSREvaluator class for trajectory evaluation
using the C2OSR Q-value calculation method.
"""

from __future__ import annotations
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

from carla_c2osr.algorithms.base import BaseAlgorithmEvaluator
from carla_c2osr.algorithms.c2osr.config import C2OSREvaluatorConfig
from carla_c2osr.algorithms.c2osr.internal import (
    GridSpec, GridMapper,
    DirichletParams, SpatialDirichletBank, MultiTimestepSpatialDirichletBank,
    OptimizedMultiTimestepSpatialDirichletBank,
    TrajectoryBuffer,
    QValueCalculator, OriginalQValueConfig, RewardCalculator,
    WorldState,
    ShapeBasedCollisionDetector,
)
from carla_c2osr.core.evaluator import EvaluationContext, EvaluationResult


class C2OSREvaluator(BaseAlgorithmEvaluator[WorldState, List[Tuple[float, float]]]):
    """C2OSR trajectory evaluator.

    Evaluates trajectories using Dirichlet-based Q-value calculation
    that considers uncertainty in agent behavior.
    """

    def __init__(
        self,
        config: C2OSREvaluatorConfig,
        trajectory_buffer: Optional[TrajectoryBuffer] = None,
        dirichlet_bank: Optional[Any] = None,
    ):
        """Initialize C2OSR evaluator.

        Args:
            config: Evaluator configuration
            trajectory_buffer: Optional pre-initialized trajectory buffer
            dirichlet_bank: Optional pre-initialized Dirichlet bank
        """
        super().__init__(config)
        self.config = config

        # Initialize grid mapper
        # Calculate grid size from bounds
        x_range = config.grid.bounds_x[1] - config.grid.bounds_x[0]
        y_range = config.grid.bounds_y[1] - config.grid.bounds_y[0]
        size_m = max(x_range, y_range)

        self.grid_spec = GridSpec(
            size_m=size_m,
            cell_m=config.grid.grid_size_m,
            macro=False,
        )
        self.grid_mapper = GridMapper(self.grid_spec)

        # Use provided buffer or create new one
        self.trajectory_buffer = trajectory_buffer
        if self.trajectory_buffer is None:
            self.trajectory_buffer = TrajectoryBuffer(
                horizon=config.q_value.horizon,
            )

        # Use provided Dirichlet bank or create new one
        if dirichlet_bank is not None:
            self.dirichlet_bank = dirichlet_bank
        else:
            dirichlet_params = DirichletParams(
                alpha_in=config.dirichlet.alpha_in,
                alpha_out=config.dirichlet.alpha_out,
            )

            # Calculate K (total number of grid cells)
            K = self.grid_spec.num_cells

            if config.dirichlet.use_optimized:
                self.dirichlet_bank = OptimizedMultiTimestepSpatialDirichletBank(
                    K=K,
                    params=dirichlet_params,
                    horizon=config.q_value.horizon,
                )
            elif config.dirichlet.use_multistep:
                self.dirichlet_bank = MultiTimestepSpatialDirichletBank(
                    K=K,
                    params=dirichlet_params,
                    horizon=config.q_value.horizon,
                )
            else:
                self.dirichlet_bank = SpatialDirichletBank(
                    K=K,
                    params=dirichlet_params,
                )

        # Initialize Q-value calculator
        # Get reward config
        try:
            from carla_c2osr.config import get_global_config
            reward_config = get_global_config().reward
        except ImportError:
            # Create default
            from dataclasses import dataclass
            @dataclass
            class DefaultRewardConfig:
                collision_penalty: float = config.reward_weights.collision_penalty
                max_comfortable_accel: float = 2.0
                acceleration_penalty_weight: float = 0.1
                jerk_penalty_weight: float = 0.05
                target_speed: float = 5.0
                speed_reward_weight: float = 0.1
                progress_reward_weight: float = 0.5
                safe_distance: float = 2.0
                distance_penalty_weight: float = 1.0

            reward_config = DefaultRewardConfig()

        self.q_value_calculator = QValueCalculator(
            config=OriginalQValueConfig(
                horizon=config.q_value.horizon,
                n_samples=config.q_value.n_samples,
                q_selection_percentile=config.q_value.selection_percentile,
            ),
            reward_config=reward_config,
        )

        # Initialize reward calculator
        try:
            from carla_c2osr.config import get_global_config
            reward_config = get_global_config().reward
            self.reward_calculator = RewardCalculator(reward_config)
        except ImportError:
            # Create with default values
            from dataclasses import dataclass
            @dataclass
            class DefaultRewardConfig:
                collision_penalty: float = config.reward_weights.collision_penalty
                max_comfortable_accel: float = 2.0
                acceleration_penalty_weight: float = 0.1
                jerk_penalty_weight: float = 0.05
                target_speed: float = 5.0
                speed_reward_weight: float = 0.1
                progress_reward_weight: float = 0.5
                min_safe_distance: float = 2.0
                safety_reward_weight: float = 1.0

            self.reward_calculator = RewardCalculator(DefaultRewardConfig())

        # Initialize collision detector
        self.collision_detector = ShapeBasedCollisionDetector()

    def evaluate(
        self,
        trajectory: List[Tuple[float, float]],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate a single trajectory.

        Args:
            trajectory: Trajectory to evaluate (list of (x, y) positions)
            context: Evaluation context containing:
                - 'current_state': WorldState - current world state
                - 'dt': float - time step (optional)
                - 'return_details': bool - whether to return detailed metrics (optional)

        Returns:
            Evaluation results containing:
                - 'q_value': float - Overall Q-value
                - 'collision_free': bool - Whether trajectory is collision-free
                - 'comfort_score': float - Comfort metric (optional)
                - 'efficiency_score': float - Efficiency metric (optional)
                - 'safety_score': float - Safety metric (optional)
        """
        current_state: WorldState = context['current_state']
        dt: float = context.get('dt', 1.0)
        return_details: bool = context.get('return_details', False)

        # Calculate Q-value
        try:
            q_values_list, _ = self.q_value_calculator.compute_q_value(
                current_world_state=current_state,
                ego_action_trajectory=trajectory,
                trajectory_buffer=self.trajectory_buffer,
                grid=self.grid_mapper,
                bank=self.dirichlet_bank,
            )

            # Select Q-value based on percentile (default is 0.0 = minimum)
            if len(q_values_list) > 0:
                percentile = self.config.q_value.selection_percentile
                if percentile == 0.0:
                    q_value = float(np.min(q_values_list))
                elif percentile == 1.0:
                    q_value = float(np.max(q_values_list))
                else:
                    q_value = float(np.percentile(q_values_list, percentile * 100))
            else:
                q_value = float('-inf')
        except Exception as e:
            if self.config.verbose:
                print(f"Warning: Q-value calculation failed: {e}")
            q_value = float('-inf')

        results = {
            'q_value': q_value,
            'success': q_value > float('-inf'),
        }

        # Calculate detailed metrics if requested
        if return_details:
            # Collision check
            collision_free = self._check_collision_free(trajectory, current_state)
            results['collision_free'] = collision_free

            # Comfort score
            try:
                comfort_reward = self.reward_calculator.calculate_comfort_reward(
                    trajectory, dt
                )
                results['comfort_score'] = comfort_reward
            except Exception as e:
                if self.config.verbose:
                    print(f"Warning: Comfort calculation failed: {e}")
                results['comfort_score'] = 0.0

            # Efficiency score
            try:
                efficiency_reward = self.reward_calculator.calculate_efficiency_reward(
                    trajectory, dt
                )
                results['efficiency_score'] = efficiency_reward
            except Exception as e:
                if self.config.verbose:
                    print(f"Warning: Efficiency calculation failed: {e}")
                results['efficiency_score'] = 0.0

            # Safety score (distance to other agents)
            agent_trajectories = {}
            for i, agent in enumerate(current_state.agents):
                # Predict agent positions (simple constant velocity)
                agent_traj = [agent.position_m]
                for _ in range(len(trajectory) - 1):
                    # Assume constant velocity
                    next_pos = (
                        agent_traj[-1][0] + agent.velocity_mps[0] * dt,
                        agent_traj[-1][1] + agent.velocity_mps[1] * dt,
                    )
                    agent_traj.append(next_pos)
                agent_trajectories[i] = agent_traj

            try:
                safety_reward = self.reward_calculator.calculate_safety_reward(
                    trajectory, agent_trajectories
                )
                results['safety_score'] = safety_reward
            except Exception as e:
                if self.config.verbose:
                    print(f"Warning: Safety calculation failed: {e}")
                results['safety_score'] = 0.0

        return results

    def _check_collision_free(
        self,
        trajectory: List[Tuple[float, float]],
        current_state: WorldState
    ) -> bool:
        """Check if trajectory is collision-free.

        Args:
            trajectory: Trajectory to check
            current_state: Current world state

        Returns:
            True if trajectory is collision-free
        """
        # Simple distance-based collision check
        for ego_pos in trajectory:
            for agent in current_state.agents:
                distance = np.linalg.norm(
                    np.array(ego_pos) - np.array(agent.position_m)
                )
                if distance < 2.0:  # 2m safety threshold
                    return False
        return True

    def evaluate_batch(
        self,
        trajectories: List[List[Tuple[float, float]]],
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Evaluate multiple trajectories in batch.

        Args:
            trajectories: List of trajectories to evaluate
            context: Evaluation context

        Returns:
            List of evaluation results
        """
        # For C2OSR, batch evaluation is just sequential evaluation
        # (Q-value calculation is already vectorized internally)
        return [self.evaluate(traj, context) for traj in trajectories]

    def save(self, path: str) -> None:
        """Save evaluator state.

        Args:
            path: Path to save directory
        """
        import pickle
        from pathlib import Path

        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save configuration
        with open(save_path / 'evaluator_config.pkl', 'wb') as f:
            pickle.dump(self.config, f)

        # Save Dirichlet bank
        with open(save_path / 'evaluator_dirichlet.pkl', 'wb') as f:
            pickle.dump(self.dirichlet_bank, f)

        # Save trajectory buffer
        with open(save_path / 'evaluator_buffer.pkl', 'wb') as f:
            pickle.dump(self.trajectory_buffer, f)

    def load(self, path: str) -> None:
        """Load evaluator state.

        Args:
            path: Path to load directory
        """
        import pickle
        from pathlib import Path

        load_path = Path(path)

        # Load Dirichlet bank
        with open(load_path / 'evaluator_dirichlet.pkl', 'rb') as f:
            self.dirichlet_bank = pickle.load(f)

        # Load trajectory buffer
        with open(load_path / 'evaluator_buffer.pkl', 'rb') as f:
            self.trajectory_buffer = pickle.load(f)

        # Recreate Q-value calculator with loaded components
        self.q_value_calculator = QValueCalculator(
            grid_mapper=self.grid_mapper,
            dirichlet_bank=self.dirichlet_bank,
            trajectory_buffer=self.trajectory_buffer,
            config=OriginalQValueConfig(
                horizon=self.config.q_value.horizon,
                n_samples=self.config.q_value.n_samples,
                q_selection_percentile=self.config.q_value.selection_percentile,
            ),
        )


__all__ = ['C2OSREvaluator']
