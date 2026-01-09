"""Constraint enforcement module for RCRL algorithm.

This module applies safety constraints (hard or soft) to candidate trajectories
based on reachability set analysis.
"""

from typing import Dict, List, Optional, Union
import torch
import numpy as np

from c2o_drive.algorithms.rcrl.config import ConstraintConfig


class ConstraintEnforcer:
    """Enforces safety constraints on candidate trajectories.

    Supports two modes:
    - Hard constraints: Filter out all unsafe trajectories
    - Soft constraints: Apply safety penalties to Q-values
    """

    def __init__(self, config: ConstraintConfig):
        """Initialize constraint enforcer.

        Args:
            config: Constraint configuration
        """
        self.config = config

    def apply_constraints(
        self,
        ego_reachable_sets: List[Dict[int, List[int]]],
        agent_reachable_sets: Dict[str, Dict[int, List[int]]],
        q_values: Optional[torch.Tensor] = None,
    ) -> Union[List[int], torch.Tensor]:
        """Apply safety constraints to trajectories.

        Args:
            ego_reachable_sets: List of ego reachable sets (one per trajectory)
            agent_reachable_sets: Agent reachable sets {agent_id: {timestep: [cells]}}
            q_values: Optional Q-values tensor [n_trajectories]

        Returns:
            If hard mode: List of safe trajectory indices
            If soft mode: Adjusted Q-values tensor
        """
        n_trajectories = len(ego_reachable_sets)

        if self.config.mode == "hard":
            return self._apply_hard_constraints(
                ego_reachable_sets, agent_reachable_sets, n_trajectories
            )
        else:  # soft mode
            if q_values is None:
                raise ValueError("Q-values required for soft constraint mode")
            return self._apply_soft_constraints(
                ego_reachable_sets, agent_reachable_sets, q_values
            )

    def _apply_hard_constraints(
        self,
        ego_reachable_sets: List[Dict[int, List[int]]],
        agent_reachable_sets: Dict[str, Dict[int, List[int]]],
        n_trajectories: int,
    ) -> List[int]:
        """Apply hard constraints (filter unsafe trajectories).

        Args:
            ego_reachable_sets: Ego reachable sets per trajectory
            agent_reachable_sets: Agent reachable sets
            n_trajectories: Total number of trajectories

        Returns:
            List of safe trajectory indices
        """
        safe_indices = []

        for traj_idx in range(n_trajectories):
            if self._is_trajectory_safe(
                ego_reachable_sets[traj_idx], agent_reachable_sets
            ):
                safe_indices.append(traj_idx)

        return safe_indices

    def _apply_soft_constraints(
        self,
        ego_reachable_sets: List[Dict[int, List[int]]],
        agent_reachable_sets: Dict[str, Dict[int, List[int]]],
        q_values: torch.Tensor,
    ) -> torch.Tensor:
        """Apply soft constraints (penalize Q-values for unsafe trajectories).

        Args:
            ego_reachable_sets: Ego reachable sets per trajectory
            agent_reachable_sets: Agent reachable sets
            q_values: Original Q-values [n_trajectories]

        Returns:
            Adjusted Q-values [n_trajectories]
        """
        n_trajectories = len(ego_reachable_sets)
        penalties = torch.zeros(n_trajectories, dtype=q_values.dtype, device=q_values.device)

        for traj_idx in range(n_trajectories):
            risk_score = self._compute_trajectory_risk(
                ego_reachable_sets[traj_idx], agent_reachable_sets
            )
            penalties[traj_idx] = risk_score * self.config.soft_penalty_weight

        # Adjust Q-values: Q_new = Q_original - penalty
        adjusted_q_values = q_values - penalties

        return adjusted_q_values

    def _is_trajectory_safe(
        self,
        ego_reachable_set: Dict[int, List[int]],
        agent_reachable_sets: Dict[str, Dict[int, List[int]]],
    ) -> bool:
        """Check if a trajectory is safe (no collision risk).

        Args:
            ego_reachable_set: Ego reachable set {timestep: [cells]}
            agent_reachable_sets: All agent reachable sets

        Returns:
            True if trajectory is safe, False otherwise
        """
        for timestep, ego_cells in ego_reachable_set.items():
            for agent_id, agent_timesteps in agent_reachable_sets.items():
                if timestep in agent_timesteps:
                    agent_cells = agent_timesteps[timestep]

                    # Check for cell overlap
                    if self._check_collision(ego_cells, agent_cells):
                        return False

        return True

    def _compute_trajectory_risk(
        self,
        ego_reachable_set: Dict[int, List[int]],
        agent_reachable_sets: Dict[str, Dict[int, List[int]]],
    ) -> float:
        """Compute risk score for a trajectory.

        Risk is computed as the maximum overlap ratio across all timesteps and agents.

        Args:
            ego_reachable_set: Ego reachable set {timestep: [cells]}
            agent_reachable_sets: All agent reachable sets

        Returns:
            Risk score in [0, 1], higher means more dangerous
        """
        max_risk = 0.0

        for timestep, ego_cells in ego_reachable_set.items():
            for agent_id, agent_timesteps in agent_reachable_sets.items():
                if timestep in agent_timesteps:
                    agent_cells = agent_timesteps[timestep]

                    # Compute overlap ratio
                    overlap_ratio = self._compute_overlap_ratio(ego_cells, agent_cells)

                    # Temporal discount: earlier collisions are more critical
                    time_weight = 1.0 / (1.0 + 0.1 * timestep)
                    weighted_risk = overlap_ratio * time_weight

                    max_risk = max(max_risk, weighted_risk)

        return max_risk

    def _check_collision(self, cells_a: List[int], cells_b: List[int]) -> bool:
        """Check if two cell sets have any overlap (collision).

        Args:
            cells_a: First set of cell indices
            cells_b: Second set of cell indices

        Returns:
            True if collision detected, False otherwise
        """
        if not cells_a or not cells_b:
            return False

        set_a = set(cells_a)
        set_b = set(cells_b)

        return len(set_a & set_b) > 0

    def _compute_overlap_ratio(self, cells_a: List[int], cells_b: List[int]) -> float:
        """Compute overlap ratio between two cell sets.

        Args:
            cells_a: First set of cell indices
            cells_b: Second set of cell indices

        Returns:
            Overlap ratio in [0, 1]
        """
        if not cells_a or not cells_b:
            return 0.0

        set_a = set(cells_a)
        set_b = set(cells_b)

        intersection = set_a & set_b
        union = set_a | set_b

        if len(union) == 0:
            return 0.0

        return len(intersection) / len(union)


__all__ = ["ConstraintEnforcer"]
