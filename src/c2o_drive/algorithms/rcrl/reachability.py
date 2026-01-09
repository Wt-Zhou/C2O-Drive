"""Reachability computation module for RCRL algorithm.

This module computes forward reachable sets for ego vehicle and agents,
leveraging the GridMapper from C2OSR for efficiency and consistency.
"""

from typing import Dict, List, Tuple
import numpy as np

from c2o_drive.core.types import AgentState, EgoState, AgentDynamicsParams
from c2o_drive.algorithms.c2osr.grid_mapper import GridMapper, GridSpec
from c2o_drive.algorithms.rcrl.config import ReachabilityConfig


class ReachabilityModule:
    """Computes forward reachable sets using grid-based approximation.

    Directly reuses C2OSR's GridMapper for agent reachability computation,
    ensuring consistency and leveraging proven implementations.
    """

    def __init__(self, config: ReachabilityConfig, world_center: Tuple[float, float] = (0.0, 0.0)):
        """Initialize reachability module.

        Args:
            config: Reachability configuration
            world_center: Center of the world grid coordinate system
        """
        self.config = config

        # Create GridMapper using config parameters
        grid_spec = GridSpec(
            size_m=config.grid_size, cell_m=config.grid_cell_size, macro=False
        )
        self.grid_mapper = GridMapper(spec=grid_spec, world_center=world_center)

    def compute_ego_reachable_set(
        self, ego_state: EgoState, trajectory: List[Tuple[float, float]]
    ) -> Dict[int, List[int]]:
        """Compute ego vehicle reachable set along planned trajectory.

        For each waypoint in the trajectory, computes the set of cells
        that the ego vehicle might occupy, accounting for control errors.

        Args:
            ego_state: Current ego vehicle state
            trajectory: Planned trajectory waypoints [(x, y), ...]

        Returns:
            Dictionary mapping timestep to list of reachable cell indices
        """
        if not self.config.use_ego_reachability:
            return {}

        reachable_sets = {}
        expand_radius = self.config.ego_expand_radius

        for timestep in range(min(len(trajectory), self.config.horizon)):
            waypoint = trajectory[timestep]

            # Convert waypoint to grid cell
            grid_xy = self.grid_mapper.to_grid_frame(waypoint)
            center_idx = self.grid_mapper.xy_to_index(grid_xy)

            # Expand neighborhood to account for control errors
            reachable_cells = self.grid_mapper.get_neighbors(
                center_idx, radius=expand_radius
            )

            reachable_sets[timestep] = reachable_cells

        return reachable_sets

    def compute_agent_reachable_sets(
        self, agents: List[AgentState]
    ) -> Dict[str, Dict[int, List[int]]]:
        """Compute reachable sets for all agents.

        Directly uses GridMapper.multi_timestep_successor_cells() for each agent,
        leveraging C2OSR's proven reachability computation with agent dynamics.

        Args:
            agents: List of agent states

        Returns:
            Dictionary mapping agent_id to {timestep: [reachable_cell_ids]}
        """
        if not self.config.use_agent_reachability:
            return {}

        agent_reachable_sets = {}

        for agent in agents:
            # Use GridMapper's multi-timestep successor computation
            # This method automatically handles different agent types
            # (pedestrian, vehicle, bicycle, motorcycle) with appropriate dynamics
            reachable_sets = self.grid_mapper.multi_timestep_successor_cells(
                agent=agent,
                horizon=self.config.horizon,
                dt=self.config.dt,
                use_numba=True,  # Use Numba acceleration if available
            )

            agent_reachable_sets[agent.agent_id] = reachable_sets

        return agent_reachable_sets

    def check_overlap(
        self, cells_a: List[int], cells_b: List[int]
    ) -> bool:
        """Check if two cell sets overlap.

        Args:
            cells_a: First set of cell indices
            cells_b: Second set of cell indices

        Returns:
            True if sets have any overlap, False otherwise
        """
        set_a = set(cells_a)
        set_b = set(cells_b)
        return len(set_a & set_b) > 0

    def compute_overlap_score(
        self, cells_a: List[int], cells_b: List[int]
    ) -> float:
        """Compute overlap score between two cell sets.

        Args:
            cells_a: First set of cell indices
            cells_b: Second set of cell indices

        Returns:
            Overlap score in [0, 1], where 1 means complete overlap
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


__all__ = ["ReachabilityModule"]
