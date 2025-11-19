"""Lightweight scenario generator for virtual replay environments."""

from __future__ import annotations

from typing import List, Dict, Tuple
import numpy as np

from c2o_drive.core.types import AgentState, EgoState, WorldState, AgentType
from c2o_drive.algorithms.c2osr.trajectory_buffer import ScenarioState


class ScenarioManager:
    """Utility that creates deterministic toy scenarios for simulations."""

    def __init__(self, grid_size_m: float = 20.0):
        self.grid_size_m = grid_size_m
        self._reference_path: List[np.ndarray] | None = None

    def create_scenario(self) -> WorldState:
        """Create a fixed mock scenario used by examples and tests."""
        ego = EgoState(position_m=(0.0, 0.0), velocity_mps=(2.0, 0.0), yaw_rad=0.0)

        agent1 = AgentState(
            agent_id="vehicle-1",
            position_m=(4.0, 4.0),
            velocity_mps=(2.0, 0.0),
            heading_rad=0.0,
            agent_type=AgentType.VEHICLE,
        )

        # Keep a single agent to stress collision reasoning by default.
        return WorldState(time_s=0.0, ego=ego, agents=[agent1])

    def generate_reference_path(
        self,
        mode: str = "straight",
        horizon: int = 10,
        ego_start: Tuple[float, float] = (0.0, 0.0),
    ) -> List[np.ndarray]:
        """Generate a reference centerline path."""
        if self._reference_path is not None:
            return self._reference_path

        path: List[np.ndarray] = []
        x0, y0 = ego_start

        if mode == "straight":
            for i in range(horizon):
                path.append(np.array([x0 + i * 1.0, y0]))
        elif mode == "curve":
            radius = 10.0
            for i in range(horizon):
                theta = i * 0.1
                x = x0 + radius * np.sin(theta)
                y = y0 + radius * (1 - np.cos(theta))
                path.append(np.array([x, y]))
        elif mode == "s_curve":
            for i in range(horizon):
                x = x0 + i * 1.0
                y = y0 + 2.0 * np.sin(i * 0.3)
                path.append(np.array([x, y]))
        else:
            raise ValueError(f"Unknown path mode: {mode}")

        self._reference_path = path
        return path

    def get_reference_path(self) -> List[np.ndarray]:
        """Return the cached reference path, generating one if needed."""
        if self._reference_path is None:
            return self.generate_reference_path()
        return self._reference_path

    def create_scenario_state(self, world: WorldState) -> ScenarioState:
        """Convert a ``WorldState`` into the buffer friendly ``ScenarioState``."""
        agents_states = []
        for agent in world.agents:
            agents_states.append(
                (
                    agent.position_m[0],
                    agent.position_m[1],
                    agent.velocity_mps[0],
                    agent.velocity_mps[1],
                    agent.heading_rad,
                    agent.agent_type.value,
                )
            )

        return ScenarioState(
            ego_position=world.ego.position_m,
            ego_velocity=world.ego.velocity_mps,
            ego_heading=world.ego.yaw_rad,
            agents_states=agents_states,
        )

    def create_world_state_from_trajectories(
        self,
        t: int,
        ego_trajectory: List[np.ndarray],
        agent_trajectories: Dict[int, List[np.ndarray]],
        world_init: WorldState,
    ) -> WorldState:
        """Create a ``WorldState`` snapshot for time ``t`` given trajectories."""
        ego_world_xy = ego_trajectory[t]
        ego = EgoState(
            position_m=tuple(ego_world_xy),
            velocity_mps=(5.0, 0.0),
            yaw_rad=0.0,
        )

        current_agents: List[AgentState] = []

        for i, agent_init in enumerate(world_init.agents):
            agent_id = i + 1
            if agent_id in agent_trajectories and t < len(agent_trajectories[agent_id]):
                agent_world_xy = agent_trajectories[agent_id][t]

                horizon = len(agent_trajectories[agent_id])
                if t < horizon - 1:
                    nxt = agent_trajectories[agent_id][t + 1]
                    vel_vec = nxt - agent_world_xy
                elif t > 0:
                    prv = agent_trajectories[agent_id][t - 1]
                    vel_vec = agent_world_xy - prv
                else:
                    vel_vec = np.array(agent_init.velocity_mps)

                vel_tuple = (float(vel_vec[0]), float(vel_vec[1]))
                heading_est = (
                    float(np.arctan2(vel_vec[1], vel_vec[0]))
                    if (vel_vec[0] ** 2 + vel_vec[1] ** 2) > 1e-9
                    else float(agent_init.heading_rad)
                )

                current_agents.append(
                    AgentState(
                        agent_id=agent_init.agent_id,
                        position_m=tuple(agent_world_xy),
                        velocity_mps=vel_tuple,
                        heading_rad=heading_est,
                        agent_type=agent_init.agent_type,
                    )
                )
            else:
                current_agents.append(agent_init)

        return WorldState(time_s=float(t), ego=ego, agents=current_agents)
