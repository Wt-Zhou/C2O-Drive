"""Simple grid-based driving environment.

This module provides a Gym-compatible wrapper around the existing
ScenarioManager for the simple 2D grid environment.
"""

from __future__ import annotations
from typing import Dict, Any, Tuple, List
import numpy as np

from carla_c2osr.core.environment import (
    DrivingEnvironment,
    StepResult,
    Box,
    Space,
    RewardFunction,
)
from carla_c2osr.env.types import WorldState, EgoState, EgoControl, AgentState
from carla_c2osr.env.scenario_manager import ScenarioManager
from carla_c2osr.environments.rewards import create_default_reward


class SimpleGridEnvironment(DrivingEnvironment[WorldState, EgoControl]):
    """Simple 2D grid environment for driving simulation.

    This environment wraps the existing ScenarioManager and provides
    a Gym-compatible interface. It uses WorldState as observations
    and EgoControl as actions.

    Observation: WorldState (ego + agents states)
    Action: EgoControl (throttle, steer, brake)
    """

    def __init__(self,
                 grid_size_m: float = 20.0,
                 dt: float = 0.1,
                 max_episode_steps: int = 100,
                 reward_fn: RewardFunction | None = None,
                 scenario_manager: ScenarioManager | None = None):
        """
        Args:
            grid_size_m: Size of the grid in meters
            dt: Timestep duration in seconds
            max_episode_steps: Maximum steps per episode
            reward_fn: Custom reward function (uses default if None)
            scenario_manager: Custom scenario manager (creates new if None)
        """
        self.grid_size_m = grid_size_m
        self.dt = dt
        self.max_episode_steps = max_episode_steps
        self.reward_fn = reward_fn or create_default_reward()

        # Create or use provided scenario manager
        self.scenario_manager = scenario_manager or ScenarioManager(grid_size_m)

        # Episode state
        self._current_state: WorldState | None = None
        self._step_count: int = 0
        self._reference_path: List[np.ndarray] | None = None
        self._episode_trajectory: List[EgoState] = []

        # Define action and observation spaces
        self._action_space = Box(
            low=np.array([-1.0, -1.0, 0.0]),  # [throttle, steer, brake]
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )

        # Observation space is more complex (WorldState), but we define
        # a simplified continuous space for ego position and velocity
        self._observation_space = Box(
            low=np.array([-50.0, -50.0, -20.0, -20.0, -np.pi]),
            high=np.array([50.0, 50.0, 20.0, 20.0, np.pi]),
            dtype=np.float32
        )

    @property
    def observation_space(self) -> Space:
        """Observation space (simplified for ego state)."""
        return self._observation_space

    @property
    def action_space(self) -> Space:
        """Action space (throttle, steer, brake)."""
        return self._action_space

    def reset(self,
              seed: int | None = None,
              options: Dict[str, Any] | None = None) -> Tuple[WorldState, Dict[str, Any]]:
        """Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility
            options: Additional options (e.g., 'path_mode' for reference path)

        Returns:
            initial_state: Initial WorldState
            info: Additional information dictionary
        """
        if seed is not None:
            self.seed(seed)

        # Reset episode state
        self._step_count = 0
        self._episode_trajectory = []

        # Create initial scenario
        self._current_state = self.scenario_manager.create_scenario()

        # Generate reference path
        path_mode = options.get('path_mode', 'straight') if options else 'straight'
        self._reference_path = self.scenario_manager.generate_reference_path(
            mode=path_mode,
            horizon=self.max_episode_steps,
            ego_start=self._current_state.ego.position_m
        )

        # Record initial state
        self._episode_trajectory.append(self._current_state.ego)

        info = {
            'step': self._step_count,
            'reference_path': self._reference_path,
        }

        return self._current_state, info

    def step(self, action: EgoControl) -> StepResult[WorldState]:
        """Execute one timestep in the environment.

        Args:
            action: Control action to execute

        Returns:
            StepResult with next observation, reward, termination flags, and info
        """
        if self._current_state is None:
            raise RuntimeError("Must call reset() before step()")

        # Store previous state for reward computation
        prev_state = self._current_state

        # Update ego state based on action
        next_ego_state = self._update_ego_state(
            self._current_state.ego,
            action,
            self.dt
        )

        # Update agent states (simple constant velocity model for now)
        next_agents = self._update_agent_states(
            self._current_state.agents,
            self.dt
        )

        # Create next world state
        next_state = WorldState(
            time_s=self._current_state.time_s + self.dt,
            ego=next_ego_state,
            agents=next_agents
        )

        self._current_state = next_state
        self._step_count += 1
        self._episode_trajectory.append(next_ego_state)

        # Check termination conditions
        collision = self._check_collision(next_ego_state, next_agents)
        out_of_bounds = self._check_out_of_bounds(next_ego_state)
        time_limit = self._step_count >= self.max_episode_steps

        terminated = collision or out_of_bounds
        truncated = time_limit and not terminated

        # Compute reward
        info = self._compute_info(prev_state, next_state, action, collision)
        reward = self.reward_fn.compute(prev_state, action, next_state, info)

        return StepResult(
            observation=next_state,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info
        )

    def _update_ego_state(self,
                          ego: EgoState,
                          control: EgoControl,
                          dt: float) -> EgoState:
        """Update ego vehicle state using simple kinematic bicycle model.

        Args:
            ego: Current ego state
            control: Control input
            dt: Timestep duration

        Returns:
            Updated ego state
        """
        # Extract current state
        x, y = ego.position_m
        vx, vy = ego.velocity_mps
        yaw = ego.yaw_rad

        # Compute speed and heading
        speed = np.sqrt(vx**2 + vy**2)

        # Apply control
        # throttle: [-1, 1] -> acceleration
        # brake: [0, 1] -> deceleration
        # steer: [-1, 1] -> yaw rate
        max_accel = 3.0  # m/s^2
        max_decel = 6.0  # m/s^2
        max_yaw_rate = 0.5  # rad/s

        if control.throttle > 0:
            accel = control.throttle * max_accel
        else:
            accel = control.throttle * max_decel

        if control.brake > 0:
            accel -= control.brake * max_decel

        yaw_rate = control.steer * max_yaw_rate

        # Update state using simple integration
        speed_next = max(0, speed + accel * dt)
        yaw_next = yaw + yaw_rate * dt

        # Update position
        x_next = x + speed_next * np.cos(yaw_next) * dt
        y_next = y + speed_next * np.sin(yaw_next) * dt

        # Update velocity
        vx_next = speed_next * np.cos(yaw_next)
        vy_next = speed_next * np.sin(yaw_next)

        return EgoState(
            position_m=(x_next, y_next),
            velocity_mps=(vx_next, vy_next),
            yaw_rad=yaw_next
        )

    def _update_agent_states(self,
                             agents: List[AgentState],
                             dt: float) -> List[AgentState]:
        """Update agent states using constant velocity model.

        Args:
            agents: List of current agent states
            dt: Timestep duration

        Returns:
            List of updated agent states
        """
        updated_agents = []
        for agent in agents:
            x, y = agent.position_m
            vx, vy = agent.velocity_mps

            # Simple constant velocity update
            x_next = x + vx * dt
            y_next = y + vy * dt

            updated_agents.append(AgentState(
                agent_id=agent.agent_id,
                position_m=(x_next, y_next),
                velocity_mps=(vx, vy),
                heading_rad=agent.heading_rad,
                agent_type=agent.agent_type
            ))

        return updated_agents

    def _check_collision(self,
                         ego: EgoState,
                         agents: List[AgentState]) -> bool:
        """Check if ego collides with any agent.

        Args:
            ego: Ego state
            agents: List of agent states

        Returns:
            True if collision detected
        """
        ego_pos = np.array(ego.position_m)
        collision_threshold = 2.0  # meters

        for agent in agents:
            agent_pos = np.array(agent.position_m)
            distance = np.linalg.norm(ego_pos - agent_pos)
            if distance < collision_threshold:
                return True

        return False

    def _check_out_of_bounds(self, ego: EgoState) -> bool:
        """Check if ego is out of bounds.

        Args:
            ego: Ego state

        Returns:
            True if out of bounds
        """
        x, y = ego.position_m
        bound = self.grid_size_m / 2.0
        return abs(x) > bound or abs(y) > bound

    def _compute_info(self,
                      prev_state: WorldState,
                      next_state: WorldState,
                      action: EgoControl,
                      collision: bool) -> Dict[str, Any]:
        """Compute additional information for this step.

        Args:
            prev_state: Previous world state
            next_state: Next world state
            action: Action taken
            collision: Whether collision occurred

        Returns:
            Info dictionary
        """
        # Compute acceleration
        prev_speed = np.linalg.norm(np.array(prev_state.ego.velocity_mps))
        next_speed = np.linalg.norm(np.array(next_state.ego.velocity_mps))
        acceleration = (next_speed - prev_speed) / self.dt

        # Get reference path y-coordinate at current x
        reference_y = 0.0
        if self._reference_path:
            # Find closest reference point
            ego_x = next_state.ego.position_m[0]
            for ref_point in self._reference_path:
                if ref_point[0] >= ego_x:
                    reference_y = ref_point[1]
                    break

        return {
            'step': self._step_count,
            'collision': collision,
            'out_of_bounds': self._check_out_of_bounds(next_state.ego),
            'acceleration': acceleration,
            'jerk': 0.0,  # Placeholder
            'reference_y': reference_y,
            'ego_position': next_state.ego.position_m,
            'ego_velocity': next_state.ego.velocity_mps,
            'num_agents': len(next_state.agents),
        }

    def close(self) -> None:
        """Clean up resources."""
        self._current_state = None
        self._reference_path = None
        self._episode_trajectory = []

    def render(self) -> None:
        """Render the environment (optional visualization)."""
        if self._current_state is None:
            return

        print(f"\nStep {self._step_count}:")
        print(f"  Ego: pos={self._current_state.ego.position_m}, "
              f"vel={self._current_state.ego.velocity_mps}")
        for i, agent in enumerate(self._current_state.agents):
            print(f"  Agent {i}: pos={agent.position_m}, vel={agent.velocity_mps}")

    def get_episode_trajectory(self) -> List[EgoState]:
        """Get the full ego trajectory for the current episode.

        Returns:
            List of ego states
        """
        return self._episode_trajectory.copy()
