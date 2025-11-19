"""C2OSR Planner implementation.

This module provides the C2OSRPlanner class that adapts the existing
C2OSR algorithm to the standard planner interface.
"""

from __future__ import annotations
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from c2o_drive.algorithms.base import EpisodicAlgorithmPlanner
from c2o_drive.algorithms.c2osr.config import C2OSRPlannerConfig
from c2o_drive.algorithms.c2osr.internal import (
    GridSpec, GridMapper,
    DirichletParams, SpatialDirichletBank, MultiTimestepSpatialDirichletBank,
    OptimizedMultiTimestepSpatialDirichletBank,
    TrajectoryBuffer, AgentTrajectoryData, ScenarioState,
    QValueCalculator, OriginalQValueConfig, RewardCalculator,
    LatticePlanner, LatticeTrajectory,
    EgoState, WorldState, EgoControl, AgentType,
    ShapeBasedCollisionDetector,
)
from c2o_drive.core.planner import Transition, UpdateMetrics


class C2OSRPlanner(EpisodicAlgorithmPlanner[WorldState, EgoControl]):
    """C2OSR planner adapted to standard interface.

    This planner uses Dirichlet process learning to model agent
    transitions and selects actions based on Q-value evaluation
    of lattice-generated trajectories.
    """

    def __init__(self, config: C2OSRPlannerConfig):
        """Initialize C2OSR planner.

        Args:
            config: C2OSR planner configuration
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
        center_x = 0.5 * (config.grid.bounds_x[0] + config.grid.bounds_x[1])
        center_y = 0.5 * (config.grid.bounds_y[0] + config.grid.bounds_y[1])
        self.grid_mapper = GridMapper(self.grid_spec, world_center=(center_x, center_y))

        # Initialize trajectory buffer
        self.trajectory_buffer = TrajectoryBuffer(
            horizon=config.lattice.horizon,
            trajectory_storage_multiplier=config.trajectory_storage_multiplier,
            capacity=config.buffer_capacity,
        )

        # Initialize Dirichlet bank
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
                horizon=config.lattice.horizon,
            )
        elif config.dirichlet.use_multistep:
            self.dirichlet_bank = MultiTimestepSpatialDirichletBank(
                K=K,
                params=dirichlet_params,
                horizon=config.lattice.horizon,
            )
        else:
            # Fallback to single-step (not recommended)
            self.dirichlet_bank = SpatialDirichletBank(
                K=K,
                params=dirichlet_params,
            )

        # Initialize lattice planner
        self.lattice_planner = LatticePlanner(
            lateral_offsets=config.lattice.lateral_offsets,
            speed_variations=config.lattice.speed_variations,
            num_trajectories=config.lattice.num_trajectories,
        )

        # Initialize Q-value calculator
        # Use config.reward_weights directly (no longer bypass to global_config)
        self.q_value_calculator = QValueCalculator(
            config=OriginalQValueConfig(
                horizon=config.q_value.horizon,
                n_samples=config.q_value.n_samples,
                q_selection_percentile=config.q_value.selection_percentile,
                gamma=config.q_value.gamma,  # Add gamma parameter
            ),
            reward_config=config.reward_weights,
        )

        # Initialize reward calculator
        # Use config.reward_weights directly (no longer bypass to global_config)
        self.reward_calculator = RewardCalculator(config.reward_weights)

        # Initialize collision detector
        self.collision_detector = ShapeBasedCollisionDetector()

        # Episode state
        self.current_reference_path: Optional[List[Tuple[float, float]]] = None
        self.episode_step_count = 0
        self.last_selected_trajectory: Optional[LatticeTrajectory] = None

        # Episode data collection for online learning
        self._episode_id = 0
        self._current_episode_timesteps: List[Tuple[ScenarioState, WorldState]] = []
        self._current_episode_ego_trajectory: List[Tuple[float, float]] = []
        # Track OBSERVED agent trajectories (key: agent_id, value: list of observed cell IDs per timestep)
        self._episode_agent_observations: Dict[int, List[int]] = {}

    def _ensure_agents_initialized(self, observation: WorldState) -> None:
        """Ensure all agents are initialized in Dirichlet bank.

        This method ensures agents are initialized once at the start,
        similar to how run_sim_cl_simple pre-initializes agents.
        This prevents alpha values from being reset on every Q-value calculation.

        Args:
            observation: Current world state containing agents
        """
        from c2o_drive.config import get_global_config

        config = get_global_config()

        for i, agent in enumerate(observation.agents):
            agent_id = i + 1

            # Skip if already initialized
            if agent_id in self.dirichlet_bank.agent_alphas:
                continue

            # Calculate reachable sets for this agent
            try:
                reachable_sets = self.grid_mapper.multi_timestep_successor_cells(
                    agent,
                    horizon=self.config.lattice.horizon,
                    dt=config.time.dt,
                    n_samples=config.sampling.reachable_set_samples,
                )

                # Initialize in bank if reachable sets exist
                if reachable_sets:
                    self.dirichlet_bank.init_agent(agent_id, reachable_sets)
                    if config.visualization.verbose_level >= 2:
                        print(f"  [Planner] Agent {agent_id} pre-initialized in Dirichlet bank "
                              f"with {len(reachable_sets)} timesteps")
            except Exception as e:
                # Log error but continue - agent will be initialized during Q-value calculation
                if config.visualization.verbose_level >= 1:
                    print(f"  [Warning] Failed to pre-initialize agent {agent_id}: {e}")

    def select_action(
        self,
        observation: WorldState,
        deterministic: bool = False,
        reference_path: Optional[List[Tuple[float, float]]] = None,
        **kwargs
    ) -> EgoControl:
        """Select action using C2OSR algorithm.

        Args:
            observation: Current world state
            deterministic: Whether to use deterministic selection (unused, always deterministic)
            reference_path: Reference path for trajectory generation
            **kwargs: Additional parameters

        Returns:
            Selected ego control action
        """
        # Ensure agents are initialized in Dirichlet bank (done once per agent)
        # This matches the behavior of run_sim_cl_simple which pre-initializes agents
        self._ensure_agents_initialized(observation)

        # Check if buffer has enough data for Q-value evaluation
        skip_q_evaluation = len(self.trajectory_buffer) < self.config.min_buffer_size

        # Store reference path if provided
        if reference_path is not None:
            self.current_reference_path = reference_path

        # Initialize action
        action = None

        if not skip_q_evaluation:
            # Generate candidate trajectories
            if self.current_reference_path is None:
                # Create a simple forward reference path if none provided
                ego_x, ego_y = observation.ego.position_m
                self.current_reference_path = [
                    (ego_x + i * 5.0, ego_y) for i in range(self.config.lattice.horizon + 1)
                ]

            ego_state_tuple = (
                observation.ego.position_m[0],
                observation.ego.position_m[1],
                observation.ego.yaw_rad,
            )

            candidate_trajectories = self.lattice_planner.generate_trajectories(
                reference_path=self.current_reference_path,
                horizon=self.config.lattice.horizon,
                dt=self.config.lattice.dt,
                ego_state=ego_state_tuple,
            )

            # If no trajectories generated, fallback to stop action
            if candidate_trajectories:
                # Evaluate trajectories using Q-values
                best_trajectory = None
                best_q_value = float('-inf')

                for trajectory in candidate_trajectories:
                    # Calculate Q-value for this trajectory
                    try:
                        q_values_list, _ = self.q_value_calculator.compute_q_value(
                            current_world_state=observation,
                            ego_action_trajectory=trajectory.waypoints,
                            trajectory_buffer=self.trajectory_buffer,
                            grid=self.grid_mapper,
                            bank=self.dirichlet_bank,
                            reference_path=self.current_reference_path,
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

                        if q_value > best_q_value:
                            best_q_value = q_value
                            best_trajectory = trajectory
                    except Exception as e:
                        # If Q-value calculation fails, skip this trajectory
                        continue

                # If valid trajectory found, convert to control
                if best_trajectory is not None:
                    self.last_selected_trajectory = best_trajectory
                    action = self._trajectory_to_control(
                        current_state=observation.ego,
                        trajectory=best_trajectory,
                    )

        # Fallback: return stop action if no action selected
        if action is None:
            action = EgoControl(throttle=0.0, steer=0.0, brake=1.0)

        self.episode_step_count += 1
        self._step_count += 1

        # Note: Data collection happens in update() to ensure it works with all planning modes
        # (batch planning, step-by-step planning, etc.)

        return action

    def _trajectory_to_control(
        self,
        current_state: EgoState,
        trajectory: LatticeTrajectory,
    ) -> EgoControl:
        """Convert trajectory to control action.

        Args:
            current_state: Current ego state
            trajectory: Selected trajectory

        Returns:
            Control action
        """
        if len(trajectory.waypoints) < 2:
            return EgoControl(throttle=0.0, steer=0.0, brake=1.0)

        # Target is the next waypoint
        target_x, target_y = trajectory.waypoints[1]
        current_x, current_y = current_state.position_m

        # Calculate heading error
        dx = target_x - current_x
        dy = target_y - current_y
        target_heading = np.arctan2(dy, dx)
        heading_error = target_heading - current_state.yaw_rad

        # Normalize heading error to [-pi, pi]
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

        # Simple P-controller for steering
        steer = np.clip(heading_error * 0.5, -1.0, 1.0)

        # Speed control based on target speed
        current_speed = np.linalg.norm(np.array(current_state.velocity_mps))
        speed_error = trajectory.target_speed - current_speed

        if speed_error > 0.5:
            throttle = 0.6
            brake = 0.0
        elif speed_error < -0.5:
            throttle = 0.0
            brake = 0.5
        else:
            throttle = 0.3
            brake = 0.0

        return EgoControl(throttle=throttle, steer=steer, brake=brake)

    def update(self, transition: Transition[WorldState, EgoControl]) -> UpdateMetrics:
        """Update planner with a transition.

        Now implements online learning by storing episode data to trajectory buffer.

        Args:
            transition: State transition to learn from

        Returns:
            Update metrics
        """
        # Log statistics for monitoring
        self._log_stat('reward', transition.reward)
        self._log_stat('step_count', float(self.episode_step_count))

        # Track collisions/successes
        if transition.terminated:
            self._log_stat('episode_terminated', 1.0)
        if transition.truncated:
            self._log_stat('episode_truncated', 1.0)

        # Record episode data for online learning (from update instead of select_action)
        # This ensures data is collected regardless of planning mode
        observation = transition.state  # WorldState
        action = transition.action  # Action ID or control

        scenario_state = self._world_state_to_scenario_state(observation, action)
        self._current_episode_timesteps.append((scenario_state, observation))
        self._current_episode_ego_trajectory.append(observation.ego.position_m)

        # Record OBSERVED agent positions as grid cells
        # This is the KEY FIX: store actual observed positions, not predictions
        for agent_idx, agent in enumerate(observation.agents):
            agent_id = agent_idx + 1  # 1-based indexing to match convention

            # Convert observed position to grid cell
            try:
                observed_cell = self.grid_mapper.world_to_cell(tuple(agent.position_m))

                # Initialize list for this agent if first observation
                if agent_id not in self._episode_agent_observations:
                    self._episode_agent_observations[agent_id] = []

                # Append observed cell for this timestep
                self._episode_agent_observations[agent_id].append(observed_cell)
            except Exception as e:
                # If position is out of bounds, skip (don't break the episode)
                # This is rare but can happen at environment boundaries
                pass

        # Store episode data when episode ends
        if transition.terminated or transition.truncated:
            self._store_episode_to_buffer()
            self._episode_id += 1

            # Clear episode data for next episode
            self._current_episode_timesteps = []
            self._current_episode_ego_trajectory = []
            self._episode_agent_observations = {}  # Clear observed trajectories

        return UpdateMetrics(
            loss=0.0,  # C2OSR doesn't have a traditional loss
            custom={
                'reward': transition.reward,
                'step_count': self.episode_step_count,
                'terminated': transition.terminated,
                'truncated': transition.truncated,
                'buffer_size': len(self.trajectory_buffer),
            }
        )

    def _store_episode_to_buffer(self):
        """Store collected episode data to trajectory buffer."""
        if len(self._current_episode_timesteps) == 0:
            return  # No data to store

        # Convert collected data into timestep_scenarios format
        timestep_scenarios = []
        horizon = self.config.lattice.horizon

        for t, (scenario_state, world_state) in enumerate(self._current_episode_timesteps):
            # Create agent trajectory data using OBSERVED positions
            # This is the KEY FIX: use real observations instead of predictions
            agent_trajectories = []

            for agent_idx, agent in enumerate(world_state.agents):
                agent_id = agent_idx + 1  # 1-based to match convention

                # Extract OBSERVED trajectory for this agent
                # Always create full horizon-length trajectory, padding with last cell if needed
                # This ensures consistent format for all timesteps and enables proper retrieval
                if agent_id in self._episode_agent_observations:
                    observed_cells_full = self._episode_agent_observations[agent_id]
                    # Create full horizon-length trajectory from current timestep
                    trajectory_cells = []
                    for offset in range(self.config.horizon):
                        # Get cell at timestep t+offset, or use last available cell if beyond episode end
                        idx = min(t + offset, len(observed_cells_full) - 1)
                        trajectory_cells.append(observed_cells_full[idx])
                    remaining_cells = trajectory_cells
                else:
                    # Agent not tracked (rare case): use current position repeated horizon times
                    try:
                        current_cell = self.grid_mapper.world_to_cell(tuple(agent.position_m))
                        remaining_cells = [current_cell] * self.config.horizon
                    except:
                        remaining_cells = [0] * self.config.horizon

                traj_data = AgentTrajectoryData(
                    agent_id=agent_id,
                    agent_type=agent.agent_type.value if hasattr(agent.agent_type, 'value') else str(agent.agent_type),
                    init_position=agent.position_m,
                    init_velocity=agent.velocity_mps,
                    init_heading=agent.heading_rad,
                    trajectory_cells=remaining_cells,  # Use OBSERVED remaining trajectory
                )
                agent_trajectories.append(traj_data)

            timestep_scenarios.append((scenario_state, agent_trajectories))

        # Store to trajectory buffer
        buffer_size_before = len(self.trajectory_buffer)
        self.trajectory_buffer.store_episode_trajectories_by_timestep(
            episode_id=self._episode_id,
            timestep_scenarios=timestep_scenarios,
            ego_trajectory=self._current_episode_ego_trajectory,
        )
        buffer_size_after = len(self.trajectory_buffer)

        # Log storage statistics
        new_data_count = buffer_size_after - buffer_size_before
        total_agents = sum(len(traj_data) for _, traj_data in timestep_scenarios)
        expected_count = len(timestep_scenarios) * total_agents * self.config.trajectory_storage_multiplier
        # Only log if verbose level is set
        from c2o_drive.config import get_global_config
        if get_global_config().visualization.verbose_level >= 1:
            print(f"[Episode {self._episode_id}] Stored {new_data_count} episodes "
                  f"(expected: {expected_count}, "
                  f"buffer: {buffer_size_before}->{buffer_size_after}, "
                  f"multiplier: {self.config.trajectory_storage_multiplier})")

        # Note: Episode data cleanup is handled in update() method when episode ends

    def _predict_agent_trajectory(
        self,
        agent: 'AgentState',
        horizon: int,
    ) -> List[int]:
        """Predict agent trajectory using constant velocity model.

        Args:
            agent: Agent state
            horizon: Prediction horizon

        Returns:
            List of predicted grid cell IDs
        """
        predicted_cells = []
        dt = self.config.lattice.dt

        current_pos = np.array(agent.position_m)
        velocity = np.array(agent.velocity_mps)

        for t in range(horizon):
            # Predict position using constant velocity
            predicted_pos = current_pos + velocity * dt * (t + 1)

            # Convert to grid cell
            try:
                cell_id = self.grid_mapper.world_to_cell(tuple(predicted_pos))
                predicted_cells.append(cell_id)
            except:
                # If out of bounds, use last valid cell or 0
                if predicted_cells:
                    predicted_cells.append(predicted_cells[-1])
                else:
                    predicted_cells.append(0)

        return predicted_cells

    def _world_state_to_scenario_state(
        self,
        world_state: WorldState,
        action: EgoControl,
    ) -> ScenarioState:
        """Convert WorldState to ScenarioState for buffer storage.

        Args:
            world_state: World state
            action: Ego control action

        Returns:
            Scenario state for buffer
        """
        # Convert agents to list of tuples: (x, y, vx, vy, heading, type)
        agents_states = []
        for agent in world_state.agents:
            agent_type_str = agent.agent_type.value if hasattr(agent, 'agent_type') else 'vehicle'
            agents_states.append((
                agent.position_m[0],  # x
                agent.position_m[1],  # y
                agent.velocity_mps[0],  # vx
                agent.velocity_mps[1],  # vy
                agent.heading_rad,  # heading
                agent_type_str,  # type
            ))

        return ScenarioState(
            ego_position=world_state.ego.position_m,
            ego_velocity=world_state.ego.velocity_mps,
            ego_heading=world_state.ego.yaw_rad,  # EgoState uses yaw_rad, not heading_rad
            agents_states=agents_states,
        )

    def plan_trajectory(
        self,
        observation: WorldState,
        horizon: int,
        reference_path: Optional[List[Tuple[float, float]]] = None,
        **kwargs
    ) -> List[EgoControl]:
        """Plan a trajectory of actions.

        Args:
            observation: Current observation
            horizon: Planning horizon (number of steps)
            reference_path: Reference path for planning
            **kwargs: Additional parameters

        Returns:
            Planned trajectory as list of actions
        """
        # For C2OSR, we generate the full trajectory at once
        # and then convert it to a sequence of actions

        if reference_path is not None:
            self.current_reference_path = reference_path

        # Select action (which internally generates and selects a trajectory)
        first_action = self.select_action(observation, reference_path=reference_path)

        # If we have a selected trajectory, convert it to action sequence
        if self.last_selected_trajectory is not None:
            trajectory = self.last_selected_trajectory
            actions = []

            # Simple approach: repeat similar actions based on trajectory shape
            num_waypoint_actions = min(horizon, max(1, len(trajectory.waypoints) - 1))
            for i in range(num_waypoint_actions):
                # This is simplified; ideally we'd simulate forward
                actions.append(first_action)

            # Fill remaining with first_action if needed
            while len(actions) < horizon:
                actions.append(first_action)

            return actions

        # Fallback: return repeated first action
        return [first_action] * horizon

    def reset(self) -> None:
        """Reset planner for new episode."""
        super().reset()
        self.episode_step_count = 0
        self.last_selected_trajectory = None
        self.current_reference_path = None

        # Clear episode data collection
        self._current_episode_timesteps = []
        self._current_episode_ego_trajectory = []
        self._episode_agent_observations = {}  # Clear observed trajectories

    def save(self, path: str) -> None:
        """Save planner state to disk.

        Args:
            path: Path to save directory
        """
        import pickle

        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save trajectory buffer
        with open(save_path / 'trajectory_buffer.pkl', 'wb') as f:
            pickle.dump(self.trajectory_buffer, f)

        # Save Dirichlet bank
        with open(save_path / 'dirichlet_bank.pkl', 'wb') as f:
            pickle.dump(self.dirichlet_bank, f)

        # Save config
        with open(save_path / 'config.pkl', 'wb') as f:
            pickle.dump(self.config, f)

        # Save statistics
        with open(save_path / 'stats.pkl', 'wb') as f:
            pickle.dump(self.get_stats(), f)

    def load(self, path: str) -> None:
        """Load planner state from disk.

        Args:
            path: Path to load directory
        """
        import pickle

        load_path = Path(path)

        # Load trajectory buffer
        with open(load_path / 'trajectory_buffer.pkl', 'rb') as f:
            self.trajectory_buffer = pickle.load(f)

        # Load Dirichlet bank
        with open(load_path / 'dirichlet_bank.pkl', 'rb') as f:
            self.dirichlet_bank = pickle.load(f)

        # Load statistics
        if (load_path / 'stats.pkl').exists():
            with open(load_path / 'stats.pkl', 'rb') as f:
                self._stats = pickle.load(f)


__all__ = ['C2OSRPlanner']
