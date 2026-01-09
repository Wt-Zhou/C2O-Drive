"""Main RCRL planner implementation.

This module implements the Reachability-Constrained RL planner,
integrating reachability analysis with deep Q-learning for safe
autonomous driving.
"""

from typing import List, Optional, Tuple, Dict
import torch
import torch.nn.functional as F
import numpy as np

from c2o_drive.core.types import WorldState, EgoControl, EgoState
from c2o_drive.core.planner import Transition, UpdateMetrics
from c2o_drive.algorithms.base import EpisodicAlgorithmPlanner
from c2o_drive.algorithms.rcrl.config import RCRLPlannerConfig
from c2o_drive.algorithms.rcrl.state_encoder import StateEncoder
from c2o_drive.algorithms.rcrl.reachability import ReachabilityModule
from c2o_drive.algorithms.rcrl.constraints import ConstraintEnforcer
from c2o_drive.algorithms.rcrl.network import DuelingDQN, StandardDQN
from c2o_drive.algorithms.rcrl.replay_buffer import ReplayBuffer, RCRLTransition


class RCRLPlanner(EpisodicAlgorithmPlanner[WorldState, EgoControl]):
    """Reachability-Constrained Reinforcement Learning Planner.

    Combines forward reachability analysis with deep Q-learning to achieve
    safe and efficient autonomous driving. Supports both hard and soft
    safety constraints.

    Architecture follows C2OSR patterns: proper inheritance, modular design,
    hierarchical configuration, and online learning via update() method.
    """

    def __init__(self, config: RCRLPlannerConfig):
        """Initialize RCRL planner.

        Args:
            config: RCRL planner configuration
        """
        super().__init__(config)
        self.config = config

        # Set device
        self.device = torch.device(config.device)

        # Initialize components
        self.state_encoder = StateEncoder(config.state_encoder)
        self.reachability_module = ReachabilityModule(config.reachability)
        self.constraint_enforcer = ConstraintEnforcer(config.constraint)

        # Initialize neural networks
        n_actions = config.get_n_actions()
        if config.network.use_dueling:
            self.q_network = DuelingDQN(config.network, n_actions).to(self.device)
            self.target_network = DuelingDQN(config.network, n_actions).to(self.device)
        else:
            self.q_network = StandardDQN(config.network, n_actions).to(self.device)
            self.target_network = StandardDQN(config.network, n_actions).to(self.device)

        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target network is always in eval mode

        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(), lr=config.training.learning_rate
        )

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(config.training.buffer_capacity)

        # Lattice planner for trajectory generation
        # Will be initialized lazily when reference path is available
        self.lattice_planner = None

        # Episode state tracking
        self._current_reference_path: Optional[List[Tuple[float, float]]] = None
        self._last_action_idx: int = 0
        self._episode_step_count: int = 0

        # Exploration epsilon
        self._epsilon = config.training.epsilon_start
        self._epsilon_decay_rate = (
            (config.training.epsilon_start - config.training.epsilon_end)
            / config.training.epsilon_decay_steps
        )

        # Training step counter
        self._training_step = 0

    def select_action(
        self,
        observation: WorldState,
        deterministic: bool = False,
        reference_path: Optional[List[Tuple[float, float]]] = None,
        **kwargs,
    ) -> EgoControl:
        """Select action using RCRL policy.

        Execution flow:
        1. Generate candidate trajectories from lattice planner
        2. Compute reachability sets (ego + agents)
        3. Apply safety constraints (hard filter or soft penalty)
        4. Encode state and run DQN forward pass
        5. Select action via epsilon-greedy or constrained greedy
        6. Convert selected trajectory to control command

        Args:
            observation: Current world state
            deterministic: If True, disable epsilon-greedy exploration
            reference_path: Reference path for planning
            **kwargs: Additional parameters

        Returns:
            Ego control command
        """
        # Update reference path if provided
        if reference_path is not None:
            self._current_reference_path = reference_path

        # Initialize lattice planner if needed
        if self.lattice_planner is None:
            from c2o_drive.utils.lattice_planner import LatticePlanner

            self.lattice_planner = LatticePlanner(
                lateral_offsets=self.config.lattice.lateral_offsets,
                speed_variations=self.config.lattice.target_speeds,
            )

        # Step 1: Generate candidate trajectories
        ego_state_tuple = (
            observation.ego.position_m[0],
            observation.ego.position_m[1],
            observation.ego.yaw_rad,
        )
        candidate_trajectories = self.lattice_planner.generate_trajectories(
            reference_path=self._current_reference_path,
            horizon=self.config.horizon,
            dt=self.config.dt,
            ego_state=ego_state_tuple,
        )

        # Add emergency brake if configured
        if self.config.lattice.include_emergency_brake:
            emergency_traj = self._create_emergency_brake_trajectory(observation.ego)
            candidate_trajectories.append(emergency_traj)

        n_trajectories = len(candidate_trajectories)

        # Step 2: Compute reachability sets
        ego_reachable_sets = []
        for traj in candidate_trajectories:
            ego_reach = self.reachability_module.compute_ego_reachable_set(
                observation.ego, traj.waypoints
            )
            ego_reachable_sets.append(ego_reach)

        agent_reachable_sets = self.reachability_module.compute_agent_reachable_sets(
            observation.agents
        )

        # Step 3: Encode state
        state_features = self.state_encoder.encode(observation, self._current_reference_path)

        # Step 4: DQN forward pass
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_features).unsqueeze(0).to(self.device)
            all_q_values = self.q_network(state_tensor).squeeze(0)  # [n_actions_max]

        # Extract Q-values for actual trajectories
        # Network outputs Q-values for all possible actions (lattice grid)
        # But we only have n_trajectories actual trajectories
        if all_q_values.shape[0] >= n_trajectories:
            q_values = all_q_values[:n_trajectories]
        else:
            # Pad if needed
            padding = torch.zeros(n_trajectories - all_q_values.shape[0], device=self.device)
            q_values = torch.cat([all_q_values, padding], dim=0)

        # Step 5: Apply constraints and select action
        if self.config.constraint.mode == "hard":
            selected_action_idx = self._select_action_hard_constraint(
                ego_reachable_sets,
                agent_reachable_sets,
                q_values,
                n_trajectories,
                deterministic,
            )
        else:  # soft constraint
            selected_action_idx = self._select_action_soft_constraint(
                ego_reachable_sets, agent_reachable_sets, q_values, deterministic
            )

        # Save action for replay buffer
        self._last_action_idx = selected_action_idx

        # Step 6: Convert to control
        selected_trajectory = candidate_trajectories[selected_action_idx]
        control = self._trajectory_to_control(observation.ego, selected_trajectory)

        # Log statistics
        self._log_stat("selected_action_idx", float(selected_action_idx))
        self._log_stat("q_value_mean", q_values.mean().item())
        self._log_stat("q_value_max", q_values.max().item())
        self._log_stat("epsilon", self._epsilon)

        self._episode_step_count += 1

        return control

    def _select_action_hard_constraint(
        self,
        ego_reachable_sets,
        agent_reachable_sets,
        q_values,
        n_trajectories,
        deterministic,
    ) -> int:
        """Select action with hard safety constraints.

        Args:
            ego_reachable_sets: Ego reachable sets per trajectory
            agent_reachable_sets: Agent reachable sets
            q_values: Q-values for all actions
            n_trajectories: Total number of trajectories
            deterministic: Whether to use deterministic policy

        Returns:
            Selected action index
        """
        # Filter safe actions
        safe_indices = self.constraint_enforcer.apply_constraints(
            ego_reachable_sets, agent_reachable_sets, q_values=None
        )

        # If no safe actions, use emergency brake (last action)
        if len(safe_indices) == 0:
            self._log_stat("emergency_brake_triggered", 1.0)
            return n_trajectories - 1  # Emergency brake is last action

        # Select from safe actions
        safe_q_values = q_values[safe_indices]

        if deterministic or np.random.random() >= self._epsilon:
            # Greedy: select best Q-value among safe actions
            best_idx = torch.argmax(safe_q_values).item()
            return safe_indices[best_idx]
        else:
            # Exploration: random safe action
            return np.random.choice(safe_indices)

    def _select_action_soft_constraint(
        self, ego_reachable_sets, agent_reachable_sets, q_values, deterministic
    ) -> int:
        """Select action with soft safety constraints.

        Args:
            ego_reachable_sets: Ego reachable sets per trajectory
            agent_reachable_sets: Agent reachable sets
            q_values: Q-values for all actions
            deterministic: Whether to use deterministic policy

        Returns:
            Selected action index
        """
        # Apply soft penalty
        constrained_q_values = self.constraint_enforcer.apply_constraints(
            ego_reachable_sets, agent_reachable_sets, q_values
        )

        # Epsilon-greedy selection
        if deterministic or np.random.random() >= self._epsilon:
            # Greedy
            return torch.argmax(constrained_q_values).item()
        else:
            # Random exploration
            return np.random.randint(len(constrained_q_values))

    def update(self, transition: Transition[WorldState, EgoControl]) -> UpdateMetrics:
        """Online learning update.

        Execution flow:
        1. Encode state and next_state
        2. Store transition in replay buffer
        3. If buffer is large enough, sample batch and train
        4. Periodically update target network
        5. Decay epsilon

        Args:
            transition: Environment transition

        Returns:
            Update metrics including loss and statistics
        """
        # Step 1: Encode states
        state_features = self.state_encoder.encode(
            transition.state, self._current_reference_path
        )
        next_state_features = self.state_encoder.encode(
            transition.next_state, self._current_reference_path
        )

        # Step 2: Store transition
        rcrl_transition = RCRLTransition(
            state=state_features,
            action=self._last_action_idx,
            reward=transition.reward,
            next_state=next_state_features,
            terminated=transition.terminated,
            truncated=transition.truncated,
        )
        self.replay_buffer.push(rcrl_transition)

        # Step 3: Training
        loss = 0.0
        if (
            len(self.replay_buffer) >= self.config.training.min_buffer_size
            and self._step_count % self.config.training.train_freq == 0
        ):
            loss = self._train_step()

        # Step 4: Update target network
        if self._step_count % self.config.training.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            self._log_stat("target_network_updated", 1.0)

        # Step 5: Decay epsilon
        self._update_epsilon()

        # Step 6: Episode cleanup
        if transition.terminated or transition.truncated:
            self._on_episode_end()

        self._step_count += 1

        return UpdateMetrics(
            loss=loss,
            custom={
                "buffer_size": len(self.replay_buffer),
                "epsilon": self._epsilon,
                "reward": transition.reward,
                "training_step": self._training_step,
            },
        )

    def _train_step(self) -> float:
        """Perform one training step using Double DQN.

        Returns:
            Training loss value
        """
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.config.training.batch_size, device=self.device
        )

        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Double DQN target
        with torch.no_grad():
            if self.config.training.double_dqn:
                # Use online network to select actions
                next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
                # Use target network to evaluate Q-values
                next_q_values = self.target_network(next_states).gather(1, next_actions)
            else:
                # Standard DQN
                next_q_values = self.target_network(next_states).max(dim=1, keepdim=True)[0]

            target_q_values = rewards.unsqueeze(1) + (
                1 - dones.unsqueeze(1)
            ) * self.config.training.gamma * next_q_values

        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if self.config.training.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.q_network.parameters(), self.config.training.grad_clip_norm
            )

        self.optimizer.step()

        self._training_step += 1
        self._log_stat("loss", loss.item())

        return loss.item()

    def _update_epsilon(self) -> None:
        """Decay epsilon for exploration."""
        if self._epsilon > self.config.training.epsilon_end:
            self._epsilon = max(
                self.config.training.epsilon_end, self._epsilon - self._epsilon_decay_rate
            )

    def _trajectory_to_control(
        self, ego_state: EgoState, trajectory
    ) -> EgoControl:
        """Convert trajectory to control command.

        Simple P-controller for lateral and longitudinal control.

        Args:
            ego_state: Current ego state
            trajectory: Selected trajectory

        Returns:
            Control command
        """
        # Get first waypoint
        if len(trajectory.waypoints) == 0:
            return EgoControl(throttle=0.0, steer=0.0, brake=1.0)

        target_waypoint = trajectory.waypoints[0]

        # Lateral control (simple P-controller)
        ego_pos = np.array(ego_state.position_m)
        target_pos = np.array(target_waypoint)
        relative_pos = target_pos - ego_pos

        # Transform to ego frame
        ego_yaw = ego_state.yaw_rad
        cos_yaw = np.cos(ego_yaw)
        sin_yaw = np.sin(ego_yaw)
        lateral_error = -relative_pos[0] * sin_yaw + relative_pos[1] * cos_yaw

        # Steer proportional to lateral error
        steer = np.clip(lateral_error * 0.5, -1.0, 1.0)

        # Longitudinal control
        current_speed = np.linalg.norm(ego_state.velocity_mps)
        target_speed = getattr(trajectory, "target_speed", 8.0)
        speed_error = target_speed - current_speed

        # Simple throttle/brake control
        if speed_error > 0.5:
            throttle = min(0.8, speed_error * 0.2)
            brake = 0.0
        elif speed_error < -0.5:
            throttle = 0.0
            brake = min(0.8, -speed_error * 0.3)
        else:
            throttle = 0.0
            brake = 0.0

        return EgoControl(throttle=throttle, steer=steer, brake=brake)

    def _create_emergency_brake_trajectory(self, ego_state: EgoState):
        """Create emergency brake trajectory.

        Args:
            ego_state: Current ego state

        Returns:
            Emergency brake trajectory
        """
        # Create simple trajectory that stays at current position
        from c2o_drive.utils.lattice_planner import LatticeTrajectory

        waypoints = [ego_state.position_m] * self.config.horizon
        return LatticeTrajectory(
            waypoints=waypoints,
            lateral_offset=0.0,
            target_speed=0.0,
            trajectory_id=-1,  # Emergency brake trajectory ID
        )

    def _on_episode_end(self) -> None:
        """Handle episode end."""
        self._episode_step_count = 0
        self._log_stat("episode_length", self._step_count)

    def reset(self) -> None:
        """Reset planner for new episode."""
        super().reset()
        self._episode_step_count = 0
        self._current_reference_path = None

    def plan_trajectory(
        self, observation: WorldState, horizon: int, **kwargs
    ) -> List[EgoControl]:
        """Plan full trajectory (required by EpisodicPlanner interface).

        Args:
            observation: Current world state
            horizon: Planning horizon
            **kwargs: Additional parameters

        Returns:
            List of control commands
        """
        # Simple implementation: repeatedly call select_action
        controls = []
        for _ in range(horizon):
            control = self.select_action(observation, deterministic=True, **kwargs)
            controls.append(control)
        return controls

    def save(self, path: str) -> None:
        """Save model to disk.

        Args:
            path: Save path
        """
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "step_count": self._step_count,
                "epsilon": self._epsilon,
                "training_step": self._training_step,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load model from disk.

        Args:
            path: Load path
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self._step_count = checkpoint["step_count"]
        self._epsilon = checkpoint["epsilon"]
        self._training_step = checkpoint["training_step"]


__all__ = ["RCRLPlanner"]
