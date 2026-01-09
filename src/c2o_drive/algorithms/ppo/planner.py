"""PPO Planner implementation.

This module provides the main PPO planner class that integrates all components
and follows the EpisodicAlgorithmPlanner interface for consistency with C2OSR.
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from c2o_drive.algorithms.base import EpisodicAlgorithmPlanner
from c2o_drive.core.types import WorldState, EgoControl
from c2o_drive.core.planner import Transition, UpdateMetrics
from c2o_drive.algorithms.ppo.config import PPOConfig
from c2o_drive.algorithms.ppo.network import ActorCriticNetwork
from c2o_drive.algorithms.ppo.rollout_buffer import RolloutBuffer
from c2o_drive.utils.lattice_planner import LatticePlanner


class PPOPlanner(EpisodicAlgorithmPlanner[WorldState, EgoControl]):
    """PPO Planner for discrete action space.

    Inheritance hierarchy:
    PPOPlanner -> EpisodicAlgorithmPlanner -> BaseAlgorithmPlanner -> BasePlanner

    This planner:
    1. Uses lattice-based discrete actions (dynamically computed from config)
    2. Generates trajectory once per episode
    3. Executes trajectory step by step
    4. Updates using PPO algorithm with GAE

    Attributes:
        config: PPO configuration
        lattice_planner: Lattice trajectory planner
        action_mapping: List of (lateral_offset, target_speed) tuples
        network: Actor-Critic network
        optimizer: Adam optimizer
        rollout_buffer: Rollout buffer for PPO
    """

    def __init__(self, config: PPOConfig):
        """Initialize PPO planner.

        Args:
            config: PPO configuration
        """
        super().__init__(config)
        self.config = config

        # Initialize Lattice Planner (reuse C2OSR logic)
        # Ensure num_trajectories covers all action combinations
        num_trajectories = max(
            config.lattice.num_trajectories,
            config.action_dim
        )
        self.lattice_planner = LatticePlanner(
            lateral_offsets=config.lattice.lateral_offsets,
            speed_variations=config.lattice.speed_variations,
            num_trajectories=num_trajectories,
        )

        # Build action space mapping (dynamically generated)
        self._build_action_space()

        # Actor-Critic network
        self.network = ActorCriticNetwork(
            state_dim=config.state_dim,
            action_dim=config.action_dim,  # Dynamically computed
            hidden_dims=config.hidden_dims,
        ).to(config.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=config.learning_rate,
        )

        # Rollout buffer
        self.rollout_buffer = RolloutBuffer(
            capacity=config.buffer_size,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda if config.use_gae else None,
        )

        # Episode state tracking
        self._current_trajectory = None
        self._trajectory_step = 0
        self._last_action_idx = None
        self._last_log_prob = None
        self._last_value = None

        # Statistics
        self._episode_count = 0

    def _build_action_space(self):
        """Build action space mapping dynamically from lattice config.

        Generates all discrete action candidates:
        action_idx -> (lateral_offset, target_speed)

        This ensures the action space adapts when lattice config changes.
        """
        self.action_mapping = []
        for lateral in self.config.lattice.lateral_offsets:
            for speed in self.config.lattice.speed_variations:
                self.action_mapping.append((lateral, speed))

        # Verify action count matches config
        assert len(self.action_mapping) == self.config.action_dim, (
            f"Action mapping size {len(self.action_mapping)} != "
            f"config.action_dim {self.config.action_dim}"
        )

    def select_action(
        self,
        observation: WorldState,
        deterministic: bool = False,
        reference_path: Optional[List[Tuple[float, float]]] = None,
        **kwargs
    ) -> EgoControl:
        """Select action (following BasePlanner interface).

        Flow:
        1. Extract state features
        2. Actor network outputs action probabilities
        3. Sample discrete action
        4. Map to lattice parameters
        5. Generate trajectory (at episode start)
        6. Return current step control

        Args:
            observation: Current world state
            deterministic: If True, select argmax action (for evaluation)
            reference_path: Reference path for trajectory generation
            **kwargs: Additional arguments

        Returns:
            Control command for current step
        """
        # Extract features
        state_features = self._extract_state_features(observation)

        # Episode start: generate trajectory
        if self._current_trajectory is None:
            # Actor network outputs action distribution
            with torch.no_grad():
                logits, value = self.network(state_features)
                action_probs = F.softmax(logits, dim=-1)

                if deterministic:
                    action_idx = torch.argmax(action_probs).item()
                    log_prob = None  # Don't need for deterministic
                else:
                    action_dist = Categorical(probs=action_probs)
                    action_idx = action_dist.sample().item()
                    log_prob = action_dist.log_prob(torch.tensor(action_idx))

            # Generate all candidate trajectories (following C2OSR pattern)
            if reference_path is None:
                # Create a simple forward reference path if none provided
                ego_x, ego_y = observation.ego.position_m
                reference_path = [
                    (ego_x + i * 5.0, ego_y) for i in range(self.config.lattice.horizon + 1)
                ]

            ego_state_tuple = (
                observation.ego.position_m[0],
                observation.ego.position_m[1],
                observation.ego.yaw_rad,
            )

            candidate_trajectories = self.lattice_planner.generate_trajectories(
                reference_path=reference_path,
                horizon=self.config.lattice.horizon,
                dt=self.config.lattice.dt,
                ego_state=ego_state_tuple,
            )

            # Select trajectory by action index
            if action_idx < len(candidate_trajectories):
                self._current_trajectory = candidate_trajectories[action_idx]
            else:
                # Fallback: use first trajectory if action_idx out of range
                self._current_trajectory = candidate_trajectories[0] if candidate_trajectories else None

            self._trajectory_step = 0

            # Store selection info (for update)
            self._last_action_idx = action_idx
            self._last_log_prob = log_prob
            self._last_value = value

        # Get current step control from trajectory
        if self._current_trajectory is None:
            # Safety: no trajectory available, apply brake
            return EgoControl(throttle=0.0, steer=0.0, brake=1.0)

        if self._trajectory_step >= len(self._current_trajectory.waypoints):
            # Safety: trajectory exhausted, apply brake
            return EgoControl(throttle=0.0, steer=0.0, brake=1.0)

        waypoint = self._current_trajectory.waypoints[self._trajectory_step]
        control = self._waypoint_to_control(observation, waypoint, self._trajectory_step)

        return control

    def update(self, transition: Transition[WorldState, EgoControl]) -> UpdateMetrics:
        """Update learning (following BasePlanner interface).

        Flow:
        1. Store transition to rollout buffer
        2. When trajectory finishes:
           - Compute GAE
           - Execute PPO update (multiple epochs)

        Args:
            transition: State transition data

        Returns:
            Update metrics (loss, entropy, etc.)
        """
        # Extract features from current state
        state_features = self._extract_state_features(transition.state)

        # Store to buffer (only if we have log_prob, i.e., not deterministic)
        if self._last_log_prob is not None:
            self.rollout_buffer.push(
                state=state_features,
                action=self._last_action_idx,
                reward=transition.reward,
                value=self._last_value,
                log_prob=self._last_log_prob,
                done=transition.terminated or transition.truncated,
            )

        # Update trajectory step
        self._trajectory_step += 1

        # Check if trajectory finished
        trajectory_finished = (
            transition.terminated
            or transition.truncated
            or (self._current_trajectory is not None and
                self._trajectory_step >= len(self._current_trajectory.waypoints))
        )

        if trajectory_finished:
            # Reset trajectory state
            self._current_trajectory = None
            self._trajectory_step = 0
            self._episode_count += 1

            # Execute PPO update if buffer is sufficient
            if len(self.rollout_buffer) >= self.config.batch_size:
                ppo_metrics = self._ppo_update()
                return UpdateMetrics(
                    loss=ppo_metrics.get('policy_loss'),
                    policy_entropy=ppo_metrics.get('entropy'),
                    custom=ppo_metrics,
                )

        return UpdateMetrics()  # Empty metrics

    def _ppo_update(self) -> Dict[str, float]:
        """Execute PPO update (multiple epochs).

        Returns:
            Dictionary of training metrics
        """
        # Compute GAE advantages
        self.rollout_buffer.compute_advantages()

        total_metrics = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy': 0.0,
            'approx_kl': 0.0,
        }
        n_updates = 0

        for epoch in range(self.config.n_epochs):
            for batch in self.rollout_buffer.sample_batches(self.config.batch_size):
                # Forward pass
                logits, values = self.network(batch['states'])
                action_dist = Categorical(logits=logits)

                # New log probabilities
                new_log_probs = action_dist.log_prob(batch['actions'])
                entropy = action_dist.entropy().mean()

                # Importance sampling ratio
                ratio = torch.exp(new_log_probs - batch['old_log_probs'])

                # PPO clipped objective
                advantages = batch['advantages']
                if self.config.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                surr1 = ratio * advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_epsilon,
                    1.0 + self.config.clip_epsilon
                ) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values.squeeze(-1), batch['returns'])

                # Total loss
                loss = (
                    policy_loss
                    + self.config.value_loss_coef * value_loss
                    - self.config.entropy_coef * entropy
                )

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(),
                    self.config.clip_grad_norm,
                )
                self.optimizer.step()

                # Accumulate metrics
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - torch.log(ratio)).mean().item()

                total_metrics['policy_loss'] += policy_loss.item()
                total_metrics['value_loss'] += value_loss.item()
                total_metrics['entropy'] += entropy.item()
                total_metrics['approx_kl'] += approx_kl
                n_updates += 1

        # Average metrics
        if n_updates > 0:
            for key in total_metrics:
                total_metrics[key] /= n_updates

        # Clear buffer
        self.rollout_buffer.clear()

        return total_metrics

    def reset(self) -> None:
        """Reset episode state."""
        self._current_trajectory = None
        self._trajectory_step = 0
        self._last_action_idx = None
        self._last_log_prob = None
        self._last_value = None

    def plan_trajectory(
        self,
        observation: WorldState,
        horizon: int,
        reference_path: Optional[List[Tuple[float, float]]] = None,
        **kwargs
    ) -> List[EgoControl]:
        """Plan complete trajectory (following EpisodicPlanner interface).

        Args:
            observation: Current world state
            horizon: Planning horizon (ignored, uses trajectory length)
            reference_path: Reference path for planning
            **kwargs: Additional arguments

        Returns:
            List of control commands for entire trajectory
        """
        # Generate action (this will create trajectory)
        control = self.select_action(
            observation,
            deterministic=True,
            reference_path=reference_path
        )

        if self._current_trajectory is None:
            return [control]

        # Return entire trajectory as control sequence
        controls = []
        for i, waypoint in enumerate(self._current_trajectory.waypoints):
            controls.append(self._waypoint_to_control(observation, waypoint, i))

        return controls

    def _extract_state_features(self, world_state: WorldState) -> torch.Tensor:
        """Extract state features from world state (reuse SAC/DQN logic).

        Args:
            world_state: Current world state

        Returns:
            Feature tensor of shape (state_dim,)
        """
        features = []

        # Ego vehicle features
        ego = world_state.ego
        ego_speed = np.linalg.norm(ego.velocity_mps)
        features.extend([
            ego.position_m[0] / 100.0,  # Normalized x position
            ego.position_m[1] / 100.0,  # Normalized y position
            ego_speed / 30.0,            # Normalized speed
            np.cos(ego.yaw_rad),         # Heading cosine
            np.sin(ego.yaw_rad),         # Heading sine
        ])

        # Goal features (if available)
        if hasattr(world_state, 'goal') and world_state.goal is not None:
            goal = world_state.goal
            rel_x = (goal.position_m[0] - ego.position_m[0]) / 100.0
            rel_y = (goal.position_m[1] - ego.position_m[1]) / 100.0
            features.extend([rel_x, rel_y])
        else:
            features.extend([0.0, 0.0])

        # Agent features (nearest N agents)
        max_agents = 10
        for i, agent in enumerate(world_state.agents[:max_agents]):
            rel_x = (agent.position_m[0] - ego.position_m[0]) / 100.0
            rel_y = (agent.position_m[1] - ego.position_m[1]) / 100.0
            agent_speed = np.linalg.norm(agent.velocity_mps)
            features.extend([
                rel_x,
                rel_y,
                agent_speed / 30.0,
                np.cos(agent.heading_rad),
                np.sin(agent.heading_rad),
            ])

        # Pad with zeros to reach state_dim
        while len(features) < self.config.state_dim:
            features.append(0.0)

        # Convert to tensor
        feature_array = np.array(features[:self.config.state_dim], dtype=np.float32)
        return torch.tensor(feature_array, dtype=torch.float32, device=self.config.device)

    def _waypoint_to_control(
        self,
        state: WorldState,
        waypoint: Tuple[float, float],
        step_idx: int
    ) -> EgoControl:
        """Convert trajectory waypoint to control command (reuse C2OSR/SAC logic).

        Args:
            state: Current world state
            waypoint: Target waypoint (x, y)
            step_idx: Current step index in trajectory

        Returns:
            Control command
        """
        # If at last waypoint, apply brake
        if step_idx + 1 >= len(self._current_trajectory.waypoints):
            return EgoControl(throttle=0.0, steer=0.0, brake=1.0)

        # Get target position
        target_x, target_y = self._current_trajectory.waypoints[step_idx + 1]
        current_x, current_y = state.ego.position_m

        # Compute steering from heading error
        dx = target_x - current_x
        dy = target_y - current_y
        target_heading = np.arctan2(dy, dx)
        heading_error = target_heading - state.ego.yaw_rad

        # Normalize heading error to [-pi, pi]
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

        # Simple proportional control for steering
        steer = np.clip(heading_error * 0.5, -1.0, 1.0)

        # Compute throttle/brake from speed error
        current_speed = np.linalg.norm(np.array(state.ego.velocity_mps))
        target_speed = self._current_trajectory.target_speed
        speed_error = target_speed - current_speed

        # Throttle if too slow, brake if too fast
        if speed_error > 1.0:
            throttle = np.clip(speed_error / 10.0, 0.0, 1.0)
            brake = 0.0
        elif speed_error < -1.0:
            throttle = 0.0
            brake = np.clip(-speed_error / 10.0, 0.0, 1.0)
        else:
            throttle = 0.3  # Maintain speed
            brake = 0.0

        return EgoControl(throttle=throttle, steer=steer, brake=brake)

    def save(self, path: str):
        """Save model to disk.

        Args:
            path: Path to save checkpoint
        """
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'episode_count': self._episode_count,
        }, path)

    def load(self, path: str):
        """Load model from disk.

        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.config.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'episode_count' in checkpoint:
            self._episode_count = checkpoint['episode_count']


__all__ = ['PPOPlanner']
