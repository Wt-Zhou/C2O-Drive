"""SAC Planner implementation for discrete action spaces.

This module provides the SACPlanner class that adapts SAC (Soft Actor-Critic)
for discrete lattice-based trajectory planning, following the EpisodicAlgorithmPlanner
interface for consistency with C2OSR and PPO.
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from copy import deepcopy

from c2o_drive.algorithms.base import EpisodicAlgorithmPlanner
from c2o_drive.core.types import WorldState, EgoControl
from c2o_drive.core.planner import Transition, UpdateMetrics
from c2o_drive.algorithms.sac.config import SACConfig
from c2o_drive.algorithms.sac.discrete_network import CategoricalActor, DiscreteQCritic
from c2o_drive.utils.lattice_planner import LatticePlanner
from c2o_drive.utils.state_encoder import UnifiedStateEncoder


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    """Soft update of target network parameters.

    θ_target = τ * θ_source + (1 - τ) * θ_target

    Args:
        target: Target network to update
        source: Source network to copy from
        tau: Soft update coefficient (typically 0.005)
    """
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


class DiscreteReplayBuffer:
    """Replay buffer for discrete action SAC.

    Stores transitions with discrete action indices instead of continuous actions.
    """

    def __init__(self, capacity: int, state_dim: int, seed: int = 42):
        """Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions
            state_dim: State feature dimension
            seed: Random seed
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.ptr = 0
        self.size = 0

        # Pre-allocate arrays (actions are discrete indices)
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)  # Discrete action indices
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        np.random.seed(seed)

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        """Add transition to buffer."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: str = 'cpu'):
        """Sample batch of transitions as torch tensors.

        Returns:
            Dict with keys: states, actions, rewards, next_states, dones
        """
        indices = np.random.randint(0, self.size, size=batch_size)

        return {
            'states': torch.FloatTensor(self.states[indices]).to(device),
            'actions': torch.LongTensor(self.actions[indices]).to(device),
            'rewards': torch.FloatTensor(self.rewards[indices]).to(device),
            'next_states': torch.FloatTensor(self.next_states[indices]).to(device),
            'dones': torch.FloatTensor(self.dones[indices]).to(device),
        }

    def __len__(self) -> int:
        return self.size


class SACPlanner(EpisodicAlgorithmPlanner[WorldState, EgoControl]):
    """SAC Planner for discrete lattice-based action spaces.

    This planner adapts SAC (Soft Actor-Critic) for discrete trajectory planning:
    - Uses Categorical policy instead of Gaussian
    - Uses discrete Q-networks
    - Preserves SAC advantages: double Q-learning, entropy regularization, automatic temperature
    - Follows EpisodicAlgorithmPlanner interface (like PPO/C2OSR)

    Workflow:
    1. Episode start: Actor selects discrete action → Generate trajectory from lattice
    2. Execute trajectory step-by-step
    3. Store transitions to replay buffer
    4. When trajectory ends: Perform SAC update

    Attributes:
        config: SAC configuration
        lattice_planner: Lattice trajectory generator
        state_encoder: Unified state feature encoder
        actor: Categorical policy network
        critic1, critic2: Double Q-networks
        critic1_target, critic2_target: Target Q-networks
        log_alpha: Learnable entropy temperature (log scale)
        replay_buffer: Experience replay buffer
    """

    def __init__(self, config: SACConfig):
        """Initialize SAC planner.

        Args:
            config: SAC configuration
        """
        super().__init__(config)
        self.config = config

        # Lattice planner (same as C2OSR/PPO)
        self.lattice_planner = LatticePlanner(
            lateral_offsets=config.lattice.lateral_offsets,
            speed_variations=config.lattice.speed_variations,
            num_trajectories=max(config.lattice.num_trajectories, config.action_dim),
        )

        # Build action space mapping
        self._build_action_space()

        # State encoder (unified across all algorithms)
        self.state_encoder = UnifiedStateEncoder()

        # Actor network (Categorical policy)
        self.actor = CategoricalActor(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dims=config.hidden_dims,
        ).to(config.device)

        # Double Q-networks (SAC signature feature)
        self.critic1 = DiscreteQCritic(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dims=config.hidden_dims,
        ).to(config.device)

        self.critic2 = DiscreteQCritic(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dims=config.hidden_dims,
        ).to(config.device)

        # Target networks (for stable learning)
        self.critic1_target = deepcopy(self.critic1)
        self.critic2_target = deepcopy(self.critic2)

        # Freeze target networks (no gradient)
        for param in self.critic1_target.parameters():
            param.requires_grad = False
        for param in self.critic2_target.parameters():
            param.requires_grad = False

        # Automatic entropy temperature adjustment
        self.log_alpha = nn.Parameter(
            torch.tensor(np.log(config.initial_alpha), dtype=torch.float32, device=config.device)
        )
        self.target_entropy = config.target_entropy

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=config.critic_lr
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.alpha_lr)

        # Replay buffer
        self.replay_buffer = DiscreteReplayBuffer(
            capacity=config.buffer_size,
            state_dim=config.state_dim,
        )

        # Episode state tracking
        self._current_trajectory = None
        self._trajectory_step = 0
        self._last_action_idx = None
        self._last_state_features = None

        # Statistics
        self._episode_count = 0
        self._update_count = 0

    def _build_action_space(self):
        """Build action space mapping (same as PPO)."""
        self.action_mapping = []
        for lateral in self.config.lattice.lateral_offsets:
            for speed in self.config.lattice.speed_variations:
                self.action_mapping.append((lateral, speed))

        assert len(self.action_mapping) == self.config.action_dim, (
            f"Action mapping size {len(self.action_mapping)} != config.action_dim {self.config.action_dim}"
        )

    @property
    def alpha(self) -> float:
        """Current entropy temperature value."""
        return self.log_alpha.exp().item()

    def select_action(
        self,
        observation: WorldState,
        deterministic: bool = False,
        reference_path: Optional[List[Tuple[float, float]]] = None,
        **kwargs
    ) -> EgoControl:
        """Select action following BasePlanner interface.

        At episode start:
        - Encode state → Actor outputs action probabilities
        - Sample discrete action → Generate trajectory from lattice

        During episode:
        - Return next control from current trajectory

        Args:
            observation: Current world state
            deterministic: If True, select argmax action
            reference_path: Reference path for trajectory generation
            **kwargs: Additional arguments

        Returns:
            Control command for current step
        """
        # Episode start: generate trajectory
        if self._current_trajectory is None:
            # 1. Encode state features
            state_features = self.state_encoder.encode(observation)
            state_tensor = torch.FloatTensor(state_features).to(self.config.device)

            # 2. Actor network outputs action distribution
            with torch.no_grad():
                action_probs = self.actor(state_tensor)

                if deterministic:
                    action_idx = torch.argmax(action_probs).item()
                else:
                    dist = Categorical(probs=action_probs)
                    action_idx = dist.sample().item()

            # 3. Generate candidate trajectories
            if reference_path is None:
                # Simple forward reference path if none provided
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

            # 4. Select trajectory by action index
            if action_idx < len(candidate_trajectories):
                self._current_trajectory = candidate_trajectories[action_idx]
            else:
                # Fallback
                self._current_trajectory = candidate_trajectories[0] if candidate_trajectories else None

            self._trajectory_step = 0

            # Store for update
            self._last_action_idx = action_idx
            self._last_state_features = state_features

        # Get current step control from trajectory
        if self._current_trajectory is None:
            # Safety: no trajectory, apply brake
            return EgoControl(throttle=0.0, steer=0.0, brake=1.0)

        if self._trajectory_step >= len(self._current_trajectory.waypoints):
            # Safety: trajectory exhausted
            return EgoControl(throttle=0.0, steer=0.0, brake=1.0)

        waypoint = self._current_trajectory.waypoints[self._trajectory_step]
        control = self._waypoint_to_control(observation, waypoint, self._trajectory_step)

        return control

    def update(self, transition: Transition[WorldState, EgoControl]) -> UpdateMetrics:
        """Update learning following BasePlanner interface.

        Flow:
        1. Encode next state
        2. Store transition to replay buffer
        3. Step trajectory counter
        4. If trajectory finished: Reset and perform SAC update

        Args:
            transition: State transition data

        Returns:
            Update metrics (loss, entropy, etc.)
        """
        # Encode next state
        next_state_features = self.state_encoder.encode(transition.next_state)

        # Store to replay buffer
        self.replay_buffer.push(
            state=self._last_state_features,
            action=self._last_action_idx,
            reward=transition.reward,
            next_state=next_state_features,
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

            # Perform SAC update if buffer is sufficient
            if len(self.replay_buffer) >= self.config.batch_size:
                sac_metrics = self._sac_update()
                return UpdateMetrics(
                    loss=sac_metrics.get('critic_loss'),
                    policy_entropy=sac_metrics.get('entropy'),
                    custom=sac_metrics,
                )

        return UpdateMetrics()

    def _sac_update(self) -> Dict[str, float]:
        """Execute SAC update (core algorithm).

        SAC update steps:
        1. Critic update: Minimize Bellman error with target networks
        2. Actor update: Maximize expected Q-value minus entropy
        3. Alpha update: Adjust temperature to match target entropy
        4. Soft update target networks

        Returns:
            Dictionary of training metrics
        """
        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(self.config.batch_size, self.config.device)

        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']

        # === 1. Critic Update ===
        with torch.no_grad():
            # Next action probabilities from current policy
            next_action_probs = self.actor(next_states)  # (batch, action_dim)

            # Target Q-values (double Q-learning)
            next_q1 = self.critic1_target(next_states)  # (batch, action_dim)
            next_q2 = self.critic2_target(next_states)
            next_q = torch.min(next_q1, next_q2)

            # Expected Q-value with entropy bonus (SAC signature)
            # V(s') = E[Q(s', a') - α * log π(a'|s')]
            next_value = (next_action_probs * (next_q - self.alpha * torch.log(next_action_probs + 1e-8))).sum(dim=1)

            # Target Q-value: r + γ * (1 - done) * V(s')
            target_q = rewards + self.config.gamma * (1 - dones) * next_value

        # Current Q-values for taken actions
        current_q1 = self.critic1.get_q_value(states, actions)
        current_q2 = self.critic2.get_q_value(states, actions)

        # Critic loss (MSE)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        # Optimize critics
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # === 2. Actor Update ===
        # Get current action probabilities
        action_probs = self.actor(states)  # (batch, action_dim)
        log_probs = torch.log(action_probs + 1e-8)

        # Q-values from both critics
        q1 = self.critic1(states)
        q2 = self.critic2(states)
        q = torch.min(q1, q2)

        # Policy loss: Maximize E[Q(s, a) - α * log π(a|s)]
        # Equivalent to minimizing: E[α * log π(a|s) - Q(s, a)]
        actor_loss = (action_probs * (self.alpha * log_probs - q)).sum(dim=1).mean()

        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # === 3. Temperature Update ===
        with torch.no_grad():
            # Re-compute action probs (detached from actor update)
            action_probs = self.actor(states)

        log_probs = torch.log(action_probs + 1e-8)
        entropy = -(action_probs * log_probs).sum(dim=1).mean()

        # Alpha loss: Adjust temperature to match target entropy
        alpha_loss = -(self.log_alpha * (entropy - self.target_entropy).detach()).mean()

        # Optimize alpha
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # === 4. Soft Update Target Networks ===
        soft_update(self.critic1_target, self.critic1, self.config.tau)
        soft_update(self.critic2_target, self.critic2, self.config.tau)

        self._update_count += 1

        # Return metrics
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': self.alpha,
            'entropy': entropy.item(),
            'target_entropy': self.target_entropy,
            'mean_q': q.mean().item(),
        }

    def reset(self) -> None:
        """Reset episode state."""
        self._current_trajectory = None
        self._trajectory_step = 0
        self._last_action_idx = None
        self._last_state_features = None

    def plan_trajectory(
        self,
        observation: WorldState,
        horizon: int,
        reference_path: Optional[List[Tuple[float, float]]] = None,
        **kwargs
    ) -> List[EgoControl]:
        """Plan complete trajectory (EpisodicPlanner interface)."""
        # Generate trajectory
        control = self.select_action(observation, deterministic=True, reference_path=reference_path)

        if self._current_trajectory is None:
            return [control]

        # Return entire trajectory as control sequence
        controls = []
        for i, waypoint in enumerate(self._current_trajectory.waypoints):
            controls.append(self._waypoint_to_control(observation, waypoint, i))

        return controls

    def _waypoint_to_control(
        self,
        state: WorldState,
        waypoint: Tuple[float, float],
        step_idx: int
    ) -> EgoControl:
        """Convert waypoint to control (same as PPO)."""
        if self._current_trajectory is None:
            return EgoControl(throttle=0.0, steer=0.0, brake=1.0)

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

        # Normalize to [-pi, pi]
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

        # Proportional steering control
        steer = np.clip(heading_error * 0.5, -1.0, 1.0)

        # Compute throttle/brake from speed error
        current_speed = np.linalg.norm(np.array(state.ego.velocity_mps))
        target_speed = self._current_trajectory.target_speed
        speed_error = target_speed - current_speed

        if speed_error > 1.0:
            throttle = np.clip(speed_error / 10.0, 0.0, 1.0)
            brake = 0.0
        elif speed_error < -1.0:
            throttle = 0.0
            brake = np.clip(-speed_error / 10.0, 0.0, 1.0)
        else:
            throttle = 0.3
            brake = 0.0

        return EgoControl(throttle=throttle, steer=steer, brake=brake)

    def save(self, path: str):
        """Save model to disk."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'critic1_target_state_dict': self.critic1_target.state_dict(),
            'critic2_target_state_dict': self.critic2_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
            'config': self.config,
            'episode_count': self._episode_count,
            'update_count': self._update_count,
        }, path)

    def load(self, path: str):
        """Load model from disk."""
        checkpoint = torch.load(path, map_location=self.config.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.critic1_target.load_state_dict(checkpoint['critic1_target_state_dict'])
        self.critic2_target.load_state_dict(checkpoint['critic2_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        self.log_alpha.data = checkpoint['log_alpha'].data
        if 'episode_count' in checkpoint:
            self._episode_count = checkpoint['episode_count']
        if 'update_count' in checkpoint:
            self._update_count = checkpoint['update_count']


__all__ = ['SACPlanner']
