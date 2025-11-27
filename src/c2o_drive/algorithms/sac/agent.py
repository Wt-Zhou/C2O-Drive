"""SAC Agent implementation."""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any, Tuple

from c2o_drive.algorithms.sac.config import SACConfig
from c2o_drive.algorithms.sac.network import Actor, Critic
from c2o_drive.algorithms.sac.replay_buffer import ReplayBuffer
from c2o_drive.core.types import WorldState, EgoControl


class SACAgent:
    """Soft Actor-Critic agent for autonomous driving.

    Implements the SAC algorithm with automatic entropy tuning.
    """

    def __init__(self, config: SACConfig):
        """Initialize SAC agent.

        Args:
            config: SAC configuration
        """
        self.config = config
        self.device = torch.device(config.device)

        # Create actor network
        self.actor = Actor(
            config.state_dim,
            config.action_dim,
            config.hidden_dims,
            config.max_action,
        ).to(self.device)

        # Create critic networks
        self.critic = Critic(
            config.state_dim,
            config.action_dim,
            config.hidden_dims,
        ).to(self.device)

        self.critic_target = Critic(
            config.state_dim,
            config.action_dim,
            config.hidden_dims,
        ).to(self.device)

        # Initialize target network
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Create optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=config.actor_lr
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=config.critic_lr
        )

        # Automatic entropy tuning
        self.target_entropy = config.target_entropy
        self.log_alpha = torch.tensor(
            np.log(config.initial_alpha),
            dtype=torch.float32,
            requires_grad=True,
            device=self.device,
        )
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.alpha_lr)

        # Create replay buffer
        self.replay_buffer = ReplayBuffer(
            config.buffer_size,
            config.state_dim,
            config.action_dim,
        )

        # Training counters
        self.step_count = 0
        self.update_count = 0

    @property
    def alpha(self) -> float:
        """Get current entropy coefficient."""
        return self.log_alpha.exp().item()

    def select_action(
        self, state: np.ndarray, training: bool = True
    ) -> np.ndarray:
        """Select action using the current policy.

        Args:
            state: Current state
            training: Whether in training mode (enables exploration)

        Returns:
            Selected action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        if training:
            action, _ = self.actor.sample(state_tensor, deterministic=False)
        else:
            action, _ = self.actor.sample(state_tensor, deterministic=True)

        return action.detach().cpu().numpy().squeeze()

    def select_lattice_params(
        self, state: np.ndarray, training: bool = True
    ) -> Tuple[float, float]:
        """Select lattice parameters (lateral_offset, target_speed).

        Args:
            state: Current state
            training: Whether in training mode (enables exploration)

        Returns:
            Tuple of (lateral_offset in meters, target_speed in m/s)
        """
        # Get normalized action from policy [-1, 1]
        action = self.select_action(state, training=training)

        # Rescale to lattice parameter ranges
        # Assumes action[0] -> lateral_offset, action[1] -> target_speed
        # Note: actual ranges should be configured externally
        lateral_offset = action[0]  # Will be rescaled in training script
        target_speed = action[1]    # Will be rescaled in training script

        return lateral_offset, target_speed

    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Store transition in replay buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.replay_buffer.push(state, action, reward, next_state, done)
        self.step_count += 1

    def train_step(self) -> Dict[str, float]:
        """Perform one training step.

        Returns:
            Dictionary of training metrics
        """
        if not self.replay_buffer.is_ready(self.config.batch_size):
            return {}

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.config.batch_size
        )

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Scale rewards
        rewards = rewards * self.config.reward_scale

        # Update critic
        with torch.no_grad():
            # Sample actions from current policy
            next_actions, next_log_probs = self.actor.sample(next_states)

            # Compute target Q-values
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_value = target_q - self.log_alpha.exp() * next_log_probs.unsqueeze(1)
            target_q_values = rewards + self.config.gamma * (1 - dones) * target_value

        # Compute current Q-values
        current_q1, current_q2 = self.critic(states, actions)

        # Critic loss
        critic_loss = F.mse_loss(current_q1, target_q_values) + F.mse_loss(
            current_q2, target_q_values
        )

        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # Update actor
        new_actions, log_probs = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, new_actions)
        q_new = torch.min(q1_new, q2_new)

        # Actor loss
        actor_loss = (self.log_alpha.exp() * log_probs.unsqueeze(1) - q_new).mean()

        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        # Update alpha (entropy coefficient)
        alpha_loss = -(
            self.log_alpha * (log_probs + self.target_entropy).detach()
        ).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Soft update target network
        self.update_count += 1
        for param, target_param in zip(
            self.critic.parameters(), self.critic_target.parameters()
        ):
            target_param.data.copy_(
                self.config.tau * param.data + (1 - self.config.tau) * target_param.data
            )

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha,
            "q_mean": q_new.mean().item(),
        }

    def _extract_state_features(self, world_state: WorldState) -> np.ndarray:
        """Extract state features from world state.

        Args:
            world_state: Current world state

        Returns:
            Feature vector for the neural network
        """
        features = []

        # Ego vehicle features
        ego = world_state.ego
        # Calculate speed from velocity vector
        ego_speed = np.linalg.norm(ego.velocity_mps)
        features.extend([
            ego.position_m[0] / 100.0,  # Normalized x position
            ego.position_m[1] / 100.0,  # Normalized y position
            ego_speed / 30.0,            # Normalized speed
            np.cos(ego.yaw_rad),         # Heading cosine
            np.sin(ego.yaw_rad),         # Heading sine
        ])

        # Goal features (if available)
        if hasattr(world_state, 'goal'):
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
            # Calculate agent speed from velocity vector
            agent_speed = np.linalg.norm(agent.velocity_mps)
            features.extend([
                rel_x,
                rel_y,
                agent_speed / 30.0,
                np.cos(agent.heading_rad),
                np.sin(agent.heading_rad),
            ])

        # Pad with zeros if fewer agents
        while len(features) < self.config.state_dim:
            features.append(0.0)

        return np.array(features[:self.config.state_dim], dtype=np.float32)

    def _action_to_control(self, action: np.ndarray) -> EgoControl:
        """Convert continuous action to control commands.

        Args:
            action: Continuous action array [steer, accel]

        Returns:
            Control commands for the ego vehicle
        """
        # Action is already bounded to [-max_action, max_action]
        steer = float(action[0])  # Range: [-1, 1]

        if len(action) > 1:
            accel = float(action[1])  # Range: [-1, 1]
        else:
            accel = 0.0

        # Convert acceleration to throttle/brake
        if accel >= 0:
            return EgoControl(throttle=accel, steer=steer, brake=0.0)
        else:
            return EgoControl(throttle=0.0, steer=steer, brake=-accel)

    def save(self, path: str):
        """Save agent state to file.

        Args:
            path: Path to save file
        """
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "alpha_optimizer": self.alpha_optimizer.state_dict(),
                "log_alpha": self.log_alpha,
                "step_count": self.step_count,
                "update_count": self.update_count,
            },
            path,
        )

    def load(self, path: str):
        """Load agent state from file.

        Args:
            path: Path to saved file
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
        self.log_alpha = checkpoint["log_alpha"]
        self.step_count = checkpoint["step_count"]
        self.update_count = checkpoint["update_count"]