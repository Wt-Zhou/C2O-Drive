"""DQN Agent implementation."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Dict, Any

from c2o_drive.algorithms.dqn.config import DQNConfig
from c2o_drive.algorithms.dqn.network import QNetwork, DuelingQNetwork
from c2o_drive.algorithms.dqn.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from c2o_drive.core.types import WorldState, EgoControl


class DQNAgent:
    """Deep Q-Network agent for autonomous driving.

    Implements the DQN algorithm with experience replay and target network.
    """

    def __init__(self, config: DQNConfig, use_dueling: bool = True, use_prioritized: bool = False):
        """Initialize DQN agent.

        Args:
            config: DQN configuration
            use_dueling: Whether to use dueling architecture
            use_prioritized: Whether to use prioritized replay
        """
        self.config = config
        self.device = torch.device(config.device)

        # Create Q-networks
        network_class = DuelingQNetwork if use_dueling else QNetwork
        self.q_network = network_class(
            config.state_dim, config.action_dim, config.hidden_dims
        ).to(self.device)
        self.target_network = network_class(
            config.state_dim, config.action_dim, config.hidden_dims
        ).to(self.device)

        # Initialize target network
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Create optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate)

        # Create replay buffer
        if use_prioritized:
            self.replay_buffer = PrioritizedReplayBuffer(config.buffer_size)
        else:
            self.replay_buffer = ReplayBuffer(config.buffer_size)
        self.use_prioritized = use_prioritized

        # Exploration parameters
        self.epsilon = config.epsilon_start
        self.epsilon_end = config.epsilon_end
        self.epsilon_decay = config.epsilon_decay

        # Training counters
        self.step_count = 0
        self.update_count = 0

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy.

        Args:
            state: Current state
            training: Whether in training mode (enables exploration)

        Returns:
            Selected action index
        """
        # Epsilon-greedy exploration
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.config.action_dim)

        # Exploitation: select best action
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            action = q_values.argmax(dim=1).item()

        return action

    def act(self, world_state: WorldState) -> EgoControl:
        """Select control action for driving.

        Args:
            world_state: Current world state

        Returns:
            Control commands for the ego vehicle
        """
        # Convert world state to feature vector
        state = self._extract_state_features(world_state)

        # Select discrete action
        action_idx = self.select_action(state, training=False)

        # Convert discrete action to continuous control
        control = self._action_to_control(action_idx)

        return control

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
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

        # Decay epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon -= (self.config.epsilon_start - self.epsilon_end) / self.epsilon_decay

    def train_step(self) -> Dict[str, float]:
        """Perform one training step.

        Returns:
            Dictionary of training metrics
        """
        if not self.replay_buffer.is_ready(self.config.batch_size):
            return {}

        # Sample batch
        if self.use_prioritized:
            states, actions, rewards, next_states, dones, weights, indices = self.replay_buffer.sample(
                self.config.batch_size
            )
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(
                self.config.batch_size
            )
            weights = torch.ones(self.config.batch_size).to(self.device)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Compute current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + self.config.gamma * next_q_values * (1 - dones)

        # Compute loss
        td_errors = target_q_values - current_q_values
        loss = (weights * td_errors.pow(2)).mean()

        # Update priorities if using prioritized replay
        if self.use_prioritized:
            priorities = td_errors.detach().cpu().numpy()
            self.replay_buffer.update_priorities(indices, np.abs(priorities))

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10)
        self.optimizer.step()

        # Update target network
        self.update_count += 1
        if self.update_count % self.config.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return {
            "loss": loss.item(),
            "q_mean": current_q_values.mean().item(),
            "epsilon": self.epsilon,
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
        features.extend([
            ego.position_m[0] / 100.0,  # Normalized x position
            ego.position_m[1] / 100.0,  # Normalized y position
            ego.speed_mps / 30.0,       # Normalized speed
            np.cos(ego.heading_rad),    # Heading cosine
            np.sin(ego.heading_rad),    # Heading sine
        ])

        # Agent features (nearest N agents)
        max_agents = 10
        for i, agent in enumerate(world_state.agents[:max_agents]):
            rel_x = (agent.position_m[0] - ego.position_m[0]) / 100.0
            rel_y = (agent.position_m[1] - ego.position_m[1]) / 100.0
            features.extend([
                rel_x,
                rel_y,
                agent.speed_mps / 30.0,
                np.cos(agent.heading_rad),
                np.sin(agent.heading_rad),
            ])

        # Pad with zeros if fewer agents
        while len(features) < self.config.state_dim:
            features.append(0.0)

        return np.array(features[:self.config.state_dim], dtype=np.float32)

    def _action_to_control(self, action_idx: int) -> EgoControl:
        """Convert discrete action index to continuous control.

        Args:
            action_idx: Discrete action index (0-8 for 3x3 grid)

        Returns:
            Continuous control commands
        """
        # 3x3 action grid: (steer, throttle/brake)
        steer_idx = action_idx % 3
        accel_idx = action_idx // 3

        # Map to continuous values
        steer_values = [-0.3, 0.0, 0.3]
        accel_values = [-1.0, 0.0, 0.5]  # brake, coast, throttle

        steer = steer_values[steer_idx]
        accel = accel_values[accel_idx]

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
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
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
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.step_count = checkpoint["step_count"]
        self.update_count = checkpoint["update_count"]