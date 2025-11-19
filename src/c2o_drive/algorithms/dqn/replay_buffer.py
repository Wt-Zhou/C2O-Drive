"""Replay buffer for DQN training."""

import numpy as np
import random
from collections import deque
from typing import Tuple, List


class ReplayBuffer:
    """Experience replay buffer for DQN.

    Stores transitions and provides random sampling for training.
    """

    def __init__(self, capacity: int, seed: int = 42):
        """Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions to store
            seed: Random seed for sampling
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        random.seed(seed)
        np.random.seed(seed)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Add a transition to the buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)

        states = np.array([t[0] for t in batch], dtype=np.float32)
        actions = np.array([t[1] for t in batch], dtype=np.int64)
        rewards = np.array([t[2] for t in batch], dtype=np.float32)
        next_states = np.array([t[3] for t in batch], dtype=np.float32)
        dones = np.array([t[4] for t in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """Return current size of the buffer."""
        return len(self.buffer)

    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples for training.

        Args:
            batch_size: Required batch size

        Returns:
            True if buffer has at least batch_size samples
        """
        return len(self.buffer) >= batch_size


class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer.

    Samples transitions based on their TD error (importance).
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 1e-4,
        seed: int = 42,
    ):
        """Initialize prioritized replay buffer.

        Args:
            capacity: Maximum number of transitions to store
            alpha: Prioritization exponent
            beta: Importance sampling exponent
            beta_increment: Beta annealing increment per sample
            seed: Random seed
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_priority = 1.0

        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)

        random.seed(seed)
        np.random.seed(seed)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Add a transition with maximum priority.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(self.max_priority)

    def sample(
        self, batch_size: int
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Sample a prioritized batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones, weights, indices)
        """
        # Calculate sampling probabilities
        priorities = np.array(self.priorities, dtype=np.float32)
        probs = priorities ** self.alpha
        probs /= probs.sum()

        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)

        # Calculate importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        # Anneal beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        # Extract batch
        batch = [self.buffer[idx] for idx in indices]

        states = np.array([t[0] for t in batch], dtype=np.float32)
        actions = np.array([t[1] for t in batch], dtype=np.int64)
        rewards = np.array([t[2] for t in batch], dtype=np.float32)
        next_states = np.array([t[3] for t in batch], dtype=np.float32)
        dones = np.array([t[4] for t in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones, weights, indices

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled transitions.

        Args:
            indices: Indices of sampled transitions
            priorities: New priority values (typically TD errors)
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6  # Small epsilon for stability
            self.max_priority = max(self.max_priority, priority)

    def __len__(self) -> int:
        """Return current size of the buffer."""
        return len(self.buffer)

    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples for training."""
        return len(self.buffer) >= batch_size