"""Replay buffer for SAC training."""

import numpy as np
import random
from collections import deque
from typing import Tuple


class ReplayBuffer:
    """Experience replay buffer for SAC.

    Stores continuous action transitions and provides random sampling for training.
    """

    def __init__(self, capacity: int, state_dim: int, action_dim: int, seed: int = 42):
        """Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions to store
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            seed: Random seed for sampling
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ptr = 0
        self.size = 0

        # Pre-allocate arrays for efficiency
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

        random.seed(seed)
        np.random.seed(seed)

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
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
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        indices = np.random.randint(0, self.size, size=batch_size)

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )

    def __len__(self) -> int:
        """Return current size of the buffer."""
        return self.size

    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples for training.

        Args:
            batch_size: Required batch size

        Returns:
            True if buffer has at least batch_size samples
        """
        return self.size >= batch_size

    def clear(self):
        """Clear the replay buffer."""
        self.ptr = 0
        self.size = 0

    def save(self, path: str):
        """Save buffer to file.

        Args:
            path: Path to save file
        """
        np.savez(
            path,
            states=self.states[:self.size],
            actions=self.actions[:self.size],
            rewards=self.rewards[:self.size],
            next_states=self.next_states[:self.size],
            dones=self.dones[:self.size],
        )

    def load(self, path: str):
        """Load buffer from file.

        Args:
            path: Path to saved file
        """
        data = np.load(path)
        self.size = len(data["states"])
        self.states[:self.size] = data["states"]
        self.actions[:self.size] = data["actions"]
        self.rewards[:self.size] = data["rewards"]
        self.next_states[:self.size] = data["next_states"]
        self.dones[:self.size] = data["dones"]
        self.ptr = self.size % self.capacity