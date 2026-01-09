"""Replay buffer module for RCRL algorithm.

This module implements experience replay for off-policy DQN learning.
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import torch
import random


@dataclass
class RCRLTransition:
    """RCRL transition dataclass.

    Attributes:
        state: Encoded state features [state_dim]
        action: Selected action index
        reward: Immediate reward
        next_state: Next state features [state_dim]
        terminated: Whether episode terminated
        truncated: Whether episode was truncated
    """

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    terminated: bool
    truncated: bool


class ReplayBuffer:
    """Simple circular replay buffer for DQN.

    Stores transitions and supports uniform random sampling.
    """

    def __init__(self, capacity: int):
        """Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions to store
        """
        self.capacity = capacity
        self.buffer: List[RCRLTransition] = []
        self.position = 0

    def push(self, transition: RCRLTransition) -> None:
        """Add transition to buffer.

        Args:
            transition: Transition to add
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition

        self.position = (self.position + 1) % self.capacity

    def sample(
        self, batch_size: int, device: str = "cpu"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample random batch of transitions.

        Args:
            batch_size: Number of transitions to sample
            device: Device to place tensors on

        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
            - states: [batch, state_dim]
            - actions: [batch]
            - rewards: [batch]
            - next_states: [batch, state_dim]
            - dones: [batch] (terminated or truncated)
        """
        batch = random.sample(self.buffer, batch_size)

        states = np.stack([t.state for t in batch])
        actions = np.array([t.action for t in batch])
        rewards = np.array([t.reward for t in batch])
        next_states = np.stack([t.next_state for t in batch])
        dones = np.array([t.terminated or t.truncated for t in batch])

        return (
            torch.FloatTensor(states).to(device),
            torch.LongTensor(actions).to(device),
            torch.FloatTensor(rewards).to(device),
            torch.FloatTensor(next_states).to(device),
            torch.FloatTensor(dones).to(device),
        )

    def __len__(self) -> int:
        """Get current buffer size.

        Returns:
            Number of transitions in buffer
        """
        return len(self.buffer)

    def clear(self) -> None:
        """Clear all transitions from buffer."""
        self.buffer.clear()
        self.position = 0


class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer.

    Samples transitions with probability proportional to their TD error.
    Provides importance sampling weights for unbiased learning.
    """

    def __init__(self, capacity: int, alpha: float = 0.6, beta_start: float = 0.4):
        """Initialize prioritized replay buffer.

        Args:
            capacity: Maximum number of transitions to store
            alpha: Priority exponent (0 = uniform, 1 = fully prioritized)
            beta_start: Initial importance sampling exponent
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.buffer: List[RCRLTransition] = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0

    def push(self, transition: RCRLTransition, priority: float = None) -> None:
        """Add transition with priority.

        Args:
            transition: Transition to add
            priority: Optional priority value (uses max if None)
        """
        if priority is None:
            priority = self.max_priority

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition

        self.priorities[self.position] = priority
        self.max_priority = max(self.max_priority, priority)

        self.position = (self.position + 1) % self.capacity

    def sample(
        self, batch_size: int, device: str = "cpu", beta: float = None
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray
    ]:
        """Sample batch with importance sampling weights.

        Args:
            batch_size: Number of transitions to sample
            device: Device to place tensors on
            beta: Importance sampling exponent (uses self.beta if None)

        Returns:
            Tuple of (states, actions, rewards, next_states, dones, weights, indices)
            - states: [batch, state_dim]
            - actions: [batch]
            - rewards: [batch]
            - next_states: [batch, state_dim]
            - dones: [batch]
            - weights: [batch] importance sampling weights
            - indices: [batch] buffer indices for priority updates
        """
        if beta is None:
            beta = self.beta

        buffer_size = len(self.buffer)

        # Compute sampling probabilities
        priorities = self.priorities[:buffer_size]
        probs = priorities ** self.alpha
        probs /= probs.sum()

        # Sample indices
        indices = np.random.choice(buffer_size, batch_size, p=probs, replace=False)

        # Compute importance sampling weights
        weights = (buffer_size * probs[indices]) ** (-beta)
        weights /= weights.max()  # Normalize by max for stability

        # Extract transitions
        batch = [self.buffer[idx] for idx in indices]

        states = np.stack([t.state for t in batch])
        actions = np.array([t.action for t in batch])
        rewards = np.array([t.reward for t in batch])
        next_states = np.stack([t.next_state for t in batch])
        dones = np.array([t.terminated or t.truncated for t in batch])

        return (
            torch.FloatTensor(states).to(device),
            torch.LongTensor(actions).to(device),
            torch.FloatTensor(rewards).to(device),
            torch.FloatTensor(next_states).to(device),
            torch.FloatTensor(dones).to(device),
            torch.FloatTensor(weights).to(device),
            indices,
        )

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """Update priorities for sampled transitions.

        Args:
            indices: Buffer indices
            priorities: New priority values (typically TD errors)
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def __len__(self) -> int:
        """Get current buffer size.

        Returns:
            Number of transitions in buffer
        """
        return len(self.buffer)

    def clear(self) -> None:
        """Clear all transitions from buffer."""
        self.buffer.clear()
        self.priorities.fill(0)
        self.position = 0
        self.max_priority = 1.0


__all__ = ["RCRLTransition", "ReplayBuffer", "PrioritizedReplayBuffer"]
