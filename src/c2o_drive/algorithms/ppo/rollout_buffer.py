"""Rollout buffer for PPO algorithm.

This module provides a rollout buffer with GAE (Generalized Advantage Estimation) support.
"""

from typing import Optional, Dict, Iterator
import torch


class RolloutBuffer:
    """Rollout buffer for PPO with GAE support.

    Stores trajectories and computes advantages using GAE or Monte Carlo returns.

    Attributes:
        capacity: Maximum buffer capacity
        gamma: Discount factor
        gae_lambda: Lambda parameter for GAE (None = use Monte Carlo)
    """

    def __init__(
        self,
        capacity: int,
        gamma: float,
        gae_lambda: Optional[float] = None
    ):
        """Initialize rollout buffer.

        Args:
            capacity: Maximum number of transitions to store
            gamma: Discount factor
            gae_lambda: Lambda for GAE (None = use simple Monte Carlo returns)
        """
        self.capacity = capacity
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # Storage lists
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

        # Computed data (after compute_advantages)
        self.returns = None
        self.advantages = None

    def push(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        value: torch.Tensor,
        log_prob: torch.Tensor,
        done: bool
    ):
        """Store a transition.

        Args:
            state: State tensor
            action: Action index
            reward: Reward value
            value: State value from critic
            log_prob: Log probability of action
            done: Whether episode ended
        """
        # Enforce capacity limit
        if len(self.states) >= self.capacity:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.values.pop(0)
            self.log_probs.pop(0)
            self.dones.pop(0)

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value.item() if isinstance(value, torch.Tensor) else value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def compute_advantages(self):
        """Compute advantages and returns using GAE or Monte Carlo.

        This method must be called before sampling batches.
        """
        if len(self.states) == 0:
            return

        if self.gae_lambda is not None:
            # GAE (Generalized Advantage Estimation)
            advantages = []
            gae = 0.0

            for t in reversed(range(len(self.rewards))):
                if t == len(self.rewards) - 1:
                    next_value = 0.0
                else:
                    next_value = self.values[t + 1]

                # TD error: delta = r + gamma * V(s') - V(s)
                delta = (
                    self.rewards[t]
                    + self.gamma * next_value * (1.0 - float(self.dones[t]))
                    - self.values[t]
                )

                # GAE: A_t = sum_{l=0}^{inf} (gamma * lambda)^l * delta_{t+l}
                gae = delta + self.gamma * self.gae_lambda * (1.0 - float(self.dones[t])) * gae
                advantages.insert(0, gae)

            self.advantages = torch.tensor(advantages, dtype=torch.float32)
            self.returns = self.advantages + torch.tensor(self.values, dtype=torch.float32)

        else:
            # Simple Monte Carlo returns
            returns = []
            R = 0.0

            for r, done in zip(reversed(self.rewards), reversed(self.dones)):
                R = r + self.gamma * R * (1.0 - float(done))
                returns.insert(0, R)

            self.returns = torch.tensor(returns, dtype=torch.float32)
            self.advantages = self.returns - torch.tensor(self.values, dtype=torch.float32)

    def sample_batches(self, batch_size: int) -> Iterator[Dict[str, torch.Tensor]]:
        """Generate mini-batches for training.

        Args:
            batch_size: Size of each mini-batch

        Yields:
            Dictionary containing batch data:
                - 'states': State tensors
                - 'actions': Action indices
                - 'old_log_probs': Log probabilities from buffer
                - 'returns': Computed returns
                - 'advantages': Computed advantages
        """
        if self.returns is None or self.advantages is None:
            raise RuntimeError("Must call compute_advantages() before sampling")

        # Random permutation of indices
        indices = torch.randperm(len(self.states))

        # Generate batches
        for start_idx in range(0, len(self.states), batch_size):
            end_idx = min(start_idx + batch_size, len(self.states))
            batch_indices = indices[start_idx:end_idx]

            yield {
                'states': torch.stack([self.states[i] for i in batch_indices]),
                'actions': torch.tensor([self.actions[i] for i in batch_indices], dtype=torch.long),
                'old_log_probs': torch.stack([self.log_probs[i] for i in batch_indices]),
                'returns': self.returns[batch_indices],
                'advantages': self.advantages[batch_indices],
            }

    def clear(self):
        """Clear all stored data."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.returns = None
        self.advantages = None

    def __len__(self) -> int:
        """Return number of stored transitions."""
        return len(self.states)


__all__ = ['RolloutBuffer']
