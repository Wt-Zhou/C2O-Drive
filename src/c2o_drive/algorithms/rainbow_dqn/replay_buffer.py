"""Prioritized Experience Replay for Rainbow DQN

Implements prioritized replay buffer using Sum Tree data structure for efficient
sampling. This is one of the six Rainbow components.

Reference:
    Schaul et al. "Prioritized Experience Replay" (2015)
"""

from typing import List, Tuple, Any
import random
import numpy as np
from dataclasses import dataclass


@dataclass
class ReplayTransition:
    """Single transition in replay buffer.

    Attributes:
        state: Current state (WorldState)
        action: Action taken (trajectory index)
        reward: Reward received
        next_state: Next state (WorldState)
        done: Whether episode terminated
    """
    state: Any
    action: int
    reward: float
    next_state: Any
    done: bool


class SumTree:
    """Sum Tree data structure for efficient prioritized sampling.

    A binary tree where each node contains the sum of its children's values.
    The root contains the sum of all priorities, enabling O(log n) sampling.

    Structure:
        - Tree array of size 2*capacity-1
        - First capacity-1 elements are internal nodes
        - Last capacity elements are leaf nodes (priorities)
        - Data array stores actual transitions

    Attributes:
        capacity: Maximum number of transitions
        tree: Priority values in tree structure
        data: Actual transition objects
        write: Current write position
        n_entries: Current number of transitions stored
    """

    def __init__(self, capacity: int):
        """Initialize Sum Tree.

        Args:
            capacity: Maximum number of transitions to store
        """
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx: int, change: float):
        """Propagate priority change up the tree.

        Args:
            idx: Tree index to start propagation from
            change: Change in priority value
        """
        parent = (idx - 1) // 2
        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        """Retrieve leaf index for given cumulative priority value.

        Args:
            idx: Current tree node index
            s: Cumulative priority value to search for

        Returns:
            Index of leaf node containing the priority
        """
        left = 2 * idx + 1
        right = left + 1

        # Reached leaf node
        if left >= len(self.tree):
            return idx

        # Traverse left or right based on cumulative priority
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        """Get total sum of all priorities.

        Returns:
            Total priority (value at root node)
        """
        return self.tree[0]

    def add(self, priority: float, data: Any):
        """Add new transition with given priority.

        Args:
            priority: Priority value for this transition
            data: Transition object to store
        """
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)

        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx: int, priority: float):
        """Update priority of a transition.

        Args:
            idx: Tree index to update
            priority: New priority value
        """
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s: float) -> Tuple[int, float, Any]:
        """Sample transition with cumulative priority s.

        Args:
            s: Cumulative priority value

        Returns:
            Tuple of (tree_index, priority, data)
        """
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

    def max(self) -> float:
        """Get maximum priority in the tree.

        Returns:
            Maximum priority value among all transitions
        """
        return np.max(self.tree[-self.capacity:])

    def __len__(self) -> int:
        """Get number of transitions currently stored.

        Returns:
            Number of valid transitions
        """
        return self.n_entries


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer.

    Samples transitions with probability proportional to their TD error.
    Uses importance sampling weights to correct for bias introduced by
    non-uniform sampling.

    Attributes:
        tree: SumTree for efficient priority-based sampling
        alpha: Prioritization exponent (0=uniform, 1=full prioritization)
        beta: Importance sampling exponent (annealed from beta_start to 1.0)
        beta_start: Initial value of beta
        beta_frames: Number of frames to anneal beta to 1.0
        frame: Current frame count (for beta annealing)
        epsilon: Small constant to prevent zero priorities
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000
    ):
        """Initialize prioritized replay buffer.

        Args:
            capacity: Maximum buffer size
            alpha: Prioritization exponent
            beta_start: Initial importance sampling exponent
            beta_frames: Number of frames to anneal beta to 1.0
        """
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 0
        self.epsilon = 1e-6  # Small constant to avoid zero priorities

    def push(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        done: bool
    ):
        """Add transition to buffer with maximum priority.

        New transitions are added with maximum priority to ensure
        they are sampled at least once.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode terminated
        """
        transition = ReplayTransition(state, action, reward, next_state, done)

        # Use maximum priority for new transitions
        max_priority = self.tree.max()
        if max_priority == 0:
            max_priority = 1.0

        self.tree.add(max_priority, transition)

    def sample(self, batch_size: int) -> Tuple[List[ReplayTransition], List[int], np.ndarray]:
        """Sample batch of transitions with importance sampling weights.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple of (batch, indices, weights)
            - batch: List of sampled transitions
            - indices: Tree indices for priority updates
            - weights: Importance sampling weights
        """
        batch = []
        indices = []
        priorities = []

        # Proportional prioritized sampling
        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)

            idx, priority, data = self.tree.get(s)

            batch.append(data)
            indices.append(idx)
            priorities.append(priority)

        # Compute importance sampling weights
        self.frame += 1
        beta = self._get_beta()

        sampling_probs = np.array(priorities) / self.tree.total()
        weights = (len(self.tree) * sampling_probs) ** (-beta)
        weights /= weights.max()  # Normalize for stability

        return batch, indices, weights

    def update_priorities(self, indices: List[int], td_errors: np.ndarray):
        """Update priorities based on TD errors.

        Args:
            indices: Tree indices of transitions to update
            td_errors: TD errors for each transition
        """
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)

    def _get_beta(self) -> float:
        """Get current beta value (linearly annealed).

        Returns:
            Current beta value
        """
        return min(1.0, self.beta_start + (1.0 - self.beta_start) * self.frame / self.beta_frames)

    def __len__(self) -> int:
        """Get current buffer size.

        Returns:
            Number of transitions in buffer
        """
        return len(self.tree)
