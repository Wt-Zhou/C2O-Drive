"""Rainbow DQN Algorithm Implementation

This module implements the Rainbow DQN algorithm as a baseline for comparison
with C2OSR. Rainbow DQN combines six independent improvements to DQN:
1. Double DQN
2. Dueling Network Architecture
3. Prioritized Experience Replay
4. Multi-step Learning
5. Distributional RL (C51)
6. Noisy Nets

The implementation follows the same code structure and abstractions as C2OSR
for fair comparison.
"""

from c2o_drive.algorithms.rainbow_dqn.planner import RainbowDQNPlanner
from c2o_drive.algorithms.rainbow_dqn.config import (
    RainbowDQNConfig,
    RainbowNetworkConfig,
    ReplayBufferConfig,
    TrainingConfig,
    LatticePlannerConfig as RainbowLatticePlannerConfig,
    GridConfig as RainbowGridConfig,
)
from c2o_drive.algorithms.rainbow_dqn.network import RainbowNetwork
from c2o_drive.algorithms.rainbow_dqn.replay_buffer import PrioritizedReplayBuffer

__all__ = [
    'RainbowDQNPlanner',
    'RainbowDQNConfig',
    'RainbowNetworkConfig',
    'ReplayBufferConfig',
    'TrainingConfig',
    'RainbowLatticePlannerConfig',
    'RainbowGridConfig',
    'RainbowNetwork',
    'PrioritizedReplayBuffer',
]
