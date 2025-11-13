"""Baseline algorithms module

Currently supported algorithms:
- SAC (Soft Actor-Critic)
- DQN (Deep Q-Network)
"""

from carla_c2osr.agents.baselines.sac_agent import SACAgent, SACConfig
from carla_c2osr.agents.baselines.dqn_agent import DQNAgent, DQNConfig

__all__ = [
    "SACAgent",
    "SACConfig",
    "DQNAgent",
    "DQNConfig",
]
