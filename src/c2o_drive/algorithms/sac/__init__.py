"""Soft Actor-Critic (SAC) algorithm implementation.

This module provides a complete SAC implementation for autonomous driving.
"""

from c2o_drive.algorithms.sac.agent import SACAgent
from c2o_drive.algorithms.sac.config import SACConfig
from c2o_drive.algorithms.sac.replay_buffer import ReplayBuffer

__all__ = [
    "SACAgent",
    "SACConfig",
    "ReplayBuffer",
]