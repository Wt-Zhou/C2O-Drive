"""Soft Actor-Critic (SAC) algorithm implementation.

This module provides SAC adapted for discrete lattice-based trajectory planning
using the unified Planner interface with Categorical policies.
"""

from c2o_drive.algorithms.sac.planner import SACPlanner
from c2o_drive.algorithms.sac.config import SACConfig
from c2o_drive.algorithms.sac.discrete_network import CategoricalActor, DiscreteQCritic

__all__ = [
    "SACPlanner",
    "SACConfig",
    "CategoricalActor",
    "DiscreteQCritic",
]