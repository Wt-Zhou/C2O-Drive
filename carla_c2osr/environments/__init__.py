"""Driving environments for the C2O-Drive framework.

This module provides Gym-compatible environment implementations:
- SimpleGridEnvironment: 2D grid-based driving simulation
- ScenarioReplayEnvironment: ScenarioManager-based replay environment
- CarlaEnvironment: CARLA 3D simulation
"""

from carla_c2osr.environments.simple_grid_env import SimpleGridEnvironment
from carla_c2osr.environments.scenario_replay_env import ScenarioReplayEnvironment
from carla_c2osr.environments.carla_env import CarlaEnvironment
from carla_c2osr.environments.rewards import (
    SafetyReward,
    ComfortReward,
    EfficiencyReward,
    CenterlineReward,
    TimeReward,
    create_default_reward,
)

__all__ = [
    "SimpleGridEnvironment",
    "ScenarioReplayEnvironment",
    "CarlaEnvironment",
    "SafetyReward",
    "ComfortReward",
    "EfficiencyReward",
    "CenterlineReward",
    "TimeReward",
    "create_default_reward",
]
