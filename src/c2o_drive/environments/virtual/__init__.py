"""Virtual environments for C2O-Drive.

This module provides lightweight virtual environments for
testing and development without requiring CARLA.
"""

from c2o_drive.environments.virtual.grid_world import SimpleGridEnvironment
from c2o_drive.environments.virtual.scenario_manager import ScenarioManager

# Backwards compatible alias
SimpleGridEnv = SimpleGridEnvironment

__all__ = [
    "SimpleGridEnvironment",
    "SimpleGridEnv",
    "ScenarioManager",
]
