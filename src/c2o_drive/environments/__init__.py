"""Environments for C2O-Drive.

This module provides various driving environments including
CARLA simulation and virtual environments for testing.
"""

# Lazy-load heavy environments so CARLA is optional.

__all__ = ["CarlaEnv", "SimpleGridEnv"]


def __getattr__(name):
    if name == "CarlaEnv":
        from c2o_drive.environments.carla.carla_env import CarlaEnv as _CarlaEnv
        return _CarlaEnv
    if name == "SimpleGridEnv":
        from c2o_drive.environments.virtual.grid_world import (
            SimpleGridEnv as _SimpleGridEnv,
        )
        return _SimpleGridEnv
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
