"""CARLA environment for C2O-Drive.

This module provides integration with the CARLA simulator.
"""

__all__ = [
    "CarlaEnv",
    "ScenarioConfig",
    "create_scenario",
    "get_available_scenarios",
]


def __getattr__(name):
    if name == "CarlaEnv":
        from c2o_drive.environments.carla.carla_env import CarlaEnv as _CarlaEnv
        return _CarlaEnv

    if name in {"ScenarioConfig", "create_scenario", "get_available_scenarios"}:
        from c2o_drive.environments.carla.scenarios import (
            ScenarioConfig as _ScenarioConfig,
            create_scenario as _create_scenario,
            get_available_scenarios as _get_available_scenarios,
        )

        mapping = {
            "ScenarioConfig": _ScenarioConfig,
            "create_scenario": _create_scenario,
            "get_available_scenarios": _get_available_scenarios,
        }
        return mapping[name]

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
