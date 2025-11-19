"""C2O-Drive: Clean Architecture for Autonomous Driving Algorithms

This package provides a modular framework for implementing and comparing
autonomous driving algorithms in both simulated and real environments.
"""

__version__ = "2.0.0"

# Make key components easily accessible
from c2o_drive.core.types import (
    WorldState,
    EgoState,
    AgentState,
    EgoControl,
    AgentType,
)

__all__ = [
    "WorldState",
    "EgoState",
    "AgentState",
    "EgoControl",
    "AgentType",
]