"""Backward-compatible CARLA type definitions.

Historically the CARLA integration exposed its own ``types`` module.
The canonical definitions now live in :mod:`c2o_drive.core.types`.  This
module simply re-exports the same symbols so existing imports continue
to work.
"""

from c2o_drive.core.types import (  # noqa: F401
    AgentType,
    AgentDynamicsParams,
    AgentState,
    EgoState,
    EgoControl,
    WorldState,
)

__all__ = [
    "AgentType",
    "AgentDynamicsParams",
    "AgentState",
    "EgoState",
    "EgoControl",
    "WorldState",
]
