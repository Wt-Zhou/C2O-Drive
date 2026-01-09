"""Factory functions for creating RCRL planner instances.

This module provides convenient factory functions for initializing
RCRL planners with different configuration sources.
"""

from typing import Optional
from c2o_drive.algorithms.rcrl.planner import RCRLPlanner
from c2o_drive.algorithms.rcrl.config import RCRLPlannerConfig


def create_rcrl_planner(
    config: Optional[RCRLPlannerConfig] = None,
    use_global_config: bool = False,
    **config_overrides,
) -> RCRLPlanner:
    """Create RCRL planner with flexible configuration.

    Args:
        config: Optional RCRLPlannerConfig instance
        use_global_config: If True, initialize from global config
        **config_overrides: Keyword arguments to override config parameters

    Returns:
        Initialized RCRLPlanner instance

    Examples:
        # Create with default config
        planner = create_rcrl_planner()

        # Create from global config
        planner = create_rcrl_planner(use_global_config=True)

        # Create with custom parameters
        planner = create_rcrl_planner(
            horizon=15,
            device="cuda",
        )

        # Create with custom config
        custom_config = RCRLPlannerConfig(horizon=20)
        planner = create_rcrl_planner(config=custom_config)
    """
    # Determine configuration source
    if config is None:
        if use_global_config:
            config = RCRLPlannerConfig.from_global_config()
        else:
            config = RCRLPlannerConfig()

    # Apply overrides
    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                raise ValueError(f"Unknown config parameter: {key}")

    # Re-run __post_init__ to sync parameters
    config.__post_init__()

    # Create planner
    planner = RCRLPlanner(config)

    return planner


def create_hard_constraint_planner(
    horizon: int = 10, device: str = "cpu", **kwargs
) -> RCRLPlanner:
    """Create RCRL planner with hard safety constraints.

    Convenience function for creating a planner with hard constraint mode,
    suitable for safety-critical scenarios.

    Args:
        horizon: Planning horizon
        device: Device for neural network
        **kwargs: Additional config parameters

    Returns:
        RCRLPlanner with hard constraints
    """
    config = RCRLPlannerConfig(horizon=horizon, device=device, **kwargs)
    config.constraint.mode = "hard"
    config.__post_init__()

    return RCRLPlanner(config)


def create_soft_constraint_planner(
    horizon: int = 10,
    device: str = "cpu",
    penalty_weight: float = 100.0,
    **kwargs,
) -> RCRLPlanner:
    """Create RCRL planner with soft safety constraints.

    Convenience function for creating a planner with soft constraint mode,
    suitable for training scenarios where exploration is needed.

    Args:
        horizon: Planning horizon
        device: Device for neural network
        penalty_weight: Soft constraint penalty weight
        **kwargs: Additional config parameters

    Returns:
        RCRLPlanner with soft constraints
    """
    config = RCRLPlannerConfig(horizon=horizon, device=device, **kwargs)
    config.constraint.mode = "soft"
    config.constraint.soft_penalty_weight = penalty_weight
    config.__post_init__()

    return RCRLPlanner(config)


__all__ = [
    "create_rcrl_planner",
    "create_hard_constraint_planner",
    "create_soft_constraint_planner",
]
