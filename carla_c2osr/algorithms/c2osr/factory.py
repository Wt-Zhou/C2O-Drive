"""Factory functions for creating C2OSR algorithm components.

This module provides convenient factory functions for creating
C2OSR planner and evaluator instances with various configurations.
"""

from typing import Optional

from carla_c2osr.algorithms.c2osr.config import (
    C2OSRPlannerConfig,
    C2OSREvaluatorConfig,
)
from carla_c2osr.algorithms.c2osr.planner import C2OSRPlanner
from carla_c2osr.algorithms.c2osr.evaluator import C2OSREvaluator


def create_c2osr_planner(
    config: Optional[C2OSRPlannerConfig] = None,
    use_global_config: bool = False,
) -> C2OSRPlanner:
    """Create a C2OSR planner instance.

    Args:
        config: Optional configuration. If None, uses default config.
        use_global_config: If True, creates config from global configuration.

    Returns:
        Initialized C2OSR planner

    Examples:
        >>> # Create with default config
        >>> planner = create_c2osr_planner()

        >>> # Create with custom config
        >>> config = C2OSRPlannerConfig(
        ...     lattice=LatticePlannerConfig(horizon=15),
        ...     gamma=0.95,
        ... )
        >>> planner = create_c2osr_planner(config)

        >>> # Create from global config
        >>> planner = create_c2osr_planner(use_global_config=True)
    """
    if config is None:
        if use_global_config:
            config = C2OSRPlannerConfig.from_global_config()
        else:
            config = C2OSRPlannerConfig()

    return C2OSRPlanner(config)


def create_c2osr_evaluator(
    config: Optional[C2OSREvaluatorConfig] = None,
    planner: Optional[C2OSRPlanner] = None,
) -> C2OSREvaluator:
    """Create a C2OSR evaluator instance.

    Args:
        config: Optional configuration. If None, uses default config.
        planner: Optional planner to share trajectory buffer and Dirichlet bank with.
                If provided, the evaluator will use the planner's components.

    Returns:
        Initialized C2OSR evaluator

    Examples:
        >>> # Create standalone evaluator
        >>> evaluator = create_c2osr_evaluator()

        >>> # Create evaluator that shares state with planner
        >>> planner = create_c2osr_planner()
        >>> evaluator = create_c2osr_evaluator(planner=planner)

        >>> # Create with custom config
        >>> config = C2OSREvaluatorConfig(
        ...     q_value=QValueConfig(n_samples=200),
        ... )
        >>> evaluator = create_c2osr_evaluator(config)
    """
    if planner is not None:
        # Share components with planner
        if config is None:
            config = C2OSREvaluatorConfig.from_planner_config(planner.config)

        return C2OSREvaluator(
            config=config,
            trajectory_buffer=planner.trajectory_buffer,
            dirichlet_bank=planner.dirichlet_bank,
        )
    else:
        # Create standalone evaluator
        if config is None:
            config = C2OSREvaluatorConfig()

        return C2OSREvaluator(config)


def create_c2osr_planner_evaluator_pair(
    planner_config: Optional[C2OSRPlannerConfig] = None,
    use_global_config: bool = False,
) -> tuple[C2OSRPlanner, C2OSREvaluator]:
    """Create a matched planner-evaluator pair that share state.

    Args:
        planner_config: Optional planner configuration
        use_global_config: If True, creates config from global configuration

    Returns:
        Tuple of (planner, evaluator) that share trajectory buffer and Dirichlet bank

    Examples:
        >>> # Create pair with default config
        >>> planner, evaluator = create_c2osr_planner_evaluator_pair()

        >>> # Create pair with custom config
        >>> config = C2OSRPlannerConfig(gamma=0.95)
        >>> planner, evaluator = create_c2osr_planner_evaluator_pair(config)

        >>> # Create pair from global config
        >>> planner, evaluator = create_c2osr_planner_evaluator_pair(
        ...     use_global_config=True
        ... )
    """
    # Create planner
    planner = create_c2osr_planner(
        config=planner_config,
        use_global_config=use_global_config,
    )

    # Create evaluator that shares state with planner
    evaluator = create_c2osr_evaluator(planner=planner)

    return planner, evaluator


__all__ = [
    'create_c2osr_planner',
    'create_c2osr_evaluator',
    'create_c2osr_planner_evaluator_pair',
]
