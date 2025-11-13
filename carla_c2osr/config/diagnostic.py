"""Configuration diagnostic tool for checking consistency between global and algorithm configs."""

from typing import Dict, Any, List, Tuple
from dataclasses import fields
import sys


def check_config_consistency() -> Dict[str, Any]:
    """Check consistency between global and algorithm configs.

    Returns:
        Dict containing:
        - consistent: bool, whether configs are consistent
        - issues: list of inconsistency descriptions
        - global_config: the global config object
        - algorithm_config: the algorithm config object
    """
    try:
        from carla_c2osr.config import get_global_config
        from carla_c2osr.algorithms.c2osr.config import C2OSRPlannerConfig
    except ImportError as e:
        return {
            'consistent': False,
            'issues': [f"Import error: {e}"],
            'global_config': None,
            'algorithm_config': None,
        }

    gc = get_global_config()
    ac = C2OSRPlannerConfig.from_global_config()

    issues = []

    # Check horizon consistency
    if gc.time.default_horizon != ac.horizon:
        issues.append(f"Horizon mismatch: global={gc.time.default_horizon}, algorithm={ac.horizon}")

    # Check if horizons are synced in subconfigs
    if ac.lattice.horizon != ac.horizon:
        issues.append(f"Lattice horizon not synced: lattice={ac.lattice.horizon}, main={ac.horizon}")

    if ac.q_value.horizon != ac.horizon:
        issues.append(f"Q-value horizon not synced: q_value={ac.q_value.horizon}, main={ac.horizon}")

    # Check grid consistency
    if gc.grid.cell_size_m != ac.grid.grid_size_m:
        issues.append(f"Grid cell size mismatch: global={gc.grid.cell_size_m}, algorithm={ac.grid.grid_size_m}")

    # Check Dirichlet parameters
    if gc.dirichlet.alpha_in != ac.dirichlet.alpha_in:
        issues.append(f"Dirichlet alpha_in mismatch: global={gc.dirichlet.alpha_in}, algorithm={ac.dirichlet.alpha_in}")

    if gc.dirichlet.alpha_out != ac.dirichlet.alpha_out:
        issues.append(f"Dirichlet alpha_out mismatch: global={gc.dirichlet.alpha_out}, algorithm={ac.dirichlet.alpha_out}")

    if gc.dirichlet.learning_rate != ac.dirichlet.learning_rate:
        issues.append(f"Dirichlet learning_rate mismatch: global={gc.dirichlet.learning_rate}, algorithm={ac.dirichlet.learning_rate}")

    # Check Q-value parameters
    if gc.sampling.q_value_samples != ac.q_value.n_samples:
        issues.append(f"Q-value samples mismatch: global={gc.sampling.q_value_samples}, algorithm={ac.q_value.n_samples}")

    if gc.c2osr.q_selection_percentile != ac.q_value.selection_percentile:
        issues.append(f"Q selection percentile mismatch: global={gc.c2osr.q_selection_percentile}, algorithm={ac.q_value.selection_percentile}")

    # Check gamma consistency
    if gc.c2osr.gamma != ac.gamma:
        issues.append(f"Gamma mismatch: global={gc.c2osr.gamma}, algorithm={ac.gamma}")

    if ac.q_value.gamma != ac.gamma:
        issues.append(f"Q-value gamma not synced: q_value={ac.q_value.gamma}, main={ac.gamma}")

    # Check trajectory storage multiplier
    if gc.matching.trajectory_storage_multiplier != ac.trajectory_storage_multiplier:
        issues.append(f"Trajectory storage multiplier mismatch: global={gc.matching.trajectory_storage_multiplier}, algorithm={ac.trajectory_storage_multiplier}")

    # Check reward weights
    reward_mappings = [
        ('collision_penalty', 'collision_penalty'),
        ('collision_threshold', 'collision_threshold'),
        ('collision_check_cell_radius', 'collision_check_cell_radius'),
        ('speed_reward_weight', 'speed_reward_weight'),
        ('speed_target_mps', 'speed_target_mps'),
        ('progress_reward_weight', 'progress_reward_weight'),
        ('acceleration_penalty_weight', 'acceleration_penalty_weight'),
        ('acceleration_penalty_threshold', 'acceleration_penalty_threshold'),
        ('jerk_penalty_weight', 'jerk_penalty_weight'),
        ('centerline_offset_penalty_weight', 'centerline_offset_penalty_weight'),
    ]

    for global_attr, algo_attr in reward_mappings:
        global_val = getattr(gc.reward, global_attr, None)
        algo_val = getattr(ac.reward_weights, algo_attr, None)
        if global_val != algo_val:
            issues.append(f"Reward {global_attr} mismatch: global={global_val}, algorithm={algo_val}")

    return {
        'consistent': len(issues) == 0,
        'issues': issues,
        'global_config': gc,
        'algorithm_config': ac,
    }


def print_config_summary():
    """Print human-readable config summary and consistency check."""
    result = check_config_consistency()

    print("=" * 60)
    print("C2O-Drive Configuration Diagnostic Report")
    print("=" * 60)

    if result['consistent']:
        print("✅ Configuration is consistent!")
    else:
        print("⚠️  Configuration has issues:")
        for issue in result['issues']:
            print(f"  ❌ {issue}")

    print("\n" + "-" * 60)
    print("Current Global Configuration Settings:")
    print("-" * 60)

    if result['global_config']:
        gc = result['global_config']
        print(f"  Horizon: {gc.time.default_horizon} steps")
        print(f"  Time step (dt): {gc.time.dt} seconds")
        print(f"  Grid size: {gc.grid.grid_size_m}m × {gc.grid.grid_size_m}m")
        print(f"  Grid cell: {gc.grid.cell_size_m}m")
        print(f"  Grid bounds: X[{gc.grid.x_min}, {gc.grid.x_max}], Y[{gc.grid.y_min}, {gc.grid.y_max}]")
        print(f"  Q-value samples: {gc.sampling.q_value_samples}")
        print(f"  Q selection percentile: {gc.c2osr.q_selection_percentile}")
        print(f"  Discount factor (gamma): {gc.c2osr.gamma}")
        print(f"  Dirichlet alpha_in: {gc.dirichlet.alpha_in}")
        print(f"  Dirichlet alpha_out: {gc.dirichlet.alpha_out}")
        print(f"  Dirichlet learning rate: {gc.dirichlet.learning_rate}")
        print(f"  Trajectory storage multiplier: {gc.matching.trajectory_storage_multiplier}")
        print(f"  Collision penalty: {gc.reward.collision_penalty}")
        print(f"  Centerline offset weight: {gc.reward.centerline_offset_penalty_weight}")

    print("\n" + "-" * 60)
    print("Algorithm Config (from_global_config):")
    print("-" * 60)

    if result['algorithm_config']:
        ac = result['algorithm_config']
        print(f"  Horizon: {ac.horizon} steps")
        print(f"  Lattice horizon: {ac.lattice.horizon} (synced: {ac.lattice.horizon == ac.horizon})")
        print(f"  Q-value horizon: {ac.q_value.horizon} (synced: {ac.q_value.horizon == ac.horizon})")
        print(f"  Gamma: {ac.gamma}")
        print(f"  Q-value gamma: {ac.q_value.gamma} (synced: {ac.q_value.gamma == ac.gamma})")

    print("\n" + "=" * 60)
    print("Recommendation:")
    if result['consistent']:
        print("✅ Your configuration is properly synchronized.")
        print("   Modify parameters in global_config.py for experiments.")
    else:
        print("⚠️  Fix inconsistencies by:")
        print("   1. Update global_config.py with desired values")
        print("   2. Use C2OSRPlannerConfig.from_global_config()")
        print("   3. Avoid direct modification of c2osr/config.py defaults")
    print("=" * 60)


def compare_configs(config1_path: str = None, config2_path: str = None):
    """Compare two configuration instances or files.

    Args:
        config1_path: Path to first config (default: global_config)
        config2_path: Path to second config (default: algorithm config from_global_config)
    """
    # This can be extended to compare arbitrary config files
    result = check_config_consistency()

    differences = []

    # Compare all numeric parameters
    if result['global_config'] and result['algorithm_config']:
        gc = result['global_config']
        ac = result['algorithm_config']

        # Time parameters
        differences.append(('horizon', gc.time.default_horizon, ac.horizon))
        differences.append(('dt', gc.time.dt, getattr(ac.lattice, 'dt', None)))

        # Grid parameters
        differences.append(('grid_cell_size', gc.grid.cell_size_m, ac.grid.grid_size_m))

        # Sampling parameters
        differences.append(('q_value_samples', gc.sampling.q_value_samples, ac.q_value.n_samples))

    return differences


def validate_config_values():
    """Validate that configuration values are within reasonable ranges."""
    result = check_config_consistency()

    warnings = []

    if result['global_config']:
        gc = result['global_config']

        # Validate horizon
        if gc.time.default_horizon < 1:
            warnings.append(f"Horizon too small: {gc.time.default_horizon} < 1")
        elif gc.time.default_horizon > 100:
            warnings.append(f"Horizon very large: {gc.time.default_horizon} > 100")

        # Validate dt
        if gc.time.dt <= 0:
            warnings.append(f"Invalid dt: {gc.time.dt} <= 0")
        elif gc.time.dt > 10:
            warnings.append(f"dt very large: {gc.time.dt} > 10 seconds")

        # Validate grid
        if gc.grid.grid_size_m <= 0:
            warnings.append(f"Invalid grid size: {gc.grid.grid_size_m} <= 0")

        if gc.grid.cell_size_m <= 0:
            warnings.append(f"Invalid cell size: {gc.grid.cell_size_m} <= 0")

        # Validate Dirichlet
        if gc.dirichlet.alpha_in <= 0:
            warnings.append(f"Invalid alpha_in: {gc.dirichlet.alpha_in} <= 0")

        if gc.dirichlet.alpha_out <= 0:
            warnings.append(f"Invalid alpha_out: {gc.dirichlet.alpha_out} <= 0")

        if gc.dirichlet.learning_rate < 0 or gc.dirichlet.learning_rate > 1:
            warnings.append(f"Learning rate out of range: {gc.dirichlet.learning_rate} not in [0, 1]")

        # Validate gamma
        if gc.c2osr.gamma < 0 or gc.c2osr.gamma > 1:
            warnings.append(f"Gamma out of range: {gc.c2osr.gamma} not in [0, 1]")

    return warnings


if __name__ == "__main__":
    # Run diagnostic when script is executed directly
    print_config_summary()

    # Also validate values
    warnings = validate_config_values()
    if warnings:
        print("\n⚠️  Value Validation Warnings:")
        for warning in warnings:
            print(f"  - {warning}")