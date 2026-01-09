#!/usr/bin/env python3
"""
Test script for PPO implementation.

This script verifies that the PPO algorithm is correctly integrated
and can perform basic training operations.
"""

import sys
from pathlib import Path

# Add project root to path
_repo_root = Path(__file__).resolve().parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

_src_path = _repo_root / "src"
if _src_path.exists() and str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

import numpy as np
from c2o_drive.algorithms.ppo import PPOPlanner, PPOConfig
from c2o_drive.algorithms.c2osr.config import LatticePlannerConfig
from c2o_drive.core.types import WorldState, EgoState, EgoControl
from c2o_drive.core.planner import Transition


def test_ppo_instantiation():
    """Test 1: PPO planner can be instantiated."""
    print("\n" + "="*70)
    print("TEST 1: PPO Instantiation")
    print("="*70)

    # Create lattice config
    lattice_config = LatticePlannerConfig(
        lateral_offsets=[-3.0, -2.0, 0.0, 2.0, 3.0],
        speed_variations=[4.0, 6.0, 8.0],
        horizon=10,
        dt=1.0,
    )

    # Create PPO config
    config = PPOConfig(
        lattice=lattice_config,
        state_dim=128,
        horizon=10,
    )

    print(f"✓ PPOConfig created")
    print(f"  Action dimension: {config.action_dim}")
    print(f"  State dimension: {config.state_dim}")

    # Instantiate planner
    planner = PPOPlanner(config)
    print(f"✓ PPOPlanner instantiated")
    print(f"  Network architecture: Actor-Critic with shared features")
    print(f"  Hidden dimensions: {config.hidden_dims}")
    print(f"  Action mapping size: {len(planner.action_mapping)}")

    return True


def test_action_selection():
    """Test 2: PPO can select actions and generate trajectories."""
    print("\n" + "="*70)
    print("TEST 2: Action Selection & Trajectory Generation")
    print("="*70)

    # Create planner
    lattice_config = LatticePlannerConfig(
        lateral_offsets=[-3.0, 0.0, 3.0],
        speed_variations=[4.0, 6.0],
        horizon=10,
        dt=1.0,
    )

    config = PPOConfig(
        lattice=lattice_config,
        state_dim=128,
        horizon=10,
    )

    planner = PPOPlanner(config)

    # Create mock WorldState
    ego = EgoState(
        position_m=(0.0, 0.0),
        velocity_mps=(5.0, 0.0),
        yaw_rad=0.0,
    )

    world_state = WorldState(
        time_s=0.0,
        ego=ego,
        agents=[],
    )

    # Create reference path
    reference_path = [(i * 5.0, 0.0) for i in range(15)]

    # Test deterministic action selection
    control_det = planner.select_action(world_state, deterministic=True, reference_path=reference_path)
    print(f"✓ Deterministic action selection successful")
    print(f"  Control: throttle={control_det.throttle:.3f}, steer={control_det.steer:.3f}, brake={control_det.brake:.3f}")
    print(f"  Trajectory ID: {planner._current_trajectory.trajectory_id}")
    print(f"  Trajectory waypoints: {len(planner._current_trajectory.waypoints)}")

    # Reset for stochastic test
    planner.reset()

    # Test stochastic action selection
    control_stoch = planner.select_action(world_state, deterministic=False, reference_path=reference_path)
    print(f"✓ Stochastic action selection successful")
    print(f"  Control: throttle={control_stoch.throttle:.3f}, steer={control_stoch.steer:.3f}, brake={control_stoch.brake:.3f}")
    print(f"  Log prob stored: {planner._last_log_prob is not None}")

    return True


def test_training_loop():
    """Test 3: PPO can perform training updates."""
    print("\n" + "="*70)
    print("TEST 3: Training Loop & PPO Updates")
    print("="*70)

    # Create planner with small batch size
    lattice_config = LatticePlannerConfig(
        lateral_offsets=[-3.0, 0.0, 3.0],
        speed_variations=[4.0, 6.0],
        horizon=10,
        dt=1.0,
    )

    config = PPOConfig(
        lattice=lattice_config,
        state_dim=128,
        horizon=10,
        batch_size=8,
        buffer_size=20,
        n_epochs=2,
    )

    planner = PPOPlanner(config)
    print(f"✓ Planner created with batch_size={config.batch_size}")

    # Simulate multiple episodes
    reference_path = [(i * 5.0, 0.0) for i in range(15)]
    update_count = 0

    for ep in range(3):
        planner.reset()

        for step in range(5):
            # Create mock WorldState
            ego = EgoState(
                position_m=(step * 1.0, 0.0),
                velocity_mps=(5.0, 0.0),
                yaw_rad=0.0,
            )

            world_state = WorldState(
                time_s=step * 1.0,
                ego=ego,
                agents=[],
            )

            # Select action
            control = planner.select_action(world_state, deterministic=False, reference_path=reference_path)

            # Create next state
            next_ego = EgoState(
                position_m=((step + 1) * 1.0, 0.0),
                velocity_mps=(5.0, 0.0),
                yaw_rad=0.0,
            )

            next_state = WorldState(
                time_s=(step + 1) * 1.0,
                ego=next_ego,
                agents=[],
            )

            # Create transition
            transition = Transition(
                state=world_state,
                action=control,
                reward=1.0 + np.random.randn() * 0.1,
                next_state=next_state,
                terminated=False,
                truncated=(step == 4),
                info={},
            )

            # Update planner
            metrics = planner.update(transition)

            # Check if PPO update happened
            if metrics.loss is not None:
                update_count += 1
                print(f"  ✓ PPO update #{update_count} triggered at episode {ep}, step {step}")
                print(f"    Policy loss: {metrics.loss:.4f}")
                print(f"    Entropy: {metrics.policy_entropy:.4f}")
                if metrics.custom:
                    print(f"    Value loss: {metrics.custom.get('value_loss', 0.0):.4f}")
                    print(f"    Approx KL: {metrics.custom.get('approx_kl', 0.0):.6f}")

    print(f"✓ Training loop completed")
    print(f"  Total PPO updates: {update_count}")
    print(f"  Final buffer size: {len(planner.rollout_buffer)}")

    return update_count > 0


def test_dynamic_action_space():
    """Test 4: Action space adapts to lattice config changes."""
    print("\n" + "="*70)
    print("TEST 4: Dynamic Action Space Adaptation")
    print("="*70)

    # Test with different lattice configurations
    configs = [
        ([-3.0, 0.0, 3.0], [4.0, 6.0]),  # 3 × 2 = 6 actions
        ([-2.0, 0.0, 2.0], [3.0, 5.0, 7.0]),  # 3 × 3 = 9 actions
        ([-3.0, -1.5, 0.0, 1.5, 3.0], [4.0, 6.0, 8.0]),  # 5 × 3 = 15 actions
    ]

    for i, (lateral_offsets, speed_variations) in enumerate(configs, 1):
        lattice_config = LatticePlannerConfig(
            lateral_offsets=lateral_offsets,
            speed_variations=speed_variations,
            horizon=10,
            dt=1.0,
        )

        config = PPOConfig(
            lattice=lattice_config,
            state_dim=128,
            horizon=10,
        )

        expected_dim = len(lateral_offsets) * len(speed_variations)
        actual_dim = config.action_dim

        print(f"  Config {i}: {len(lateral_offsets)} lateral × {len(speed_variations)} speed")
        print(f"    Expected action_dim: {expected_dim}")
        print(f"    Actual action_dim: {actual_dim}")

        if actual_dim != expected_dim:
            print(f"    ✗ FAILED: action_dim mismatch!")
            return False

        # Verify planner uses correct action space
        planner = PPOPlanner(config)
        if len(planner.action_mapping) != expected_dim:
            print(f"    ✗ FAILED: action_mapping size mismatch!")
            return False

        print(f"    ✓ Passed")

    print("✓ Dynamic action space adaptation verified")
    return True


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print(" PPO IMPLEMENTATION TEST SUITE")
    print("="*70)

    tests = [
        ("PPO Instantiation", test_ppo_instantiation),
        ("Action Selection", test_action_selection),
        ("Training Loop", test_training_loop),
        ("Dynamic Action Space", test_dynamic_action_space),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ TEST FAILED: {test_name}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Summary
    print("\n" + "="*70)
    print(" TEST SUMMARY")
    print("="*70)

    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status}: {test_name}")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All tests passed! PPO implementation is working correctly.")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
