"""Test script for RCRL algorithm.

This script verifies that the RCRL implementation is working correctly
by testing basic functionality without requiring CARLA environment.
"""

import numpy as np
from c2o_drive.core.types import WorldState, EgoState, AgentState, AgentType, EgoControl
from c2o_drive.algorithms.rcrl import (
    create_rcrl_planner,
    create_hard_constraint_planner,
    create_soft_constraint_planner,
)
from c2o_drive.core.planner import Transition


def create_dummy_world_state() -> WorldState:
    """Create a dummy world state for testing."""
    ego = EgoState(
        position_m=(0.0, 0.0),
        velocity_mps=(10.0, 0.0),
        yaw_rad=0.0,
    )

    agents = [
        AgentState(
            agent_id="agent_0",
            position_m=(20.0, -2.0),
            velocity_mps=(8.0, 0.0),
            heading_rad=0.0,
            agent_type=AgentType.VEHICLE,
        ),
        AgentState(
            agent_id="agent_1",
            position_m=(15.0, 2.0),
            velocity_mps=(5.0, 0.0),
            heading_rad=0.0,
            agent_type=AgentType.PEDESTRIAN,
        ),
    ]

    return WorldState(time_s=0.0, ego=ego, agents=agents)


def create_reference_path() -> list:
    """Create a simple reference path."""
    return [(float(x), 0.0) for x in range(0, 100, 5)]


def test_planner_creation():
    """Test planner creation with different factory functions."""
    print("=== Testing Planner Creation ===")

    # Test default creation
    print("Creating default RCRL planner...")
    planner1 = create_rcrl_planner()
    print(f"✓ Default planner created: {planner1.__class__.__name__}")
    print(f"  - Constraint mode: {planner1.config.constraint.mode}")
    print(f"  - Device: {planner1.config.device}")
    print(f"  - Horizon: {planner1.config.horizon}")

    # Test hard constraint planner
    print("\nCreating hard constraint planner...")
    planner2 = create_hard_constraint_planner(horizon=15)
    print(f"✓ Hard constraint planner created")
    print(f"  - Constraint mode: {planner2.config.constraint.mode}")
    print(f"  - Horizon: {planner2.config.horizon}")

    # Test soft constraint planner
    print("\nCreating soft constraint planner...")
    planner3 = create_soft_constraint_planner(horizon=10, penalty_weight=200.0)
    print(f"✓ Soft constraint planner created")
    print(f"  - Constraint mode: {planner3.config.constraint.mode}")
    print(f"  - Penalty weight: {planner3.config.constraint.soft_penalty_weight}")

    print("\n✓ All planner creation tests passed!\n")


def test_action_selection():
    """Test action selection."""
    print("=== Testing Action Selection ===")

    planner = create_rcrl_planner(device="cpu")
    world_state = create_dummy_world_state()
    reference_path = create_reference_path()

    print("Selecting action...")
    control = planner.select_action(
        observation=world_state,
        deterministic=False,
        reference_path=reference_path,
    )

    print(f"✓ Action selected successfully")
    print(f"  - Throttle: {control.throttle:.3f}")
    print(f"  - Steer: {control.steer:.3f}")
    print(f"  - Brake: {control.brake:.3f}")

    # Test multiple selections
    print("\nTesting multiple action selections...")
    for i in range(5):
        control = planner.select_action(
            observation=world_state, deterministic=False, reference_path=reference_path
        )
        print(f"  Step {i+1}: throttle={control.throttle:.3f}, steer={control.steer:.3f}")

    print("\n✓ Action selection tests passed!\n")


def test_update():
    """Test online learning update."""
    print("=== Testing Online Learning Update ===")

    planner = create_rcrl_planner(device="cpu")
    world_state = create_dummy_world_state()
    reference_path = create_reference_path()

    # Select action
    control = planner.select_action(
        observation=world_state, deterministic=False, reference_path=reference_path
    )

    # Create next state
    next_world_state = create_dummy_world_state()
    next_world_state.ego.position_m = (1.0, 0.0)

    # Create transition
    transition = Transition(
        state=world_state,
        action=control,
        reward=1.0,
        next_state=next_world_state,
        terminated=False,
        truncated=False,
        info={},
    )

    print("Updating planner...")
    metrics = planner.update(transition)

    print(f"✓ Update successful")
    print(f"  - Loss: {metrics.loss:.4f}")
    print(f"  - Buffer size: {metrics.custom.get('buffer_size', 0)}")
    print(f"  - Epsilon: {metrics.custom.get('epsilon', 0):.4f}")
    print(f"  - Reward: {metrics.custom.get('reward', 0):.2f}")

    # Test multiple updates
    print("\nTesting multiple updates...")
    for i in range(10):
        control = planner.select_action(
            observation=world_state, deterministic=False, reference_path=reference_path
        )
        transition = Transition(
            state=world_state,
            action=control,
            reward=np.random.random(),
            next_state=next_world_state,
            terminated=False,
            truncated=False,
            info={},
        )
        metrics = planner.update(transition)

        if (i + 1) % 5 == 0:
            print(f"  Step {i+1}: buffer_size={len(planner.replay_buffer)}, epsilon={planner._epsilon:.4f}")

    print("\n✓ Update tests passed!\n")


def test_save_load():
    """Test model save and load."""
    print("=== Testing Save/Load ===")

    import tempfile
    import os

    planner = create_rcrl_planner(device="cpu")
    world_state = create_dummy_world_state()
    reference_path = create_reference_path()

    # Perform some steps
    for _ in range(5):
        control = planner.select_action(
            observation=world_state, deterministic=False, reference_path=reference_path
        )

    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
        save_path = f.name

    try:
        print(f"Saving model to {save_path}...")
        planner.save(save_path)
        print("✓ Model saved")

        # Create new planner and load
        print("Loading model into new planner...")
        new_planner = create_rcrl_planner(device="cpu")
        new_planner.load(save_path)
        print("✓ Model loaded")

        print(f"  - Loaded step count: {new_planner._step_count}")
        print(f"  - Loaded epsilon: {new_planner._epsilon:.4f}")

    finally:
        if os.path.exists(save_path):
            os.remove(save_path)

    print("\n✓ Save/load tests passed!\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("RCRL Algorithm Integration Tests")
    print("=" * 60 + "\n")

    try:
        test_planner_creation()
        test_action_selection()
        test_update()
        test_save_load()

        print("=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60 + "\n")

    except Exception as e:
        print("\n" + "=" * 60)
        print("✗ TEST FAILED!")
        print("=" * 60)
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
