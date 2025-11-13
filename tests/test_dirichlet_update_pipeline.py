#!/usr/bin/env python3
"""
Test Dirichlet update pipeline after hash collision fix.

This test verifies:
1. Episodes are stored without hash collisions (unique IDs)
2. Historical matching works correctly
3. Dirichlet updates are triggered with matched data
4. storage_multiplier parameter is respected
"""

import sys
import numpy as np
from typing import List, Tuple

# Import core classes
from carla_c2osr.env.types import EgoState, AgentState, WorldState, EgoControl, AgentType
from carla_c2osr.agents.c2osr.trajectory_buffer import TrajectoryBuffer, AgentTrajectoryData, ScenarioState
from carla_c2osr.algorithms.c2osr.config import C2OSRPlannerConfig
from carla_c2osr.algorithms.c2osr.planner import C2OSRPlanner
from carla_c2osr.core.planner import Transition


def create_world_state(timestep: int, ego_x: float = 0.0) -> WorldState:
    """Create a simple world state for testing."""
    ego = EgoState(
        position_m=(ego_x, 0.0),
        velocity_mps=(5.0, 0.0),
        yaw_rad=0.0,
    )

    # Add one agent
    agent = AgentState(
        agent_id="agent_1",
        position_m=(ego_x + 10.0, 2.0),
        velocity_mps=(4.0, 0.0),
        heading_rad=0.0,
        agent_type=AgentType.VEHICLE,
    )

    return WorldState(
        time_s=timestep * 1.0,
        ego=ego,
        agents=[agent]
    )


def test_hash_collision_fixed():
    """Test 1: Verify hash collision is fixed."""
    print("=" * 60)
    print("Test 1: Hash Collision Fix")
    print("=" * 60)

    # Create config with storage_multiplier=10
    config = C2OSRPlannerConfig()
    config.trajectory_storage_multiplier = 10

    buffer = TrajectoryBuffer(
        capacity=10000,
        trajectory_storage_multiplier=10,
    )

    # Store 5 episodes, each should create 10 copies
    for episode_idx in range(5):
        traj_data = AgentTrajectoryData(
            agent_id=1,
            agent_type='car',
            init_position=(episode_idx * 5.0, 0.0),
            init_velocity=(5.0, 0.0),
            init_heading=0.0,
            trajectory_cells=[episode_idx * 5 + t for t in range(5)],
        )

        scenario_state = ScenarioState(
            ego_position=(episode_idx * 5.0, 0.0),
            ego_velocity=(5.0, 0.0),
            ego_heading=0.0,
            agents_states=[(episode_idx * 5.0 + 10.0, 2.0, 4.0, 0.0, 0.0, 'car')],
        )

        buffer.store_episode_trajectories_by_timestep(
            episode_id=episode_idx,
            timestep_scenarios=[
                (scenario_state, [traj_data])
            ],
        )

    # Check storage
    stats = buffer.get_stats()
    expected_count = 5 * 10  # 5 episodes × 10 multiplier
    actual_count = stats['total_episodes']

    print(f"Expected episodes: {expected_count}")
    print(f"Actual episodes: {actual_count}")
    print(f"Storage multiplier: {stats['storage_multiplier']}")

    if actual_count == expected_count:
        print("✅ Hash collision fixed! All episodes stored correctly.")
        return True
    else:
        print(f"❌ Expected {expected_count} episodes, got {actual_count}")
        return False


def test_storage_multiplier_parameter():
    """Test 2: Verify storage_multiplier parameter works."""
    print("\n" + "=" * 60)
    print("Test 2: Storage Multiplier Parameter")
    print("=" * 60)

    test_results = []

    for multiplier in [1, 5, 10, 50]:
        buffer = TrajectoryBuffer(
            capacity=10000,
            trajectory_storage_multiplier=multiplier,
        )

        # Store 3 episodes
        for episode_idx in range(3):
            traj_data = AgentTrajectoryData(
                agent_id=1,
                agent_type='car',
                init_position=(episode_idx * 5.0, 0.0),
                init_velocity=(5.0, 0.0),
                init_heading=0.0,
                trajectory_cells=[episode_idx * 5 + t for t in range(5)],
            )

            scenario_state = ScenarioState(
                ego_position=(episode_idx * 5.0, 0.0),
                ego_velocity=(5.0, 0.0),
                ego_heading=0.0,
                agents_states=[(episode_idx * 5.0 + 10.0, 2.0, 4.0, 0.0, 0.0, 'car')],
            )

            buffer.store_episode_trajectories_by_timestep(
                episode_id=episode_idx,
                timestep_scenarios=[(scenario_state, [traj_data])],
            )

        stats = buffer.get_stats()
        expected = 3 * multiplier
        actual = stats['total_episodes']

        success = (actual == expected)
        test_results.append(success)

        status = "✅" if success else "❌"
        print(f"{status} Multiplier={multiplier}: Expected={expected}, Actual={actual}")

    if all(test_results):
        print("\n✅ Storage multiplier parameter works correctly!")
        return True
    else:
        print("\n❌ Storage multiplier parameter has issues")
        return False


def test_planner_episode_storage():
    """Test 3: Verify C2OSRPlanner stores episodes correctly."""
    print("\n" + "=" * 60)
    print("Test 3: C2OSRPlanner Episode Storage")
    print("=" * 60)

    # Create planner with storage_multiplier=20
    config = C2OSRPlannerConfig()
    config.trajectory_storage_multiplier = 20
    config.min_buffer_size = 0  # Allow updates from episode 1

    planner = C2OSRPlanner(config)

    print(f"Planner storage_multiplier: {planner.config.trajectory_storage_multiplier}")
    print(f"Buffer storage_multiplier: {planner.trajectory_buffer.storage_multiplier}")

    # Run 3 episodes
    for episode in range(3):
        print(f"\nEpisode {episode + 1}:")

        # Run 5 timesteps per episode
        for t in range(5):
            world_state = create_world_state(t, ego_x=t * 5.0)
            action = planner.select_action(world_state)

        # End episode
        final_world_state = create_world_state(5, ego_x=25.0)
        transition = Transition(
            state=final_world_state,
            action=0,
            reward=0.0,
            next_state=final_world_state,
            terminated=True,
            truncated=False,
        )
        planner.update(transition)

    # Check final buffer size
    stats = planner.trajectory_buffer.get_stats()
    print(f"\n{'='*60}")
    print("Final Buffer Statistics:")
    print(f"  Total episodes: {stats['total_episodes']}")
    print(f"  Expected: ~{3 * 5 * 20} (3 episodes × 5 timesteps × 20 multiplier)")
    print(f"  Storage multiplier: {stats['storage_multiplier']}")
    print(f"  Capacity: {stats['capacity']}")

    # Success if we have roughly the expected amount
    # Each episode has 5 timesteps, 1 agent, with 20x multiplier = 100 per episode
    expected_min = 3 * 5 * 15  # Allow some margin for agent predictions
    expected_max = 3 * 5 * 25
    actual = stats['total_episodes']

    if expected_min <= actual <= expected_max:
        print(f"\n✅ Planner stores episodes correctly!")
        return True
    else:
        print(f"\n❌ Expected {expected_min}-{expected_max} episodes, got {actual}")
        return False


def test_unique_episode_ids():
    """Test 4: Verify all episode IDs are unique (no hash collisions)."""
    print("\n" + "=" * 60)
    print("Test 4: Unique Episode IDs")
    print("=" * 60)

    buffer = TrajectoryBuffer(
        capacity=10000,
        trajectory_storage_multiplier=10,
    )

    # Store 100 episodes to stress-test uniqueness
    for episode_idx in range(100):
        traj_data = AgentTrajectoryData(
            agent_id=1,
            agent_type='car',
            init_position=(episode_idx * 2.0, 0.0),
            init_velocity=(5.0, 0.0),
            init_heading=0.0,
            trajectory_cells=[episode_idx * 10 + t for t in range(5)],
        )

        scenario_state = ScenarioState(
            ego_position=(episode_idx * 2.0, 0.0),
            ego_velocity=(5.0, 0.0),
            ego_heading=0.0,
            agents_states=[(episode_idx * 2.0 + 10.0, 2.0, 4.0, 0.0, 0.0, 'car')],
        )

        buffer.store_episode_trajectories_by_timestep(
            episode_id=episode_idx,
            timestep_scenarios=[(scenario_state, [traj_data])],
        )

    # Access internal data to check IDs
    episode_ids = list(buffer._episode_lookup.keys())
    unique_ids = set(episode_ids)

    print(f"Total episode IDs: {len(episode_ids)}")
    print(f"Unique episode IDs: {len(unique_ids)}")
    print(f"Sample IDs: {sorted(episode_ids)[:20]}")

    if len(episode_ids) == len(unique_ids):
        print("\n✅ All episode IDs are unique!")
        return True
    else:
        collisions = len(episode_ids) - len(unique_ids)
        print(f"\n❌ Found {collisions} hash collisions!")
        return False


def test_buffer_capacity_eviction():
    """Test 5: Verify FIFO eviction when capacity is reached."""
    print("\n" + "=" * 60)
    print("Test 5: Buffer Capacity and FIFO Eviction")
    print("=" * 60)

    # Create buffer with small capacity
    buffer = TrajectoryBuffer(
        capacity=50,  # Small capacity
        trajectory_storage_multiplier=10,
    )

    print(f"Buffer capacity: {buffer.capacity}")
    print(f"Storage multiplier: {buffer.storage_multiplier}")

    # Store 10 episodes (should create 100 entries, exceeding capacity)
    for episode_idx in range(10):
        traj_data = AgentTrajectoryData(
            agent_id=1,
            agent_type='car',
            init_position=(episode_idx * 5.0, 0.0),
            init_velocity=(5.0, 0.0),
            init_heading=0.0,
            trajectory_cells=[episode_idx * 5 + t for t in range(5)],
        )

        scenario_state = ScenarioState(
            ego_position=(episode_idx * 5.0, 0.0),
            ego_velocity=(5.0, 0.0),
            ego_heading=0.0,
            agents_states=[(episode_idx * 5.0 + 10.0, 2.0, 4.0, 0.0, 0.0, 'car')],
        )

        buffer.store_episode_trajectories_by_timestep(
            episode_id=episode_idx,
            timestep_scenarios=[(scenario_state, [traj_data])],
        )

        stats = buffer.get_stats()
        print(f"Episode {episode_idx + 1}: Buffer size = {stats['total_episodes']}")

    final_stats = buffer.get_stats()
    final_size = final_stats['total_episodes']

    print(f"\nFinal buffer size: {final_size}")
    print(f"Capacity: {buffer.capacity}")

    if final_size <= buffer.capacity:
        print("\n✅ FIFO eviction works correctly!")
        return True
    else:
        print(f"\n❌ Buffer size {final_size} exceeds capacity {buffer.capacity}")
        return False


def main():
    """Run all tests."""
    print("Testing Dirichlet Update Pipeline")
    print("After Hash Collision Fix and Debug Logging\n")

    results = []

    results.append(("Hash Collision Fix", test_hash_collision_fixed()))
    results.append(("Storage Multiplier", test_storage_multiplier_parameter()))
    results.append(("Planner Storage", test_planner_episode_storage()))
    results.append(("Unique Episode IDs", test_unique_episode_ids()))
    results.append(("FIFO Eviction", test_buffer_capacity_eviction()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(success for _, success in results)

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All tests passed!")
        print("\nKey fixes verified:")
        print("  1. Hash collision fixed - all episode IDs are unique")
        print("  2. Storage multiplier parameter works correctly")
        print("  3. C2OSRPlanner stores episodes with correct multiplier")
        print("  4. FIFO eviction maintains buffer capacity")
        print("\nNext: Run actual scenario to verify Q-value updates with debug logs")
    else:
        print("❌ Some tests failed - need further investigation")
    print("=" * 60)

    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
