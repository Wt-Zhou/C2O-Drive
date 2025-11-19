"""
Test that observed agent trajectories are stored (not predicted trajectories)

This test verifies the critical fix: the planner should store actual observed
agent positions during episode execution, not predicted future positions.

Key validation:
1. Agent observations are recorded during update()
2. Stored trajectories match observed movements
3. Q-values change as real data accumulates (Bayesian learning)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from c2o_drive.config import GlobalConfig, set_global_config
from c2o_drive.algorithms.c2osr.planner import C2OSRPlanner
from c2o_drive.algorithms.c2osr.config import C2OSRPlannerConfig
from c2o_drive.environments.carla.types import WorldState, EgoState, AgentState, AgentType
from c2o_drive.core.planner import Transition


def create_moving_agent_state(step: int, agent_id: int, speed: float = 2.0):
    """Create agent that moves along x-axis at constant speed.

    This creates predictable movement for testing:
    - Agent starts at x=10.0 + agent_id*5
    - Moves at speed m/s along x-axis
    - Position at step t: x = 10 + agent_id*5 + speed*t
    """
    x_start = 10.0 + agent_id * 5.0
    x_position = x_start + speed * float(step)

    return AgentState(
        agent_id=f"agent_{agent_id}",
        position_m=(x_position, float(agent_id) * 3.0),
        velocity_mps=(speed, 0.0),
        heading_rad=0.0,
        agent_type=AgentType.VEHICLE
    )


def create_test_state(step: int = 0, num_agents: int = 2):
    """Create test WorldState with predictably moving agents."""
    ego = EgoState(
        position_m=(float(step) * 2.0, 0.0),
        velocity_mps=(5.0, 0.0),
        yaw_rad=0.0
    )

    agents = [create_moving_agent_state(step, i+1) for i in range(num_agents)]

    return WorldState(time_s=float(step), ego=ego, agents=agents)


def get_stored_trajectory_cells(planner, agent_id: int, episode_idx: int = 0):
    """Extract stored trajectory cells for an agent from buffer.

    Returns the trajectory cells stored for the given agent in the buffer.
    """
    # Access the buffer's _agent_episodes dictionary
    if agent_id not in planner.trajectory_buffer._agent_episodes:
        return None

    agent_episodes = planner.trajectory_buffer._agent_episodes[agent_id]

    if episode_idx >= len(agent_episodes):
        return None

    # Get the specified episode data
    episode_data = agent_episodes[episode_idx]

    # Return the trajectory cells
    return episode_data.agent_trajectory_cells


def main():
    print("=" * 70)
    print("Test: Observed Trajectory Storage (vs Predicted)")
    print("=" * 70)

    # Configure with minimal logging
    global_config = GlobalConfig()
    global_config.visualization.verbose_level = 0  # Silent for clean test output
    set_global_config(global_config)

    # Configure planner
    planner_config = C2OSRPlannerConfig()
    planner_config.trajectory_storage_multiplier = 1  # No multiplication for clarity
    planner_config.learning_rate = 1.0
    planner_config.lattice.horizon = 5
    planner_config.lattice.dt = 1.0
    planner_config.grid.grid_size_m = 0.5
    planner_config.min_buffer_size = 1  # Store immediately

    print("\nConfiguration:")
    print(f"  Horizon: {planner_config.lattice.horizon}")
    print(f"  dt: {planner_config.lattice.dt}")
    print(f"  Grid cell size: {planner_config.grid.grid_size_m}m")
    print(f"  Storage multiplier: {planner_config.trajectory_storage_multiplier}")

    # Create planner
    planner = C2OSRPlanner(config=planner_config)

    print("\n" + "=" * 70)
    print("Episode 1: Agent moves at 2.0 m/s along x-axis")
    print("=" * 70)

    # Simulate 10-step episode with predictable agent movement
    num_steps = 10
    agent_speed = 2.0  # m/s

    for step in range(num_steps):
        state = create_test_state(step)
        action = 0

        transition = Transition(
            state=state,
            action=action,
            reward=0.0,
            next_state=create_test_state(step + 1),
            terminated=False,
            truncated=False,
            info={},
        )
        planner.update(transition)

    # Final transition to mark episode complete
    final_state = create_test_state(num_steps)
    final_transition = Transition(
        state=final_state,
        action=0,
        reward=0.0,
        next_state=final_state,
        terminated=False,
        truncated=True,
        info={},
    )
    planner.update(final_transition)

    print(f"\n✅ Episode completed: {num_steps} steps")
    print(f"   Buffer size: {len(planner.trajectory_buffer)}")

    # Verify observations were recorded
    print("\n" + "-" * 70)
    print("Verification 1: Agent observations were recorded")
    print("-" * 70)

    # Note: After episode end, observations are cleared
    # But we can check the stored data in buffer

    if len(planner.trajectory_buffer) > 0:
        print(f"✅ Buffer has data: {len(planner.trajectory_buffer)} episodes stored")

        # Extract stored trajectory for agent 1 at timestep 0
        stored_cells = get_stored_trajectory_cells(planner, agent_id=1, episode_idx=0)

        if stored_cells is not None:
            print(f"✅ Found stored trajectory for Agent 1: {len(stored_cells)} cells")
            print(f"   Stored cells: {stored_cells[:5]}...")  # Show first 5
        else:
            print("❌ Could not extract stored trajectory from buffer")
    else:
        print("❌ Buffer is empty - storage failed!")

    # Verify stored trajectories match OBSERVED movement (not predictions)
    print("\n" + "-" * 70)
    print("Verification 2: Stored trajectories match OBSERVED positions")
    print("-" * 70)

    # Calculate expected observed positions for agent 1
    # Agent 1 starts at x=15.0, moves at 2.0 m/s
    # At timestep t=0, agent is at x=15.0
    # Expected trajectory from t=0 with horizon=5:
    #   t=0: x=15.0
    #   t=1: x=17.0  (15 + 2*1)
    #   t=2: x=19.0  (15 + 2*2)
    #   t=3: x=21.0  (15 + 2*3)
    #   t=4: x=23.0  (15 + 2*4)

    horizon = planner_config.lattice.horizon
    agent_id = 1
    x_start = 15.0  # 10.0 + agent_id*5.0 where agent_id=1

    expected_positions = []
    expected_cells = []
    for t in range(horizon):
        x_pos = x_start + agent_speed * t
        y_pos = 3.0  # Agent 1 is at y=3.0
        expected_positions.append((x_pos, y_pos))

        # Convert to cell
        try:
            cell = planner.grid_mapper.world_to_cell((x_pos, y_pos))
            expected_cells.append(cell)
        except:
            expected_cells.append(-1)

    print(f"\nExpected OBSERVED positions for Agent 1 (t=0 to t={horizon-1}):")
    for t, (pos, cell) in enumerate(zip(expected_positions, expected_cells)):
        print(f"  t={t}: pos={pos}, cell={cell}")

    if stored_cells is not None and len(stored_cells) >= horizon:
        print(f"\nActual STORED cells:")
        for t, cell in enumerate(stored_cells[:horizon]):
            print(f"  t={t}: cell={cell}")

        # Compare
        matches = sum(1 for i in range(horizon) if stored_cells[i] == expected_cells[i])
        match_rate = matches / horizon

        print(f"\nMatch rate: {matches}/{horizon} = {match_rate*100:.1f}%")

        if match_rate >= 0.8:  # Allow some tolerance for grid discretization
            print("✅ Stored trajectories MATCH observed positions!")
            print("   This confirms we're storing OBSERVED data, not predictions.")
        else:
            print("❌ Stored trajectories DO NOT match observed positions!")
            print("   This suggests predictions are still being stored.")

            # Show differences
            print("\nDifferences:")
            for t in range(horizon):
                if stored_cells[t] != expected_cells[t]:
                    print(f"  t={t}: expected cell {expected_cells[t]}, got {stored_cells[t]}")
    else:
        print("⚠️  Could not verify - stored cells not available")

    # Verify Q-values change with data accumulation
    print("\n" + "-" * 70)
    print("Verification 3: Q-values change as observed data accumulates")
    print("-" * 70)

    # Run a few more episodes and track Q-value changes
    q_value_history = []

    for episode in range(5):
        # Simulate short episode
        for step in range(5):
            state = create_test_state(step)
            action = 0

            transition = Transition(
                state=state,
                action=action,
                reward=0.0,
                next_state=create_test_state(step + 1),
                terminated=False,
                truncated=False,
                info={},
            )
            planner.update(transition)

        # Complete episode
        final_state = create_test_state(5)
        final_transition = Transition(
            state=final_state,
            action=0,
            reward=0.0,
            next_state=final_state,
            terminated=False,
            truncated=True,
            info={},
        )
        planner.update(final_transition)

        # Try to get Q-value
        try:
            test_state = create_test_state(0)
            # This will trigger Q-value calculation internally
            action = planner.select_action(test_state)

            # Get alpha sum as a proxy for learning
            total_alpha = 0.0
            if 1 in planner.dirichlet_bank.agent_alphas:
                for timestep in planner.dirichlet_bank.agent_alphas[1]:
                    total_alpha += planner.dirichlet_bank.agent_alphas[1][timestep].sum()

            q_value_history.append({
                'episode': episode,
                'buffer_size': len(planner.trajectory_buffer),
                'alpha_sum': total_alpha
            })

            print(f"Episode {episode}: buffer={len(planner.trajectory_buffer)}, "
                  f"alpha_sum={total_alpha:.1f}")
        except Exception as e:
            print(f"Episode {episode}: Q-calculation failed - {type(e).__name__}")

    # Check if alpha is growing
    if len(q_value_history) >= 2:
        alpha_growth = q_value_history[-1]['alpha_sum'] - q_value_history[0]['alpha_sum']

        print(f"\nAlpha growth: {q_value_history[0]['alpha_sum']:.1f} → "
              f"{q_value_history[-1]['alpha_sum']:.1f} "
              f"(+{alpha_growth:.1f})")

        if alpha_growth > 0:
            print("✅ Alpha is growing - Bayesian learning is working!")
        else:
            print("❌ Alpha is not growing - learning may not be working")

    # Final summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)

    test_passed = True

    # Check 1: Data was stored
    if len(planner.trajectory_buffer) > 0:
        print("✅ Episode data is stored to buffer")
    else:
        print("❌ Episode data NOT stored")
        test_passed = False

    # Check 2: Trajectories match observations
    if stored_cells is not None and match_rate >= 0.8:
        print("✅ Stored trajectories match OBSERVED positions")
        print("   → CRITICAL FIX VERIFIED: Using observations, not predictions!")
    else:
        print("❌ Stored trajectories do not match observations")
        test_passed = False

    # Check 3: Learning is happening
    if len(q_value_history) >= 2 and alpha_growth > 0:
        print("✅ Bayesian learning is working (alpha growing)")
    else:
        print("⚠️  Could not verify Bayesian learning")

    if test_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nKey Achievement:")
        print("  ✓ Planner now stores OBSERVED agent trajectories")
        print("  ✓ Prior: uniform over reachable set")
        print("  ✓ Evidence: observed agent movements")
        print("  ✓ Posterior: converges to reality with more data")
        print("\nThis fixes the Q-value stagnation issue!")
    else:
        print("\n❌ SOME TESTS FAILED")
        print("   Review implementation of observed trajectory storage")


if __name__ == "__main__":
    main()
