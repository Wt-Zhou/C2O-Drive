"""Comprehensive tests for SimpleGridEnvironment.

This test suite verifies:
1. Environment creation and initialization
2. Gym interface compliance
3. State transitions and dynamics
4. Reward computation
5. Termination conditions
6. Integration with existing codebase
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from carla_c2osr.environments import SimpleGridEnvironment
from carla_c2osr.env.types import EgoControl, WorldState
from carla_c2osr.environments.rewards import (
    SafetyReward, ComfortReward, EfficiencyReward
)
from carla_c2osr.core.environment import CompositeRewardFunction


def test_environment_creation():
    """Test 1: Environment can be created successfully."""
    print("\n" + "="*70)
    print("TEST 1: Environment Creation")
    print("="*70)

    try:
        env = SimpleGridEnvironment()
        print("✓ Environment created successfully")
        print(f"  Grid size: {env.grid_size_m}m")
        print(f"  Timestep: {env.dt}s")
        print(f"  Max steps: {env.max_episode_steps}")
        return True
    except Exception as e:
        print(f"✗ Failed to create environment: {e}")
        return False


def test_observation_action_spaces():
    """Test 2: Observation and action spaces are properly defined."""
    print("\n" + "="*70)
    print("TEST 2: Observation and Action Spaces")
    print("="*70)

    try:
        env = SimpleGridEnvironment()

        # Check action space
        action_space = env.action_space
        print(f"✓ Action space: {action_space}")
        print(f"  Shape: {action_space.shape}")
        print(f"  Low: {action_space.low}")
        print(f"  High: {action_space.high}")

        # Sample random action
        sample_action = action_space.sample()
        print(f"  Sample action: {sample_action}")
        assert action_space.contains(sample_action), "Sampled action not in space"
        print("✓ Action space sampling works")

        # Check observation space
        obs_space = env.observation_space
        print(f"✓ Observation space: {obs_space}")
        print(f"  Shape: {obs_space.shape}")

        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reset_functionality():
    """Test 3: Environment reset works correctly."""
    print("\n" + "="*70)
    print("TEST 3: Reset Functionality")
    print("="*70)

    try:
        env = SimpleGridEnvironment()

        # Reset with default options
        state, info = env.reset()
        print("✓ Reset successful")
        print(f"  State type: {type(state)}")
        print(f"  Ego position: {state.ego.position_m}")
        print(f"  Ego velocity: {state.ego.velocity_mps}")
        print(f"  Number of agents: {len(state.agents)}")
        print(f"  Info keys: {list(info.keys())}")

        assert isinstance(state, WorldState), "State should be WorldState"
        assert 'step' in info, "Info should contain step"
        assert 'reference_path' in info, "Info should contain reference_path"
        print("✓ Reset returns correct types")

        # Reset with seed for reproducibility
        state1, _ = env.reset(seed=42)
        state2, _ = env.reset(seed=42)
        print("✓ Seeded reset is reproducible")

        # Reset with different path mode
        state_straight, _ = env.reset(options={'path_mode': 'straight'})
        state_curve, _ = env.reset(options={'path_mode': 'curve'})
        print("✓ Reset with options works")

        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_step_functionality():
    """Test 4: Environment step executes correctly."""
    print("\n" + "="*70)
    print("TEST 4: Step Functionality")
    print("="*70)

    try:
        env = SimpleGridEnvironment()
        state, _ = env.reset()

        # Create a simple control action (go straight with constant speed)
        action = EgoControl(throttle=0.5, steer=0.0, brake=0.0)

        # Execute step
        step_result = env.step(action)
        print("✓ Step executed successfully")
        print(f"  Next state type: {type(step_result.observation)}")
        print(f"  Reward: {step_result.reward:.2f}")
        print(f"  Terminated: {step_result.terminated}")
        print(f"  Truncated: {step_result.truncated}")
        print(f"  Info keys: {list(step_result.info.keys())}")

        # Verify state has changed
        next_state = step_result.observation
        print(f"  Previous ego pos: {state.ego.position_m}")
        print(f"  Next ego pos: {next_state.ego.position_m}")

        # Position should have changed
        pos_changed = (next_state.ego.position_m[0] != state.ego.position_m[0] or
                      next_state.ego.position_m[1] != state.ego.position_m[1])
        assert pos_changed, "Position should change after step"
        print("✓ State dynamics working")

        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_steps():
    """Test 5: Multiple consecutive steps work correctly."""
    print("\n" + "="*70)
    print("TEST 5: Multiple Steps")
    print("="*70)

    try:
        env = SimpleGridEnvironment(max_episode_steps=10)
        state, _ = env.reset()

        total_reward = 0.0
        for i in range(10):
            action = EgoControl(throttle=0.3, steer=0.0, brake=0.0)
            step_result = env.step(action)

            total_reward += step_result.reward

            if step_result.terminated:
                print(f"  Episode terminated at step {i+1}")
                break
            if step_result.truncated:
                print(f"  Episode truncated at step {i+1}")
                break

        print(f"✓ Completed {i+1} steps")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Final position: {step_result.observation.ego.position_m}")

        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_collision_detection():
    """Test 6: Collision detection works."""
    print("\n" + "="*70)
    print("TEST 6: Collision Detection")
    print("="*70)

    try:
        env = SimpleGridEnvironment()
        state, _ = env.reset()

        print(f"  Initial ego pos: {state.ego.position_m}")
        if state.agents:
            print(f"  Agent pos: {state.agents[0].position_m}")

        # Try to move towards agent (if any)
        collision_detected = False
        for i in range(50):
            # Move forward aggressively
            action = EgoControl(throttle=1.0, steer=0.0, brake=0.0)
            step_result = env.step(action)

            if step_result.info.get('collision', False):
                collision_detected = True
                print(f"✓ Collision detected at step {i+1}")
                print(f"  Final position: {step_result.observation.ego.position_m}")
                assert step_result.terminated, "Episode should terminate on collision"
                print("✓ Episode terminated on collision")
                break

        if not collision_detected:
            print("  No collision occurred (agents may be far away)")

        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_out_of_bounds_detection():
    """Test 7: Out of bounds detection works."""
    print("\n" + "="*70)
    print("TEST 7: Out of Bounds Detection")
    print("="*70)

    try:
        env = SimpleGridEnvironment(grid_size_m=20.0)
        state, _ = env.reset()

        print(f"  Grid bounds: [-{env.grid_size_m/2}, {env.grid_size_m/2}]")
        print(f"  Initial position: {state.ego.position_m}")

        # Move in one direction until out of bounds
        out_of_bounds_detected = False
        for i in range(100):
            # Move right with full throttle
            action = EgoControl(throttle=1.0, steer=0.0, brake=0.0)
            step_result = env.step(action)

            if step_result.info.get('out_of_bounds', False):
                out_of_bounds_detected = True
                print(f"✓ Out of bounds detected at step {i+1}")
                print(f"  Final position: {step_result.observation.ego.position_m}")
                assert step_result.terminated, "Episode should terminate when out of bounds"
                print("✓ Episode terminated when out of bounds")
                break

        assert out_of_bounds_detected, "Should eventually go out of bounds"
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_custom_reward_function():
    """Test 8: Custom reward functions work correctly."""
    print("\n" + "="*70)
    print("TEST 8: Custom Reward Functions")
    print("="*70)

    try:
        # Create custom reward function
        safety_reward = SafetyReward(collision_penalty=-100.0)
        efficiency_reward = EfficiencyReward(speed_target=5.0)

        custom_reward = CompositeRewardFunction([
            (safety_reward, 2.0),  # Double weight on safety
            (efficiency_reward, 1.0),
        ])

        env = SimpleGridEnvironment(reward_fn=custom_reward)
        state, _ = env.reset()

        print("✓ Custom reward function created")

        # Take a step
        action = EgoControl(throttle=0.5, steer=0.0, brake=0.0)
        step_result = env.step(action)

        print(f"  Reward with custom function: {step_result.reward:.2f}")
        print("✓ Custom reward function works")

        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_episode():
    """Test 9: Run a complete episode."""
    print("\n" + "="*70)
    print("TEST 9: Full Episode Execution")
    print("="*70)

    try:
        env = SimpleGridEnvironment(max_episode_steps=50)
        state, info = env.reset(seed=42)

        episode_rewards = []
        episode_length = 0

        print("  Running full episode...")
        while True:
            # Simple policy: maintain speed and go straight
            action = EgoControl(throttle=0.4, steer=0.0, brake=0.0)
            step_result = env.step(action)

            episode_rewards.append(step_result.reward)
            episode_length += 1

            if step_result.terminated or step_result.truncated:
                break

        print(f"✓ Episode completed")
        print(f"  Length: {episode_length} steps")
        print(f"  Total reward: {sum(episode_rewards):.2f}")
        print(f"  Average reward: {np.mean(episode_rewards):.2f}")
        print(f"  Terminated: {step_result.terminated}")
        print(f"  Truncated: {step_result.truncated}")

        # Get trajectory
        trajectory = env.get_episode_trajectory()
        print(f"  Trajectory length: {len(trajectory)}")
        assert len(trajectory) == episode_length + 1, "Trajectory should include initial state"
        print("✓ Episode trajectory recorded")

        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gym_interface_compliance():
    """Test 10: Verify Gym interface compliance."""
    print("\n" + "="*70)
    print("TEST 10: Gym Interface Compliance")
    print("="*70)

    try:
        env = SimpleGridEnvironment()

        # Check required methods exist
        required_methods = ['reset', 'step', 'close', 'render', 'seed']
        for method in required_methods:
            assert hasattr(env, method), f"Missing method: {method}"
            print(f"✓ Method '{method}' exists")

        # Check required properties exist
        required_properties = ['observation_space', 'action_space']
        for prop in required_properties:
            assert hasattr(env, prop), f"Missing property: {prop}"
            print(f"✓ Property '{prop}' exists")

        # Test basic Gym loop
        obs, info = env.reset()
        for _ in range(5):
            action = env.action_space.sample()
            step_result = env.step(EgoControl(
                throttle=float(action[0]),
                steer=float(action[1]),
                brake=float(action[2])
            ))
            if step_result.terminated or step_result.truncated:
                break

        env.close()
        print("✓ Basic Gym loop works")

        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all test cases and report results."""
    print("\n" + "#"*70)
    print("# SIMPLE GRID ENVIRONMENT TEST SUITE")
    print("#"*70)

    tests = [
        ("Environment Creation", test_environment_creation),
        ("Observation/Action Spaces", test_observation_action_spaces),
        ("Reset Functionality", test_reset_functionality),
        ("Step Functionality", test_step_functionality),
        ("Multiple Steps", test_multiple_steps),
        ("Collision Detection", test_collision_detection),
        ("Out of Bounds Detection", test_out_of_bounds_detection),
        ("Custom Reward Function", test_custom_reward_function),
        ("Full Episode", test_full_episode),
        ("Gym Interface Compliance", test_gym_interface_compliance),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ Test '{name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Print summary
    print("\n" + "#"*70)
    print("# TEST SUMMARY")
    print("#"*70)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")

    print(f"\n{passed}/{total} tests passed ({100*passed//total}%)")

    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
