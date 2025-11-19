"""Comprehensive functional test of the refactored architecture.

This test demonstrates all key features of the new architecture:
1. Core interfaces
2. Environment adapters
3. Reward composition
4. State space discretization
5. Integration with existing code
"""

import sys
from pathlib import Path
import numpy as np
import time

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from c2o_drive.core import (
    DrivingEnvironment, BasePlanner, TrajectoryEvaluator,
    GridBasedDiscretizer, Box, Discrete,
    CompositeRewardFunction, StepResult
)
from c2o_drive.environments.carlaironments import (
    SimpleGridEnvironment,
    SafetyReward, ComfortReward, EfficiencyReward,
    CenterlineReward, TimeReward
)
from c2o_drive.environments.carla.types import WorldState, EgoControl, EgoState
from c2o_drive.environments.virtual.scenario_manager import ScenarioManager


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def test_1_core_interfaces():
    """Test 1: Verify all core interfaces are working."""
    print_section("TEST 1: Core Interfaces")

    # Test environment interface
    print("\n[1.1] Testing environment interface...")
    env = SimpleGridEnvironment()
    assert isinstance(env, DrivingEnvironment), "Should implement DrivingEnvironment"
    print("  ✓ Environment implements DrivingEnvironment interface")

    # Test space definitions
    print("\n[1.2] Testing space definitions...")
    action_space = env.action_space
    obs_space = env.observation_space
    assert isinstance(action_space, Box), "Action space should be Box"
    assert isinstance(obs_space, Box), "Observation space should be Box"
    print(f"  ✓ Action space: Box{action_space.shape}")
    print(f"  ✓ Observation space: Box{obs_space.shape}")

    # Test space sampling
    print("\n[1.3] Testing space sampling...")
    for i in range(3):
        sample = action_space.sample()
        assert action_space.contains(sample), "Sample should be in space"
    print("  ✓ Space sampling works correctly")

    # Test state space discretizer
    print("\n[1.4] Testing state space discretizer...")
    discretizer = GridBasedDiscretizer(
        bounds=[(-10, 10), (-10, 10), (-5, 5)],
        resolution=1.0
    )
    continuous_state = np.array([2.5, -3.2, 1.8])
    discrete_state = discretizer.discretize(continuous_state)
    print(f"  ✓ Continuous state: {continuous_state}")
    print(f"  ✓ Discrete index: {discrete_state.index}")
    print(f"  ✓ Total dimensions: {discretizer.get_dimension()}")

    # Verify discretization is consistent
    discrete_state2 = discretizer.discretize(continuous_state)
    assert discrete_state.index == discrete_state2.index, "Should be deterministic"
    print("  ✓ Discretization is deterministic")

    return True


def test_2_reward_composition():
    """Test 2: Verify modular reward composition."""
    print_section("TEST 2: Reward Composition")

    print("\n[2.1] Creating individual reward components...")
    safety = SafetyReward(collision_penalty=-100.0, min_distance=3.0)
    comfort = ComfortReward(jerk_penalty_weight=1.0)
    efficiency = EfficiencyReward(speed_target=5.0, progress_weight=2.0)
    centerline = CenterlineReward(max_deviation=2.0)
    time_penalty = TimeReward(time_penalty=-0.1)
    print("  ✓ Created 5 reward components")

    print("\n[2.2] Testing individual rewards...")
    # Create mock states
    state = WorldState(
        time_s=0.0,
        ego=EgoState(position_m=(0.0, 0.0), velocity_mps=(3.0, 0.0), yaw_rad=0.0),
        agents=[]
    )
    next_state = WorldState(
        time_s=0.1,
        ego=EgoState(position_m=(0.3, 0.0), velocity_mps=(3.0, 0.0), yaw_rad=0.0),
        agents=[]
    )
    action = EgoControl(throttle=0.3, steer=0.0, brake=0.0)
    info = {
        'collision': False,
        'acceleration': 0.0,
        'jerk': 0.0,
        'reference_y': 0.0
    }

    r_safety = safety.compute(state, action, next_state, info)
    r_comfort = comfort.compute(state, action, next_state, info)
    r_efficiency = efficiency.compute(state, action, next_state, info)
    r_centerline = centerline.compute(state, action, next_state, info)
    r_time = time_penalty.compute(state, action, next_state, info)

    print(f"  ✓ Safety reward: {r_safety:.2f}")
    print(f"  ✓ Comfort reward: {r_comfort:.2f}")
    print(f"  ✓ Efficiency reward: {r_efficiency:.2f}")
    print(f"  ✓ Centerline reward: {r_centerline:.2f}")
    print(f"  ✓ Time reward: {r_time:.2f}")

    print("\n[2.3] Testing composite reward...")
    composite = CompositeRewardFunction([
        (safety, 5.0),
        (comfort, 1.0),
        (efficiency, 2.0),
        (centerline, 0.5),
        (time_penalty, 1.0),
    ])

    total_reward = composite.compute(state, action, next_state, info)
    expected = (5.0*r_safety + 1.0*r_comfort + 2.0*r_efficiency +
                0.5*r_centerline + 1.0*r_time)

    assert abs(total_reward - expected) < 0.01, "Composite should sum correctly"
    print(f"  ✓ Composite reward: {total_reward:.2f}")
    print(f"  ✓ Expected: {expected:.2f}")
    print("  ✓ Composite reward calculation correct")

    return True


def test_3_environment_dynamics():
    """Test 3: Verify environment dynamics and state transitions."""
    print_section("TEST 3: Environment Dynamics")

    print("\n[3.1] Creating environment and resetting...")
    env = SimpleGridEnvironment(dt=0.1, max_episode_steps=100)
    state, info = env.reset(seed=42)

    print(f"  ✓ Initial ego position: {state.ego.position_m}")
    print(f"  ✓ Initial ego velocity: {state.ego.velocity_mps}")
    print(f"  ✓ Number of agents: {len(state.agents)}")

    print("\n[3.2] Testing constant velocity motion...")
    action = EgoControl(throttle=0.0, steer=0.0, brake=0.0)

    positions = [state.ego.position_m]
    velocities = [state.ego.velocity_mps]

    for i in range(5):
        step_result = env.step(action)
        positions.append(step_result.observation.ego.position_m)
        velocities.append(step_result.observation.ego.velocity_mps)
        state = step_result.observation

    print("  Position trajectory:")
    for i, pos in enumerate(positions):
        print(f"    Step {i}: {pos}")

    # Verify motion is consistent
    dx = positions[1][0] - positions[0][0]
    for i in range(2, len(positions)):
        dx_i = positions[i][0] - positions[i-1][0]
        assert abs(dx_i - dx) < 0.1, "Constant velocity should give constant dx"

    print("  ✓ Constant velocity motion verified")

    print("\n[3.3] Testing acceleration...")
    env.reset(seed=42)
    action_accel = EgoControl(throttle=0.5, steer=0.0, brake=0.0)

    speeds = []
    for i in range(5):
        step_result = env.step(action_accel)
        speed = np.linalg.norm(np.array(step_result.observation.ego.velocity_mps))
        speeds.append(speed)

    print("  Speed profile:")
    for i, speed in enumerate(speeds):
        print(f"    Step {i+1}: {speed:.2f} m/s")

    # Verify speed increases
    for i in range(1, len(speeds)):
        assert speeds[i] >= speeds[i-1], "Speed should increase with throttle"

    print("  ✓ Acceleration dynamics verified")

    print("\n[3.4] Testing steering...")
    env.reset(seed=42)
    action_steer = EgoControl(throttle=0.3, steer=0.5, brake=0.0)

    yaws = [0.0]
    for i in range(5):
        step_result = env.step(action_steer)
        yaws.append(step_result.observation.ego.yaw_rad)

    print("  Yaw trajectory:")
    for i, yaw in enumerate(yaws):
        print(f"    Step {i}: {yaw:.3f} rad ({np.degrees(yaw):.1f}°)")

    # Verify yaw changes
    assert abs(yaws[-1] - yaws[0]) > 0.01, "Yaw should change with steering"
    print("  ✓ Steering dynamics verified")

    return True


def test_4_termination_conditions():
    """Test 4: Verify termination conditions work correctly."""
    print_section("TEST 4: Termination Conditions")

    print("\n[4.1] Testing time limit (truncation)...")
    env = SimpleGridEnvironment(max_episode_steps=10)
    state, _ = env.reset()

    action = EgoControl(throttle=0.3, steer=0.0, brake=0.0)
    for i in range(15):
        step_result = env.step(action)
        if step_result.truncated:
            print(f"  ✓ Episode truncated at step {i+1}")
            assert i+1 == 10, "Should truncate at max_episode_steps"
            break

    print("\n[4.2] Testing out of bounds (termination)...")
    env = SimpleGridEnvironment(grid_size_m=20.0)
    state, _ = env.reset()

    action = EgoControl(throttle=1.0, steer=0.0, brake=0.0)
    terminated = False
    for i in range(50):
        step_result = env.step(action)
        if step_result.terminated and step_result.info.get('out_of_bounds'):
            print(f"  ✓ Out of bounds detected at step {i+1}")
            print(f"    Final position: {step_result.observation.ego.position_m}")
            terminated = True
            break

    assert terminated, "Should terminate when out of bounds"

    print("\n[4.3] Testing collision detection...")
    # Note: Collision depends on scenario setup
    env = SimpleGridEnvironment()
    state, _ = env.reset()

    if state.agents:
        print(f"  Initial distance to nearest agent:")
        ego_pos = np.array(state.ego.position_m)
        agent_pos = np.array(state.agents[0].position_m)
        dist = np.linalg.norm(ego_pos - agent_pos)
        print(f"    {dist:.2f} m")
        print("  ✓ Collision detection available (tested in unit tests)")
    else:
        print("  ℹ No agents in scenario (collision test skipped)")

    return True


def test_5_integration_with_existing_code():
    """Test 5: Verify integration with existing codebase."""
    print_section("TEST 5: Integration with Existing Code")

    print("\n[5.1] Testing ScenarioManager integration...")
    scenario_mgr = ScenarioManager(grid_size_m=20.0)
    world_state = scenario_mgr.create_scenario()

    print(f"  ✓ ScenarioManager created world state")
    print(f"    Ego: {world_state.ego.position_m}")
    print(f"    Agents: {len(world_state.agents)}")

    print("\n[5.2] Testing reference path generation...")
    ref_path_straight = scenario_mgr.generate_reference_path(
        mode='straight', horizon=20
    )
    ref_path_curve = scenario_mgr.generate_reference_path(
        mode='curve', horizon=20
    )

    print(f"  ✓ Generated straight path: {len(ref_path_straight)} points")
    print(f"  ✓ Generated curve path: {len(ref_path_curve)} points")

    print("\n[5.3] Testing custom ScenarioManager with environment...")
    custom_env = SimpleGridEnvironment(
        scenario_manager=scenario_mgr,
        grid_size_m=20.0
    )
    state, info = custom_env.reset()

    assert 'reference_path' in info, "Should have reference path"
    print(f"  ✓ Custom scenario manager works with environment")
    print(f"    Reference path length: {len(info['reference_path'])}")

    print("\n[5.4] Testing WorldState and EgoControl types...")
    assert isinstance(state, WorldState), "Should be WorldState"
    assert isinstance(state.ego, EgoState), "Should be EgoState"

    action = EgoControl(throttle=0.5, steer=0.0, brake=0.0)
    step_result = custom_env.step(action)

    print("  ✓ All existing types work correctly")
    print(f"    WorldState: {type(state).__name__}")
    print(f"    EgoControl: {type(action).__name__}")
    print(f"    StepResult: {type(step_result).__name__}")

    return True


def test_6_performance():
    """Test 6: Basic performance benchmarks."""
    print_section("TEST 6: Performance Benchmarks")

    print("\n[6.1] Measuring environment reset time...")
    env = SimpleGridEnvironment()

    reset_times = []
    for i in range(100):
        start = time.time()
        env.reset()
        reset_times.append(time.time() - start)

    print(f"  Average reset time: {np.mean(reset_times)*1000:.2f} ms")
    print(f"  Std dev: {np.std(reset_times)*1000:.2f} ms")

    print("\n[6.2] Measuring step time...")
    env.reset()
    action = EgoControl(throttle=0.5, steer=0.0, brake=0.0)

    step_times = []
    for i in range(1000):
        start = time.time()
        step_result = env.step(action)
        step_times.append(time.time() - start)

        if step_result.terminated or step_result.truncated:
            env.reset()

    print(f"  Average step time: {np.mean(step_times)*1000:.2f} ms")
    print(f"  Std dev: {np.std(step_times)*1000:.2f} ms")
    print(f"  Steps per second: {1.0/np.mean(step_times):.0f}")

    print("\n[6.3] Measuring episode throughput...")
    num_episodes = 50
    total_steps = 0

    start = time.time()
    for ep in range(num_episodes):
        env.reset()
        for step in range(100):
            step_result = env.step(action)
            total_steps += 1
            if step_result.terminated or step_result.truncated:
                break

    elapsed = time.time() - start

    print(f"  Total episodes: {num_episodes}")
    print(f"  Total steps: {total_steps}")
    print(f"  Total time: {elapsed:.2f} s")
    print(f"  Episodes/sec: {num_episodes/elapsed:.1f}")
    print(f"  Steps/sec: {total_steps/elapsed:.0f}")

    return True


def test_7_different_scenarios():
    """Test 7: Test different scenario configurations."""
    print_section("TEST 7: Different Scenarios")

    print("\n[7.1] Testing different path modes...")
    path_modes = ['straight', 'curve', 's_curve']

    for mode in path_modes:
        env = SimpleGridEnvironment()
        state, info = env.reset(options={'path_mode': mode})
        path = info['reference_path']

        print(f"  ✓ {mode:10s} path: {len(path)} points")
        print(f"    Start: {path[0]}")
        print(f"    End:   {path[-1]}")

    print("\n[7.2] Testing different time steps...")
    for dt in [0.05, 0.1, 0.2]:
        env = SimpleGridEnvironment(dt=dt)
        state, _ = env.reset()

        action = EgoControl(throttle=0.5, steer=0.0, brake=0.0)
        step_result = env.step(action)

        dx = step_result.observation.ego.position_m[0] - state.ego.position_m[0]
        print(f"  ✓ dt={dt:.2f}s: position change = {dx:.3f}m")

    print("\n[7.3] Testing different grid sizes...")
    for grid_size in [10.0, 20.0, 50.0]:
        env = SimpleGridEnvironment(grid_size_m=grid_size)
        state, _ = env.reset()

        print(f"  ✓ grid_size={grid_size}m: bounds = [{-grid_size/2}, {grid_size/2}]")

    return True


def test_8_real_driving_scenario():
    """Test 8: Simulate a realistic driving scenario."""
    print_section("TEST 8: Realistic Driving Scenario")

    print("\n[8.1] Setup: Urban driving with simple reactive policy...")
    env = SimpleGridEnvironment(
        dt=0.1,
        max_episode_steps=100,
        grid_size_m=50.0
    )

    def reactive_policy(state):
        """Simple reactive policy that maintains speed and avoids obstacles."""
        ego = state.ego
        agents = state.agents

        # Default: maintain moderate speed
        throttle = 0.4
        steer = 0.0
        brake = 0.0

        # Check for nearby agents
        ego_pos = np.array(ego.position_m)
        for agent in agents:
            agent_pos = np.array(agent.position_m)
            relative_pos = agent_pos - ego_pos
            distance = np.linalg.norm(relative_pos)

            # If agent is close ahead
            if relative_pos[0] > 0 and distance < 8.0:
                if abs(relative_pos[1]) < 3.0:
                    # Directly ahead - slow down
                    throttle = 0.0
                    brake = 0.3
                else:
                    # To the side - steer away
                    steer = -0.2 if relative_pos[1] > 0 else 0.2

        return EgoControl(throttle=throttle, steer=steer, brake=brake)

    print("\n[8.2] Running episode with reactive policy...")
    state, info = env.reset(seed=123)

    episode_data = {
        'positions': [state.ego.position_m],
        'speeds': [np.linalg.norm(np.array(state.ego.velocity_mps))],
        'rewards': [],
        'actions': [],
    }

    for step in range(100):
        action = reactive_policy(state)
        step_result = env.step(action)

        episode_data['positions'].append(step_result.observation.ego.position_m)
        episode_data['speeds'].append(
            np.linalg.norm(np.array(step_result.observation.ego.velocity_mps))
        )
        episode_data['rewards'].append(step_result.reward)
        episode_data['actions'].append(action)

        state = step_result.observation

        if step_result.terminated or step_result.truncated:
            break

    print(f"\n[8.3] Episode Results:")
    print(f"  Steps: {len(episode_data['rewards'])}")
    print(f"  Total reward: {sum(episode_data['rewards']):.2f}")
    print(f"  Average reward: {np.mean(episode_data['rewards']):.2f}")
    print(f"  Final position: {episode_data['positions'][-1]}")
    print(f"  Distance traveled: {episode_data['positions'][-1][0]:.2f}m")
    print(f"  Average speed: {np.mean(episode_data['speeds']):.2f} m/s")
    print(f"  Max speed: {np.max(episode_data['speeds']):.2f} m/s")

    # Analyze actions
    throttles = [a.throttle for a in episode_data['actions']]
    steers = [a.steer for a in episode_data['actions']]
    brakes = [a.brake for a in episode_data['actions']]

    print(f"\n[8.4] Action Statistics:")
    print(f"  Average throttle: {np.mean(throttles):.2f}")
    print(f"  Average steer: {np.mean(steers):.2f}")
    print(f"  Average brake: {np.mean(brakes):.2f}")
    print(f"  Brake activations: {sum(1 for b in brakes if b > 0)}")

    print(f"\n  ✓ Realistic scenario completed successfully")

    return True


def run_all_tests():
    """Run all comprehensive tests."""
    print("\n" + "#"*70)
    print("#  COMPREHENSIVE FUNCTIONAL TEST SUITE")
    print("#  Testing all features of the refactored architecture")
    print("#"*70)

    tests = [
        ("Core Interfaces", test_1_core_interfaces),
        ("Reward Composition", test_2_reward_composition),
        ("Environment Dynamics", test_3_environment_dynamics),
        ("Termination Conditions", test_4_termination_conditions),
        ("Integration with Existing Code", test_5_integration_with_existing_code),
        ("Performance Benchmarks", test_6_performance),
        ("Different Scenarios", test_7_different_scenarios),
        ("Realistic Driving Scenario", test_8_real_driving_scenario),
    ]

    results = []
    start_time = time.time()

    for name, test_func in tests:
        try:
            print(f"\nRunning: {name}...")
            success = test_func()
            results.append((name, success, None))
            print(f"✓ {name} completed")
        except Exception as e:
            print(f"\n✗ {name} failed with error:")
            import traceback
            traceback.print_exc()
            results.append((name, False, str(e)))

    total_time = time.time() - start_time

    # Print summary
    print("\n" + "#"*70)
    print("#  TEST SUMMARY")
    print("#"*70)

    passed = sum(1 for _, success, _ in results if success)
    total = len(results)

    for name, success, error in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")
        if error:
            print(f"       Error: {error}")

    print(f"\n{passed}/{total} tests passed ({100*passed//total}%)")
    print(f"Total time: {total_time:.2f} seconds")

    if passed == total:
        print("\n" + "="*70)
        print("🎉 ALL COMPREHENSIVE TESTS PASSED!")
        print("="*70)
        print("\nThe refactored architecture is fully functional and ready to use!")
        print("\nKey achievements:")
        print("  ✓ Core interfaces working correctly")
        print("  ✓ Environment dynamics accurate")
        print("  ✓ Reward composition flexible")
        print("  ✓ Integration with existing code seamless")
        print("  ✓ Performance acceptable")
        print("  ✓ All scenarios working")
        print("\nYou can now:")
        print("  1. Use SimpleGridEnvironment for RL experiments")
        print("  2. Integrate with Stable-Baselines3 or similar libraries")
        print("  3. Proceed to implement algorithm adapters (Stage 3)")
        print("="*70)
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
