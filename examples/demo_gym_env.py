"""Demonstration of the new Gym-style environment interface.

This script shows how to use the SimpleGridEnvironment with the
standardized Gym interface, making it easy to integrate with
existing RL algorithms.
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from carla_c2osr.environments import SimpleGridEnvironment
from carla_c2osr.env.types import EgoControl


def simple_policy(observation):
    """A simple hand-crafted policy: maintain speed and avoid obstacles.

    Args:
        observation: Current WorldState

    Returns:
        EgoControl action
    """
    ego = observation.ego
    agents = observation.agents

    # Default action: maintain moderate speed, go straight
    throttle = 0.5
    steer = 0.0
    brake = 0.0

    # Check if any agent is close in front
    ego_pos = np.array(ego.position_m)
    for agent in agents:
        agent_pos = np.array(agent.position_m)
        relative_pos = agent_pos - ego_pos

        # If agent is in front and close
        if relative_pos[0] > 0 and np.linalg.norm(relative_pos) < 5.0:
            # Slow down or steer away
            if abs(relative_pos[1]) < 3.0:
                # Agent directly ahead - brake
                throttle = 0.0
                brake = 0.5
            else:
                # Agent to the side - steer away
                steer = 0.3 if relative_pos[1] < 0 else -0.3

    return EgoControl(throttle=throttle, steer=steer, brake=brake)


def demo_basic_usage():
    """Demo 1: Basic usage of the environment."""
    print("\n" + "="*70)
    print("DEMO 1: Basic Environment Usage")
    print("="*70)

    # Create environment
    env = SimpleGridEnvironment(max_episode_steps=50)
    print("✓ Environment created")

    # Reset to get initial state
    state, info = env.reset(seed=42)
    print(f"✓ Environment reset")
    print(f"  Initial ego position: {state.ego.position_m}")
    print(f"  Number of agents: {len(state.agents)}")

    # Run a few steps
    print("\nRunning 5 steps...")
    for i in range(5):
        action = EgoControl(throttle=0.4, steer=0.0, brake=0.0)
        step_result = env.step(action)

        print(f"  Step {i+1}: "
              f"pos={step_result.observation.ego.position_m}, "
              f"reward={step_result.reward:.2f}")

        if step_result.terminated or step_result.truncated:
            break

    env.close()
    print("✓ Demo 1 completed\n")


def demo_full_episode():
    """Demo 2: Run a complete episode with simple policy."""
    print("\n" + "="*70)
    print("DEMO 2: Full Episode with Simple Policy")
    print("="*70)

    env = SimpleGridEnvironment(max_episode_steps=100)
    state, info = env.reset(seed=123)

    episode_reward = 0.0
    step_count = 0

    print("Running episode with simple policy...")
    while True:
        # Use simple policy to select action
        action = simple_policy(state)

        # Execute action
        step_result = env.step(action)

        episode_reward += step_result.reward
        step_count += 1

        # Optional: print every 10 steps
        if step_count % 10 == 0:
            print(f"  Step {step_count}: reward={episode_reward:.2f}, "
                  f"pos={step_result.observation.ego.position_m}")

        # Update state
        state = step_result.observation

        # Check termination
        if step_result.terminated:
            print(f"\n✗ Episode terminated at step {step_count}")
            print(f"  Reason: collision or out of bounds")
            break
        elif step_result.truncated:
            print(f"\n✓ Episode completed {step_count} steps (time limit)")
            break

    print(f"  Total reward: {episode_reward:.2f}")
    print(f"  Final position: {state.ego.position_m}")

    # Get full trajectory
    trajectory = env.get_episode_trajectory()
    print(f"  Trajectory length: {len(trajectory)}")

    env.close()
    print("✓ Demo 2 completed\n")


def demo_custom_reward():
    """Demo 3: Using custom reward function."""
    print("\n" + "="*70)
    print("DEMO 3: Custom Reward Function")
    print("="*70)

    from carla_c2osr.environments.rewards import (
        SafetyReward, EfficiencyReward
    )
    from carla_c2osr.core.environment import CompositeRewardFunction

    # Create custom reward with high safety priority
    custom_reward = CompositeRewardFunction([
        (SafetyReward(collision_penalty=-200.0), 5.0),  # High weight on safety
        (EfficiencyReward(speed_target=3.0), 0.5),      # Low weight on efficiency
    ])

    env = SimpleGridEnvironment(
        reward_fn=custom_reward,
        max_episode_steps=50
    )

    state, _ = env.reset(seed=456)
    episode_reward = 0.0

    print("Running episode with safety-focused reward...")
    for i in range(50):
        action = simple_policy(state)
        step_result = env.step(action)

        episode_reward += step_result.reward
        state = step_result.observation

        if step_result.terminated or step_result.truncated:
            break

    print(f"  Steps: {i+1}")
    print(f"  Total reward: {episode_reward:.2f}")
    print(f"  (Compare with default reward to see difference)")

    env.close()
    print("✓ Demo 3 completed\n")


def demo_gym_compatibility():
    """Demo 4: Show Gym interface compatibility."""
    print("\n" + "="*70)
    print("DEMO 4: Gym Interface Compatibility")
    print("="*70)

    env = SimpleGridEnvironment()

    # Standard Gym interface
    print("Using standard Gym interface:")

    # Get spaces
    print(f"  Action space: {env.action_space}")
    print(f"  Observation space: {env.observation_space}")

    # Sample random action from action space
    random_action_array = env.action_space.sample()
    print(f"  Random action: {random_action_array}")

    # Reset
    obs, info = env.reset()
    print(f"  Reset: got observation (type={type(obs).__name__})")

    # Step with random actions
    print("\n  Taking 3 random steps:")
    for i in range(3):
        # Convert action array to EgoControl
        action = EgoControl(
            throttle=float(random_action_array[0]),
            steer=float(random_action_array[1]),
            brake=float(random_action_array[2])
        )

        step_result = env.step(action)
        print(f"    Step {i+1}: reward={step_result.reward:.2f}, "
              f"done={step_result.terminated}")

        if step_result.terminated:
            break

        # Sample new action
        random_action_array = env.action_space.sample()

    env.close()
    print("✓ Demo 4 completed\n")


def demo_multiple_episodes():
    """Demo 5: Run multiple episodes for statistics."""
    print("\n" + "="*70)
    print("DEMO 5: Multiple Episodes for Statistics")
    print("="*70)

    env = SimpleGridEnvironment(max_episode_steps=50)

    num_episodes = 5
    episode_rewards = []
    episode_lengths = []

    print(f"Running {num_episodes} episodes...")

    for ep in range(num_episodes):
        state, _ = env.reset(seed=ep * 100)
        ep_reward = 0.0
        ep_length = 0

        while True:
            action = simple_policy(state)
            step_result = env.step(action)

            ep_reward += step_result.reward
            ep_length += 1
            state = step_result.observation

            if step_result.terminated or step_result.truncated:
                break

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)

        print(f"  Episode {ep+1}: reward={ep_reward:.2f}, length={ep_length}")

    print(f"\nStatistics over {num_episodes} episodes:")
    print(f"  Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Average length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")

    env.close()
    print("✓ Demo 5 completed\n")


def main():
    """Run all demonstrations."""
    print("\n" + "#"*70)
    print("# SIMPLE GRID ENVIRONMENT DEMONSTRATIONS")
    print("#"*70)

    demos = [
        demo_basic_usage,
        demo_full_episode,
        demo_custom_reward,
        demo_gym_compatibility,
        demo_multiple_episodes,
    ]

    for demo in demos:
        try:
            demo()
        except Exception as e:
            print(f"\n✗ Demo failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "#"*70)
    print("# ALL DEMONSTRATIONS COMPLETED")
    print("#"*70)
    print("\nThe SimpleGridEnvironment is now ready to use with:")
    print("  - Gym-compatible RL algorithms")
    print("  - Custom reward functions")
    print("  - Simple or complex policies")
    print("  - Multiple episodes for training\n")


if __name__ == "__main__":
    main()
