"""Visual demonstration of the SimpleGridEnvironment.

This script shows a complete episode with detailed output
including ASCII visualization of the driving scenario.
"""

import sys
from pathlib import Path
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from carla_c2osr.environments import SimpleGridEnvironment
from carla_c2osr.env.types import EgoControl


def visualize_scenario(state, step_num, reward, action):
    """Create ASCII visualization of current scenario."""
    grid_size = 20  # meters
    display_size = 40  # characters

    # Create grid
    grid = [[' ' for _ in range(display_size)] for _ in range(display_size)]

    # Helper to convert coordinates
    def to_grid(x, y):
        gx = int((x + grid_size/2) / grid_size * display_size)
        gy = int((y + grid_size/2) / grid_size * display_size)
        return max(0, min(display_size-1, gx)), max(0, min(display_size-1, gy))

    # Draw reference line (centerline)
    for x in range(display_size):
        real_x = (x / display_size) * grid_size - grid_size/2
        gx, gy = to_grid(real_x, 0)
        if 0 <= gy < display_size:
            grid[gy][gx] = '·'

    # Draw agents
    for i, agent in enumerate(state.agents):
        ax, ay = agent.position_m
        gx, gy = to_grid(ax, ay)
        if 0 <= gx < display_size and 0 <= gy < display_size:
            grid[gy][gx] = 'A'

    # Draw ego (with direction indicator)
    ex, ey = state.ego.position_m
    gx, gy = to_grid(ex, ey)
    if 0 <= gx < display_size and 0 <= gy < display_size:
        # Show direction
        yaw = state.ego.yaw_rad
        if abs(yaw) < np.pi/8:
            grid[gy][gx] = '→'
        elif abs(yaw - np.pi/2) < np.pi/8:
            grid[gy][gx] = '↑'
        elif abs(yaw + np.pi/2) < np.pi/8:
            grid[gy][gx] = '↓'
        else:
            grid[gy][gx] = 'E'

    # Print visualization
    print(f"\n┌{'─' * (display_size + 2)}┐")
    print(f"│ Step {step_num:3d} {' ' * (display_size - 10)}│")
    print(f"├{'─' * (display_size + 2)}┤")

    for row in reversed(grid):  # Reverse to have y-axis pointing up
        print(f"│ {''.join(row)} │")

    print(f"└{'─' * (display_size + 2)}┘")

    # Print state info
    speed = np.linalg.norm(np.array(state.ego.velocity_mps))
    print(f"\n  Position: ({ex:6.2f}, {ey:6.2f}) m")
    print(f"  Speed:    {speed:6.2f} m/s")
    print(f"  Heading:  {np.degrees(state.ego.yaw_rad):6.1f}°")
    print(f"  Action:   throttle={action.throttle:5.2f}, steer={action.steer:5.2f}, brake={action.brake:5.2f}")
    print(f"  Reward:   {reward:6.2f}")
    print(f"  Agents:   {len(state.agents)}")


def simple_controller(state, target_speed=5.0):
    """Simple P-controller for speed and heading."""
    current_speed = np.linalg.norm(np.array(state.ego.velocity_mps))

    # Speed control
    speed_error = target_speed - current_speed
    throttle = np.clip(0.5 + 0.1 * speed_error, 0.0, 1.0)
    brake = 0.0

    # Heading control (try to stay straight)
    heading_error = -state.ego.yaw_rad
    steer = np.clip(0.5 * heading_error, -1.0, 1.0)

    # Simple obstacle avoidance
    ego_pos = np.array(state.ego.position_m)
    for agent in state.agents:
        agent_pos = np.array(agent.position_m)
        relative_pos = agent_pos - ego_pos
        distance = np.linalg.norm(relative_pos)

        # If agent is ahead and close
        if relative_pos[0] > 0 and distance < 10.0:
            if abs(relative_pos[1]) < 4.0:
                # Too close - slow down
                throttle = 0.0
                brake = 0.5
            else:
                # Steer away from agent
                steer_away = -0.3 if relative_pos[1] > 0 else 0.3
                steer = np.clip(steer + steer_away, -1.0, 1.0)

    return EgoControl(throttle=throttle, steer=steer, brake=brake)


def run_visual_demo():
    """Run a complete episode with visualization."""
    print("\n" + "="*60)
    print("  VISUAL DEMONSTRATION - SimpleGridEnvironment")
    print("="*60)

    print("\nLegend:")
    print("  → ↑ ↓  = Ego vehicle (direction)")
    print("  A      = Other agents")
    print("  ·      = Reference centerline")
    print("  Space  = Empty road")

    print("\nScenario: Urban driving with obstacle avoidance")
    print("Controller: Simple P-controller with reactive avoidance")

    input("\nPress Enter to start the simulation...")

    # Create environment
    env = SimpleGridEnvironment(
        dt=0.2,  # Slower for visualization
        max_episode_steps=30,
        grid_size_m=20.0
    )

    # Reset
    state, info = env.reset(seed=42)
    action = EgoControl(0.0, 0.0, 0.0)

    # Initial visualization
    visualize_scenario(state, 0, 0.0, action)
    input("\nPress Enter for next step...")

    # Run episode
    total_reward = 0.0
    step = 0

    while True:
        # Get action from controller
        action = simple_controller(state, target_speed=5.0)

        # Step environment
        step_result = env.step(action)
        step += 1
        total_reward += step_result.reward

        # Visualize
        visualize_scenario(
            step_result.observation,
            step,
            step_result.reward,
            action
        )

        # Update state
        state = step_result.observation

        # Check termination
        if step_result.terminated:
            print("\n" + "!"*60)
            print("  EPISODE TERMINATED")
            if step_result.info.get('collision'):
                print("  Reason: COLLISION")
            elif step_result.info.get('out_of_bounds'):
                print("  Reason: OUT OF BOUNDS")
            print("!"*60)
            break
        elif step_result.truncated:
            print("\n" + "="*60)
            print("  EPISODE COMPLETED (time limit)")
            print("="*60)
            break

        # Wait for user input (or auto-advance after first few steps)
        if step < 5:
            input("\nPress Enter for next step...")
        else:
            print("\n(Auto-advancing...)")
            import time
            time.sleep(0.5)

    # Final statistics
    print("\n" + "="*60)
    print("  EPISODE SUMMARY")
    print("="*60)
    print(f"  Total steps:      {step}")
    print(f"  Total reward:     {total_reward:.2f}")
    print(f"  Average reward:   {total_reward/step:.2f}")
    print(f"  Final position:   ({state.ego.position_m[0]:.2f}, {state.ego.position_m[1]:.2f}) m")
    print(f"  Distance traveled: {state.ego.position_m[0]:.2f} m")

    # Get trajectory
    trajectory = env.get_episode_trajectory()
    print(f"  Trajectory length: {len(trajectory)} states")

    print("\n  Trajectory overview:")
    print("  " + "-"*56)
    print("  Step |    Position (m)     |   Velocity (m/s)   | Heading")
    print("  " + "-"*56)

    for i, ego_state in enumerate(trajectory[::5]):  # Show every 5th
        pos = ego_state.position_m
        vel = ego_state.velocity_mps
        yaw = ego_state.yaw_rad
        print(f"  {i*5:4d} | ({pos[0]:6.2f}, {pos[1]:6.2f}) | "
              f"({vel[0]:5.2f}, {vel[1]:5.2f}) | {np.degrees(yaw):6.1f}°")

    print("  " + "-"*56)

    env.close()

    print("\n" + "="*60)
    print("  ✓ Visual demonstration completed!")
    print("="*60)


if __name__ == "__main__":
    try:
        run_visual_demo()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\n✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()
