#!/usr/bin/env python3
"""
Rainbow DQN + CARLA Integration Script

Integrates Rainbow DQN algorithm with CARLA simulation environment.

Features:
- Uses RainbowDQNPlanner (following standard planner interface)
- Uses CarlaEnvironment (Gym standard interface)
- Discrete lattice-based action space
- Supports predefined scenario library
- TensorBoard logging
- Model checkpointing

Prerequisites:
- CARLA server must be running (default: localhost:2000)
- carla python package must be installed or path configured
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import time
import torch

# Add project root to path
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

_src_path = _repo_root / "src"
if _src_path.exists() and str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

# Rainbow DQN components
from c2o_drive.algorithms.rainbow_dqn import RainbowDQNPlanner, RainbowDQNConfig

# CARLA environment components
from c2o_drive.environments.carla_env import CarlaEnvironment
from c2o_drive.environments.carla.scenarios import (
    get_scenario,
    list_scenarios
)
from c2o_drive.core.planner import Transition
from c2o_drive.core.types import EgoControl
from c2o_drive.config import get_global_config

# Metrics collection
from c2o_drive.utils.metrics_collector import MetricsCollector

# TensorBoard logging
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: TensorBoard not available. Install with: pip install tensorboard")


class RainbowDQNTrainer:
    """Rainbow DQN Trainer for CARLA environment."""

    def __init__(
        self,
        planner: RainbowDQNPlanner,
        env: CarlaEnvironment,
        output_dir: Path,
        log_dir: Optional[Path] = None,
        save_interval: int = 50,
        verbose: bool = True,
        scenario_name: Optional[str] = None,
    ):
        """Initialize trainer.

        Args:
            planner: Rainbow DQN planner instance
            env: CARLA environment
            output_dir: Directory for checkpoints
            log_dir: Directory for TensorBoard logs
            save_interval: Save checkpoint every N episodes
            verbose: Print training progress
            scenario_name: Scenario name for metrics tracking
        """
        self.planner = planner
        self.env = env
        self.output_dir = output_dir
        self.save_interval = save_interval
        self.verbose = verbose
        self.scenario_name = scenario_name or "unknown"

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard writer
        self.writer = None
        if TENSORBOARD_AVAILABLE and log_dir:
            self.writer = SummaryWriter(log_dir=str(log_dir))

        # Statistics
        self.episode_rewards = []
        self.episode_lengths = []

        # Metrics collector
        self.metrics = MetricsCollector(
            algorithm_name="RainbowDQN",
            scenario_name=self.scenario_name,
            output_dir=str(self.output_dir),
        )

    def _trajectory_to_control(self, current_state, trajectory, step_idx) -> EgoControl:
        """Convert trajectory waypoint to control command (aligned with C2OSR).

        Args:
            current_state: Current world state
            trajectory: Selected lattice trajectory
            step_idx: Current step index in trajectory

        Returns:
            Control command for this step
        """
        if step_idx + 1 >= len(trajectory.waypoints):
            return EgoControl(throttle=0.0, steer=0.0, brake=1.0)

        target_x, target_y = trajectory.waypoints[step_idx + 1]
        current_x, current_y = current_state.ego.position_m

        # Calculate heading error
        dx = target_x - current_x
        dy = target_y - current_y
        target_heading = np.arctan2(dy, dx)
        heading_error = target_heading - current_state.ego.yaw_rad
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

        # P-controller for steering
        steer = np.clip(heading_error * 0.5, -1.0, 1.0)

        # Speed control
        current_speed = np.linalg.norm(np.array(current_state.ego.velocity_mps))
        speed_error = trajectory.target_speed - current_speed

        if speed_error > 0.5:
            throttle = 0.6
            brake = 0.0
        elif speed_error < -0.5:
            throttle = 0.0
            brake = 0.5
        else:
            throttle = 0.3
            brake = 0.0

        return EgoControl(throttle=throttle, steer=steer, brake=brake)

    def _compute_q_statistics(self, state_features) -> Dict[str, float]:
        """Compute Q-value statistics (using Q-network for Rainbow DQN).

        Args:
            state_features: State features tensor

        Returns:
            Dictionary with Q-value statistics (mean, std, min, max)
        """
        with torch.no_grad():
            q_values = self.planner.q_network(state_features)
            q_mean = q_values.mean(dim=2)  # Average over atoms
            q_np = q_mean.cpu().numpy().flatten()

        return {
            'mean': float(np.mean(q_np)),
            'std': float(np.std(q_np)),
            'min': float(np.min(q_np)),
            'max': float(np.max(q_np)),
        }

    def _compute_min_distance(self, episode_states) -> float:
        """Compute minimum distance to any agent during episode.

        Args:
            episode_states: List of world states during episode

        Returns:
            Minimum distance to any agent (in meters)
        """
        min_dist = float('inf')

        for state in episode_states:
            ego_pos = np.array(state.ego.position_m)

            for agent in state.agents:
                agent_pos = np.array(agent.position_m)
                dist = np.linalg.norm(ego_pos - agent_pos)
                min_dist = min(min_dist, dist)

        return min_dist if min_dist != float('inf') else 100.0

    def run_episode(
        self,
        episode_id: int,
        max_steps: int,
        scenario_name: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run single episode.

        Args:
            episode_id: Episode number
            max_steps: Maximum steps per episode
            scenario_name: Scenario name (optional)
            seed: Random seed

        Returns:
            Episode statistics
        """
        # Reset environment
        scenario_def = None
        if scenario_name:
            scenario_def = get_scenario(scenario_name)

        reset_options = {}
        if scenario_def:
            reset_options['scenario_config'] = {
                'scenario': scenario_def,
                'scenario_name': scenario_name,
            }

        state, info = self.env.reset(seed=seed, options=reset_options)
        reference_path = info.get('reference_path', [])

        # Reset planner
        self.planner.reset()

        # 1. Generate all candidate trajectories
        if reference_path is None or len(reference_path) == 0:
            # Create a simple forward reference path if none provided
            ego_x, ego_y = state.ego.position_m
            reference_path = [
                (ego_x + i * 5.0, ego_y) for i in range(self.planner.config.lattice.horizon + 1)
            ]

        ego_state_tuple = (
            state.ego.position_m[0],
            state.ego.position_m[1],
            state.ego.yaw_rad,
        )

        candidate_trajectories = self.planner.lattice_planner.generate_trajectories(
            reference_path=reference_path,
            horizon=self.planner.config.lattice.horizon,
            dt=self.planner.config.lattice.dt,
            ego_state=ego_state_tuple,
        )

        if not candidate_trajectories:
            # No valid trajectories, return immediately
            return {
                'episode': episode_id,
                'reward': 0.0,
                'steps': 0,
                'collision': False,
            }

        # 2. Select trajectory using Rainbow DQN policy (epsilon-greedy)
        state_features = self.planner._extract_state_features(state)
        with torch.no_grad():
            q_values = self.planner.q_network(state_features)
            q_mean = q_values.mean(dim=2)  # Average over atoms for distributed Q
            action_idx = torch.argmax(q_mean).item()

        # Ensure action index is valid
        if action_idx >= len(candidate_trajectories):
            action_idx = 0

        selected_trajectory = candidate_trajectories[action_idx]
        num_waypoints = len(selected_trajectory.waypoints)

        # Limit episode length to trajectory length
        max_steps = min(max_steps, num_waypoints - 1)

        # 3. Execute trajectory waypoint by waypoint
        episode_reward = 0.0
        episode_steps = 0
        collision = False
        episode_states = [state]  # Track all states for metrics
        episode_start_time = time.time()

        for step in range(max_steps):
            # Convert waypoint to control (using step+1 as target)
            control = self._trajectory_to_control(state, selected_trajectory, step)

            # Execute in environment
            step_result = self.env.step(control)

            # Create transition
            transition = Transition(
                state=state,
                action=control,
                reward=step_result.reward,
                next_state=step_result.observation,
                terminated=step_result.terminated,
                truncated=step_result.truncated,
                info=step_result.info,
            )

            # Update planner
            metrics = self.planner.update(transition)

            # Update state
            state = step_result.observation
            episode_states.append(state)  # Track states
            episode_reward += step_result.reward
            episode_steps += 1

            # Check termination
            if step_result.terminated or step_result.truncated:
                collision = step_result.info.get('collision', False)
                break

        episode_time = time.time() - episode_start_time

        # Determine outcome
        if collision:
            outcome = 'collision'
        elif episode_steps >= max_steps:
            outcome = 'timeout'
        elif step_result.terminated and not collision:
            outcome = 'success'
        else:
            outcome = 'success'

        # Compute metrics
        q_stats = self._compute_q_statistics(state_features)
        min_distance = self._compute_min_distance(episode_states)
        # Near-miss判定（包含collision情况：碰撞本身就是最严重的near-miss）
        global_config = get_global_config()
        near_miss = (min_distance < global_config.safety.near_miss_threshold_m) or collision

        # Record episode data for metrics
        episode_data = {
            'episode_id': episode_id,
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'steps': episode_steps,
            'total_reward': episode_reward,
            'episode_time': episode_time,
            'outcome': outcome,
            'collision': collision,
            'near_miss': near_miss,
            'min_distance_to_agents': min_distance,
            'q_value_mean': q_stats['mean'],
            'q_value_std': q_stats['std'],
            'q_value_min': q_stats['min'],
            'q_value_max': q_stats['max'],
            'selected_action_idx': action_idx,
            'q_loss': getattr(metrics, 'q_loss', None) if metrics else None,
            'epsilon': self.planner.epsilon if hasattr(self.planner, 'epsilon') else None,
        }

        self.metrics.add_episode(episode_data)

        # Episode statistics (for compatibility)
        stats = {
            'episode': episode_id,
            'reward': episode_reward,
            'steps': episode_steps,
            'collision': collision,
        }

        return stats

    def train(
        self,
        num_episodes: int,
        max_steps: int,
        scenario_name: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        """Train Rainbow DQN agent.

        Args:
            num_episodes: Number of training episodes
            max_steps: Maximum steps per episode
            scenario_name: Scenario name for training
            seed: Random seed
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print(f" Rainbow DQN Training")
            print(f"{'='*70}")
            print(f"  Episodes: {num_episodes}")
            print(f"  Max steps: {max_steps}")
            print(f"  Scenario: {scenario_name or 'default'}")
            print(f"  Action dim: {self.planner.config.network.num_actions}")
            print(f"{'='*70}\n")

        start_time = time.time()

        for episode in range(1, num_episodes + 1):
            # Run episode
            episode_start = time.time()
            stats = self.run_episode(
                episode_id=episode,
                max_steps=max_steps,
                scenario_name=scenario_name,
                seed=seed + episode if seed else None,
            )
            episode_time = time.time() - episode_start

            # Store statistics
            self.episode_rewards.append(stats['reward'])
            self.episode_lengths.append(stats['steps'])

            # Compute moving averages
            window = min(100, len(self.episode_rewards))
            avg_reward = np.mean(self.episode_rewards[-window:])
            avg_length = np.mean(self.episode_lengths[-window:])

            # Log to TensorBoard
            if self.writer:
                self.writer.add_scalar('Train/EpisodeReward', stats['reward'], episode)
                self.writer.add_scalar('Train/EpisodeLength', stats['steps'], episode)
                self.writer.add_scalar('Train/AvgReward', avg_reward, episode)
                self.writer.add_scalar('Train/Collision', int(stats['collision']), episode)

            # Print progress
            if self.verbose:
                collision_str = "[COLLISION]" if stats['collision'] else ""
                print(f"Episode {episode}/{num_episodes} | "
                      f"Reward: {stats['reward']:.2f} | "
                      f"Avg: {avg_reward:.2f} | "
                      f"Steps: {stats['steps']} | "
                      f"Time: {episode_time:.1f}s {collision_str}")

            # Save checkpoint
            if episode % self.save_interval == 0:
                checkpoint_path = self.output_dir / f"rainbow_dqn_episode_{episode}.pt"
                self.planner.save(str(checkpoint_path))
                if self.verbose:
                    print(f"  → Saved checkpoint: {checkpoint_path.name}")

        # Final summary
        total_time = time.time() - start_time
        if self.verbose:
            print(f"\n{'='*70}")
            print(f" Training Complete")
            print(f"{'='*70}")
            print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
            print(f"  Avg reward: {avg_reward:.2f}")
            print(f"  Avg length: {avg_length:.1f}")
            print(f"{'='*70}\n")

        # Save final model
        final_path = self.output_dir / "rainbow_dqn_final.pt"
        self.planner.save(str(final_path))
        print(f"✓ Final model saved: {final_path}")

        # Save metrics
        metrics_path = self.output_dir / "metrics.json"
        self.metrics.save(str(metrics_path))
        if self.verbose:
            print(f"✓ Metrics saved: {metrics_path}")
            # Print metrics summary
            self.metrics.print_summary()

        # Close writer
        if self.writer:
            self.writer.close()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Rainbow DQN + CARLA Training Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # CARLA settings
    parser.add_argument("--host", type=str, default="localhost",
                       help="CARLA server host")
    parser.add_argument("--port", type=int, default=2000,
                       help="CARLA server port")
    parser.add_argument("--town", type=str, default="Town03",
                       help="CARLA town/map")
    parser.add_argument("--no-rendering", action="store_true",
                       help="Disable CARLA rendering")

    # Scenario settings
    parser.add_argument("--scenario", type=str, default="s1",
                       help="Scenario name (default: s1)")
    parser.add_argument("--list-scenarios", action="store_true",
                       help="List available scenarios and exit")

    # Training settings
    parser.add_argument("--episodes", type=int, default=1000,
                       help="Number of training episodes")
    parser.add_argument("--max-steps", type=int, default=100,
                       help="Maximum steps per episode")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")

    # Rainbow DQN hyperparameters
    parser.add_argument("--lr", type=float, default=6.25e-5,
                       help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                       help="Discount factor")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Mini-batch size")
    parser.add_argument("--buffer-size", type=int, default=100000,
                       help="Replay buffer size")

    # Output settings
    parser.add_argument("--output-dir", type=str, default="outputs/rainbow_dqn_carla",
                       help="Output directory for checkpoints")
    parser.add_argument("--log-dir", type=str, default="logs/rainbow_dqn_carla",
                       help="TensorBoard log directory")
    parser.add_argument("--save-interval", type=int, default=50,
                       help="Save checkpoint every N episodes")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress verbose output")

    # Use global config
    parser.add_argument("--use-global-config", action="store_true",
                       help="Use GlobalConfig for lattice parameters")

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()

    # Handle --list-scenarios
    if args.list_scenarios:
        print("\nAvailable Scenarios:")
        print("="*60)
        for scenario_name in list_scenarios():
            scenario = get_scenario(scenario_name)
            print(f"\n{scenario_name}:")
            print(f"  Description: {scenario.description}")
            print(f"  Map: {scenario.town}")
            print(f"  Difficulty: {scenario.difficulty}")
        print("\n")
        return

    # Print configuration
    print(f"\n{'='*70}")
    print(f" Rainbow DQN + CARLA Training")
    print(f"{'='*70}")
    print(f"\nCARLA Configuration:")
    print(f"  Server: {args.host}:{args.port}")
    print(f"  Town: {args.town}")
    print(f"  Scenario: {args.scenario}")
    print(f"  Rendering: {'Disabled' if args.no_rendering else 'Enabled'}")
    print(f"\nRainbow DQN Configuration:")
    print(f"  Episodes: {args.episodes}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Gamma: {args.gamma}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Buffer size: {args.buffer_size}")
    print(f"\nOutput:")
    print(f"  Checkpoints: {args.output_dir}")
    print(f"  Logs: {args.log_dir}")
    print(f"{'='*70}\n")

    # Create output directories
    output_dir = Path(args.output_dir)
    log_dir = Path(args.log_dir) if TENSORBOARD_AVAILABLE else None

    # Connect to CARLA
    print("Connecting to CARLA server...")
    try:
        env = CarlaEnvironment(
            host=args.host,
            port=args.port,
            town=args.town,
            dt=1.0,
            max_episode_steps=args.max_steps,
            no_rendering=args.no_rendering,
        )
        print("✓ Successfully connected to CARLA server\n")
    except Exception as e:
        print(f"✗ Failed to connect to CARLA: {e}")
        print(f"\nPlease ensure CARLA server is running:")
        print(f"  cd /path/to/CARLA")
        print(f"  ./CarlaUE4.sh")
        return

    # Create Rainbow DQN planner
    print("Creating Rainbow DQN planner...")
    if args.use_global_config:
        config = RainbowDQNConfig.from_global_config()
        print("  Using GlobalConfig for lattice parameters")
    else:
        config = RainbowDQNConfig(
            seed=args.seed,
        )
        config.training.learning_rate = args.lr
        config.training.gamma = args.gamma
        config.training.batch_size = args.batch_size
        config.replay.capacity = args.buffer_size

    planner = RainbowDQNPlanner(config)
    print(f"✓ Planner created")
    print(f"  Action space: {config.network.num_actions} discrete actions")
    print(f"  State space: {config.network.state_feature_dim} features")
    print(f"  Lattice: {len(config.lattice.lateral_offsets)} lateral × "
          f"{len(config.lattice.speed_variations)} speeds\n")

    # Create trainer
    trainer = RainbowDQNTrainer(
        planner=planner,
        env=env,
        output_dir=output_dir,
        log_dir=log_dir,
        save_interval=args.save_interval,
        verbose=not args.quiet,
        scenario_name=args.scenario,
    )

    # Train
    try:
        trainer.train(
            num_episodes=args.episodes,
            max_steps=args.max_steps,
            scenario_name=args.scenario,
            seed=args.seed,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        # Save current model
        interrupt_path = output_dir / "rainbow_dqn_interrupted.pt"
        planner.save(str(interrupt_path))
        print(f"✓ Model saved: {interrupt_path}")
    finally:
        env.close()
        print("\n✓ Environment closed")


if __name__ == "__main__":
    main()
