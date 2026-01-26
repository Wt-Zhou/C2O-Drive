#!/usr/bin/env python3
"""
PPO + CARLA Integration Script

Integrates PPO algorithm with CARLA simulation environment.

Features:
- Uses PPOPlanner (following standard planner interface)
- Uses CarlaEnvironment (Gym standard interface)
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
from typing import Dict, Any, Optional, List
import numpy as np
import time
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt

# Add project root to path
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

_src_path = _repo_root / "src"
if _src_path.exists() and str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

# PPO components
from c2o_drive.algorithms.ppo import PPOPlanner, PPOConfig
from c2o_drive.algorithms.c2osr.config import LatticePlannerConfig

# CARLA environment components
from c2o_drive.environments.carla_env import CarlaEnvironment
from c2o_drive.environments.carla.scenarios import (
    CarlaScenarioLibrary,
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


class PPOTrainer:
    """PPO Trainer for CARLA environment."""

    def __init__(
        self,
        planner: PPOPlanner,
        env: CarlaEnvironment,
        output_dir: Path,
        log_dir: Optional[Path] = None,
        save_interval: int = 50,
        verbose: bool = True,
        scenario_name: Optional[str] = None,
    ):
        """Initialize trainer.

        Args:
            planner: PPO planner instance
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
        self.episode_stats = []  # 记录每个episode的统计数据

        # Reward日志文件
        self.reward_log_path = self.output_dir / "reward_breakdown.txt"
        self.summary_log_path = self.output_dir / "episode_summary.csv"

        # 初始化breakdown日志
        with open(self.reward_log_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("PPO Training Reward Breakdown Log\n")
            f.write(f"Scenario: {self.scenario_name}\n")
            f.write(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

        # 初始化统计表格CSV
        with open(self.summary_log_path, 'w') as f:
            f.write("Episode,Reward,Collision,NearMiss,Steps,Action,Outcome\n")

        # Metrics collector
        self.metrics = MetricsCollector(
            algorithm_name="PPO",
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
        """Compute Q-value statistics (using critic network for PPO).

        Args:
            state_features: State features tensor

        Returns:
            Dictionary with Q-value statistics (mean, std, min, max)
        """
        with torch.no_grad():
            _, value = self.planner.network(state_features)
            value_np = value.cpu().numpy().flatten()

        return {
            'mean': float(np.mean(value_np)),
            'std': float(np.std(value_np)),
            'min': float(np.min(value_np)),
            'max': float(np.max(value_np)),
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

    def _save_training_curve(self, window: int = 10) -> None:
        """Save training reward curve as PNG image.

        Args:
            window: Moving average window size (default: 10)
        """
        if not self.metrics.episodes:
            return

        episodes = self.metrics.episodes
        episode_ids = [ep['episode_id'] for ep in episodes]
        rewards = [ep['total_reward'] for ep in episodes]

        # Compute moving average
        moving_avg = []
        for i in range(len(rewards)):
            start_idx = max(0, i - window + 1)
            moving_avg.append(np.mean(rewards[start_idx:i+1]))

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot raw rewards and moving average
        ax.plot(episode_ids, rewards, alpha=0.3, color='steelblue',
                linewidth=1, label='Episode Reward')
        ax.plot(episode_ids, moving_avg, linewidth=2.5, color='darkblue',
                label=f'Moving Average (window={window})')
        ax.fill_between(episode_ids, rewards, moving_avg, alpha=0.1, color='blue')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)

        # Formatting
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Total Reward', fontsize=12)
        ax.set_title(f'PPO Training: Episode Reward Progression ({self.scenario_name})',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(True, alpha=0.3)

        # Add summary statistics
        summary = self.metrics.compute_summary()
        summary_text = (
            f"Episodes: {len(rewards)} | "
            f"Mean: {summary['avg_reward']:.2f} | "
            f"Final 100: {summary['final_100_avg_reward']:.2f} | "
            f"To 90%: {summary.get('episodes_to_90_percent', 'N/A')}"
        )
        ax.text(0.02, 0.98, summary_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        plt.tight_layout()

        # Save figure
        figure_path = self.output_dir / "training_curve.png"
        plt.savefig(figure_path, dpi=300, bbox_inches='tight')
        plt.close()

        if self.verbose:
            print(f"✓ Training curve saved: {figure_path}")

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

        # 2. Select trajectory using PPO policy
        state_features = self.planner._extract_state_features(state)
        with torch.no_grad():
            logits, value = self.planner.network(state_features)
            action_probs = F.softmax(logits, dim=-1)
            action_dist = Categorical(probs=action_probs)
            action_idx = action_dist.sample().item()
            log_prob = action_dist.log_prob(torch.tensor(action_idx))

        # 关键：设置planner的内部变量，否则update()不会存储到buffer！
        self.planner._last_action_idx = action_idx
        self.planner._last_log_prob = log_prob
        self.planner._last_value = value

        # Ensure action index is valid
        if action_idx >= len(candidate_trajectories):
            action_idx = 0

        selected_trajectory = candidate_trajectories[action_idx]
        num_waypoints = len(selected_trajectory.waypoints)

        original_max_steps = max_steps
        # Limit episode length to trajectory length
        max_steps = min(max_steps, num_waypoints - 1)

        print(f"📊 轨迹信息: 轨迹点数={num_waypoints}, 原始max_steps={original_max_steps}, 限制后max_steps={max_steps}")

        # 3. Execute trajectory waypoint by waypoint
        episode_reward = 0.0
        episode_steps = 0
        collision = False
        episode_states = [state]  # Track all states for metrics
        episode_start_time = time.time()

        # 累积各reward组件的值
        reward_breakdown_accum = {}

        # 记录每步的min_distance和near_miss
        step_min_distances = []  # 中心点距离
        step_obb_distances = []  # OBB距离
        step_near_miss_flags = []  # 每步的near_miss标志
        episode_near_miss = False  # 整个episode是否触发过near_miss

        for step in range(max_steps):
            # Convert waypoint to control (using step+1 as target)
            control = self._trajectory_to_control(state, selected_trajectory, step)

            # Execute in environment
            step_result = self.env.step(control)

            # 不再每个step调用planner.update()
            # PPO学习的是"轨迹选择决策"，应该在episode结束后存储一条记录

            # Update state
            state = step_result.observation
            episode_states.append(state)  # Track states
            episode_reward += step_result.reward
            episode_steps += 1

            # 获取CARLA的OBB检测结果
            step_near_miss = step_result.info.get('near_miss', False)
            obb_min_dist = step_result.info.get('min_distance_to_agents', float('inf'))

            # 如果这一步触发near_miss，标记整个episode
            if step_near_miss:
                episode_near_miss = True

            # 同时记录中心点距离（用于对比）
            current_min_dist = float('inf')
            ego_pos = np.array(state.ego.position_m)
            for agent in state.agents:
                agent_pos = np.array(agent.position_m)
                dist = np.linalg.norm(ego_pos - agent_pos)
                current_min_dist = min(current_min_dist, dist)

            # 记录所有距离信息
            step_min_distances.append(current_min_dist)
            step_obb_distances.append(obb_min_dist)
            step_near_miss_flags.append(step_near_miss)

            # 打印near_miss检测结果（OBB检测）
            if step_near_miss and self.verbose:
                print(f"  ⚠️ NEAR-MISS检测！Step {step}, OBB_dist={obb_min_dist:.2f}m, center_dist={current_min_dist:.2f}m")

            # 检查碰撞（每个step都检查，不只是terminated时）
            if step_result.info.get('collision', False):
                collision = True
                print(f"  ⚠️ 碰撞检测！Step {step}, OBB_dist={obb_min_dist:.2f}m, center_dist={current_min_dist:.2f}m, Reward: {step_result.reward:.2f}, Total: {episode_reward:.2f}")

            # 累积各reward组件
            if 'reward_breakdown' in step_result.info:
                for comp_name, comp_data in step_result.info['reward_breakdown'].items():
                    if comp_name not in reward_breakdown_accum:
                        reward_breakdown_accum[comp_name] = {'raw': 0.0, 'weighted': 0.0, 'weight': comp_data['weight']}
                    reward_breakdown_accum[comp_name]['raw'] += comp_data['raw']
                    reward_breakdown_accum[comp_name]['weighted'] += comp_data['weighted']

            # Check termination
            if step_result.terminated or step_result.truncated:
                break

        episode_time = time.time() - episode_start_time

        # 4. Episode结束：存储一条PPO训练记录
        # (初始state, 轨迹选择action, 总episode reward) 三者匹配
        if self.planner._last_log_prob is not None:
            self.planner.rollout_buffer.push(
                state=state_features,          # 初始状态特征
                action=action_idx,             # 选择的轨迹索引
                reward=episode_reward,         # 整个episode的总reward
                value=self.planner._last_value,
                log_prob=self.planner._last_log_prob,
                done=True,
            )

        # 检查是否需要执行PPO更新
        metrics = None
        buffer_len = len(self.planner.rollout_buffer)
        if buffer_len >= self.planner.config.batch_size:
            print(f"  🔄 PPO更新! buffer={buffer_len}, batch_size={self.planner.config.batch_size}")
            metrics = self.planner._ppo_update()
            if metrics:
                print(f"     policy_loss={metrics.get('policy_loss', 0):.4f}, "
                      f"value_loss={metrics.get('value_loss', 0):.4f}, "
                      f"entropy={metrics.get('entropy', 0):.4f}")

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

        # Near-miss判定：使用CARLA的OBB检测结果
        # 碰撞本身也算near-miss（最严重的情况）
        near_miss = episode_near_miss or collision

        # 每个episode都打印距离信息，方便调试
        if self.verbose:
            global_config = get_global_config()
            print(f"  📏 Episode Summary: min_center_distance={min_distance:.2f}m, "
                  f"threshold={global_config.safety.near_miss_threshold_m}m, "
                  f"OBB_near_miss={episode_near_miss}, collision={collision}, final_near_miss={near_miss}")

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
            'policy_loss': getattr(metrics, 'policy_loss', None) if metrics else None,
            'value_loss': getattr(metrics, 'value_loss', None) if metrics else None,
            'policy_entropy': getattr(metrics, 'entropy', None) if metrics else None,
        }

        self.metrics.add_episode(episode_data)

        # 写入reward breakdown到日志文件
        with open(self.reward_log_path, 'a') as f:
            f.write(f"Episode {episode_id}\n")
            f.write("-" * 60 + "\n")
            f.write(f"  Selected Action: {action_idx} (trajectory index)\n")
            f.write(f"  Action Probs: {action_probs.numpy().round(3).tolist()}\n")
            f.write(f"  Steps: {episode_steps}, Outcome: {outcome}, Collision: {collision}, NearMiss: {near_miss}\n")
            f.write(f"  Total Reward: {episode_reward:.4f}\n")
            f.write("\n  Reward Breakdown:\n")
            f.write(f"  {'Component':<20} {'Weight':<8} {'Raw Sum':<12} {'Weighted Sum':<12}\n")
            f.write(f"  {'-'*52}\n")
            for comp_name, comp_data in reward_breakdown_accum.items():
                f.write(f"  {comp_name:<20} {comp_data['weight']:<8.2f} {comp_data['raw']:<12.4f} {comp_data['weighted']:<12.4f}\n")

            # 写入每步的距离和near_miss信息
            f.write("\n  Step-by-Step Distance Analysis:\n")
            f.write(f"  {'Step':<8} {'Center Dist(m)':<18} {'OBB Dist(m)':<18} {'Near-Miss':<12}\n")
            f.write(f"  {'-'*56}\n")
            for step_idx in range(len(step_min_distances)):
                center_dist = step_min_distances[step_idx]
                obb_dist = step_obb_distances[step_idx] if step_idx < len(step_obb_distances) else float('inf')
                near_miss_flag = step_near_miss_flags[step_idx] if step_idx < len(step_near_miss_flags) else False

                center_str = f"{center_dist:.2f}" if center_dist != float('inf') else "No agents"
                obb_str = f"{obb_dist:.2f}" if obb_dist != float('inf') else "N/A"
                near_miss_str = "YES" if near_miss_flag else "No"

                f.write(f"  {step_idx:<8} {center_str:<18} {obb_str:<18} {near_miss_str:<12}\n")
            f.write("\n")

        # 写入统计表格CSV
        with open(self.summary_log_path, 'a') as f:
            collision_flag = 1 if collision else 0
            near_miss_flag = 1 if near_miss else 0
            f.write(f"{episode_id},{episode_reward:.4f},{collision_flag},{near_miss_flag},{episode_steps},{action_idx},{outcome}\n")

        # 记录统计数据
        self.episode_stats.append({
            'episode': episode_id,
            'reward': episode_reward,
            'collision': collision_flag,
            'near_miss': near_miss_flag,
            'steps': episode_steps,
            'action': action_idx,
            'outcome': outcome,
        })

        # Episode statistics (for compatibility)
        stats = {
            'episode': episode_id,
            'reward': episode_reward,
            'steps': episode_steps,
            'collision': collision,
            'min_distance': min_distance,
            'near_miss': near_miss,
            'reward_breakdown': reward_breakdown_accum,  # 也返回breakdown供其他用途
        }

        return stats

    def train(
        self,
        num_episodes: int,
        max_steps: int,
        scenario_name: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        """Train PPO agent.

        Args:
            num_episodes: Number of training episodes
            max_steps: Maximum steps per episode
            scenario_name: Scenario name for training
            seed: Random seed
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print(f" PPO Training")
            print(f"{'='*70}")
            print(f"  Episodes: {num_episodes}")
            print(f"  Max steps: {max_steps}")
            print(f"  Scenario: {scenario_name or 'default'}")
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
                near_miss_str = "[NEAR-MISS]" if stats.get('near_miss', False) and not stats['collision'] else ""
                sim_time = stats['steps'] * self.planner.config.lattice.dt
                min_dist = stats.get('min_distance', 0)
                print(f"Episode {episode}/{num_episodes} | "
                      f"Reward: {stats['reward']:.2f} | "
                      f"Avg: {avg_reward:.2f} | "
                      f"MinDist: {min_dist:.2f}m | "
                      f"Steps: {stats['steps']} | "
                      f"WallTime: {episode_time:.1f}s {collision_str}{near_miss_str}")

            # Save checkpoint
            if episode % self.save_interval == 0:
                checkpoint_path = self.output_dir / f"ppo_episode_{episode}.pt"
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
        final_path = self.output_dir / "ppo_final.pt"
        self.planner.save(str(final_path))
        print(f"✓ Final model saved: {final_path}")

        # Save metrics
        metrics_path = self.output_dir / "metrics.json"
        self.metrics.save(str(metrics_path))
        if self.verbose:
            print(f"✓ Metrics saved: {metrics_path}")
            # Print metrics summary
            self.metrics.print_summary()

        # Auto-generate training curve
        self._save_training_curve()

        # 写入汇总统计表格到日志文件末尾
        self._write_summary_table()

        # Close writer
        if self.writer:
            self.writer.close()

    def _write_summary_table(self):
        """写入汇总统计表格到日志文件末尾"""
        if not self.episode_stats:
            return

        total_episodes = len(self.episode_stats)
        total_collisions = sum(s['collision'] for s in self.episode_stats)
        total_near_miss = sum(s['near_miss'] for s in self.episode_stats)
        avg_reward = np.mean([s['reward'] for s in self.episode_stats])

        with open(self.reward_log_path, 'a') as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write("TRAINING SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            # 汇总统计
            f.write(f"Total Episodes: {total_episodes}\n")
            f.write(f"Total Collisions: {total_collisions} ({100*total_collisions/total_episodes:.1f}%)\n")
            f.write(f"Total Near-Miss: {total_near_miss} ({100*total_near_miss/total_episodes:.1f}%)\n")
            f.write(f"Average Reward: {avg_reward:.4f}\n\n")

            # 详细表格
            f.write("Episode Statistics Table:\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'Episode':<10} {'Reward':<12} {'Collision':<12} {'NearMiss':<12} {'Steps':<8} {'Action':<8}\n")
            f.write("-" * 70 + "\n")
            for s in self.episode_stats:
                f.write(f"{s['episode']:<10} {s['reward']:<12.4f} {s['collision']:<12} {s['near_miss']:<12} {s['steps']:<8} {s['action']:<8}\n")
            f.write("-" * 70 + "\n")

        print(f"✓ Summary table saved to: {self.reward_log_path}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="PPO + CARLA Training Script",
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
    parser.add_argument("--scenario", type=str, default="s4_wrong_way",
                       help="Scenario name (default: s4_wrong_way)")
    parser.add_argument("--list-scenarios", action="store_true",
                       help="List available scenarios and exit")

    # Training settings
    parser.add_argument("--episodes", type=int, default=1000,
                       help="Number of training episodes")
    parser.add_argument("--max-steps", type=int, default=100,
                       help="Maximum steps per episode")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")

    # PPO hyperparameters
    parser.add_argument("--lr", type=float, default=3e-4,
                       help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                       help="Discount factor")
    parser.add_argument("--clip-epsilon", type=float, default=0.2,
                       help="PPO clipping parameter")
    parser.add_argument("--n-epochs", type=int, default=10,
                       help="PPO update epochs")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Mini-batch size")

    # Output settings
    parser.add_argument("--output-dir", type=str, default="outputs/ppo_carla",
                       help="Output directory for checkpoints")
    parser.add_argument("--log-dir", type=str, default="logs/ppo_carla",
                       help="TensorBoard log directory")
    parser.add_argument("--save-interval", type=int, default=50,
                       help="Save checkpoint every N episodes")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress verbose output")

    # 评估模式
    parser.add_argument("--eval", action="store_true",
                       help="Evaluation mode (no training, deterministic actions)")
    parser.add_argument("--load", type=str, default=None,
                       help="Path to load model checkpoint (e.g., outputs/s5_xxx/ppo_final.pt)")

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

    # Create PPO config from GlobalConfig
    print("Creating PPO configuration from GlobalConfig...")
    config = PPOConfig.from_global_config()
    # Apply command-line overrides
    config.learning_rate = args.lr
    config.gamma = args.gamma
    config.clip_epsilon = args.clip_epsilon
    config.n_epochs = args.n_epochs
    config.batch_size = args.batch_size
    print("✓ PPO configuration created\n")

    # Print configuration
    print(f"\n{'='*70}")
    print(f" PPO + CARLA Training")
    print(f"{'='*70}")
    print(f"\nCARLA Configuration:")
    print(f"  Server: {args.host}:{args.port}")
    print(f"  Town: {args.town}")
    print(f"  Scenario: {args.scenario}")
    print(f"  Rendering: {'Disabled' if args.no_rendering else 'Enabled'}")
    print(f"\nPPO Configuration:")
    print(f"  Episodes: {args.episodes}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Gamma: {config.gamma}")
    print(f"  Clip epsilon: {config.clip_epsilon}")
    print(f"  Batch size: {config.batch_size}")
    print(f"\nLattice Configuration (from GlobalConfig):")
    print(f"  Lateral offsets: {config.lattice.lateral_offsets}")
    print(f"  Speed variations: {config.lattice.speed_variations}")
    print(f"  Horizon: {config.horizon}, dt: {config.lattice.dt}s")
    print(f"  Action dim: {config.action_dim}")
    # Create output directories with scenario name and timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.scenario}_{timestamp}"
    output_dir = Path(args.output_dir) / run_name
    log_dir = (Path(args.log_dir) / run_name) if TENSORBOARD_AVAILABLE else None

    print(f"\nOutput:")
    print(f"  Checkpoints: {output_dir}")
    print(f"  Logs: {log_dir}")
    print(f"{'='*70}\n")

    # Connect to CARLA
    print("Connecting to CARLA server...")
    try:
        env = CarlaEnvironment(
            host=args.host,
            port=args.port,
            town=args.town,
            dt=config.lattice.dt,  # 环境时间步长：1.0s
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

    # Create PPO planner
    print("Creating PPO planner...")
    planner = PPOPlanner(config)
    print(f"✓ Planner created")
    print(f"  Action space: {config.action_dim} discrete actions")
    print(f"  State space: {config.state_dim} features\n")

    # 加载模型（如果指定）
    if args.load:
        print(f"Loading model from: {args.load}")
        planner.load(args.load)
        print("✓ Model loaded successfully\n")

    # 评估模式
    if args.eval:
        print("=" * 70)
        print("EVALUATION MODE (deterministic actions)")
        print("=" * 70)

        try:
            total_reward = 0.0
            total_collisions = 0
            for ep in range(args.episodes):
                print(f"\n{'='*50}")
                print(f"Starting Episode {ep+1}/{args.episodes}")
                print(f"{'='*50}")

                try:
                    # Reset environment
                    reset_options = {}
                    scenario_def = CarlaScenarioLibrary.get_scenario(args.scenario)
                    if scenario_def:
                        reset_options['scenario_config'] = {
                            'scenario': scenario_def,
                            'scenario_name': args.scenario,
                        }
                    state, info = env.reset(options=reset_options)
                    reference_path = info.get('reference_path', [])
                except Exception as e:
                    print(f"❌ Episode {ep} reset失败: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

                # Generate trajectories
                ego_state_tuple = (state.ego.position_m[0], state.ego.position_m[1], state.ego.yaw_rad)
                candidate_trajectories = planner.lattice_planner.generate_trajectories(
                    reference_path=reference_path,
                    horizon=planner.config.lattice.horizon,
                    dt=planner.config.lattice.dt,
                    ego_state=ego_state_tuple,
                )

                # Select action (deterministic)
                state_features = planner._extract_state_features(state)
                with torch.no_grad():
                    logits, _ = planner.network(state_features)
                    action_probs = F.softmax(logits, dim=-1)
                    action_idx = torch.argmax(action_probs).item()

                if action_idx >= len(candidate_trajectories):
                    action_idx = 0
                selected_trajectory = candidate_trajectories[action_idx]

                # Execute episode（使用和训练相同的控制逻辑）
                episode_reward = 0.0
                collision = False
                for step in range(min(args.max_steps, len(selected_trajectory.waypoints) - 1)):
                    # Convert trajectory waypoint to control（复用训练逻辑）
                    if step + 1 >= len(selected_trajectory.waypoints):
                        control = EgoControl(throttle=0.0, steer=0.0, brake=1.0)
                    else:
                        target_x, target_y = selected_trajectory.waypoints[step + 1]
                        current_x, current_y = state.ego.position_m

                        # Calculate heading error
                        dx = target_x - current_x
                        dy = target_y - current_y
                        target_heading = np.arctan2(dy, dx)
                        heading_error = target_heading - state.ego.yaw_rad
                        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

                        # P-controller for steering（和训练一致）
                        steer = np.clip(heading_error * 0.5, -1.0, 1.0)

                        # Speed control（和训练一致）
                        current_speed = np.linalg.norm(np.array(state.ego.velocity_mps))
                        speed_error = selected_trajectory.target_speed - current_speed

                        if speed_error > 0.5:
                            throttle = 0.6
                            brake = 0.0
                        elif speed_error < -0.5:
                            throttle = 0.0
                            brake = 0.5
                        else:
                            throttle = 0.3
                            brake = 0.0

                        control = EgoControl(throttle=throttle, steer=steer, brake=brake)

                    step_result = env.step(control)
                    episode_reward += step_result.reward
                    state = step_result.observation

                    # 每个step都检查collision（和训练一致）
                    if step_result.info.get('collision', False):
                        collision = True

                    if step_result.terminated or step_result.truncated:
                        break

                total_reward += episode_reward
                if collision:
                    total_collisions += 1

                print(f"Episode {ep}: reward={episode_reward:.2f}, collision={collision}, action={action_idx}, probs={action_probs.numpy().round(3)}")

            print(f"\n{'='*70}")
            print(f"Evaluation Results ({args.episodes} episodes):")
            print(f"  Average Reward: {total_reward/args.episodes:.2f}")
            print(f"  Collision Rate: {total_collisions}/{args.episodes} ({100*total_collisions/args.episodes:.1f}%)")
            print(f"{'='*70}")

        finally:
            env.close()
            print("\n✓ Environment closed")
        return

    # Create trainer
    trainer = PPOTrainer(
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
        interrupt_path = output_dir / "ppo_interrupted.pt"
        planner.save(str(interrupt_path))
        print(f"✓ Model saved: {interrupt_path}")
        # Save training curve
        if 'trainer' in dir():
            trainer._save_training_curve()
            print(f"✓ Training curve saved")
    finally:
        env.close()
        print("\n✓ Environment closed")


if __name__ == "__main__":
    main()
