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
import matplotlib.pyplot as plt

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
        debug_q: bool = False,
        debug_q_topk: int = 5,
        random_episodes: int = 0,
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
        self.debug_q = debug_q
        self.debug_q_topk = debug_q_topk
        self.random_episodes = random_episodes

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
        self.action_reward_stats = {}  # action_idx -> list of rewards
        self._restored_noisy_sigma = False

        # Reward日志文件（与PPO一致）
        self.reward_log_path = self.output_dir / "reward_breakdown.txt"
        self.summary_log_path = self.output_dir / "episode_summary.csv"

        # 初始化breakdown日志
        with open(self.reward_log_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("Rainbow DQN Training Reward Breakdown Log\n")
            f.write(f"Scenario: {self.scenario_name}\n")
            f.write(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

        # 初始化统计表格CSV
        with open(self.summary_log_path, 'w') as f:
            f.write("Episode,Reward,Collision,NearMiss,Steps,Action,Outcome\n")

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

    def _compute_q_statistics(self, state) -> Dict[str, float]:
        """Compute Q-value statistics (using Q-network for Rainbow DQN).

        Args:
            state: WorldState object

        Returns:
            Dictionary with Q-value statistics (mean, std, min, max)
        """
        with torch.no_grad():
            q_dist, q_values = self.planner.q_network([state])  # 直接使用WorldState列表
            q_np = q_values.cpu().numpy().flatten()

        return {
            'mean': float(np.mean(q_np)),
            'std': float(np.std(q_np)),
            'min': float(np.min(q_np)),
            'max': float(np.max(q_np)),
        }

    def _save_training_curve(self, window: int = 10) -> None:
        """Save training reward curve as PNG image (与PPO一致).

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
        ax.set_title(f'Rainbow DQN Training: Episode Reward Progression ({self.scenario_name})',
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

    # 已弃用：现在使用CARLA的OBB检测结果
    # def _compute_min_distance(self, episode_states) -> float:
    #     """Compute minimum distance to any agent during episode (DEPRECATED).
    #
    #     现在使用CARLA提供的OBB距离，不再需要手动计算中心点距离。
    #     """
    #     min_dist = float('inf')
    #     for state in episode_states:
    #         ego_pos = np.array(state.ego.position_m)
    #         for agent in state.agents:
    #             agent_pos = np.array(agent.position_m)
    #             dist = np.linalg.norm(ego_pos - agent_pos)
    #             min_dist = min(min_dist, dist)
    #     return min_dist if min_dist != float('inf') else 100.0

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

        # 2. Select trajectory using Rainbow DQN policy
        # 保存初始state（用于episode-level transition）
        initial_state = state

        # 使用planner的select_action来获得探索（Noisy Nets）
        # 需要手动调用网络来获取Q值和action索引
        # 训练模式：启用Noisy Nets探索
        # Warmup阶段使用更高的noisy_sigma增强探索
        warmup_steps = self.planner.config.training.warmup_steps
        if len(self.planner.replay_buffer) < warmup_steps:
            self.planner.q_network.set_noisy_sigma(self.planner.config.network.warmup_noisy_sigma)
        elif not self._restored_noisy_sigma:
            self.planner.q_network.set_noisy_sigma(self.planner.config.network.noisy_sigma)
            self._restored_noisy_sigma = True
        self.planner.q_network.reset_noise()  # 重置噪声
        self.planner.q_network.train()  # 训练模式

        with torch.no_grad():
            q_dist, q_values = self.planner.q_network([state])
            action_idx = q_values.argmax(dim=1).item()

        if episode_id <= self.random_episodes:
            action_idx = np.random.randint(len(candidate_trajectories))

        if self.debug_q:
            q_np = q_values.cpu().numpy().flatten()
            topk = min(self.debug_q_topk, q_np.shape[0])
            top_indices = np.argsort(-q_np)[:topk]
            print("  Q-value Top-K (action_idx, q, lateral_offset, target_speed):")
            for idx in top_indices:
                if idx < len(candidate_trajectories):
                    traj = candidate_trajectories[idx]
                    print(f"    {idx:>2d} | {q_np[idx]:>8.4f} | {traj.lateral_offset:>6.2f} | {traj.target_speed:>5.2f}")
                else:
                    print(f"    {idx:>2d} | {q_np[idx]:>8.4f} | N/A   | N/A")

        # Ensure action index is valid
        if action_idx >= len(candidate_trajectories):
            action_idx = 0

        selected_trajectory = candidate_trajectories[action_idx]

        # 保存到planner（用于后续可能需要）
        self.planner.last_trajectory_idx = action_idx
        self.planner.last_selected_trajectory = selected_trajectory
        num_waypoints = len(selected_trajectory.waypoints)

        # Limit episode length to trajectory length
        original_max_steps = max_steps
        max_steps = min(max_steps, num_waypoints - 1)
        if self.verbose:
            print(f"  📊 轨迹信息: 轨迹点数={num_waypoints}, 原始max_steps={original_max_steps}, 限制后max_steps={max_steps}")

        # 3. Execute trajectory waypoint by waypoint
        episode_reward = 0.0
        episode_steps = 0
        collision = False
        episode_states = [state]  # Track all states for metrics
        episode_start_time = time.time()

        # 累积各reward组件的值（与PPO一致）
        reward_breakdown_accum = {}

        # 记录每步的距离信息（OBB vs center distance）
        step_min_distances = []  # 中心点距离（用于对比）
        step_obb_distances = []  # OBB距离（CARLA提供）
        step_near_miss_flags = []  # 每步的near_miss标志
        episode_near_miss = False  # 整个episode是否触发过near_miss

        for step in range(max_steps):
            # Convert waypoint to control (using step+1 as target)
            control = self._trajectory_to_control(state, selected_trajectory, step)

            # Execute in environment
            step_result = self.env.step(control)

            # 不再每个step调用planner.update()
            # Rainbow DQN学习的是"轨迹选择决策"，应该在episode结束后存储一条记录

            # 获取CARLA的OBB检测结果
            step_near_miss = step_result.info.get('near_miss', False)
            obb_min_dist = step_result.info.get('min_distance_to_agents', float('inf'))

            if step_near_miss:
                episode_near_miss = True

            # 同时计算中心点距离（用于对比）
            current_min_dist = float('inf')
            ego_pos = np.array(state.ego.position_m)
            for agent in state.agents:
                agent_pos = np.array(agent.position_m)
                dist = np.linalg.norm(ego_pos - agent_pos)
                current_min_dist = min(current_min_dist, dist)

            # 记录距离信息
            step_min_distances.append(current_min_dist)
            step_obb_distances.append(obb_min_dist)
            step_near_miss_flags.append(step_near_miss)

            # 打印near-miss检测（实时）
            if step_near_miss and self.verbose:
                print(f"  ⚠️ NEAR-MISS检测！Step {step}, OBB_dist={obb_min_dist:.2f}m, center_dist={current_min_dist:.2f}m")

            # 检测碰撞
            if step_result.info.get('collision', False):
                collision = True
                if self.verbose:
                    print(f"  ⚠️ 碰撞检测！Step {step}, OBB_dist={obb_min_dist:.2f}m, center_dist={current_min_dist:.2f}m")

            # 累积各reward组件（与PPO一致）
            if 'reward_breakdown' in step_result.info:
                for comp_name, comp_data in step_result.info['reward_breakdown'].items():
                    if comp_name not in reward_breakdown_accum:
                        reward_breakdown_accum[comp_name] = {'raw': 0.0, 'weighted': 0.0, 'weight': comp_data['weight']}
                    reward_breakdown_accum[comp_name]['raw'] += comp_data['raw']
                    reward_breakdown_accum[comp_name]['weighted'] += comp_data['weighted']

            # Update state
            state = step_result.observation
            episode_states.append(state)  # Track states
            episode_reward += step_result.reward
            episode_steps += 1

            # Check termination
            if step_result.terminated or step_result.truncated:
                if not collision:  # collision已经在上面检测过
                    collision = step_result.info.get('collision', False)
                break

        episode_time = time.time() - episode_start_time

        # 4. Episode结束：存储一条episode-level transition
        # Rainbow DQN的replay buffer需要episode-level transition
        final_state = state

        self.planner.replay_buffer.push(
            state=initial_state,
            action=action_idx,
            reward=episode_reward,  # 整个episode的总reward
            next_state=final_state,
            done=True
        )

        # 定期训练（类似PPO）
        metrics = None
        buffer_len = len(self.planner.replay_buffer)
        if buffer_len >= self.planner.config.training.batch_size:
            # 调用planner的训练步骤
            if hasattr(self.planner, '_train_step'):
                if self.verbose:
                    print(f"  🔄 Rainbow DQN更新! buffer={buffer_len}")
                metrics = self.planner._train_step()
                # 打印训练metrics（与PPO一致）
                if metrics and self.verbose:
                    loss = metrics.loss if hasattr(metrics, 'loss') else metrics.get('loss', None)
                    q_value = metrics.q_value if hasattr(metrics, 'q_value') else metrics.get('q_value', None)
                    custom = metrics.custom if hasattr(metrics, 'custom') else metrics.get('custom', {})
                    td_error = custom.get('td_error_mean', None)

                    loss_str = f"{loss:.4f}" if loss is not None else "N/A"
                    q_value_str = f"{q_value:.4f}" if q_value is not None else "N/A"
                    td_error_str = f"{td_error:.4f}" if td_error is not None else "N/A"
                    print(f"     loss={loss_str}, q_value={q_value_str}, td_error={td_error_str}")
            # 如果没有_train_step方法，暂时跳过训练

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
        q_stats = self._compute_q_statistics(initial_state)

        # 使用OBB检测结果
        min_distance_obb = min(step_obb_distances) if step_obb_distances else float('inf')
        near_miss = episode_near_miss or collision

        # Record episode data for metrics
        episode_data = {
            'episode_id': episode_id,
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'steps': episode_steps,
            'total_reward': episode_reward,
            'episode_time': episode_time,
            'outcome': outcome,
            'collision': collision,
            'near_miss': near_miss,  # 使用OBB检测结果
            'min_distance_to_agents': min_distance_obb,  # 使用OBB距离
            'q_value_mean': q_stats['mean'],
            'q_value_std': q_stats['std'],
            'q_value_min': q_stats['min'],
            'q_value_max': q_stats['max'],
            'selected_action_idx': action_idx,
            'q_loss': getattr(metrics, 'q_loss', None) if metrics else None,
            'epsilon': self.planner.epsilon if hasattr(self.planner, 'epsilon') else None,
        }

        # Track reward distribution per action
        self.action_reward_stats.setdefault(action_idx, []).append(episode_reward)

        self.metrics.add_episode(episode_data)

        # 写入reward breakdown到日志文件（与PPO一致）
        with open(self.reward_log_path, 'a') as f:
            f.write(f"Episode {episode_id}\n")
            f.write("-" * 60 + "\n")
            f.write(f"  Selected Action: {action_idx} (trajectory index)\n")
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

        # 写入统计表格CSV（与PPO一致）
        with open(self.summary_log_path, 'a') as f:
            f.write(f"{episode_id},{episode_reward:.2f},{int(collision)},{int(near_miss)},{episode_steps},{action_idx},{outcome}\n")

        # 打印step-by-step距离分析（详细日志）
        if self.verbose and len(step_min_distances) > 0:
            print(f"\n  Step-by-Step Distance Analysis:")
            print(f"  {'Step':<8} {'Center Dist(m)':<18} {'OBB Dist(m)':<18} {'Near-Miss':<12}")
            print(f"  {'-'*56}")
            for step_idx in range(len(step_min_distances)):
                center_dist = step_min_distances[step_idx]
                obb_dist = step_obb_distances[step_idx]
                near_miss_flag = "✓" if step_near_miss_flags[step_idx] else ""
                print(f"  {step_idx:<8} {center_dist:<18.2f} {obb_dist:<18.2f} {near_miss_flag:<12}")
            print()

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

        # Print action reward distribution summary
        if self.verbose and self.action_reward_stats:
            print("\nAction Reward Distribution (mean ± std, count):")
            for action_idx in sorted(self.action_reward_stats.keys()):
                rewards = self.action_reward_stats[action_idx]
                mean_r = float(np.mean(rewards))
                std_r = float(np.std(rewards))
                print(f"  Action {action_idx:>2d}: {mean_r:>7.2f} ± {std_r:>6.2f} (n={len(rewards)})")

        # Save metrics
        metrics_path = self.output_dir / "metrics.json"
        self.metrics.save(str(metrics_path))
        if self.verbose:
            print(f"✓ Metrics saved: {metrics_path}")
            # Print metrics summary
            self.metrics.print_summary()

        # Save training curve (align with PPO)
        self._save_training_curve()

        # 打印日志文件位置（与PPO一致）
        if self.verbose:
            print(f"✓ Reward breakdown log saved to: {self.reward_log_path}")
            print(f"✓ Summary table saved to: {self.summary_log_path}")

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
    parser.add_argument("--lr", type=float, default=3e-5,
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
    parser.add_argument("--debug-q", action="store_true",
                       help="Print Q-value top-k actions each episode")
    parser.add_argument("--debug-q-topk", type=int, default=5,
                       help="Top-k actions to print when --debug-q is enabled")
    parser.add_argument("--random-episodes", type=int, default=0,
                       help="Number of initial episodes to sample actions uniformly at random")

    # 评估模式
    parser.add_argument("--eval", action="store_true",
                       help="Evaluation mode (no training, deterministic actions)")
    parser.add_argument("--load", type=str, default=None,
                       help="Path to load model checkpoint (e.g., outputs/s5_xxx/rainbow_dqn_final.pt)")

    # Use global config
    parser.add_argument("--no-global-config", action="store_true",
                       help="Disable GlobalConfig and use local RainbowDQNConfig defaults")

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
    gc = get_global_config()
    print(f"\nGlobalConfig Lattice:")
    print(f"  Horizon: {gc.lattice.horizon}")
    print(f"  Dt: {gc.lattice.dt}")
    print(f"  Lateral offsets: {gc.lattice.lateral_offsets}")
    print(f"  Speed variations: {gc.lattice.speed_variations}")

    # Create output directories with scenario name and timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.scenario}_{timestamp}"
    output_dir = Path(args.output_dir) / run_name
    log_dir = (Path(args.log_dir) / run_name) if TENSORBOARD_AVAILABLE else None

    print(f"\nOutput:")
    print(f"  Checkpoints: {output_dir}")
    print(f"  Logs: {log_dir}")
    print(f"{'='*70}\n")

    # Create Rainbow DQN config
    print("Creating Rainbow DQN planner...")
    if args.no_global_config:
        config = RainbowDQNConfig(
            seed=args.seed,
        )
        config.training.learning_rate = args.lr
        config.training.gamma = args.gamma
        config.training.batch_size = args.batch_size
        config.replay.capacity = args.buffer_size
        print("  Using local RainbowDQNConfig defaults (no GlobalConfig)")
    else:
        config = RainbowDQNConfig.from_global_config()
        print("  Using GlobalConfig for lattice parameters")

    print(f"  Env dt: {config.lattice.dt}s")

    # Connect to CARLA
    print("Connecting to CARLA server...")
    try:
        env = CarlaEnvironment(
            host=args.host,
            port=args.port,
            town=args.town,
            dt=config.lattice.dt,
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
    planner = RainbowDQNPlanner(config)
    print(f"✓ Planner created")
    print(f"  Action space: {config.network.num_actions} discrete actions")
    print(f"  State space: {config.network.state_feature_dim} features")
    print(f"  Lattice: {len(config.lattice.lateral_offsets)} lateral × "
          f"{len(config.lattice.speed_variations)} speeds\n")
    print(f"  Lattice horizon: {config.lattice.horizon}")

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
                    scenario_def = get_scenario(args.scenario)
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

                # Generate trajectories (same as training)
                if reference_path is None or len(reference_path) == 0:
                    ego_x, ego_y = state.ego.position_m
                    reference_path = [
                        (ego_x + i * 5.0, ego_y) for i in range(planner.config.lattice.horizon + 1)
                    ]

                ego_state_tuple = (state.ego.position_m[0], state.ego.position_m[1], state.ego.yaw_rad)
                candidate_trajectories = planner.lattice_planner.generate_trajectories(
                    reference_path=reference_path,
                    horizon=planner.config.lattice.horizon,
                    dt=planner.config.lattice.dt,
                    ego_state=ego_state_tuple,
                )

                # Select action (deterministic)
                planner.q_network.eval()
                with torch.no_grad():
                    _, q_values = planner.q_network([state])
                    action_idx = q_values.argmax(dim=1).item()

                if action_idx >= len(candidate_trajectories):
                    action_idx = 0
                selected_trajectory = candidate_trajectories[action_idx]

                # Execute episode（使用和训练相同的控制逻辑）
                episode_reward = 0.0
                collision = False
                max_steps = min(args.max_steps, len(selected_trajectory.waypoints) - 1)
                for step in range(max_steps):
                    # Convert trajectory waypoint to control（和训练一致）
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

                print(f"Episode {ep}: reward={episode_reward:.2f}, collision={collision}, action={action_idx}")

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
    trainer = RainbowDQNTrainer(
        planner=planner,
        env=env,
        output_dir=output_dir,
        log_dir=log_dir,
        save_interval=args.save_interval,
        verbose=not args.quiet,
        scenario_name=args.scenario,
        debug_q=args.debug_q,
        debug_q_topk=args.debug_q_topk,
        random_episodes=args.random_episodes,
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
