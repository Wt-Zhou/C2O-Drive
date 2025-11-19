#!/usr/bin/env python3
"""
C2OSR + ScenarioManager 运行脚本（新架构版本）

使用新架构的 C2OSRPlanner 和 ScenarioReplayEnvironment，
复现原 run_sim_cl_simple.py 的功能，但基于标准 Gym 接口。

特性：
- 使用 C2OSRPlanner 适配器（来自 algorithms/c2osr/）
- 使用 ScenarioReplayEnvironment（Gym 标准接口）
- 支持多 episodes 运行
- 生成可视化和统计报告
- 配置预设支持
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import time

# 添加项目根目录到路径
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

_src_path = _repo_root / "src"
if _src_path.exists() and str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

# 新架构组件
from c2o_drive.algorithms.c2osr.factory import create_c2osr_planner
from c2o_drive.algorithms.c2osr.config import (
    C2OSRPlannerConfig,
    LatticePlannerConfig,
    QValueConfig,
    GridConfig,
    DirichletConfig,
    RewardWeightsConfig,
)
from c2o_drive.environments.scenario_replay_env import ScenarioReplayEnvironment
from c2o_drive.environments.virtual.scenario_manager import ScenarioManager
from c2o_drive.core.planner import Transition
from c2o_drive.core.types import EgoControl
from c2o_drive.config import get_global_config

# 可视化组件
from visualization_utils import (
    EpisodeVisualizer,
    GlobalVisualizer,
    create_visualization_pipeline,
)


class EpisodeRunner:
    """Episode 运行器（批量轨迹规划模式）"""

    def __init__(
        self,
        planner,
        env,
        q_tracker=None,
        global_visualizer: Optional[GlobalVisualizer] = None,
        enable_visualization: bool = True,
        output_dir: Optional[Path] = None,
        verbose: bool = True,
        visualize_distributions: bool = False,
        vis_interval: int = 5,
    ):
        self.planner = planner
        self.env = env
        self.q_tracker = q_tracker
        self.global_visualizer = global_visualizer
        self.enable_visualization = enable_visualization
        self.output_dir = output_dir or Path("outputs/c2osr_scenario")
        self.verbose = verbose
        self.visualize_distributions = visualize_distributions
        self.vis_interval = vis_interval

    def run_episode(
        self,
        episode_id: int,
        max_steps: int,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """运行单个 episode（批量轨迹规划模式）

        该方法使用批量轨迹规划：
        1. 在 episode 开始时生成并评估所有候选轨迹
        2. 选择最优轨迹
        3. 执行该轨迹，无需在每步重新规划

        这与原始 run_sim_cl_simple.py 的行为一致，比逐步规划快约 20 倍。

        Args:
            episode_id: Episode ID
            max_steps: 最大步数
            seed: 随机种子

        Returns:
            Episode 统计数据字典，包含轨迹信息
        """
        # Reset 环境
        state, info = self.env.reset(seed=seed)
        reference_path = info.get('reference_path', [])

        if self.verbose:
            print(f"\n{'='*70}")
            print(f" Episode {episode_id + 1}")
            print(f"{'='*70}")

        # Reset 规划器
        self.planner.reset()

        # 存储 reference_path 到规划器
        if reference_path:
            self.planner.current_reference_path = [(p[0], p[1]) for p in reference_path]

        episode_start_time = time.time()

        # ===== 批量轨迹规划：一次性生成并评估所有候选轨迹 =====
        if self.verbose:
            print(f"  生成候选轨迹...")

        # 生成并评估所有候选轨迹
        trajectory_q_values = self._generate_and_evaluate_trajectories(state, reference_path)

        if not trajectory_q_values:
            if self.verbose:
                print(f"  警告：没有有效轨迹生成")
            return {
                'episode_id': episode_id,
                'steps': 0,
                'total_reward': 0.0,
                'avg_reward': 0.0,
                'outcome': 'failure',
                'time': time.time() - episode_start_time,
                'trajectory_q_values': [],
                'selected_trajectory_info': None,
            }

        # 选择最优轨迹
        selected_trajectory, selected_info = self._select_optimal_trajectory(trajectory_q_values)

        if self.verbose:
            print(f"  选中轨迹 {selected_info['trajectory_id']}: "
                  f"Min_Q={selected_info['min_q']:.2f}, "
                  f"Mean_Q={selected_info['mean_q']:.2f}, "
                  f"P{int(selected_info['selection_percentile']*100)}_Q={selected_info['percentile_q']:.2f}")

        # ===== 可视化和统计 =====
        episode_visualizer = None
        if self.enable_visualization:
            # 创建 episode 可视化器
            episode_visualizer = EpisodeVisualizer(
                episode_id=episode_id,
                output_dir=self.output_dir,
                grid_mapper=self.planner.grid_mapper,
                world_state=state,
                horizon=self.planner.config.lattice.horizon,
                verbose=self.verbose,
            )

            if self.verbose:
                print(f"  生成轨迹选择可视化...")

            # 可视化轨迹选择
            episode_visualizer.visualize_trajectory_selection(
                trajectory_q_values, selected_info
            )

            # 可选：可视化 Transition/Dirichlet 分布（每 N 个 episode）
            if (hasattr(self, 'visualize_distributions') and
                self.visualize_distributions and
                episode_id % self.vis_interval == 0):
                episode_visualizer.visualize_distributions(
                    q_calculator=self.planner.q_value_calculator,
                    world_state=state,
                    ego_action_trajectory=selected_info['waypoints'],
                    trajectory_buffer=self.planner.trajectory_buffer,
                    bank=self.planner.dirichlet_bank,
                )

        # 记录所有轨迹的 Q 值到 tracker
        if self.q_tracker is not None:
            self.q_tracker.add_all_trajectories_data(episode_id, trajectory_q_values)

            # 可视化 Q 值演化
            if self.global_visualizer is not None:
                self.global_visualizer.visualize_q_evolution(episode_id)

        # ===== 执行选中的轨迹 =====
        if self.verbose:
            print(f"  执行轨迹...")

        # Episode 统计
        total_reward = 0.0
        steps = 0
        outcome = 'success'

        # 执行轨迹（最多 max_steps 步）
        num_waypoints = len(selected_trajectory.waypoints)

        # 自动调整max_steps到轨迹长度
        if max_steps > num_waypoints - 1:
            if self.verbose:
                print(f"  调整: max_steps从{max_steps}调整为{num_waypoints - 1}（轨迹长度限制）")
            max_steps = num_waypoints - 1

        actual_steps = min(max_steps, num_waypoints - 1)  # -1 because we start from current position

        direct_waypoint_follow = hasattr(self.env, "step_to_waypoint")
        last_action: EgoControl = EgoControl(throttle=0.0, steer=0.0, brake=0.0)

        for step in range(actual_steps):
            if direct_waypoint_follow:
                target_wp = selected_trajectory.waypoints[step + 1]
                step_result = self.env.step_to_waypoint(target_wp)
                action = EgoControl(throttle=0.0, steer=0.0, brake=0.0)
            else:
                action = self._trajectory_to_control(state, selected_trajectory, step)
                step_result = self.env.step(action)

            last_action = action

            # ========== 可视化当前时间步 ==========
            if episode_visualizer is not None:
                try:
                    # 准备可视化数据
                    prob_grid, reachable_sets = self._prepare_timestep_visualization(
                        state=step_result.observation,
                        step=step,
                    )

                    # 获取统计信息
                    buffer_size = len(self.planner.trajectory_buffer)

                    # 计算总alpha值（所有agents的alpha总和）
                    total_alpha = 0.0
                    for agent_id in self.planner.dirichlet_bank.agent_alphas:
                        for timestep_idx in self.planner.dirichlet_bank.agent_alphas[agent_id]:
                            total_alpha += self.planner.dirichlet_bank.agent_alphas[agent_id][timestep_idx].sum()

                    # 渲染热力图帧（从 t=1 开始编号，与原版本一致）
                    frame_path = episode_visualizer.render_timestep_heatmap(
                        timestep=step + 1,
                        current_world_state=step_result.observation,
                        prob_grid=prob_grid,
                        multi_timestep_reachable_sets=reachable_sets,
                        buffer_size=buffer_size,
                        matched_transitions=None,  # 可以后续添加
                        total_alpha=total_alpha,
                    )
                except Exception as e:
                    if self.verbose:
                        print(f"    警告: 时间步 {step+1} 可视化失败: {e}")
            # =======================================

            # 更新规划器（仅用于统计）
            transition = Transition(
                state=state,
                action=action,
                reward=step_result.reward,
                next_state=step_result.observation,
                terminated=step_result.terminated,
                truncated=step_result.truncated,
                info=step_result.info,
            )
            self.planner.update(transition)

            # 更新统计
            total_reward += step_result.reward
            steps += 1
            state = step_result.observation

            if self.verbose and (step + 1) % 5 == 0:
                print(f"  Step {step+1}/{actual_steps}: reward={step_result.reward:.2f}, "
                      f"total={total_reward:.2f}")

            # 检查终止
            if step_result.terminated:
                outcome = 'collision'
                if self.verbose:
                    print(f"  ✗ 碰撞！Episode 在第 {steps} 步结束")
                break
            if step_result.truncated:
                outcome = 'timeout'
                if self.verbose:
                    print(f"  ⏱ 超时！Episode 在第 {steps} 步结束")
                break

        # Send final transition for successful completions to trigger buffer storage
        if outcome == 'success':
            final_transition = Transition(
                state=state,
                action=last_action,
                reward=0.0,
                next_state=state,
                terminated=False,
                truncated=True,  # Mark episode as complete
                info={},
            )
            self.planner.update(final_transition)

        episode_time = time.time() - episode_start_time

        if outcome == 'success' and self.verbose:
            print(f"  ✓ 成功完成 {steps} 步！")

        # ===== 生成 episode GIF =====
        gif_path = ""
        if episode_visualizer is not None and episode_visualizer.frame_paths:
            if self.verbose:
                print(f"  生成 episode GIF...")

            gif_path = episode_visualizer.generate_episode_gif()

            # 添加最后一帧到汇总
            if self.global_visualizer is not None and episode_visualizer.frame_paths:
                last_frame = episode_visualizer.frame_paths[-1]
                self.global_visualizer.add_summary_frame(last_frame)

        # 返回统计数据（包含轨迹信息）
        result = {
            'episode_id': episode_id,
            'steps': steps,
            'total_reward': total_reward,
            'avg_reward': total_reward / steps if steps > 0 else 0.0,
            'outcome': outcome,
            'time': episode_time,
            'trajectory_q_values': trajectory_q_values,
            'selected_trajectory_info': selected_info,
            'gif_path': gif_path,
        }

        if self.verbose:
            print(f"  总奖励: {total_reward:.2f}, 平均: {result['avg_reward']:.2f}")
            print(f"  耗时: {episode_time:.2f}s")
            if gif_path:
                print(f"  GIF: {gif_path}")

        return result

    def _generate_and_evaluate_trajectories(
        self,
        state: Any,
        reference_path: List,
    ) -> List[Dict[str, Any]]:
        """生成并评估所有候选轨迹

        Args:
            state: 当前状态
            reference_path: 参考路径

        Returns:
            包含每条轨迹 Q 值信息的列表
        """
        # 生成候选轨迹
        ego_state_tuple = (
            state.ego.position_m[0],
            state.ego.position_m[1],
            state.ego.yaw_rad,
        )

        ref_path = [(p[0], p[1]) for p in reference_path] if reference_path else None
        if ref_path is None:
            # 创建简单前向路径
            ego_x, ego_y = state.ego.position_m
            ref_path = [
                (ego_x + i * 5.0, ego_y) for i in range(self.planner.config.lattice.horizon + 1)
            ]

        candidate_trajectories = self.planner.lattice_planner.generate_trajectories(
            reference_path=ref_path,
            horizon=self.planner.config.lattice.horizon,
            dt=self.planner.config.lattice.dt,
            ego_state=ego_state_tuple,
        )

        if self.verbose:
            print(f"    生成 {len(candidate_trajectories)} 条候选轨迹")

        # 评估每条候选轨迹
        trajectory_q_values = []

        for traj_idx, trajectory in enumerate(candidate_trajectories):
            try:
                # 计算 Q 值
                q_values_list, detailed_info = self.planner.q_value_calculator.compute_q_value(
                    current_world_state=state,
                    ego_action_trajectory=trajectory.waypoints,
                    trajectory_buffer=self.planner.trajectory_buffer,
                    grid=self.planner.grid_mapper,
                    bank=self.planner.dirichlet_bank,
                    reference_path=ref_path,
                )

                if len(q_values_list) > 0:
                    # 计算 Q 值统计
                    min_q = float(np.min(q_values_list))
                    mean_q = float(np.mean(q_values_list))
                    max_q = float(np.max(q_values_list))

                    # 计算百分位 Q 值
                    percentile = self.planner.config.q_value.selection_percentile
                    if percentile == 0.0:
                        percentile_q = min_q
                    elif percentile == 1.0:
                        percentile_q = max_q
                    else:
                        percentile_q = float(np.percentile(q_values_list, percentile * 100))

                    # 获取碰撞率
                    collision_rate = detailed_info.get('reward_breakdown', {}).get('collision_rate', 0.0)

                    trajectory_info = {
                        'trajectory_id': traj_idx,
                        'trajectory': trajectory,
                        'waypoints': trajectory.waypoints,
                        'lateral_offset': trajectory.lateral_offset,
                        'target_speed': trajectory.target_speed,
                        'min_q': min_q,
                        'mean_q': mean_q,
                        'max_q': max_q,
                        'percentile_q': percentile_q,
                        'selection_percentile': percentile,
                        'collision_rate': collision_rate,
                        'q_values': q_values_list,
                        'detailed_info': detailed_info,
                    }

                    trajectory_q_values.append(trajectory_info)

            except Exception as e:
                if self.verbose:
                    print(f"    警告：轨迹 {traj_idx} 的 Q 值计算失败: {e}")
                continue

        return trajectory_q_values

    def _select_optimal_trajectory(
        self,
        trajectory_q_values: List[Dict[str, Any]],
    ) -> tuple:
        """根据百分位 Q 值选择最优轨迹

        Args:
            trajectory_q_values: 轨迹 Q 值列表

        Returns:
            (selected_trajectory, selected_info)
        """
        if not trajectory_q_values:
            return None, None

        # 选择 percentile_q 最大的轨迹
        best_trajectory_info = max(trajectory_q_values, key=lambda x: x['percentile_q'])

        return best_trajectory_info['trajectory'], best_trajectory_info

    def _trajectory_to_control(self, current_state, trajectory, step_idx):
        """将轨迹的某一步转换为控制动作

        Args:
            current_state: 当前状态
            trajectory: 轨迹对象
            step_idx: 步索引

        Returns:
            控制动作
        """
        # 使用 planner 的转换方法
        return self.planner._trajectory_to_control(current_state.ego, trajectory)

    def _prepare_timestep_visualization(self, state, step: int):
        """准备时间步可视化数据

        Args:
            state: 当前世界状态
            step: 时间步索引

        Returns:
            (prob_grid, multi_timestep_reachable_sets)
        """
        # 1. 计算 agents 的多时间步可达集
        config = self.planner.config
        multi_timestep_reachable_sets = {}

        for i, agent in enumerate(state.agents):
            agent_id = i + 1
            try:
                multi_reachable = self.planner.grid_mapper.multi_timestep_successor_cells(
                    agent,
                    horizon=config.lattice.horizon,
                    dt=config.lattice.dt,
                    n_samples=50,  # 采样数
                )
                multi_timestep_reachable_sets[agent_id] = multi_reachable
            except Exception as e:
                # 如果计算失败，使用当前位置作为可达集
                current_cell = self.planner.grid_mapper.world_to_cell(agent.position_m)
                multi_timestep_reachable_sets[agent_id] = {
                    t: [current_cell] for t in range(1, config.lattice.horizon + 1)
                }

        # 2. 获取概率分布（用于热力图）
        # 简化版本：使用均匀分布
        # 完整版本需要从 Dirichlet bank 中获取后验均值
        K = self.planner.grid_mapper.K
        prob_grid = np.ones(K) / K

        return prob_grid, multi_timestep_reachable_sets


class StatisticsCollector:
    """统计收集器"""

    def __init__(self):
        self.episodes_results: List[Dict[str, Any]] = []

    def add_episode(self, result: Dict[str, Any]):
        """添加 episode 结果"""
        self.episodes_results.append(result)

    def print_summary(self):
        """打印汇总统计"""
        if not self.episodes_results:
            print("\n没有完成的 episode")
            return

        print(f"\n{'='*70}")
        print(" 实验汇总")
        print(f"{'='*70}")

        # 基本统计
        num_episodes = len(self.episodes_results)
        total_steps = sum(r['steps'] for r in self.episodes_results)
        total_time = sum(r['time'] for r in self.episodes_results)

        # 结果统计
        outcomes = [r['outcome'] for r in self.episodes_results]
        success_count = outcomes.count('success')
        collision_count = outcomes.count('collision')
        timeout_count = outcomes.count('timeout')

        # 奖励统计
        rewards = [r['total_reward'] for r in self.episodes_results]
        avg_reward = np.mean(rewards)
        std_reward = np.std(rewards)

        # 步数统计
        steps_list = [r['steps'] for r in self.episodes_results]
        avg_steps = np.mean(steps_list)

        print(f"\nEpisodes: {num_episodes}")
        print(f"成功率: {success_count/num_episodes*100:.1f}% "
              f"({success_count} 成功, {collision_count} 碰撞, {timeout_count} 超时)")
        print(f"\n平均奖励: {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"平均步数: {avg_steps:.1f}")
        print(f"总耗时: {total_time:.2f}s")
        print(f"平均速度: {total_steps/total_time:.1f} steps/s")

        # 最佳和最差 episode
        best_idx = np.argmax(rewards)
        worst_idx = np.argmin(rewards)

        print(f"\n最佳 Episode: #{self.episodes_results[best_idx]['episode_id']+1}")
        print(f"  奖励: {self.episodes_results[best_idx]['total_reward']:.2f}, "
              f"步数: {self.episodes_results[best_idx]['steps']}, "
              f"结果: {self.episodes_results[best_idx]['outcome']}")

        print(f"\n最差 Episode: #{self.episodes_results[worst_idx]['episode_id']+1}")
        print(f"  奖励: {self.episodes_results[worst_idx]['total_reward']:.2f}, "
              f"步数: {self.episodes_results[worst_idx]['steps']}, "
              f"结果: {self.episodes_results[worst_idx]['outcome']}")

        print(f"\n{'='*70}")


def create_planner_config(args) -> C2OSRPlannerConfig:
    """创建规划器配置"""

    gc = get_global_config()
    grid_half = args.grid_size / 2.0
    dt = args.dt if args.dt is not None else gc.time.dt

    def build_dirichlet_config() -> DirichletConfig:
        return DirichletConfig(
            alpha_in=gc.dirichlet.alpha_in,
            alpha_out=gc.dirichlet.alpha_out,
            learning_rate=gc.dirichlet.learning_rate,
            use_multistep=True,
            use_optimized=True,
        )

    def build_reward_weights() -> RewardWeightsConfig:
        r = gc.reward
        return RewardWeightsConfig(
            collision_penalty=r.collision_penalty,
            collision_threshold=r.collision_threshold,
            collision_check_cell_radius=r.collision_check_cell_radius,
            comfort_weight=r.comfort_weight,
            efficiency_weight=r.efficiency_weight,
            safety_weight=r.safety_weight,
            max_accel_penalty=r.max_accel_penalty,
            max_jerk_penalty=r.max_jerk_penalty,
            acceleration_penalty_weight=r.acceleration_penalty_weight,
            jerk_penalty_weight=r.jerk_penalty_weight,
            max_comfortable_accel=r.max_comfortable_accel,
            speed_reward_weight=r.speed_reward_weight,
            target_speed=r.target_speed,
            progress_reward_weight=r.progress_reward_weight,
            safe_distance=r.safe_distance,
            distance_penalty_weight=r.distance_penalty_weight,
            centerline_offset_penalty_weight=r.centerline_offset_penalty_weight,
        )

    def build_q_value_config(n_samples: int | None = None) -> QValueConfig:
        return QValueConfig(
            n_samples=n_samples if n_samples is not None else gc.sampling.q_value_samples,
            selection_percentile=gc.c2osr.q_selection_percentile,
            gamma=gc.c2osr.gamma,
        )

    common_kwargs = dict(
        horizon=args.horizon,
        dirichlet=build_dirichlet_config(),
        reward_weights=build_reward_weights(),
        trajectory_storage_multiplier=gc.matching.trajectory_storage_multiplier,
        learning_rate=gc.dirichlet.learning_rate,
        gamma=gc.c2osr.gamma,
    )

    if args.config_preset == "fast":
        config = C2OSRPlannerConfig(
            grid=GridConfig(
                grid_size_m=1.0,
                bounds_x=(-grid_half, grid_half),
                bounds_y=(-grid_half, grid_half),
            ),
            lattice=LatticePlannerConfig(
                lateral_offsets=[-2.0, 0.0, 2.0],
                speed_variations=[3.0, 5.0],
                num_trajectories=6,
                dt=dt,
            ),
            q_value=build_q_value_config(n_samples=gc.sampling.q_value_samples),
            **common_kwargs,
        )
    elif args.config_preset == "high-precision":
        config = C2OSRPlannerConfig(
            grid=GridConfig(
                grid_size_m=0.5,
                bounds_x=(-grid_half, grid_half),
                bounds_y=(-grid_half, grid_half),
            ),
            lattice=LatticePlannerConfig(
                lateral_offsets=[-4.0, -2.0, 0.0, 2.0, 4.0],
                speed_variations=[2.0, 3.0, 5.0, 7.0],
                num_trajectories=20,
                dt=dt,
            ),
            q_value=build_q_value_config(n_samples=gc.sampling.q_value_samples),
            **common_kwargs,
        )
    else:
        config = C2OSRPlannerConfig(
            grid=GridConfig(
                grid_size_m=gc.grid.cell_size_m,
                bounds_x=(-grid_half, grid_half),
                bounds_y=(-grid_half, grid_half),
            ),
            lattice=LatticePlannerConfig(
                dt=dt,
            ),
            q_value=build_q_value_config(),
            **common_kwargs,
        )

    return config


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="C2OSR + ScenarioManager 实验（新架构版本）"
    )

    # 基本运行参数
    parser.add_argument("--episodes", type=int, default=10,
                       help="执行 episode 数")
    parser.add_argument("--max-steps", type=int, default=50,
                       help="每个 episode 的最大步数")
    parser.add_argument("--seed", type=int, default=2025,
                       help="随机种子")

    # 场景参数
    parser.add_argument("--reference-path-mode",
                       choices=["straight", "curve", "s_curve"],
                       default="straight",
                       help="参考路径模式")

    # 配置预设
    parser.add_argument("--config-preset",
                       choices=["default", "fast", "high-precision"],
                       default="default",
                       help="配置预设")

    # 规划参数
    parser.add_argument("--horizon", type=int, default=None,
                       help="规划时域（步数），默认读取 global_config.time.default_horizon")
    parser.add_argument("--dt", type=float, default=0.5,
                       help="时间步长（秒）")
    parser.add_argument("--grid-size", type=float, default=50.0,
                       help="网格大小（米）")

    # 输出参数
    parser.add_argument("--output-dir", type=str,
                       default="outputs/c2osr_scenario",
                       help="输出目录")
    parser.add_argument("--quiet", action="store_true",
                       help="静默模式（减少输出）")

    # 可视化参数
    parser.add_argument("--visualize-distributions", action="store_true",
                       default=True,  # 默认启用分布可视化
                       help="生成 Transition/Dirichlet 分布可视化")
    parser.add_argument("--vis-interval", type=int, default=5,
                       help="分布可视化的间隔（每 N 个 episode 生成一次）")

    return parser.parse_args()


def main():
    """主函数"""
    # 1. 解析参数
    args = parse_arguments()
    if args.horizon is None:
        args.horizon = get_global_config().time.default_horizon

    print(f"{'='*70}")
    print(f" C2OSR + ScenarioManager 实验（新架构版本）")
    print(f"{'='*70}")
    print(f"\n配置:")
    print(f"  Episodes: {args.episodes}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Horizon: {args.horizon}, dt: {args.dt}s")
    print(f"  Reference path: {args.reference_path_mode}")
    print(f"  Config preset: {args.config_preset}")
    print(f"  Grid size: {args.grid_size}m")
    print(f"  Seed: {args.seed}")

    # 2. 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output: {output_dir}")

    # 3. 创建可视化管道
    enable_visualization = not args.quiet
    q_tracker, global_visualizer = create_visualization_pipeline(
        output_dir=output_dir,
        enable_visualization=enable_visualization,
    )
    if enable_visualization:
        print(f"  ✓ 可视化管道创建完成")

    # 4. 创建组件
    print(f"\n创建组件...")

    # 创建场景管理器
    scenario_manager = ScenarioManager(grid_size_m=args.grid_size)

    # 创建环境
    env = ScenarioReplayEnvironment(
        scenario_manager=scenario_manager,
        reference_path_mode=args.reference_path_mode,
        dt=args.dt,
        max_episode_steps=args.max_steps,
        horizon=args.horizon,
    )

    # 创建规划器配置
    planner_config = create_planner_config(args)

    # 创建规划器
    planner = create_c2osr_planner(planner_config)

    print(f"  ✓ ScenarioManager 创建完成")
    print(f"  ✓ ScenarioReplayEnvironment 创建完成")
    print(f"  ✓ C2OSRPlanner 创建完成")

    # 5. 创建运行器和统计收集器
    runner = EpisodeRunner(
        planner=planner,
        env=env,
        q_tracker=q_tracker,
        global_visualizer=global_visualizer,
        enable_visualization=enable_visualization,
        output_dir=output_dir,
        verbose=not args.quiet,
        visualize_distributions=args.visualize_distributions,
        vis_interval=args.vis_interval,
    )
    stats_collector = StatisticsCollector()

    # 5. 运行 episodes
    print(f"\n开始运行 {args.episodes} 个 episodes...")

    experiment_start_time = time.time()

    for episode_id in range(args.episodes):
        try:
            result = runner.run_episode(
                episode_id=episode_id,
                max_steps=args.max_steps,
                seed=args.seed + episode_id,
            )
            stats_collector.add_episode(result)

        except Exception as e:
            print(f"\n✗ Episode {episode_id + 1} 失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    experiment_time = time.time() - experiment_start_time

    # 7. 打印汇总
    stats_collector.print_summary()

    # 8. 生成全局可视化
    if enable_visualization and global_visualizer is not None:
        print(f"\n生成全局可视化...")

        # 生成汇总 GIF
        summary_gif = global_visualizer.generate_summary_gif()
        if summary_gif:
            print(f"  ✓ 汇总 GIF: {summary_gif}")

        # 生成最终统计图
        global_visualizer.generate_final_plots()

    print(f"\n实验总耗时: {experiment_time:.2f}s")
    print(f"\n✓ 实验完成！结果保存在 {output_dir}")

    # 9. 关闭环境
    env.close()


if __name__ == "__main__":
    main()
