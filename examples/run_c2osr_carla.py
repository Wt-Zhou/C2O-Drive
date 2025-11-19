#!/usr/bin/env python3
"""
C2OSR + CARLA 集成运行脚本

将C2OSR算法与CARLA仿真环境集成，支持真实的3D仿真场景。

特性：
- 使用C2OSRPlanner（标准化算法接口）
- 使用CarlaEnvironment（Gym标准接口）
- 支持预定义场景库
- 支持自定义CARLA配置
- 实时性能监控
- 可视化和数据导出

使用前提：
- CARLA服务器必须已经启动（默认localhost:2000）
- 确保carla python包已安装或路径已配置
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import time

# 添加项目根目录到路径
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
    
_src_path = _repo_root / "src"
if _src_path.exists() and str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

# C2OSR算法组件
from c2o_drive.algorithms.c2osr.factory import create_c2osr_planner
from c2o_drive.algorithms.c2osr.config import (
    C2OSRPlannerConfig,
    LatticePlannerConfig,
    QValueConfig,
    GridConfig,
    DirichletConfig,
    RewardWeightsConfig,
)

# CARLA环境组件
from c2o_drive.environments.carla_env import CarlaEnvironment
from c2o_drive.environments.carla.scenarios import CarlaScenarioLibrary, get_scenario, list_scenarios
from c2o_drive.core.planner import Transition
from c2o_drive.core.types import EgoControl
from c2o_drive.config import get_global_config
from visualization_utils import EpisodeVisualizer, create_visualization_pipeline


class CarlaEpisodeRunner:
    """CARLA环境下的Episode运行器"""

    def __init__(
        self,
        planner,
        env: CarlaEnvironment,
        output_dir: Optional[Path] = None,
        verbose: bool = True,
        q_tracker=None,
        global_visualizer=None,
        visualize_distributions: bool = True,
        vis_interval: int = 5,
    ):
        self.planner = planner
        self.env = env
        self.output_dir = output_dir or Path("outputs/c2osr_carla")
        self.verbose = verbose
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.q_tracker = q_tracker
        self.global_visualizer = global_visualizer
        self.visualize_distributions = visualize_distributions
        self.vis_interval = max(1, vis_interval)

    def run_episode(
        self,
        episode_id: int,
        max_steps: int,
        scenario_name: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """运行单个episode

        Args:
            episode_id: Episode ID
            max_steps: 最大步数
            scenario_name: 场景名称（可选）
            seed: 随机种子

        Returns:
            Episode统计数据字典
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Episode {episode_id}")
            if scenario_name:
                print(f"场景: {scenario_name}")
            print(f"{'='*60}")

        episode_start_time = time.time()

        # 重置环境和规划器
        options = {}
        if scenario_name:
            scenario_def = get_scenario(scenario_name)
            options['scenario_config'] = {
                'scenario': scenario_def,
                'scenario_name': scenario_name,
            }
            reference_path = CarlaScenarioLibrary.get_reference_path(
                scenario_def,
                horizon=self.planner.config.horizon,
                dt=self.planner.config.lattice.dt,
            )
            options['reference_path'] = reference_path
            if self.verbose:
                print(f"场景: {scenario_def.description}")
                print(f"难度: {scenario_def.difficulty}")

        state, info = self.env.reset(seed=seed, options=options)
        initial_world_state = state
        self.planner.reset()

        reference_path = info.get('reference_path', [])
        if reference_path:
            reference_path = [(float(p[0]), float(p[1])) for p in reference_path]
            self.planner.current_reference_path = reference_path

        if self.verbose:
            print("  生成候选轨迹...")

        trajectory_q_values = self._generate_and_evaluate_trajectories(state, reference_path)
        if not trajectory_q_values:
            if self.verbose:
                print("  ✗ 没有有效轨迹可执行")
            return {
                'episode_id': episode_id,
                'steps': 0,
                'total_reward': 0.0,
                'outcome': 'planning_failed',
                'collision': False,
                'episode_time': 0.0,
                'scenario_name': scenario_name or 'default',
                'selected_trajectory_info': None,
                'gif_path': '',
                'trajectory_q_values': [],
            }

        selected_trajectory, selected_info = self._select_optimal_trajectory(trajectory_q_values)
        if self.verbose:
            print(f"  选中轨迹 {selected_info['trajectory_id']}: "
                  f"P{int(selected_info['selection_percentile']*100)}_Q="
                  f"{selected_info['percentile_q']:.2f}, "
                  f"Min_Q={selected_info['min_q']:.2f}, "
                  f"Mean_Q={selected_info['mean_q']:.2f}")

        episode_visualizer = EpisodeVisualizer(
            episode_id=episode_id,
            output_dir=self.output_dir,
            grid_mapper=self.planner.grid_mapper,
            world_state=initial_world_state,
            horizon=self.planner.config.lattice.horizon,
            verbose=self.verbose,
        )
        episode_visualizer.visualize_trajectory_selection(trajectory_q_values, selected_info)

        if self.visualize_distributions and (episode_id % self.vis_interval == 0):
            episode_visualizer.visualize_distributions(
                q_calculator=self.planner.q_value_calculator,
                world_state=initial_world_state,
                ego_action_trajectory=selected_info['waypoints'],
                trajectory_buffer=self.planner.trajectory_buffer,
                bank=self.planner.dirichlet_bank,
            )

        if self.q_tracker is not None:
            self.q_tracker.add_all_trajectories_data(episode_id, trajectory_q_values)
            self.q_tracker.add_episode_data(
                episode_id=episode_id,
                q_value=selected_info.get('percentile_q', 0.0),
                q_distribution=list(selected_info.get('q_values', [])),
                collision_rate=selected_info.get('collision_rate', 0.0),
                detailed_info=selected_info.get('detailed_info', {}),
            )
            if self.global_visualizer is not None:
                self.global_visualizer.visualize_q_evolution(episode_id)

        num_waypoints = len(selected_trajectory.waypoints)
        if num_waypoints < 2:
            if self.verbose:
                print("  ✗ 选中轨迹缺少waypoints")
            return {
                'episode_id': episode_id,
                'steps': 0,
                'total_reward': 0.0,
                'outcome': 'planning_failed',
                'collision': False,
                'episode_time': time.time() - episode_start_time,
                'scenario_name': scenario_name or 'default',
                'selected_trajectory_info': selected_info,
                'gif_path': '',
                'trajectory_q_values': trajectory_q_values,
            }

        if max_steps > num_waypoints - 1:
            if self.verbose:
                print(f"  调整: max_steps从{max_steps}调整为{num_waypoints - 1}（轨迹长度限制）")
            max_steps = num_waypoints - 1

        total_reward = 0.0
        steps = 0
        outcome = 'success'
        collision_occurred = False
        last_action = EgoControl(throttle=0.0, steer=0.0, brake=0.0)

        for step in range(max_steps):
            action = self._trajectory_to_control(state, selected_trajectory, step)
            last_action = action
            step_result = self.env.step(action)

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

            if episode_visualizer is not None:
                prob_grid, reachable_sets = self._prepare_timestep_visualization(step_result.observation)
                buffer_size = len(self.planner.trajectory_buffer)
                total_alpha = self._compute_total_alpha()
                episode_visualizer.render_timestep_heatmap(
                    timestep=step + 1,
                    current_world_state=step_result.observation,
                    prob_grid=prob_grid,
                    multi_timestep_reachable_sets=reachable_sets,
                    buffer_size=buffer_size,
                    matched_transitions=None,
                    total_alpha=total_alpha,
                )

            total_reward += step_result.reward
            steps += 1
            state = step_result.observation

            if self.verbose:
                print(f"  Step {step+1}/{max_steps}: reward={step_result.reward:.2f}, "
                      f"total={total_reward:.2f}")

            if step_result.terminated:
                outcome = 'collision'
                collision_occurred = True
                if self.verbose:
                    print(f"  ✗ 碰撞！Episode在第{steps}步结束")
                break

            if step_result.truncated:
                outcome = 'timeout'
                if self.verbose:
                    print(f"  ⏱ Episode在第{steps}步因truncated结束")
                break

        if outcome == 'success':
            final_transition = Transition(
                state=state,
                action=last_action,
                reward=0.0,
                next_state=state,
                terminated=False,
                truncated=True,
                info={},
            )
            self.planner.update(final_transition)

        episode_time = time.time() - episode_start_time

        if outcome == 'success' and self.verbose:
            print(f"  ✓ 成功完成{steps}步！")

        gif_path = ""
        if episode_visualizer is not None:
            gif_path = episode_visualizer.generate_episode_gif()
            if self.global_visualizer is not None and episode_visualizer.frame_paths:
                self.global_visualizer.add_summary_frame(episode_visualizer.frame_paths[-1])

        # 统计信息
        stats = {
            'episode_id': episode_id,
            'steps': steps,
            'total_reward': total_reward,
            'outcome': outcome,
            'collision': collision_occurred,
            'episode_time': episode_time,
            'scenario_name': scenario_name or 'default',
            'selected_trajectory_info': selected_info,
            'gif_path': gif_path,
            'trajectory_q_values': trajectory_q_values,
        }

        if self.verbose:
            print(f"  完成时间: {episode_time:.2f}s")
            print(f"  平均步时: {episode_time/max(steps, 1):.3f}s/step")

        return stats

    def _generate_and_evaluate_trajectories(
        self,
        state,
        reference_path: List,
    ) -> List[Dict[str, Any]]:
        """生成并评估候选轨迹"""
        ego_state_tuple = (
            state.ego.position_m[0],
            state.ego.position_m[1],
            state.ego.yaw_rad,
        )

        ref_path = reference_path or None
        if ref_path is None:
            ego_x, ego_y = state.ego.position_m
            ref_path = [
                (ego_x + i * 5.0, ego_y)
                for i in range(self.planner.config.lattice.horizon + 1)
            ]

        candidate_trajectories = self.planner.lattice_planner.generate_trajectories(
            reference_path=ref_path,
            horizon=self.planner.config.lattice.horizon,
            dt=self.planner.config.lattice.dt,
            ego_state=ego_state_tuple,
        )

        trajectory_q_values: List[Dict[str, Any]] = []

        for traj_idx, trajectory in enumerate(candidate_trajectories):
            try:
                q_values_list, detailed_info = self.planner.q_value_calculator.compute_q_value(
                    current_world_state=state,
                    ego_action_trajectory=trajectory.waypoints,
                    trajectory_buffer=self.planner.trajectory_buffer,
                    grid=self.planner.grid_mapper,
                    bank=self.planner.dirichlet_bank,
                    reference_path=ref_path,
                )

                if len(q_values_list) == 0:
                    continue

                min_q = float(np.min(q_values_list))
                mean_q = float(np.mean(q_values_list))
                max_q = float(np.max(q_values_list))

                percentile = self.planner.config.q_value.selection_percentile
                if percentile == 0.0:
                    percentile_q = min_q
                elif percentile == 1.0:
                    percentile_q = max_q
                else:
                    percentile_q = float(np.percentile(q_values_list, percentile * 100))

                collision_rate = detailed_info.get('reward_breakdown', {}).get('collision_rate', 0.0)

                trajectory_q_values.append({
                    'trajectory_id': traj_idx,
                    'trajectory': trajectory,
                    'waypoints': trajectory.waypoints,
                    'lateral_offset': getattr(trajectory, 'lateral_offset', 0.0),
                    'target_speed': getattr(trajectory, 'target_speed', 0.0),
                    'min_q': min_q,
                    'mean_q': mean_q,
                    'max_q': max_q,
                    'percentile_q': percentile_q,
                    'selection_percentile': percentile,
                    'collision_rate': collision_rate,
                    'q_values': q_values_list,
                    'detailed_info': detailed_info,
                })
            except Exception:
                continue

        return trajectory_q_values

    def _select_optimal_trajectory(
        self,
        trajectory_q_values: List[Dict[str, Any]],
    ):
        """根据百分位Q值选择最优轨迹"""
        best = max(trajectory_q_values, key=lambda t: t['percentile_q'])
        return best['trajectory'], best

    def _trajectory_to_control(
        self,
        current_state,
        trajectory,
        step_idx: int,
    ) -> EgoControl:
        """将轨迹waypoint转换为控制指令"""
        if step_idx + 1 >= len(trajectory.waypoints):
            return EgoControl(throttle=0.0, steer=0.0, brake=1.0)

        target_x, target_y = trajectory.waypoints[step_idx + 1]
        current_x, current_y = current_state.ego.position_m

        dx = target_x - current_x
        dy = target_y - current_y
        target_heading = np.arctan2(dy, dx)
        heading_error = target_heading - current_state.ego.yaw_rad
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

        steer = np.clip(heading_error * 0.5, -1.0, 1.0)

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

    def _prepare_timestep_visualization(self, state) -> tuple[np.ndarray, Dict[int, Dict[int, List[int]]]]:
        """准备多时间步可视化数据"""
        config = self.planner.config
        multi_timestep_reachable_sets: Dict[int, Dict[int, List[int]]] = {}

        for i, agent in enumerate(state.agents):
            agent_id = i + 1
            try:
                multi_reachable = self.planner.grid_mapper.multi_timestep_successor_cells(
                    agent,
                    horizon=config.lattice.horizon,
                    dt=config.lattice.dt,
                    n_samples=50,
                )
                multi_timestep_reachable_sets[agent_id] = multi_reachable
            except Exception:
                current_cell = self.planner.grid_mapper.world_to_cell(agent.position_m)
                multi_timestep_reachable_sets[agent_id] = {
                    t: [current_cell] for t in range(1, config.lattice.horizon + 1)
                }

        K = self.planner.grid_mapper.K
        prob_grid = np.ones(K, dtype=float) / float(K)
        return prob_grid, multi_timestep_reachable_sets

    def _compute_total_alpha(self) -> float:
        """计算Dirichlet bank中的总alpha值"""
        bank = getattr(self.planner, "dirichlet_bank", None)
        if bank is None:
            return 0.0

        total_alpha = 0.0
        for agent_data in bank.agent_alphas.values():
            for alpha_vec in agent_data.values():
                total_alpha += float(alpha_vec.sum())
        return total_alpha


def create_planner_config(args, grid_center: Optional[Tuple[float, float]] = None) -> C2OSRPlannerConfig:
    """根据命令行参数创建规划器配置"""

    gc = get_global_config()
    grid_half = args.grid_size / 2.0
    dt = args.dt if args.dt is not None else gc.time.dt
    center_x, center_y = grid_center if grid_center is not None else (0.0, 0.0)
    bounds_x = (center_x - grid_half, center_x + grid_half)
    bounds_y = (center_y - grid_half, center_y + grid_half)

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

    def build_q_value_config(n_samples: Optional[int] = None) -> QValueConfig:
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
        lattice_config = LatticePlannerConfig(
            lateral_offsets=[-2.0, 0.0, 2.0],
            speed_variations=[4.0],
            dt=dt,
        )
        q_value_config = build_q_value_config(n_samples=20)
    elif args.config_preset == "high-precision":
        lattice_config = LatticePlannerConfig(
            lateral_offsets=[-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0],
            speed_variations=[3.0, 4.0, 5.0],
            dt=dt,
        )
        q_value_config = build_q_value_config(n_samples=100)
    else:
        lattice_config = LatticePlannerConfig(
            lateral_offsets=[-3.0, -2.0, 0.0, 2.0, 3.0],
            speed_variations=[4.0],
            dt=dt,
        )
        q_value_config = build_q_value_config()

    config = C2OSRPlannerConfig(
        grid=GridConfig(
            grid_size_m=gc.grid.cell_size_m,
            bounds_x=bounds_x,
            bounds_y=bounds_y,
        ),
        lattice=lattice_config,
        q_value=q_value_config,
        **common_kwargs,
    )

    return config


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="C2OSR + CARLA集成实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本运行
  python run_c2osr_carla.py

  # 指定场景
  python run_c2osr_carla.py --scenario s4_wrong_way

  # 自定义配置
  python run_c2osr_carla.py --town Town04 --num-vehicles 20 --episodes 5

  # 查看所有可用场景
  python run_c2osr_carla.py --list-scenarios

可用场景: s4_wrong_way
        """
    )

    # 基本运行参数
    parser.add_argument("--episodes", type=int, default=5,
                       help="执行episode数")
    parser.add_argument("--max-steps", type=int, default=None,
                       help="每个episode的最大步数（默认等于horizon）")
    parser.add_argument("--seed", type=int, default=2025,
                       help="随机种子")

    # CARLA连接参数
    parser.add_argument("--host", type=str, default="localhost",
                       help="CARLA服务器地址")
    parser.add_argument("--port", type=int, default=2000,
                       help="CARLA服务器端口")

    # CARLA场景参数
    parser.add_argument("--town", type=str, default="Town03",
                       help="CARLA地图名称 (Town01-Town10)")
    parser.add_argument("--scenario", type=str, default=None,
                       help="预定义场景名称")
    parser.add_argument("--list-scenarios", action="store_true",
                       help="列出所有可用场景并退出")
    parser.add_argument("--num-vehicles", type=int, default=10,
                       help="环境车辆数量")
    parser.add_argument("--num-pedestrians", type=int, default=5,
                       help="行人数量")
    parser.add_argument("--no-rendering", action="store_true",
                       help="禁用CARLA渲染（提升性能）")

    # C2OSR配置参数
    parser.add_argument("--config-preset",
                       choices=["default", "fast", "high-precision"],
                       default="default",
                       help="配置预设")
    parser.add_argument("--horizon", type=int, default=None,
                       help="规划时域（步数），默认读取global_config")
    parser.add_argument("--dt", type=float, default=None,
                       help="时间步长（秒），默认读取global_config")
    parser.add_argument("--grid-size", type=float, default=200.0,
                       help="网格大小（米）")

    # 输出参数
    parser.add_argument("--output-dir", type=str,
                       default="outputs/c2osr_carla",
                       help="输出目录")
    parser.add_argument("--visualize-distributions", dest="visualize_distributions",
                       action="store_true",
                       help="生成Transition/Dirichlet分布可视化（默认开启）")
    parser.add_argument("--no-visualize-distributions", dest="visualize_distributions",
                       action="store_false",
                       help="禁用Transition/Dirichlet分布可视化")
    parser.add_argument("--vis-interval", type=int, default=5,
                       help="分布可视化间隔（episode数）")
    parser.add_argument("--quiet", action="store_true",
                       help="静默模式（减少输出）")

    parser.set_defaults(visualize_distributions=True)

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_arguments()
    gc = get_global_config()
    if args.horizon is None:
        args.horizon = gc.time.default_horizon
    if args.dt is None:
        args.dt = gc.time.dt
    if args.max_steps is None:
        args.max_steps = args.horizon

    # 处理--list-scenarios
    if args.list_scenarios:
        print("\n可用场景:")
        print("="*60)
        for scenario_name in list_scenarios():
            scenario = get_scenario(scenario_name)
            print(f"\n{scenario_name}:")
            print(f"  描述: {scenario.description}")
            print(f"  难度: {scenario.difficulty}")
            print(f"  路径模式: {scenario.reference_path_mode}")
        print("\n")
        return

    # 打印配置
    print(f"\n{'='*70}")
    print(f" C2OSR + CARLA 集成实验")
    print(f"{'='*70}")
    print(f"\nCARLA配置:")
    print(f"  服务器: {args.host}:{args.port}")
    print(f"  地图: {args.town}")
    print(f"  场景: {args.scenario or '默认'}")
    print(f"  车辆数: {args.num_vehicles}")
    print(f"  行人数: {args.num_pedestrians}")
    print(f"  渲染: {'禁用' if args.no_rendering else '启用'}")

    print(f"\nC2OSR配置:")
    print(f"  Episodes: {args.episodes}")
    print(f"  最大步数: {args.max_steps}")
    print(f"  Horizon: {args.horizon}, dt: {args.dt}s")
    print(f"  配置预设: {args.config_preset}")
    print(f"  网格大小: {args.grid_size}m")
    print(f"  种子: {args.seed}")

    scenario_def_for_planner = None
    grid_center = None
    if args.scenario:
        try:
            scenario_def_for_planner = get_scenario(args.scenario)
            grid_center = (
                float(scenario_def_for_planner.ego_spawn[0]),
                float(scenario_def_for_planner.ego_spawn[1]),
            )
        except Exception:
            grid_center = None

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n输出目录: {output_dir}")

    enable_visualization = not args.quiet
    q_tracker, global_visualizer = create_visualization_pipeline(
        output_dir=output_dir,
        enable_visualization=enable_visualization,
    )
    if enable_visualization:
        print(f"✓ 可视化管道已创建")

    # 创建环境
    print(f"\n正在连接CARLA服务器...")
    try:
        env = CarlaEnvironment(
            host=args.host,
            port=args.port,
            town=args.town,
            dt=args.dt,
            max_episode_steps=args.max_steps,
            num_vehicles=args.num_vehicles,
            num_pedestrians=args.num_pedestrians,
            no_rendering=args.no_rendering,
        )
        print(f"✓ 成功连接到CARLA服务器")
    except Exception as e:
        print(f"✗ 连接CARLA失败: {e}")
        print(f"\n请确保CARLA服务器已启动：")
        print(f"  cd /path/to/CARLA")
        print(f"  ./CarlaUE4.sh")
        return

    # 创建规划器
    print(f"\n创建C2OSR规划器...")
    planner_config = create_planner_config(args, grid_center=grid_center)
    planner = create_c2osr_planner(planner_config)
    print(f"✓ 规划器创建完成")

    # 创建运行器
    runner = CarlaEpisodeRunner(
        planner=planner,
        env=env,
        output_dir=output_dir,
        verbose=not args.quiet,
        q_tracker=q_tracker,
        global_visualizer=global_visualizer,
        visualize_distributions=args.visualize_distributions,
        vis_interval=args.vis_interval,
    )

    # 运行episodes
    print(f"\n{'='*70}")
    print(f"开始运行 {args.episodes} 个episodes")
    print(f"{'='*70}")

    all_stats = []
    total_start_time = time.time()

    for episode_id in range(args.episodes):
        episode_seed = args.seed + episode_id if args.seed else None

        try:
            stats = runner.run_episode(
                episode_id=episode_id,
                max_steps=args.max_steps,
                scenario_name=args.scenario,
                seed=episode_seed,
            )
            all_stats.append(stats)

        except KeyboardInterrupt:
            print(f"\n\n用户中断，提前结束")
            break
        except Exception as e:
            print(f"\n✗ Episode {episode_id} 失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    total_time = time.time() - total_start_time

    # 打印总结
    if len(all_stats) > 0:
        print(f"\n{'='*70}")
        print(f" 实验总结")
        print(f"{'='*70}")

        total_episodes = len(all_stats)
        successful = sum(1 for s in all_stats if s['outcome'] == 'success')
        collisions = sum(1 for s in all_stats if s['collision'])
        avg_reward = np.mean([s['total_reward'] for s in all_stats])
        avg_steps = np.mean([s['steps'] for s in all_stats])

        print(f"\n完成Episodes: {total_episodes}/{args.episodes}")
        print(f"成功: {successful} ({successful/total_episodes*100:.1f}%)")
        print(f"碰撞: {collisions} ({collisions/total_episodes*100:.1f}%)")
        print(f"平均奖励: {avg_reward:.2f}")
        print(f"平均步数: {avg_steps:.1f}")
        print(f"总用时: {total_time:.1f}s")
        print(f"平均episode用时: {total_time/total_episodes:.1f}s")

    if enable_visualization and global_visualizer is not None:
        print(f"\n生成全局可视化...")
        summary_gif = global_visualizer.generate_summary_gif()
        if summary_gif:
            print(f"  ✓ 汇总 GIF: {summary_gif}")
        global_visualizer.generate_final_plots()

    # 清理
    print(f"\n清理资源...")
    env.close()
    print(f"✓ 完成")


if __name__ == "__main__":
    main()
