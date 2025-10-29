#!/usr/bin/env python3
"""
基于Lattice规划器的多次场景执行（简化重构版）

这是replay_openloop_lattice.py的简化重构版本：
- 将复杂的run_episode函数拆分为多个独立模块
- 使用EpisodeContext封装参数，提高可读性
- 保持与原版本完全相同的功能

演示lattice轨迹规划与Q值评估结合的贝叶斯学习过程：
- 每个episode使用lattice planner生成候选轨迹
- 为每条候选轨迹计算Q值
- 使用百分位准则选择最优轨迹
- 跟踪Q值随episode的改进情况
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
from typing import Any, Dict

# 添加项目根目录到路径
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from carla_c2osr.agents.c2osr.grid import GridSpec, GridMapper
from carla_c2osr.agents.c2osr.spatial_dirichlet import DirichletParams, OptimizedMultiTimestepSpatialDirichletBank
from carla_c2osr.agents.c2osr.trajectory_buffer import TrajectoryBuffer
from carla_c2osr.evaluation.buffer_analyzer import BufferAnalyzer
from carla_c2osr.evaluation.q_evaluator import QEvaluator
from carla_c2osr.evaluation.q_distribution_tracker import QDistributionTracker
from carla_c2osr.evaluation.q_value_calculator import QValueConfig
from carla_c2osr.utils.simple_trajectory_generator import SimpleTrajectoryGenerator
from carla_c2osr.utils.lattice_planner import LatticePlanner
from carla_c2osr.utils.checkpoint_manager import CheckpointManager
from carla_c2osr.env.scenario_manager import ScenarioManager
from carla_c2osr.config import get_global_config, set_global_config, ConfigPresets

# 导入重构的模块
from carla_c2osr.runner.refactored import (
    EpisodeContext,
    TrajectoryEvaluator,
    TimestepExecutor,
    VisualizationManager,
    DataManager
)


def setup_output_dirs(base_dir: str = "outputs/replay_experiment") -> Path:
    """创建输出目录结构"""
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    return base_path


def run_episode(episode_id: int,
                horizon: int,
                reference_path,
                world_init,
                grid,
                bank,
                trajectory_buffer,
                scenario_state,
                rng,
                output_dir,
                sigma: float,
                lattice_planner=None,
                q_evaluator=None,
                trajectory_generator=None,
                scenario_manager=None,
                buffer_analyzer=None,
                q_tracker=None) -> Dict[str, Any]:
    """
    运行单个episode（简化重构版）

    使用模块化设计，将原来492行的函数简化到约80行。
    """
    # 1. 创建Episode上下文
    ctx = EpisodeContext.create(
        episode_id=episode_id,
        horizon=horizon,
        reference_path=reference_path,
        world_init=world_init,
        grid=grid,
        bank=bank,
        trajectory_buffer=trajectory_buffer,
        scenario_state=scenario_state,
        rng=rng,
        output_dir=output_dir,
        sigma=sigma,
        lattice_planner=lattice_planner,
        q_evaluator=q_evaluator,
        trajectory_generator=trajectory_generator,
        scenario_manager=scenario_manager,
        buffer_analyzer=buffer_analyzer,
        q_tracker=q_tracker
    )

    # 2. 生成并评估候选轨迹
    evaluator = TrajectoryEvaluator(ctx)
    trajectory_q_values = evaluator.generate_and_evaluate_trajectories()

    # 3. 选择最优轨迹
    ego_trajectory, selected_trajectory_info = evaluator.select_optimal_trajectory(trajectory_q_values)

    # 4. 可视化轨迹选择
    vis_manager = VisualizationManager(ctx)
    vis_manager.visualize_trajectory_selection(trajectory_q_values, selected_trajectory_info)

    # 5. 生成agent轨迹
    data_manager = DataManager(ctx)
    agent_trajectories, agent_trajectory_cells = data_manager.generate_agent_trajectories()

    # 6. 执行所有时间步
    timestep_executor = TimestepExecutor(ctx)
    episode_stats, frame_paths = timestep_executor.execute_all_timesteps(
        ego_trajectory, agent_trajectories
    )

    # 7. 生成episode GIF
    gif_path = vis_manager.generate_episode_gif(frame_paths)

    # 8. 记录所有轨迹的Q值数据
    if ctx.q_tracker is not None:
        ctx.q_tracker.add_all_trajectories_data(episode_id, trajectory_q_values)
        # 可视化所有轨迹Q值演化（从第1个episode到当前episode）
        vis_manager.visualize_q_evolution()

    # 9. 存储轨迹数据到buffer
    data_manager.store_episode_trajectories(
        ego_trajectory, agent_trajectories, agent_trajectory_cells
    )

    return {
        'episode_id': episode_id,
        'frame_paths': frame_paths,
        'gif_path': gif_path,
        'stats': episode_stats,
        'selected_trajectory': selected_trajectory_info,
        'all_trajectories': trajectory_q_values
    }


def initialize_components(args, world_init, output_dir):
    """初始化所有组件"""
    config = get_global_config()
    ego_start_pos = world_init.ego.position_m

    # 创建网格
    grid_spec = GridSpec(
        size_m=config.grid.grid_size_m,
        cell_m=config.grid.cell_size_m,
        macro=True
    )
    grid = GridMapper(grid_spec, world_center=ego_start_pos)

    # 创建轨迹生成器
    grid_half_size = grid.size_m / 2.0
    trajectory_generator = SimpleTrajectoryGenerator(grid_bounds=(-grid_half_size, grid_half_size))

    # 创建Dirichlet Bank
    dirichlet_params = DirichletParams(
        alpha_in=config.dirichlet.alpha_in,
        alpha_out=config.dirichlet.alpha_out,
        delta=config.dirichlet.delta,
        cK=config.dirichlet.cK
    )
    bank = OptimizedMultiTimestepSpatialDirichletBank(grid.K, dirichlet_params, horizon=config.time.default_horizon)
    print(f"🚀 使用终极优化版本的Dirichlet Bank - 维度自适应，零采样计算")

    # 创建其他组件
    trajectory_buffer = TrajectoryBuffer(horizon=config.time.default_horizon)
    q_evaluator = QEvaluator()
    buffer_analyzer = BufferAnalyzer(trajectory_buffer)
    q_tracker = QDistributionTracker()
    lattice_planner = LatticePlanner.from_config(config)
    scenario_manager = ScenarioManager()

    # 初始化agent的Dirichlet分布
    for i, agent in enumerate(world_init.agents):
        agent_id = i + 1
        multi_reachable = grid.multi_timestep_successor_cells(
            agent,
            horizon=config.time.default_horizon,
            dt=config.time.dt,
            n_samples=config.sampling.reachable_set_samples
        )
        if not multi_reachable:
            current_cell = grid.world_to_cell(agent.position_m)
            multi_reachable = {t: [current_cell] for t in range(1, config.time.default_horizon + 1)}
        bank.init_agent(agent_id, multi_reachable)

    return {
        'grid': grid,
        'bank': bank,
        'trajectory_buffer': trajectory_buffer,
        'trajectory_generator': trajectory_generator,
        'q_evaluator': q_evaluator,
        'buffer_analyzer': buffer_analyzer,
        'q_tracker': q_tracker,
        'lattice_planner': lattice_planner,
        'scenario_manager': scenario_manager
    }


def run_all_episodes(args, components, reference_path, world_init, scenario_state, output_dir,
                     checkpoint_manager=None, start_episode=0):
    """运行所有episodes"""
    all_episodes = []
    summary_frames = []

    for e in range(start_episode, args.episodes):
        try:
            rng = np.random.default_rng(args.seed + e)

            print(f"\nRunning Episode {e+1}/{args.episodes}")
            config = get_global_config()
            episode_result = run_episode(
                e, config.time.default_horizon, reference_path, world_init,
                components['grid'], components['bank'], components['trajectory_buffer'],
                scenario_state, rng, output_dir, args.sigma,
                lattice_planner=components['lattice_planner'],
                q_evaluator=components['q_evaluator'],
                trajectory_generator=components['trajectory_generator'],
                scenario_manager=components['scenario_manager'],
                buffer_analyzer=components['buffer_analyzer'],
                q_tracker=components['q_tracker']
            )
            all_episodes.append(episode_result)

            # 收集最后一帧用于汇总GIF
            if episode_result['frame_paths']:
                summary_frames.append(episode_result['frame_paths'][-1])

            # 打印完成状态
            if episode_result['stats']:
                final_stats = episode_result['stats'][-1]
                print(f"  完成: alpha_sum={final_stats['alpha_sum']:.1f}, "
                      f"nz_cells={final_stats['nz_cells']}")

            # 每10个episode清理matplotlib内存
            if (e + 1) % 10 == 0:
                import matplotlib.pyplot as plt
                plt.close('all')
                print(f"  内存清理: Episode {e+1}")

            # 定期保存checkpoint
            if checkpoint_manager and args.checkpoint_interval > 0 and (e + 1) % args.checkpoint_interval == 0:
                try:
                    # 准备配置字典
                    config_dict = {
                        'time': config.time.__dict__,
                        'sampling': config.sampling.__dict__,
                        'grid': config.grid.__dict__,
                        'dirichlet': config.dirichlet.__dict__,
                        'matching': config.matching.__dict__,
                        'reward': config.reward.__dict__,
                        'lattice': config.lattice.__dict__,
                        'visualization': config.visualization.__dict__
                    }

                    checkpoint_manager.save_checkpoint(
                        episode_id=e,
                        trajectory_buffer=components['trajectory_buffer'],
                        dirichlet_bank=components['bank'],
                        q_tracker=components['q_tracker'],
                        config=config_dict,
                        metadata={
                            'episodes_total': args.episodes,
                            'checkpoint_interval': args.checkpoint_interval
                        }
                    )
                except Exception as checkpoint_ex:
                    print(f"  ⚠️ Checkpoint保存失败: {checkpoint_ex}")

        except Exception as ex:
            print(f"Episode {e+1} 执行失败: {ex}")
            print("继续执行下一个episode...")
            continue

    return all_episodes, summary_frames


def print_summary(all_episodes, components, output_dir):
    """打印执行摘要"""
    print(f"\n=== 完成 ===")
    print(f"输出目录: {output_dir}")

    # 打印轨迹选择改进趋势
    selected_trajectories = [ep['selected_trajectory'] for ep in all_episodes if ep['selected_trajectory']]

    if selected_trajectories:
        first_selected = selected_trajectories[0]
        last_selected = selected_trajectories[-1]

        q_config = QValueConfig.from_global_config()
        percentile = q_config.q_selection_percentile

        print(f"\n轨迹选择改进:")
        print(f"  第1个episode: 轨迹{first_selected['trajectory_id']}")
        print(f"    Min_Q={first_selected['min_q']:.2f}, "
              f"Mean_Q={first_selected['mean_q']:.2f}, "
              f"P{int(percentile*100)}_Q={first_selected['percentile_q']:.2f}, "
              f"碰撞率={first_selected['collision_rate']:.3f}")
        print(f"  第{len(selected_trajectories)}个episode: 轨迹{last_selected['trajectory_id']}")
        print(f"    Min_Q={last_selected['min_q']:.2f}, "
              f"Mean_Q={last_selected['mean_q']:.2f}, "
              f"P{int(percentile*100)}_Q={last_selected['percentile_q']:.2f}, "
              f"碰撞率={last_selected['collision_rate']:.3f}")

        percentile_q_improvement = last_selected['percentile_q'] - first_selected['percentile_q']
        collision_rate_improvement = first_selected['collision_rate'] - last_selected['collision_rate']

        print(f"  P{int(percentile*100)}_Q改进: {percentile_q_improvement:+.2f}, "
              f"碰撞率降低: {collision_rate_improvement:+.3f}")

    # 打印学习趋势
    first_stats = all_episodes[0]['stats'][-1]
    last_stats = all_episodes[-1]['stats'][-1]
    print(f"\nDirichlet学习: Alpha {first_stats['alpha_sum']:.1f} -> {last_stats['alpha_sum']:.1f}, "
          f"非零单元 {first_stats['nz_cells']} -> {last_stats['nz_cells']}")

    # 打印buffer统计
    buffer_stats = components['buffer_analyzer'].get_buffer_stats()
    config = get_global_config()
    storage_multiplier = config.matching.trajectory_storage_multiplier
    actual_episodes = buffer_stats['total_episodes'] // storage_multiplier if storage_multiplier > 1 else buffer_stats['total_episodes']
    print(f"\nBuffer: {buffer_stats['total_agents']} agents, "
          f"{buffer_stats['total_episodes']} 条存储记录 (实际{actual_episodes}个episode × {storage_multiplier}倍), "
          f"{buffer_stats['total_agent_episodes']} agent-episodes")

    # 生成Q值分布可视化
    q_tracker = components['q_tracker']
    if len(q_tracker.q_value_history) > 0:
        q_evolution_path = output_dir / "q_distribution_evolution.png"
        collision_rate_path = output_dir / "collision_rate_evolution.png"
        q_data_path = output_dir / "q_distribution_data.json"

        try:
            q_tracker.plot_q_distribution_evolution(str(q_evolution_path))
            q_tracker.plot_collision_rate_evolution(str(collision_rate_path))
            q_tracker.save_data(str(q_data_path))

            print(f"\n可视化已生成: {q_evolution_path.name}, {collision_rate_path.name}, {q_data_path.name}")

        except Exception as e:
            print(f"警告: Q值分布可视化失败: {e}")
    else:
        print(f"\n警告: 没有Q值数据")


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="基于Lattice规划器的Q值优化实验（简化重构版）")

    # 基本运行参数
    parser.add_argument("--episodes", type=int, default=20, help="执行episode数")
    parser.add_argument("--seed", type=int, default=2025, help="随机种子")
    parser.add_argument("--gif-fps", type=int, default=2, help="GIF帧率")
    parser.add_argument("--ego-mode", choices=["straight", "fixed-traj"],
                       default="straight", help="自车运动模式")
    parser.add_argument("--sigma", type=float, default=0.5, help="软计数核宽度")

    # Checkpoint参数
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint保存目录")
    parser.add_argument("--checkpoint-interval", type=int, default=0, help="Checkpoint保存间隔（每N个episode，0表示不定期保存）")
    parser.add_argument("--resume-from", type=str, help="从指定checkpoint恢复训练")

    # 配置预设参数
    parser.add_argument("--config-preset", choices=["default", "fast", "high-precision", "long-horizon"],
                       default="default", help="预设配置模板")

    # 可选覆盖参数
    parser.add_argument("--dt", type=float, help="覆盖时间步长（秒）")
    parser.add_argument("--horizon", type=int, help="覆盖预测时间步数")
    parser.add_argument("--reachable-samples", type=int, help="覆盖可达集采样数量")
    parser.add_argument("--q-samples", type=int, help="覆盖Q值采样数量")

    return parser.parse_args()


def configure_system(args):
    """配置系统参数"""
    # 应用预设配置
    if args.config_preset == "fast":
        config = ConfigPresets.fast_testing()
    elif args.config_preset == "high-precision":
        config = ConfigPresets.high_precision()
    elif args.config_preset == "long-horizon":
        config = ConfigPresets.long_horizon()
    else:
        config = get_global_config()

    # 覆盖特定参数
    if args.dt is not None:
        config.time.dt = args.dt
    if args.horizon is not None:
        config.time.default_horizon = args.horizon
    if args.reachable_samples is not None:
        config.sampling.reachable_set_samples = args.reachable_samples
    if args.q_samples is not None:
        config.sampling.q_value_samples = args.q_samples

    # 设置运行参数
    config.random_seed = args.seed
    config.visualization.gif_fps = args.gif_fps

    set_global_config(config)

    # 打印配置信息
    print(f"=== Lattice规划器 + Q值优化实验（简化重构版） ===")
    print(f"Episodes: {args.episodes}, Horizon: {config.time.default_horizon}")
    print(f"Ego mode: {args.ego_mode}, Sigma: {args.sigma}")
    print(f"Seed: {args.seed}")
    print(f"配置预设: {args.config_preset}")
    print(f"时间步长: {config.time.dt}s, 预测时间: {config.time.horizon_seconds:.1f}s")
    print(f"可达集采样: {config.sampling.reachable_set_samples}, Q值采样: {config.sampling.q_value_samples}")
    print(f"Lattice轨迹数: {config.lattice.num_trajectories}, "
          f"横向偏移: {config.lattice.lateral_offsets}, "
          f"速度变化: {config.lattice.speed_variations}")
    print(f"轨迹存储倍数: {config.matching.trajectory_storage_multiplier}x (数据增强)")

    # 设置numpy随机种子
    np.random.seed(args.seed)

    return config


def main():
    # 1. 解析参数
    args = parse_arguments()

    # 2. 配置系统
    config = configure_system(args)

    # 3. 设置输出目录
    output_dir = setup_output_dirs()

    # 4. 创建初始场景
    scenario_manager = ScenarioManager()
    world_init = scenario_manager.create_scenario()
    scenario_state = scenario_manager.create_scenario_state(world_init)

    # 5. 初始化组件
    components = initialize_components(args, world_init, output_dir)

    # 6. 生成reference path
    reference_path = components['scenario_manager'].generate_reference_path(
        mode=args.ego_mode,
        horizon=config.time.default_horizon,
        ego_start=world_init.ego.position_m
    )
    print(f"\n生成Reference Path: {len(reference_path)} 个waypoints (mode={args.ego_mode})")

    # 6.5. Checkpoint管理和恢复
    checkpoint_manager = CheckpointManager(checkpoint_dir=args.checkpoint_dir)
    start_episode = 0

    if args.resume_from:
        print(f"\n🔄 从checkpoint恢复训练...")
        checkpoint_data = checkpoint_manager.load_checkpoint(args.resume_from)

        # 恢复training_state
        start_episode = checkpoint_data['training_state']['episode_id'] + 1

        # 恢复组件状态
        components['trajectory_buffer'] = TrajectoryBuffer.from_dict(checkpoint_data['trajectory_buffer_data'])
        components['bank'] = OptimizedMultiTimestepSpatialDirichletBank.from_dict(checkpoint_data['dirichlet_bank_data'])

        # 恢复QDistributionTracker
        q_tracker_data = checkpoint_data['q_tracker_data']
        components['q_tracker'].episode_data = q_tracker_data.get('episode_data', [])
        components['q_tracker'].q_value_history = q_tracker_data.get('q_value_history', [])
        components['q_tracker'].percentile_q_history = q_tracker_data.get('percentile_q_history', [])
        components['q_tracker'].collision_rate_history = q_tracker_data.get('collision_rate_history', [])
        components['q_tracker'].all_trajectories_history = q_tracker_data.get('all_trajectories_history', [])  # 新增：恢复所有轨迹Q值历史
        components['q_tracker'].q_distribution_history = [ep['q_distribution'] for ep in components['q_tracker'].episode_data]
        components['q_tracker'].detailed_info_history = [ep.get('detailed_info', {}) for ep in components['q_tracker'].episode_data]

        # 更新buffer_analyzer
        components['buffer_analyzer'] = BufferAnalyzer(components['trajectory_buffer'])

        print(f"✅ 已恢复到Episode {start_episode}，继续训练...")

    # 7. 运行所有episodes
    all_episodes, summary_frames = run_all_episodes(
        args, components, reference_path, world_init, scenario_state, output_dir,
        checkpoint_manager=checkpoint_manager, start_episode=start_episode
    )

    # 8. 生成汇总GIF
    if summary_frames:
        VisualizationManager.generate_summary_gif(summary_frames, output_dir)
    else:
        print("\n警告: 没有成功的episode，跳过汇总GIF生成")

    # 9. 打印摘要
    if all_episodes:
        print_summary(all_episodes, components, output_dir)
    else:
        print("\n警告: 所有episode都失败了")

    # 10. 保存最终checkpoint
    if all_episodes:
        try:
            print(f"\n💾 保存最终checkpoint...")
            # 准备配置字典
            config_dict = {
                'time': config.time.__dict__,
                'sampling': config.sampling.__dict__,
                'grid': config.grid.__dict__,
                'dirichlet': config.dirichlet.__dict__,
                'matching': config.matching.__dict__,
                'reward': config.reward.__dict__,
                'lattice': config.lattice.__dict__,
                'visualization': config.visualization.__dict__
            }

            checkpoint_manager.save_checkpoint(
                episode_id=args.episodes - 1,
                trajectory_buffer=components['trajectory_buffer'],
                dirichlet_bank=components['bank'],
                q_tracker=components['q_tracker'],
                config=config_dict,
                metadata={
                    'episodes_total': args.episodes,
                    'is_final': True
                }
            )
        except Exception as checkpoint_ex:
            print(f"  ⚠️ 最终checkpoint保存失败: {checkpoint_ex}")


if __name__ == "__main__":
    main()
