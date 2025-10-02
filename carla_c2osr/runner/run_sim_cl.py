#!/usr/bin/env python3
"""
基于Lattice规划器的多次场景执行（Q值优化版）

演示lattice轨迹规划与Q值评估结合的贝叶斯学习过程：
- 每个episode使用lattice planner生成候选轨迹
- 为每条候选轨迹计算Q值
- 使用min-max准则选择最优轨迹（最大化最小Q值）
- 跟踪Q值随episode的改进情况
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import yaml
from typing import Any, List, Dict

# 添加项目根目录到路径
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from carla_c2osr.env.types import AgentState, EgoState, WorldState, AgentType
from carla_c2osr.agents.c2osr.grid import GridSpec, GridMapper
from carla_c2osr.agents.c2osr.spatial_dirichlet import DirichletParams, SpatialDirichletBank, MultiTimestepSpatialDirichletBank, OptimizedMultiTimestepSpatialDirichletBank
from carla_c2osr.agents.c2osr.trajectory_buffer import TrajectoryBuffer, AgentTrajectoryData, ScenarioState
from carla_c2osr.agents.c2osr.risk import compose_union_singlelayer
from carla_c2osr.visualization.vis import grid_heatmap, make_gif
from carla_c2osr.visualization.transition_visualizer import visualize_transition_distributions, visualize_dirichlet_distributions
from carla_c2osr.visualization.lattice_visualizer import visualize_lattice_selection, visualize_lattice_trajectories_detailed

# 导入重构后的模块
from carla_c2osr.evaluation.rewards import RewardCalculator, DistanceBasedCollisionDetector
from carla_c2osr.evaluation.q_evaluator import QEvaluator
from carla_c2osr.evaluation.buffer_analyzer import BufferAnalyzer
from carla_c2osr.evaluation.q_value_calculator import QValueCalculator, QValueConfig
from carla_c2osr.evaluation.q_distribution_tracker import QDistributionTracker
from carla_c2osr.utils.simple_trajectory_generator import SimpleTrajectoryGenerator
from carla_c2osr.utils.lattice_planner import LatticePlanner
from carla_c2osr.utils.checkpoint_manager import CheckpointManager
from carla_c2osr.env.scenario_manager import ScenarioManager
from carla_c2osr.config import get_global_config, update_dt, update_horizon, get_dt, get_horizon


def setup_output_dirs(base_dir: str = "outputs/replay_experiment") -> Path:
    """创建输出目录结构。"""
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    return base_path


def get_percentile_q_value(q_values: np.ndarray, percentile: float) -> float:
    """
    根据百分位数获取Q值

    Args:
        q_values: Q值数组
        percentile: 百分位数 [0.0, 1.0]，0.0表示最小值，1.0表示最大值

    Returns:
        百分位对应的Q值（使用线性插值）

    Examples:
        >>> q = np.array([1, 2, 3, 4, 5])
        >>> get_percentile_q_value(q, 0.0)  # 最小值
        1.0
        >>> get_percentile_q_value(q, 0.5)  # 中位数
        3.0
        >>> get_percentile_q_value(q, 0.25)  # 25%分位
        2.0
    """
    if len(q_values) == 0:
        return 0.0

    if len(q_values) == 1:
        return float(q_values[0])

    # 排序Q值
    sorted_q = np.sort(q_values)
    n = len(sorted_q)

    # 计算精确位置（0-based index）
    position = percentile * (n - 1)

    # 下界和上界索引
    lower_idx = int(np.floor(position))
    upper_idx = int(np.ceil(position))

    # 如果正好是整数位置
    if lower_idx == upper_idx:
        return float(sorted_q[lower_idx])

    # 线性插值
    weight = position - lower_idx
    return float(sorted_q[lower_idx] * (1 - weight) + sorted_q[upper_idx] * weight)


def run_episode(episode_id: int,
                horizon: int,
                reference_path: List[np.ndarray],
                world_init: WorldState,
                grid: GridMapper,
                bank: SpatialDirichletBank,
                trajectory_buffer: TrajectoryBuffer,
                scenario_state: ScenarioState,
                rng: np.random.Generator,
                output_dir: Path,
                sigma: float,
                lattice_planner: LatticePlanner = None,
                q_evaluator: QEvaluator = None,
                trajectory_generator: TrajectoryGenerator = None,
                scenario_manager: ScenarioManager = None,
                buffer_analyzer: BufferAnalyzer = None,
                q_tracker: QDistributionTracker = None) -> Dict[str, Any]:
    """运行单个episode - 使用lattice规划器选择最优轨迹。"""

    # 初始化组件
    if q_evaluator is None:
        q_evaluator = QEvaluator()
    if scenario_manager is None:
        scenario_manager = ScenarioManager()
    if buffer_analyzer is None:
        buffer_analyzer = BufferAnalyzer(trajectory_buffer)
    if lattice_planner is None:
        lattice_planner = LatticePlanner.from_config(get_global_config())

    # 创建episode输出目录
    ep_dir = output_dir / f"ep_{episode_id:02d}"
    ep_dir.mkdir(exist_ok=True)

    # ===== 第1步：生成候选轨迹 =====
    config = get_global_config()

    # 从world_init获取自车当前状态
    ego_state = (
        world_init.ego.position_m[0],
        world_init.ego.position_m[1],
        world_init.ego.yaw_rad
    )

    candidate_trajectories = lattice_planner.generate_trajectories(
        reference_path=reference_path,
        horizon=horizon,
        dt=config.time.dt,
        ego_state=ego_state  # 传入自车状态
    )
    print(f"  生成 {len(candidate_trajectories)} 条候选轨迹")

    # ===== 第2步：为每条候选轨迹计算Q值 =====
    trajectory_q_values = []

    for traj in candidate_trajectories:
        # 构造自车动作轨迹
        ego_action_trajectory = traj.waypoints

        # 创建Q值配置和计算器
        q_config = QValueConfig.from_global_config()
        reward_config = config.reward
        q_calculator = QValueCalculator(q_config, reward_config)

        # 计算Q值（传入reference_path用于中心线偏移惩罚）
        try:
            q_values, detailed_info = q_calculator.compute_q_value(
                current_world_state=world_init,
                ego_action_trajectory=ego_action_trajectory,
                trajectory_buffer=trajectory_buffer,
                grid=grid,
                bank=bank,
                rng=rng,
                reference_path=reference_path
            )

            # 计算Q值统计指标
            min_q = np.min(q_values)
            mean_q = np.mean(q_values)
            percentile_q = get_percentile_q_value(q_values, q_config.q_selection_percentile)

            trajectory_q_values.append({
                'trajectory_id': traj.trajectory_id,
                'lateral_offset': traj.lateral_offset,
                'target_speed': traj.target_speed,
                'min_q': min_q,
                'mean_q': mean_q,
                'percentile_q': percentile_q,
                'q_values': q_values,
                'collision_rate': detailed_info['reward_breakdown']['collision_rate'],
                'trajectory': traj.waypoints
            })

        except Exception as e:
            print(f"  警告: 轨迹{traj.trajectory_id}计算失败: {e}")
            continue

    # ===== 第3步：使用百分位Q值选择最优轨迹 =====
    if not trajectory_q_values:
        print(f"  警告: 没有有效轨迹，使用reference path")
        ego_trajectory = reference_path
        selected_trajectory_info = None
    else:
        # 选择percentile_q最大的轨迹
        best_trajectory = max(trajectory_q_values, key=lambda x: x['percentile_q'])
        ego_trajectory = best_trajectory['trajectory']
        selected_trajectory_info = best_trajectory

        # 获取百分位数配置用于日志
        q_config = QValueConfig.from_global_config()
        percentile = q_config.q_selection_percentile

        print(f"  选中轨迹{best_trajectory['trajectory_id']}: "
              f"偏移={best_trajectory['lateral_offset']:.1f}m, "
              f"速度={best_trajectory['target_speed']:.1f}m/s")
        print(f"    Min_Q={best_trajectory['min_q']:.2f}, "
              f"Mean_Q={best_trajectory['mean_q']:.2f}, "
              f"P{int(percentile*100)}_Q={best_trajectory['percentile_q']:.2f}, "
              f"碰撞率={best_trajectory['collision_rate']:.3f}")

        # ===== 可视化lattice轨迹选择 =====
        try:
            # 主要可视化：轨迹 + Q值柱状图
            visualize_lattice_selection(
                trajectory_q_values=trajectory_q_values,
                selected_trajectory_info=selected_trajectory_info,
                current_world_state=world_init,
                grid=grid,
                episode_idx=episode_id,
                output_dir=ep_dir
            )

            # 详细可视化：Q值分布箱线图
            visualize_lattice_trajectories_detailed(
                trajectory_q_values=trajectory_q_values,
                selected_trajectory_info=selected_trajectory_info,
                current_world_state=world_init,
                grid=grid,
                episode_idx=episode_id,
                output_dir=ep_dir
            )

        except Exception as e:
            print(f"  警告: Lattice可视化失败: {e}")

    # 为每个环境智能体生成固定的动力学轨迹
    agent_trajectories = {}
    agent_trajectory_cells = {}  # 用于存储到buffer的轨迹单元ID

    # 轨迹生成随机源：从配置读取
    # 注意：轨迹模式现在可以通过 config.agent_trajectory.mode 设置 ("dynamic", "straight", "stationary")
    rng = np.random.default_rng(config.agent_trajectory.random_seed)

    for i, agent in enumerate(world_init.agents):
        agent_id = i + 1
        try:
            # 生成符合动力学约束的轨迹
            trajectory = trajectory_generator.generate_agent_trajectory(agent, horizon)
            agent_trajectories[agent_id] = trajectory

            # 将轨迹转换为网格单元ID
            trajectory_cells = []
            for pos in trajectory:
                cell_id = grid.world_to_cell(tuple(pos))
                trajectory_cells.append(cell_id)
            agent_trajectory_cells[agent_id] = trajectory_cells

        except Exception as e:
            print(f"  警告: Agent {agent_id} 轨迹生成失败: {e}")
            # 使用简单的直线轨迹作为后备
            fallback_trajectory = []
            fallback_cells = []
            start_pos = np.array(agent.position_m)
            for t in range(horizon):
                next_pos = start_pos + np.array([0.5 * t, 0.1 * t])  # 简单移动
                grid_half_size = grid.size_m / 2.0
                next_pos = np.clip(next_pos, -grid_half_size, grid_half_size)
                fallback_trajectory.append(next_pos)
                fallback_cells.append(grid.world_to_cell(tuple(next_pos)))
            agent_trajectories[agent_id] = fallback_trajectory
            agent_trajectory_cells[agent_id] = fallback_cells
    
    # 逐时刻执行和可视化
    frame_paths = []
    episode_stats = []
    
    for t in range(horizon):
        # 创建当前世界状态
        world_current = scenario_manager.create_world_state_from_trajectories(
            t, ego_trajectory, agent_trajectories, world_init
        )
        
        # 基于当前时刻状态创建ScenarioState（用于查询历史数据）
        current_scenario_state = scenario_manager.create_scenario_state(world_current)
        
        # 计算每个智能体当前位置的多时间步可达集
        config = get_global_config()
        current_reachable = {}
        multi_timestep_reachable = {}
        for i, agent in enumerate(world_current.agents):
            agent_id = i + 1
            # 计算单时间步可达集（向后兼容）
            reachable = grid.successor_cells(agent, n_samples=config.sampling.reachable_set_samples_legacy)
            current_reachable[agent_id] = reachable
            # 计算多时间步可达集
            multi_reachable = grid.multi_timestep_successor_cells(
                agent, 
                horizon=horizon, 
                dt=config.time.dt, 
                n_samples=config.sampling.reachable_set_samples
            )
            multi_timestep_reachable[agent_id] = multi_reachable
        
        # 使用新的Q值计算器进行Q值评估
        if t == 0:  # 只在第一个时间步计算Q值
            # 构造自车未来动作轨迹
            ego_action_trajectory = []
            for action_t in range(t, min(t + horizon, len(ego_trajectory))):
                ego_action_trajectory.append(tuple(ego_trajectory[action_t]))

            try:
                # 创建Q值配置
                q_config = QValueConfig.from_global_config()

                # 直接使用全局配置中的奖励配置
                global_config = get_global_config()
                reward_config = global_config.reward

                # 创建Q值计算器
                q_calculator = QValueCalculator(q_config, reward_config)

                # 计算Q值（传入持久的Dirichlet Bank和reference_path）
                q_values, detailed_info = q_calculator.compute_q_value(
                    current_world_state=world_current,
                    ego_action_trajectory=ego_action_trajectory,
                    trajectory_buffer=trajectory_buffer,
                    grid=grid,
                    bank=bank,  # 传入持久的Bank，确保学习累积
                    rng=rng,
                    reference_path=reference_path
                )

                # 计算平均Q值用于显示
                avg_q_value = np.mean(q_values)

                # 记录Q值分布数据
                if q_tracker is not None:
                    q_distribution = detailed_info['reward_breakdown']['all_q_values']
                    collision_rate = detailed_info['reward_breakdown']['collision_rate']
                    q_tracker.add_episode_data(
                        episode_id=episode_id,
                        q_value=avg_q_value,
                        q_distribution=q_distribution,
                        collision_rate=collision_rate,
                        detailed_info=detailed_info
                    )

                # 生成transition分布和Dirichlet分布可视化（仅在第一个时刻）
                if t == 0 and episode_id % 5 == 0:  # 每5个episode生成一次可视化
                    try:
                        print(f"  生成分布可视化...")
                        # 获取transition分布数据（从Q值计算器内部获取）
                        agent_transition_samples = q_calculator._build_agent_transition_distributions(
                            world_current, ego_action_trajectory, trajectory_buffer, grid, bank, horizon
                        )

                        # 可视化transition分布
                        visualize_transition_distributions(
                            agent_transition_samples=agent_transition_samples,
                            current_world_state=world_current,
                            grid=grid,
                            episode_idx=episode_id,
                            output_dir=output_dir
                        )

                        # 可视化Dirichlet分布
                        visualize_dirichlet_distributions(
                            bank=bank,
                            current_world_state=world_current,
                            grid=grid,
                            episode_idx=episode_id,
                            output_dir=output_dir
                        )
                        print(f"  分布可视化完成")

                    except Exception as e:
                        print(f"  警告: 可视化生成失败: {e}")

            except Exception as e:
                print(f"  警告: Q值计算失败: {e}")
                # 即使Q值计算失败，也要记录失败信息
                if q_tracker is not None:
                    # 创建与成功情况相同长度的零分布
                    config = get_global_config()
                    n_samples = config.sampling.q_value_samples
                    q_tracker.add_episode_data(
                        episode_id=episode_id,
                        q_value=0.0,
                        q_distribution=[0.0] * n_samples,  # 保持一致的长度
                        collision_rate=0.0,
                        detailed_info={'error': str(e)}
                    )
        
        # 为可视化初始化MultiTimestepSpatialDirichletBank
        for i, agent in enumerate(world_current.agents):
            agent_id = i + 1
            try:
                # 计算多时间步可达集用于初始化
                agent_multi_reachable = grid.multi_timestep_successor_cells(
                    agent,
                    horizon=len(ego_action_trajectory),
                    dt=config.time.dt,
                    n_samples=config.sampling.reachable_set_samples
                )
                if agent_multi_reachable and agent_id not in bank.agent_alphas:
                    bank.init_agent(agent_id, agent_multi_reachable)
            except Exception as e:
                print(f"  警告: Agent {agent_id} 初始化失败: {e}")
                continue
        
        # 计算当前"计数图"或概率图用于可视化
        # 这里根据vis_mode选择：
        # - qmax / pmean-*: 显示概率
        # - counts-agent1/2/avg: 显示计数（alpha - alpha_init）归一化到[0,1]
        # 简化的统一可视化模式：显示agent可达集、历史轨迹和自车未来轨迹
        # 1. 构造自车未来动作轨迹
        ego_action_trajectory = []
        for action_t in range(t, min(t + horizon, len(ego_trajectory))):
            ego_action_trajectory.append(tuple(ego_trajectory[action_t]))
        
        # 2. 获取历史轨迹数据
        current_ego_state = (world_current.ego.position_m[0], world_current.ego.position_m[1], world_current.ego.yaw_rad)
        current_agents_states = []
        for agent in world_current.agents:
            current_agents_states.append((agent.position_m[0], agent.position_m[1], 
                                        agent.velocity_mps[0], agent.velocity_mps[1], 
                                        agent.heading_rad, agent.agent_type.value))
        
        # 3. 初始化可视化数据
        c = np.zeros(grid.spec.num_cells)

        # 4. 处理每个Agent（与Q值计算完全对齐的可视化）
        config = get_global_config()
        multi_timestep_reachable = {}  # 收集所有agent的多时间步可达集
        historical_data_sets = {}  # 收集所有agent的历史轨迹数据

        for i, agent in enumerate(world_current.agents):
            agent_id = i + 1

            # 4a. 计算Agent的多时间步可达集（与Q值计算使用相同参数）
            agent_multi_reachable = grid.multi_timestep_successor_cells(
                agent,
                horizon=len(ego_action_trajectory),
                dt=config.time.dt,
                n_samples=config.sampling.reachable_set_samples
            )

            if not agent_multi_reachable:
                continue

            # 保存到可视化数据结构
            multi_timestep_reachable[agent_id] = agent_multi_reachable

            # 4b. 将多时间步可达集添加到可视化（按时间步分权重）
            for timestep, reachable_cells in agent_multi_reachable.items():
                # 时间步越远，权重越低
                timestep_weight = 0.3 / (timestep + 1)  # t=0: 0.3, t=1: 0.15, t=2: 0.1...
                for cell in reachable_cells:
                    if 0 <= cell < grid.spec.num_cells:
                        c[cell] += timestep_weight

            # 4c. 获取Agent的历史轨迹数据（使用配置中的阈值）
            agent_historical_data = trajectory_buffer.get_agent_historical_transitions_strict_matching(
                agent_id=agent_id,
                current_ego_state=current_ego_state,
                current_agents_states=current_agents_states,
                ego_action_trajectory=ego_action_trajectory,
                ego_state_threshold=config.matching.ego_state_threshold,
                agents_state_threshold=config.matching.agents_state_threshold,
                ego_action_threshold=config.matching.ego_action_threshold
            )

            # 保存历史数据到可视化数据结构（不再混入概率图）
            historical_data_sets[agent_id] = agent_historical_data

        # 5. 将自车未来轨迹添加到可视化（高权重）
        for step_idx, ego_pos in enumerate(ego_action_trajectory):
            ego_cell = grid.world_to_cell(ego_pos)
            if 0 <= ego_cell < grid.spec.num_cells:
                c[ego_cell] += 1.0  # 自车轨迹用高权重显示
        
        # 6. 归一化可视化数据
        p_plot = c / (np.max(c) + 1e-12)
        
        # 转换坐标用于可视化（转换到网格坐标系）
        ego_grid = grid.to_grid_frame(world_current.ego.position_m)
        agents_grid = []
        for agent in world_current.agents:
            agent_grid = grid.to_grid_frame(agent.position_m)
            agents_grid.append(np.array(agent_grid))
        
        # 渲染热力图
        frame_path = ep_dir / f"t_{t+1:02d}.png"
        title = f"Episode {episode_id+1}, t={t+1}s: 可达集+历史轨迹+自车轨迹"
        try:
            # 传入多时间步可达集数据和历史数据进行可视化
            grid_heatmap(
                p_plot,
                grid.N,
                np.array(ego_grid),
                agents_grid,
                title,
                str(frame_path),
                grid.size_m,
                multi_timestep_reachable_sets=multi_timestep_reachable,
                historical_data_sets=historical_data_sets,
            )
            frame_paths.append(str(frame_path))
        except Exception as e:
            print(f"  警告: 渲染失败 t={t+1}: {e}")
            continue
        
        # 统计信息
        # 动态获取所有已初始化的agent ID
        initialized_agent_ids = list(bank.agent_alphas.keys()) if hasattr(bank, 'agent_alphas') else []

        # 计算Alpha总和和真实非零单元数（兼容不同的Bank类型）
        if isinstance(bank, (MultiTimestepSpatialDirichletBank, OptimizedMultiTimestepSpatialDirichletBank)):
            alpha_sum = 0.0
            bank_nonzero_cells = 0
            alpha_out_threshold = bank.params.alpha_out

            for aid in initialized_agent_ids:
                if aid in bank.agent_alphas:
                    for timestep, alpha in bank.agent_alphas[aid].items():
                        alpha_sum += alpha.sum()
                        # 统计超过先验值的单元（真正学到了知识）
                        bank_nonzero_cells += int(np.count_nonzero(alpha > alpha_out_threshold))
        else:
            # 对于旧版本的单时间步Bank
            alpha_sum = sum(bank.get_agent_alpha(aid).sum() for aid in initialized_agent_ids)
            # 简化统计
            bank_nonzero_cells = sum(int(np.count_nonzero(bank.get_agent_alpha(aid) > 1e-6))
                                    for aid in initialized_agent_ids)

        stats = {
            't': t + 1,
            'alpha_sum': alpha_sum,
            'qmax_max': float(np.max(p_plot)),
            'nz_cells': bank_nonzero_cells,  # 改为Bank真实非零单元统计
            'reachable_cells': {aid: len(current_reachable[aid]) for aid in current_reachable.keys()}
        }
        episode_stats.append(stats)
    
    # 生成episode GIF
    gif_path = output_dir / f"episode_{episode_id:02d}.gif"
    make_gif(frame_paths, str(gif_path), fps=2)
    
    # 将轨迹数据存储到buffer（按时间步存储）
    timestep_scenarios = []
    
    # 为每个时刻创建轨迹数据
    for t in range(horizon):
        # 获取当前时刻的世界状态
        world_current = scenario_manager.create_world_state_from_trajectories(
            t, ego_trajectory, agent_trajectories, world_init
        )
        
        # 创建当前时刻的场景状态
        current_scenario_state = scenario_manager.create_scenario_state(world_current)
        
        # 创建当前时刻的轨迹数据（只包含下一步）
        timestep_trajectory_data = []
        for i, agent in enumerate(world_current.agents):
            agent_id = i + 1
            if agent_id in agent_trajectory_cells and t < len(agent_trajectory_cells[agent_id]):
                # 只存储从当前时刻开始的剩余轨迹
                remaining_cells = agent_trajectory_cells[agent_id][t:]
                traj_data = AgentTrajectoryData(
                    agent_id=agent_id,
                    agent_type=agent.agent_type.value,
                    init_position=agent.position_m,
                    init_velocity=agent.velocity_mps,
                    init_heading=agent.heading_rad,
                    trajectory_cells=remaining_cells
                )
                timestep_trajectory_data.append(traj_data)
        
        timestep_scenarios.append((current_scenario_state, timestep_trajectory_data))
    
    # 存储按时间步组织的数据，传入自车轨迹
    ego_trajectory_tuples = [tuple(pos) for pos in ego_trajectory]
    trajectory_buffer.store_episode_trajectories_by_timestep(episode_id, timestep_scenarios, ego_trajectory_tuples)
    
    return {
        'episode_id': episode_id,
        'frame_paths': frame_paths,
        'gif_path': str(gif_path),
        'stats': episode_stats,
        'selected_trajectory': selected_trajectory_info,
        'all_trajectories': trajectory_q_values
    }


def main():
    parser = argparse.ArgumentParser(description="基于Lattice规划器的Q值优化实验")
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

    # 配置预设参数（主要配置方式）
    parser.add_argument("--config-preset", choices=["default", "fast", "high-precision", "long-horizon"],
                       default="default", help="预设配置模板")

    # 可选覆盖参数（仅在需要时使用，不设置默认值避免覆盖预设配置）
    parser.add_argument("--dt", type=float, help="覆盖时间步长（秒）")
    parser.add_argument("--horizon", type=int, help="覆盖预测时间步数")
    parser.add_argument("--reachable-samples", type=int, help="覆盖可达集采样数量")
    parser.add_argument("--q-samples", type=int, help="覆盖Q值采样数量")
    # 已简化为单一可视化模式，不再需要vis-mode参数

    args = parser.parse_args()
    
    # 设置全局配置
    from carla_c2osr.config import ConfigPresets, set_global_config
    
    # 首先应用预设配置
    if args.config_preset == "fast":
        config = ConfigPresets.fast_testing()
    elif args.config_preset == "high-precision":
        config = ConfigPresets.high_precision()
    elif args.config_preset == "long-horizon":
        config = ConfigPresets.long_horizon()
    else:
        config = get_global_config()
    
    # 仅在用户明确指定时才覆盖预设配置
    if args.dt is not None:
        config.time.dt = args.dt
    if args.horizon is not None:
        config.time.default_horizon = args.horizon
    if args.reachable_samples is not None:
        config.sampling.reachable_set_samples = args.reachable_samples
    if args.q_samples is not None:
        config.sampling.q_value_samples = args.q_samples
    
    # 这些参数总是从命令行获取（因为它们不是预设配置的一部分）
    config.random_seed = args.seed
    config.visualization.gif_fps = args.gif_fps
    
    set_global_config(config)
    
    print(f"=== Lattice规划器 + Q值优化实验 ===")
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
    
    # 设置随机种子
    np.random.seed(args.seed)
    
    # 初始化组件
    scenario_manager = ScenarioManager()
    # 先创建网格，然后使用正确的边界初始化轨迹生成器
    world_init = scenario_manager.create_scenario()
    ego_start_pos = world_init.ego.position_m

    # 从全局配置读取网格参数
    grid_spec = GridSpec(
        size_m=config.grid.grid_size_m,
        cell_m=config.grid.cell_size_m,
        macro=True
    )
    grid = GridMapper(grid_spec, world_center=ego_start_pos)

    # 使用正确的网格边界初始化轨迹生成器
    grid_half_size = grid.size_m / 2.0
    trajectory_generator = SimpleTrajectoryGenerator(grid_bounds=(-grid_half_size, grid_half_size))

    # 创建场景状态
    scenario_state = scenario_manager.create_scenario_state(world_init)

    # 从全局配置读取Dirichlet参数
    dirichlet_params = DirichletParams(
        alpha_in=config.dirichlet.alpha_in,
        alpha_out=config.dirichlet.alpha_out,
        delta=config.dirichlet.delta,
        cK=config.dirichlet.cK
    )
    # 使用终极优化版本的Bank - 支持直接期望计算，零采样
    bank = OptimizedMultiTimestepSpatialDirichletBank(grid.K, dirichlet_params, horizon=config.time.default_horizon)
    print(f"🚀 使用终极优化版本的Dirichlet Bank - 维度自适应，零采样计算")
    
    # 初始化轨迹缓冲区
    trajectory_buffer = TrajectoryBuffer(horizon=config.time.default_horizon)
    
    # 初始化评估器和分析器
    q_evaluator = QEvaluator()
    buffer_analyzer = BufferAnalyzer(trajectory_buffer)
    q_tracker = QDistributionTracker()  # 创建Q值分布跟踪器

    # 初始化Lattice规划器
    lattice_planner = LatticePlanner.from_config(config)

    # 从场景管理器生成reference path（作为lattice规划和Q值计算的统一中心线）
    reference_path = scenario_manager.generate_reference_path(
        mode=args.ego_mode,
        horizon=config.time.default_horizon,
        ego_start=world_init.ego.position_m
    )
    print(f"\n生成Reference Path: {len(reference_path)} 个waypoints (mode={args.ego_mode})")
    
    # 只对环境智能体初始化Dirichlet分布（不包括自车）
    for i, agent in enumerate(world_init.agents):
        agent_id = i + 1
        # 使用多时间步可达集进行初始化（适配优化版Bank）
        # 使用全局配置中的采样数
        multi_reachable = grid.multi_timestep_successor_cells(
            agent,
            horizon=config.time.default_horizon,
            dt=config.time.dt,
            n_samples=config.sampling.reachable_set_samples
        )
        if not multi_reachable:
            # 如果没有可达集，为每个时间步添加当前位置作为可达
            current_cell = grid.world_to_cell(agent.position_m)
            multi_reachable = {t: [current_cell] for t in range(1, config.time.default_horizon + 1)}

        bank.init_agent(agent_id, multi_reachable)
    
    # 设置输出目录
    output_dir = setup_output_dirs()

    # Checkpoint管理
    checkpoint_manager = CheckpointManager(checkpoint_dir=args.checkpoint_dir)
    start_episode = 0

    # 从checkpoint恢复（如果指定）
    if args.resume_from:
        print(f"\n🔄 从checkpoint恢复训练...")
        checkpoint_data = checkpoint_manager.load_checkpoint(args.resume_from)

        # 恢复training_state
        start_episode = checkpoint_data['training_state']['episode_id'] + 1

        # 恢复TrajectoryBuffer
        trajectory_buffer = TrajectoryBuffer.from_dict(checkpoint_data['trajectory_buffer_data'])

        # 恢复DirichletBank
        bank = OptimizedMultiTimestepSpatialDirichletBank.from_dict(checkpoint_data['dirichlet_bank_data'])

        # 恢复QDistributionTracker
        q_tracker = QDistributionTracker()
        q_tracker_data = checkpoint_data['q_tracker_data']
        q_tracker.episode_data = q_tracker_data.get('episode_data', [])
        q_tracker.q_value_history = q_tracker_data.get('q_value_history', [])
        q_tracker.percentile_q_history = q_tracker_data.get('percentile_q_history', [])
        q_tracker.collision_rate_history = q_tracker_data.get('collision_rate_history', [])
        q_tracker.q_distribution_history = [ep['q_distribution'] for ep in q_tracker.episode_data]
        q_tracker.detailed_info_history = [ep.get('detailed_info', {}) for ep in q_tracker.episode_data]

        # 更新buffer_analyzer
        buffer_analyzer = BufferAnalyzer(trajectory_buffer)

        print(f"✅ 已恢复到Episode {start_episode}，继续训练...")

    # 运行所有episodes
    all_episodes = []
    summary_frames = []

    for e in range(start_episode, args.episodes):
        try:
            rng = np.random.default_rng(args.seed + e)
            
            print(f"\nRunning Episode {e+1}/{args.episodes}")
            episode_result = run_episode(
                e, config.time.default_horizon, reference_path, world_init, grid, bank,
                trajectory_buffer, scenario_state, rng, output_dir, args.sigma,
                lattice_planner=lattice_planner,
                q_evaluator=q_evaluator,
                trajectory_generator=trajectory_generator,
                scenario_manager=scenario_manager,
                buffer_analyzer=buffer_analyzer,
                q_tracker=q_tracker
            )
            all_episodes.append(episode_result)
            
            # 收集最后一帧用于汇总GIF（如果存在）
            if episode_result['frame_paths']:
                summary_frames.append(episode_result['frame_paths'][-1])
            
            # 打印episode完成状态（简化输出）
            if episode_result['stats']:
                final_stats = episode_result['stats'][-1]
                print(f"  完成: alpha_sum={final_stats['alpha_sum']:.1f}, "
                      f"nz_cells={final_stats['nz_cells']}")
            
            # 每10个episode清理一次matplotlib内存
            if (e + 1) % 10 == 0:
                import matplotlib.pyplot as plt
                plt.close('all')
                print(f"  内存清理: Episode {e+1}")

            # 定期保存checkpoint
            if args.checkpoint_interval > 0 and (e + 1) % args.checkpoint_interval == 0:
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
                        trajectory_buffer=trajectory_buffer,
                        dirichlet_bank=bank,
                        q_tracker=q_tracker,
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
    
    # 生成汇总GIF
    if summary_frames:
        summary_gif_path = output_dir / "summary.gif"
        make_gif(summary_frames, str(summary_gif_path), fps=1)
        print(f"\n=== 完成 ===")
        print(f"输出目录: {output_dir}")
    else:
        print(f"\n=== 警告：所有episode都失败，没有生成GIF ===")
        print(f"输出目录: {output_dir}")

    # 打印轨迹选择改进趋势
    selected_trajectories = [ep['selected_trajectory'] for ep in all_episodes if ep['selected_trajectory']]

    if selected_trajectories:
        first_selected = selected_trajectories[0]
        last_selected = selected_trajectories[-1]

        # 获取百分位数配置
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

        # 计算改进
        percentile_q_improvement = last_selected['percentile_q'] - first_selected['percentile_q']
        collision_rate_improvement = first_selected['collision_rate'] - last_selected['collision_rate']

        print(f"  P{int(percentile*100)}_Q改进: {percentile_q_improvement:+.2f}, 碰撞率降低: {collision_rate_improvement:+.3f}")

    # 打印学习趋势
    first_stats = all_episodes[0]['stats'][-1]
    last_stats = all_episodes[-1]['stats'][-1]
    print(f"\nDirichlet学习: Alpha {first_stats['alpha_sum']:.1f} -> {last_stats['alpha_sum']:.1f}, "
          f"非零单元 {first_stats['nz_cells']} -> {last_stats['nz_cells']}")
    
    # 打印轨迹buffer统计
    buffer_stats = buffer_analyzer.get_buffer_stats()
    storage_multiplier = config.matching.trajectory_storage_multiplier
    actual_episodes = buffer_stats['total_episodes'] // storage_multiplier if storage_multiplier > 1 else buffer_stats['total_episodes']
    print(f"\nBuffer: {buffer_stats['total_agents']} agents, "
          f"{buffer_stats['total_episodes']} 条存储记录 (实际{actual_episodes}个episode × {storage_multiplier}倍), "
          f"{buffer_stats['total_agent_episodes']} agent-episodes")

    # 生成Q值分布可视化
    if len(q_tracker.q_value_history) > 0:
        q_evolution_path = output_dir / "q_distribution_evolution.png"
        collision_rate_path = output_dir / "collision_rate_evolution.png"
        q_data_path = output_dir / "q_distribution_data.json"

        try:
            # 生成Q值分布演化图（所有Q值随episode变化）
            q_tracker.plot_q_distribution_evolution(str(q_evolution_path))

            # 生成碰撞率变化图
            q_tracker.plot_collision_rate_evolution(str(collision_rate_path))

            # 保存数据
            q_tracker.save_data(str(q_data_path))

            print(f"\n可视化已生成: {q_evolution_path.name}, {collision_rate_path.name}, {q_data_path.name}")

        except Exception as e:
            print(f"警告: Q值分布可视化失败: {e}")
    else:
        print(f"\n警告: 没有Q值数据")

    # 保存最终checkpoint
    if all_episodes:  # 只在有成功的episodes时保存
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
                trajectory_buffer=trajectory_buffer,
                dirichlet_bank=bank,
                q_tracker=q_tracker,
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
