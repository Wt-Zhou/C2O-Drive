#!/usr/bin/env python3
"""
多次场景执行的概率热力图可视化（重构版）

演示固定场景下的贝叶斯学习过程：
- 自车动作固定，环境智能体按采样的转移分布滚动
- 每秒更新Dirichlet分布并渲染概率热力图
- 生成逐帧PNG和动画GIF
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

from c2o_drive.environments.carla.types import AgentState, EgoState, WorldState, AgentType
from c2o_drive.algorithms.c2osr.grid import GridSpec, GridMapper
from c2o_drive.algorithms.c2osr.spatial_dirichlet import DirichletParams, SpatialDirichletBank, MultiTimestepSpatialDirichletBank, OptimizedMultiTimestepSpatialDirichletBank
from c2o_drive.algorithms.c2osr.trajectory_buffer import TrajectoryBuffer, AgentTrajectoryData, ScenarioState
from c2o_drive.algorithms.c2osr.risk import compose_union_singlelayer
from c2o_drive.visualization.vis import grid_heatmap, make_gif
from c2o_drive.visualization.transition_visualizer import visualize_transition_distributions, visualize_dirichlet_distributions

# 导入重构后的模块
from c2o_drive.algorithms.c2osr.rewards import RewardCalculator, DistanceBasedCollisionDetector
from c2o_drive.algorithms.c2osr.q_evaluator import QEvaluator
from c2o_drive.evaluation.buffer_analyzer import BufferAnalyzer
from c2o_drive.algorithms.c2osr.q_value import QValueCalculator, QValueConfig
from c2o_drive.evaluation.q_distribution_tracker import QDistributionTracker
from c2o_drive.utils.simple_trajectory_generator import SimpleTrajectoryGenerator
from c2o_drive.environments.virtual.scenario_manager import ScenarioManager
from c2o_drive.config import get_global_config, update_dt, update_horizon, get_dt, get_horizon


def setup_output_dirs(base_dir: str = "outputs/replay_experiment") -> Path:
    """创建输出目录结构。"""
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    return base_path


def run_episode(episode_id: int, 
                horizon: int, 
                ego_trajectory: List[np.ndarray],
                world_init: WorldState, 
                grid: GridMapper, 
                bank: SpatialDirichletBank,
                trajectory_buffer: TrajectoryBuffer, 
                scenario_state: ScenarioState,
                rng: np.random.Generator, 
                output_dir: Path, 
                sigma: float,
                q_evaluator: QEvaluator = None,
                trajectory_generator: TrajectoryGenerator = None,
                scenario_manager: ScenarioManager = None,
                buffer_analyzer: BufferAnalyzer = None,
                q_tracker: QDistributionTracker = None) -> Dict[str, Any]:
    """运行单个episode。"""
    
    # 初始化组件
    if q_evaluator is None:
        q_evaluator = QEvaluator()
    if scenario_manager is None:
        scenario_manager = ScenarioManager()
    if buffer_analyzer is None:
        buffer_analyzer = BufferAnalyzer(trajectory_buffer)
    
    # 创建episode输出目录
    ep_dir = output_dir / f"ep_{episode_id:02d}"
    ep_dir.mkdir(exist_ok=True)
    
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
            
            print(f"  Agent {agent_id} ({agent.agent_type.value}) 轨迹生成: {len(trajectory)} 步")
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
            
            print(f"    开始Q值计算: 自车动作序列长度={len(ego_action_trajectory)}")
            
            try:
                # 创建Q值配置
                q_config = QValueConfig.from_global_config()

                # 直接使用全局配置中的奖励配置
                global_config = get_global_config()
                reward_config = global_config.reward

                # 创建Q值计算器
                q_calculator = QValueCalculator(q_config, reward_config)
                
                # 计算Q值（传入持久的Dirichlet Bank）
                q_values, detailed_info = q_calculator.compute_q_value(
                    current_world_state=world_current,
                    ego_action_trajectory=ego_action_trajectory,
                    trajectory_buffer=trajectory_buffer,
                    grid=grid,
                    bank=bank,  # 传入持久的Bank，确保学习累积
                    rng=rng
                )
                
                # 计算平均Q值用于显示
                avg_q_value = np.mean(q_values)
                print(f"    Q值计算结果: {avg_q_value:.2f}")
                print(f"    所有Q值: {[f'{q:.2f}' for q in q_values]}")
                print(f"    碰撞率: {detailed_info['reward_breakdown']['collision_rate']:.3f}")
                print(f"    Q值标准差: {detailed_info['reward_breakdown']['q_value_std']:.2f}")
                
                # 动态显示所有agent的信息
                for agent_id, agent_info in detailed_info.get('agent_info', {}).items():
                    reachable_total = agent_info.get('reachable_cells_total', 0)
                    historical_total = agent_info.get('historical_data_count', 0)
                    print(f"    Agent {agent_id}: 可达集(总计)={reachable_total}, 历史数据={historical_total}")
                
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
                        print(f"  🎨 生成transition分布和Dirichlet分布可视化...")
                        
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
                        
                    except Exception as e:
                        print(f"  ⚠️ 可视化生成失败: {e}")
                        import traceback
                        traceback.print_exc()
                
                # 打印每个智能体的信息
                for agent_id, info in detailed_info['agent_info'].items():
                    # reachable_cells_per_timestep的values已经是长度值（整数），不需要再调用len()
                    total_reachable = sum(info['reachable_cells_per_timestep'].values())
                    print(f"    Agent {agent_id}: 可达集(总计)={total_reachable}, "
                          f"历史数据={info['total_historical_data']}")
                    
            except Exception as e:
                print(f"    Q值计算失败: {e}")
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
                    total_reachable = sum(len(cells) for cells in agent_multi_reachable.values())
                    print(f"    Agent {agent_id}: 多时间步可达集={total_reachable}")
            except Exception as e:
                print(f"    错误: Agent {agent_id} 初始化失败: {e}")
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
        
        print(f"    === t={t+1}: Agent可达集 + 历史轨迹 + 自车轨迹 ===")
        
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
                print(f"    Agent {agent_id} ({agent.agent_type.value}): 无法计算可达集")
                continue
            
            # 保存到可视化数据结构
            multi_timestep_reachable[agent_id] = agent_multi_reachable
                
            total_reachable = sum(len(cells) for cells in agent_multi_reachable.values())
            print(f"    Agent {agent_id} ({agent.agent_type.value}): 未来{len(ego_action_trajectory)}步可达集={total_reachable}个单元")
            
            # 4b. 将多时间步可达集添加到可视化（按时间步分权重）
            for timestep, reachable_cells in agent_multi_reachable.items():
                # 时间步越远，权重越低
                timestep_weight = 0.3 / (timestep + 1)  # t=0: 0.3, t=1: 0.15, t=2: 0.1...
                for cell in reachable_cells:
                    if 0 <= cell < grid.spec.num_cells:
                        c[cell] += timestep_weight
                print(f"      时间步{timestep}: {len(reachable_cells)}个可达单元 (权重: {timestep_weight:.2f})")
            
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
            
            # 4d. 统计历史数据（与Q值计算相同的逻辑）
            total_historical = sum(len(cells) for cells in agent_historical_data.values())
            print(f"      历史轨迹数据: {total_historical}个位置")
            
            # 4e. 统计有效历史数据（与Q值计算逻辑一致）
            for timestep, agent_cells in agent_historical_data.items():
                if len(agent_cells) > 0 and timestep in agent_multi_reachable:
                    # 只统计在可达集内的历史数据（与Q值计算逻辑一致）
                    timestep_reachable = agent_multi_reachable[timestep]
                    valid_historical_cells = [cell for cell in agent_cells if cell in timestep_reachable]
                    
                    if valid_historical_cells:
                        print(f"        时间步{timestep}: {len(agent_cells)}个历史位置 -> {len(valid_historical_cells)}个有效位置")
                    else:
                        print(f"        时间步{timestep}: {len(agent_cells)}个历史位置 -> 0个有效位置（不在可达集内）")
        
        # 5. 将自车未来轨迹添加到可视化（高权重）
        print(f"    自车未来轨迹: {len(ego_action_trajectory)}步")
        for step_idx, ego_pos in enumerate(ego_action_trajectory):
            ego_cell = grid.world_to_cell(ego_pos)
            if 0 <= ego_cell < grid.spec.num_cells:
                c[ego_cell] += 1.0  # 自车轨迹用高权重显示
            print(f"      步骤{step_idx}: 位置{ego_pos} -> cell {ego_cell}")
        
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
            print(f"    错误: 热力图渲染失败 t={t+1}: {e}")
            continue
        
        # 统计信息
        # 动态获取所有已初始化的agent ID
        initialized_agent_ids = list(bank.agent_alphas.keys()) if hasattr(bank, 'agent_alphas') else []
        
        # 计算Alpha总和（兼容不同的Bank类型）
        if isinstance(bank, (MultiTimestepSpatialDirichletBank, OptimizedMultiTimestepSpatialDirichletBank)):
            alpha_sum = 0.0
            for aid in initialized_agent_ids:
                if aid in bank.agent_alphas:
                    alpha_sum += sum(alpha.sum() for alpha in bank.agent_alphas[aid].values())
        else:
            # 对于旧版本的单时间步Bank
            alpha_sum = sum(bank.get_agent_alpha(aid).sum() for aid in initialized_agent_ids)
        
        stats = {
            't': t + 1,
            'alpha_sum': alpha_sum,
            'qmax_max': float(np.max(p_plot)),
            'nz_cells': int(np.count_nonzero(p_plot > 1e-6)),
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
        'stats': episode_stats
    }


def main():
    parser = argparse.ArgumentParser(description="多次场景执行的概率热力图可视化（重构版）")
    # 基本运行参数
    parser.add_argument("--episodes", type=int, default=20, help="执行episode数")
    parser.add_argument("--seed", type=int, default=2025, help="随机种子")
    parser.add_argument("--gif-fps", type=int, default=2, help="GIF帧率")
    parser.add_argument("--ego-mode", choices=["straight", "fixed-traj"], 
                       default="straight", help="自车运动模式")
    parser.add_argument("--sigma", type=float, default=0.5, help="软计数核宽度")
    
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
    from c2o_drive.config import ConfigPresets, set_global_config
    
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
    
    print(f"=== 多场景贝叶斯学习可视化（重构版）===")
    print(f"Episodes: {args.episodes}, Horizon: {config.time.default_horizon}")
    print(f"Ego mode: {args.ego_mode}, Sigma: {args.sigma}")
    print(f"Seed: {args.seed}")
    print(f"配置预设: {args.config_preset}")
    print(f"时间步长: {config.time.dt}s, 预测时间: {config.time.horizon_seconds:.1f}s")
    print(f"可达集采样: {config.sampling.reachable_set_samples}, Q值采样: {config.sampling.q_value_samples}")
    
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
    
    # 生成自车轨迹
    ego_trajectory = trajectory_generator.generate_ego_trajectory(args.ego_mode, config.time.default_horizon)
    
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
        total_cells = sum(len(cells) for cells in multi_reachable.values())
        print(f"Agent {agent_id} ({agent.agent_type.value}): 多时间步可达集 {total_cells} cells总计")
    
    # 设置输出目录
    output_dir = setup_output_dirs()
    
    # 运行所有episodes
    all_episodes = []
    summary_frames = []
    
    for e in range(args.episodes):
        try:
            rng = np.random.default_rng(args.seed + e)
            
            print(f"\nRunning Episode {e+1}/{args.episodes}")
            episode_result = run_episode(
                e, config.time.default_horizon, ego_trajectory, world_init, grid, bank,
                trajectory_buffer, scenario_state, rng, output_dir, args.sigma,
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
            
            # 打印episode统计
            if episode_result['stats']:
                final_stats = episode_result['stats'][-1]
                # 动态获取所有agent的可达集信息
                if 'reachable_cells' in final_stats:
                    reachable_info = ", ".join([f"Agent{aid}={final_stats['reachable_cells'][aid]}" 
                                               for aid in final_stats['reachable_cells']])
                    print(f"  Final: alpha_sum={final_stats['alpha_sum']:.1f}, "
                          f"qmax_max={final_stats['qmax_max']:.4f}, "
                          f"nz_cells={final_stats['nz_cells']}, "
                          f"可达集: {reachable_info}")
                else:
                    print(f"  Final: alpha_sum={final_stats['alpha_sum']:.1f}, "
                          f"qmax_max={final_stats['qmax_max']:.4f}, "
                          f"nz_cells={final_stats['nz_cells']}")
            
            # 每10个episode清理一次matplotlib内存
            if (e + 1) % 10 == 0:
                import matplotlib.pyplot as plt
                plt.close('all')
                print(f"  内存清理: Episode {e+1}")
                
        except Exception as ex:
            print(f"Episode {e+1} 执行失败: {ex}")
            print("继续执行下一个episode...")
            continue
    
    # 生成汇总GIF
    summary_gif_path = output_dir / "summary.gif"
    make_gif(summary_frames, str(summary_gif_path), fps=1)
    
    print(f"\n=== 完成 ===")
    print(f"输出目录: {output_dir}")
    print(f"Episode GIFs: episode_00.gif - episode_{args.episodes-1:02d}.gif")
    print(f"汇总GIF: summary.gif")
    
    # 打印学习趋势
    print(f"\n学习趋势:")
    first_stats = all_episodes[0]['stats'][-1]
    last_stats = all_episodes[-1]['stats'][-1]
    print(f"  Alpha总量: {first_stats['alpha_sum']:.1f} -> {last_stats['alpha_sum']:.1f}")
    print(f"  Q_max峰值: {first_stats['qmax_max']:.4f} -> {last_stats['qmax_max']:.4f}")
    print(f"  非零单元: {first_stats['nz_cells']} -> {last_stats['nz_cells']}")
    
    # 打印轨迹buffer统计
    buffer_stats = buffer_analyzer.get_buffer_stats()
    print(f"\n轨迹Buffer统计:")
    print(f"  Agent数: {buffer_stats['total_agents']}")
    print(f"  Episode数: {buffer_stats['total_episodes']}")
    print(f"  Agent Episodes: {buffer_stats['total_agent_episodes']}")
    print(f"  索引统计 - Agent数量索引: {buffer_stats['agent_count_index_size']}")
    print(f"  索引统计 - 空间索引: {buffer_stats['spatial_index_size']}")
    print(f"  索引统计 - 动作索引: {buffer_stats['action_index_size']}")
    
    # 生成Q值分布可视化
    if len(q_tracker.q_value_history) > 0:
        print(f"\n=== Q值分布分析 ===")
        
        # 打印统计摘要
        q_tracker.print_summary()
        
        # 生成可视化图表
        q_evolution_path = output_dir / "q_distribution_evolution.png"
        q_boxplot_path = output_dir / "q_distribution_boxplot.png"
        q_data_path = output_dir / "q_distribution_data.json"
        
        try:
            # 生成Q值分布演化图（所有Q值随episode变化）
            q_tracker.plot_q_distribution_evolution(str(q_evolution_path))
            
            # 生成碰撞率变化图
            collision_rate_path = output_dir / "collision_rate_evolution.png"
            q_tracker.plot_collision_rate_evolution(str(collision_rate_path))
            
            # 保存数据
            q_tracker.save_data(str(q_data_path))
            
            print(f"\nQ值分布可视化已生成:")
            print(f"  Q值演化图: {q_evolution_path.name}")
            print(f"  碰撞率变化图: {collision_rate_path.name}")
            print(f"  数据文件: {q_data_path.name}")
            
        except Exception as e:
            print(f"Q值分布可视化生成失败: {e}")
    else:
        print(f"\n警告: 没有有效的Q值数据进行可视化")


if __name__ == "__main__":
    main()
