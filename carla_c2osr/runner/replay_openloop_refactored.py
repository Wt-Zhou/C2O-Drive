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

from carla_c2osr.env.types import AgentState, EgoState, WorldState, AgentType
from carla_c2osr.agents.c2osr.grid import GridSpec, GridMapper
from carla_c2osr.agents.c2osr.spatial_dirichlet import DirichletParams, SpatialDirichletBank
from carla_c2osr.agents.c2osr.trajectory_buffer import TrajectoryBuffer, AgentTrajectoryData, ScenarioState
from carla_c2osr.agents.c2osr.risk import compose_union_singlelayer
from carla_c2osr.evaluation.vis import grid_heatmap, make_gif

# 导入重构后的模块
from carla_c2osr.evaluation.rewards import RewardCalculator, CollisionDetector
from carla_c2osr.evaluation.q_evaluator import QEvaluator
from carla_c2osr.evaluation.buffer_analyzer import BufferAnalyzer
from carla_c2osr.utils.simple_trajectory_generator import SimpleTrajectoryGenerator
from carla_c2osr.utils.scenario_manager import ScenarioManager


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
                vis_mode: str = "qmax",
                q_evaluator: QEvaluator = None,
                trajectory_generator: TrajectoryGenerator = None,
                scenario_manager: ScenarioManager = None,
                buffer_analyzer: BufferAnalyzer = None) -> Dict[str, Any]:
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
    
    # 轨迹生成随机源：若希望每个episode不同，这里使用episode_id派生的种子；
    # 若希望每个episode完全一致，可以固定一个常数种子。
    rng = np.random.default_rng(42)  # 固定轨迹：把42改成固定常量；如需变化改为 (base_seed + episode_id)
    
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
        
        # 计算每个智能体当前位置的下一时刻可达集
        current_reachable = {}
        for i, agent in enumerate(world_current.agents):
            agent_id = i + 1
            reachable = grid.successor_cells(agent, n_samples=50)
            current_reachable[agent_id] = reachable
        
        # 每个时刻重新初始化Dirichlet Bank，基于当前可达集和历史数据
        for i, agent in enumerate(world_current.agents):
            agent_id = i + 1
            try:
                # 获取当前位置的可达集
                reachable = current_reachable[agent_id]
                
                if len(reachable) > 0:
                    # 重新初始化该智能体的Dirichlet分布
                    bank.init_agent(agent_id, reachable)
                    
                    # 获取历史转移数据（基于当前状态，timestep=0表示下一秒）
                    # 使用模糊匹配获取相似状态下的历史数据
                    historical_transitions = trajectory_buffer.get_agent_fuzzy_historical_transitions(
                        current_scenario_state, agent_id, timestep=0,
                        position_threshold=10.0,  # 位置阈值3米
                        velocity_threshold=10.0,  # 速度阈值2m/s
                        heading_threshold=3.14    # 朝向阈值0.8弧度（约45度）
                    )
                    
                    # 如果有历史数据，计算软计数并更新alpha
                    if len(historical_transitions) > 0:
                        # 创建基于可达集的软计数
                        w = np.zeros(grid.K, dtype=float)
                        # 将历史转移数据加入软计数
                        for hist_cell in historical_transitions:
                            if hist_cell in reachable:
                                w[hist_cell] += 1.0
                        
                        # 归一化到可达集
                        if w.sum() > 0:
                            w = w / w.sum()
                            # 直接设置alpha（基于历史数据）
                            bank.agent_alphas[agent_id] = bank.params.alpha_in * w
                    
                    print(f"    Agent {agent_id}: 历史={len(historical_transitions)}, "
                          f"可达集={len(reachable)}")
                    
                    # Q值评估：从Dirichlet分布采样并计算reward
                    if t < horizon - 1:  # 不是最后一步
                        # 获取自车下一状态
                        ego_next_world_xy = ego_trajectory[t + 1]
                        ego_next = EgoState(position_m=tuple(ego_next_world_xy), velocity_mps=(5.0, 0.0), yaw_rad=0.0)
                        
                        # 计算自车轨迹的格子ID
                        ego_trajectory_cells = []
                        for future_t in range(t + 1, min(t + 3, horizon)):  # 看未来2步
                            if future_t < len(ego_trajectory):
                                ego_future_xy = ego_trajectory[future_t]
                                ego_cell = grid.world_to_cell(tuple(ego_future_xy))
                                ego_trajectory_cells.append(ego_cell)
                        
                        print(f"    Agent {agent_id} Q值评估:")
                        rewards = q_evaluator.evaluate_q_values(
                            bank=bank,
                            agent_id=agent_id,
                            reachable=reachable,
                            ego_state=world_current.ego,
                            ego_next_state=ego_next,
                            agent_state=agent,
                            grid=grid,
                            ego_trajectory_cells=ego_trajectory_cells,
                            n_samples=5,  # 采样5次
                            rng=rng,
                            verbose=True
                        )
                        
                        # 计算平均reward
                        avg_reward = np.mean(rewards)
                        print(f"    Agent {agent_id} 平均reward: {avg_reward:.2f}")
                
            except Exception as e:
                print(f"    错误: Agent {agent_id} 初始化失败: {e}")
                continue
        
        # 计算当前"计数图"或概率图用于可视化
        # 这里根据vis_mode选择：
        # - qmax / pmean-*: 显示概率
        # - counts-agent1/2/avg: 显示计数（alpha - alpha_init）归一化到[0,1]
        if vis_mode == "qmax":
            p_plot = bank.conservative_qmax_union([1, 2])
        elif vis_mode == "pmean-agent1":
            p_plot = bank.posterior_mean(1)
        elif vis_mode == "pmean-agent2":
            p_plot = bank.posterior_mean(2)
        elif vis_mode == "pmean-avg":
            p_plot = 0.5 * (bank.posterior_mean(1) + bank.posterior_mean(2))
        elif vis_mode == "counts-agent1":
            # 使用Trajectory Buffer的计数（基于当前状态）
            buffer_counts = buffer_analyzer.calculate_buffer_counts(current_scenario_state, [1], 0, grid)
            c = buffer_counts[1]
            p_plot = c / (np.max(c) + 1e-12)
        elif vis_mode == "counts-agent2":
            # 使用Trajectory Buffer的计数（基于当前状态）
            buffer_counts = buffer_analyzer.calculate_buffer_counts(current_scenario_state, [2], 0, grid)
            c = buffer_counts[2]
            p_plot = c / (np.max(c) + 1e-12)
        elif vis_mode == "counts-avg":
            # 使用Trajectory Buffer的计数（基于当前状态）
            buffer_counts = buffer_analyzer.calculate_buffer_counts(current_scenario_state, [1, 2], 0, grid)
            c1 = buffer_counts[1]
            c2 = buffer_counts[2]
            # 叠加两个agent的计数，保持原始值
            c = c1 + c2
            p_plot = c / (np.max(c) + 1e-12)
        elif vis_mode == "current-counts":
            # 显示当前时刻状态下的历史transition计数
            buffer_counts = buffer_analyzer.calculate_buffer_counts(current_scenario_state, [1, 2], 0, grid)
            c1 = buffer_counts[1]
            c2 = buffer_counts[2]
            c = c1 + c2
            p_plot = c / (np.max(c) + 1e-12)
        elif vis_mode == "fuzzy-counts":
            # 显示模糊匹配的历史transition计数
            buffer_counts = buffer_analyzer.calculate_fuzzy_buffer_counts(current_scenario_state, [1, 2], 0, grid, 10, 10, 3.14)
            c1 = buffer_counts[1]
            c2 = buffer_counts[2]
            c = c1 + c2
            p_plot = c / (np.max(c) + 1e-12)
        else:
            p_plot = bank.conservative_qmax_union([1, 2])
        
        # 转换坐标用于可视化（转换到网格坐标系）
        ego_grid = grid.to_grid_frame(world_current.ego.position_m)
        agents_grid = []
        for agent in world_current.agents:
            agent_grid = grid.to_grid_frame(agent.position_m)
            agents_grid.append(np.array(agent_grid))
        
        # 渲染热力图
        frame_path = ep_dir / f"t_{t+1:02d}.png"
        title = f"Episode {episode_id+1}, t={t+1}s, vis={vis_mode}"
        try:
            # 传入每个智能体的可达集以叠加轮廓
            reachable_sets = [current_reachable.get(1, []), current_reachable.get(2, [])]
            grid_heatmap(
                p_plot,
                grid.N,
                np.array(ego_grid),
                agents_grid,
                title,
                str(frame_path),
                grid.size_m,
                reachable_sets=reachable_sets,
                reachable_colors=["cyan", "magenta"],
            )
            frame_paths.append(str(frame_path))
        except Exception as e:
            print(f"    错误: 热力图渲染失败 t={t+1}: {e}")
            continue
        
        # 统计信息
        stats = {
            't': t + 1,
            'alpha_sum': sum(bank.get_agent_alpha(aid).sum() for aid in [1, 2]),
            'qmax_max': float(np.max(p_plot)),
            'nz_cells': int(np.count_nonzero(p_plot > 1e-6)),
            'reachable_cells': {aid: len(current_reachable[aid]) for aid in [1, 2]}
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
    
    # 存储按时间步组织的数据
    trajectory_buffer.store_episode_trajectories_by_timestep(episode_id, timestep_scenarios)
    
    return {
        'episode_id': episode_id,
        'frame_paths': frame_paths,
        'gif_path': str(gif_path),
        'stats': episode_stats
    }


def main():
    parser = argparse.ArgumentParser(description="多次场景执行的概率热力图可视化（重构版）")
    parser.add_argument("--episodes", type=int, default=10, help="执行episode数")
    parser.add_argument("--horizon", type=int, default=8, help="每个episode时长")
    parser.add_argument("--seed", type=int, default=2025, help="随机种子")
    parser.add_argument("--gif-fps", type=int, default=2, help="GIF帧率")
    parser.add_argument("--ego-mode", choices=["straight", "fixed-traj"], 
                       default="straight", help="自车运动模式")
    parser.add_argument("--sigma", type=float, default=0.5, help="软计数核宽度")
    parser.add_argument(
        "--vis-mode",
        choices=[
            "qmax",
            "pmean-agent1", "pmean-agent2", "pmean-avg",
            "counts-agent1", "counts-agent2", "counts-avg",
            "current-counts",
            "fuzzy-counts"
        ],
        default="qmax",
        help=(
            "可视化模式：qmax(保守并集上界)；pmean-* 为后验均值；"
            "counts-* 为计数(α-α_prior)归一化；"
            "current-counts为当前状态下的历史transition计数；"
            "fuzzy-counts为模糊匹配的历史transition计数"
        )
    )
    
    args = parser.parse_args()
    
    print(f"=== 多场景贝叶斯学习可视化（重构版）===")
    print(f"Episodes: {args.episodes}, Horizon: {args.horizon}")
    print(f"Ego mode: {args.ego_mode}, Sigma: {args.sigma}")
    print(f"Seed: {args.seed}")
    
    # 设置随机种子
    np.random.seed(args.seed)
    
    # 初始化组件
    scenario_manager = ScenarioManager()
    # 先创建网格，然后使用正确的边界初始化轨迹生成器
    world_init = scenario_manager.create_scenario()
    ego_start_pos = world_init.ego.position_m
    grid_spec = GridSpec(size_m=100.0, cell_m=0.5, macro=True)
    grid = GridMapper(grid_spec, world_center=ego_start_pos)
    
    # 使用正确的网格边界初始化轨迹生成器
    grid_half_size = grid.size_m / 2.0
    trajectory_generator = SimpleTrajectoryGenerator(grid_bounds=(-grid_half_size, grid_half_size))
    
    # 创建场景状态
    scenario_state = scenario_manager.create_scenario_state(world_init)
    
    dirichlet_params = DirichletParams(alpha_in=30.0, alpha_out=1e-6, delta=0.05, cK=1.0)
    bank = SpatialDirichletBank(grid.K, dirichlet_params)
    
    # 初始化轨迹缓冲区
    trajectory_buffer = TrajectoryBuffer()
    
    # 初始化评估器和分析器
    q_evaluator = QEvaluator()
    buffer_analyzer = BufferAnalyzer(trajectory_buffer)
    
    # 生成自车轨迹
    ego_trajectory = trajectory_generator.generate_ego_trajectory(args.ego_mode, args.horizon)
    
    # 只对环境智能体初始化Dirichlet分布（不包括自车）
    for i, agent in enumerate(world_init.agents):
        agent_id = i + 1
        # 使用初始位置计算下一步可达集进行初始化
        reachable = grid.successor_cells(agent, n_samples=100)
        if len(reachable) == 0:
            # 如果没有可达集，添加当前位置作为可达
            current_cell = grid.world_to_cell(agent.position_m)
            reachable = [current_cell]
        bank.init_agent(agent_id, reachable)
        print(f"Agent {agent_id} ({agent.agent_type.value}): 初始可达集 {len(reachable)} cells")
    
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
                e, args.horizon, ego_trajectory, world_init, grid, bank,
                trajectory_buffer, scenario_state, rng, output_dir, args.sigma,
                vis_mode=args.vis_mode,
                q_evaluator=q_evaluator,
                trajectory_generator=trajectory_generator,
                scenario_manager=scenario_manager,
                buffer_analyzer=buffer_analyzer
            )
            all_episodes.append(episode_result)
            
            # 收集最后一帧用于汇总GIF（如果存在）
            if episode_result['frame_paths']:
                summary_frames.append(episode_result['frame_paths'][-1])
            
            # 打印episode统计
            if episode_result['stats']:
                final_stats = episode_result['stats'][-1]
                reachable_info = ", ".join([f"Agent{aid}={final_stats['reachable_cells'][aid]}" 
                                           for aid in final_stats['reachable_cells']])
                print(f"  Final: alpha_sum={final_stats['alpha_sum']:.1f}, "
                      f"qmax_max={final_stats['qmax_max']:.4f}, "
                      f"nz_cells={final_stats['nz_cells']}, "
                      f"可达集: {reachable_info}")
            
            # 每10个episode清理一次matplotlib内存
            if (e + 1) % 10 == 0:
                import matplotlib.pyplot as plt
                plt.close('all')
                print(f"  内存清理: Episode {e+1}")
                
        except Exception as e:
            print(f"Episode {e+1} 执行失败: {str(e)}")
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
    print(f"  场景数: {buffer_stats['total_scenarios']}")
    print(f"  Episode数: {buffer_stats['total_episodes']}")
    print(f"  总轨迹数: {buffer_stats['total_trajectories']}")


if __name__ == "__main__":
    main()
