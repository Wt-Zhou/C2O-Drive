#!/usr/bin/env python3
"""
多次场景执行的概率热力图可视化

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


def create_scenario(grid_size_m: float = 20.0) -> WorldState:
    """创建固定的mock场景。"""
    ego = EgoState(position_m=(0.0, 0.0), velocity_mps=(5.0, 0.0), yaw_rad=0.0)
    
    # 确保智能体位置在网格范围内 [-grid_size_m/2, grid_size_m/2] = [-10, 10]
    agent1 = AgentState(
        agent_id="vehicle-1",
        position_m=(6.0, 12.0),
        velocity_mps=(4.0, 0.0),
        heading_rad=0.0,
        agent_type=AgentType.VEHICLE
    )
    
    agent2 = AgentState(
        agent_id="pedestrian-1", 
        position_m=(14.0, -8.0),
        velocity_mps=(0.8, 0.3),
        heading_rad=0.3,
        agent_type=AgentType.PEDESTRIAN
    )
    
    return WorldState(time_s=0.0, ego=ego, agents=[agent1, agent2])


def create_scenario_state(world: WorldState) -> ScenarioState:
    """从WorldState创建ScenarioState用于buffer索引"""
    agents_states = []
    for agent in world.agents:
        agents_states.append((
            agent.position_m[0], agent.position_m[1],
            agent.velocity_mps[0], agent.velocity_mps[1], 
            agent.heading_rad, agent.agent_type.value
        ))
    
    return ScenarioState(
        ego_position=world.ego.position_m,
        ego_velocity=world.ego.velocity_mps,
        ego_heading=world.ego.yaw_rad,
        agents_states=agents_states
    )


def generate_ego_trajectory(ego_mode: str, horizon: int, ego_speed: float = 5.0) -> List[np.ndarray]:
    """生成自车固定轨迹。"""
    if ego_mode == "straight":
        # 匀速直行
        return [np.array([ego_speed * t, 0.0]) for t in range(1, horizon + 1)]
    elif ego_mode == "fixed-traj":
        # 预设轨迹
        trajectory = []
        for t in range(1, horizon + 1):
            x = ego_speed * t
            y = 0
            trajectory.append(np.array([x, y]))
        return trajectory
    else:
        raise ValueError(f"Unknown ego_mode: {ego_mode}")


def setup_output_dirs(base_dir: str = "outputs/replay_experiment") -> Path:
    """创建输出目录结构。"""
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    return base_path


def generate_agent_trajectory(agent: AgentState, horizon: int, dt: float = 1.0, 
                             grid_bounds: tuple = (-9.0, 9.0)) -> List[np.ndarray]:
    """为智能体生成符合动力学约束的固定轨迹。
    
    Args:
        agent: 智能体状态
        horizon: 轨迹长度（秒）
        dt: 时间步长
        grid_bounds: 网格边界，防止智能体移动到网格外
        
    Returns:
        轨迹列表，每个元素为世界坐标np.ndarray([x, y])
    """
    from carla_c2osr.env.types import AgentDynamicsParams
    import math
    
    trajectory = []
    current_pos = np.array(agent.position_m, dtype=float)
    current_vel = np.array(agent.velocity_mps, dtype=float)
    current_heading = agent.heading_rad
    current_speed = math.sqrt(current_vel[0]**2 + current_vel[1]**2)
    
    # 获取动力学参数
    dynamics = AgentDynamicsParams.for_agent_type(agent.agent_type)
    min_bound, max_bound = grid_bounds
    
    for t in range(horizon):
        if agent.agent_type == AgentType.PEDESTRIAN:
            # 行人：随机游走，偶尔改变方向
            if t % 3 == 0:  # 每3秒改变一次方向
                angle_change = np.random.uniform(-np.pi/3, np.pi/3)  # ±60度
                current_heading += angle_change
            
            # 速度变化：偶尔加速或减速
            if t % 2 == 0:
                speed_change = np.random.uniform(-0.5, 0.5)
                current_speed = np.clip(current_speed + speed_change, 0.1, dynamics.max_speed_mps)
            
            # 计算下一位置
            next_x = current_pos[0] + current_speed * math.cos(current_heading) * dt
            next_y = current_pos[1] + current_speed * math.sin(current_heading) * dt
            
        else:  # 车辆类型
            # 车辆：更平滑的运动，遵循道路行为
            if t % 4 == 0:  # 每4秒轻微调整
                # 轻微转向（阿克曼约束）
                max_steer = math.atan(dynamics.max_yaw_rate_rps * dynamics.wheelbase_m / max(current_speed, 0.1))
                max_steer = min(max_steer, math.pi / 12)  # 最大15度
                steer_change = np.random.uniform(-max_steer, max_steer)
                yaw_rate = current_speed * math.tan(steer_change) / dynamics.wheelbase_m
                current_heading += yaw_rate * dt
                
                # 速度调整
                accel_change = np.random.uniform(-1.0, 2.0)  # 偏向加速
                current_speed = np.clip(current_speed + accel_change * dt, 0.5, dynamics.max_speed_mps)
            
            # 计算下一位置
            next_x = current_pos[0] + current_speed * math.cos(current_heading) * dt
            next_y = current_pos[1] + current_speed * math.sin(current_heading) * dt
        
        # 边界检查：确保智能体不会移动到网格外
        next_x = np.clip(next_x, min_bound, max_bound)
        next_y = np.clip(next_y, min_bound, max_bound)
        
        # 如果碰到边界，改变方向
        if next_x <= min_bound or next_x >= max_bound:
            current_heading = math.pi - current_heading  # 水平反弹
        if next_y <= min_bound or next_y >= max_bound:
            current_heading = -current_heading  # 垂直反弹
        
        current_pos = np.array([next_x, next_y])
        trajectory.append(current_pos.copy())
    
    return trajectory


def run_episode(episode_id: int, horizon: int, ego_trajectory: List[np.ndarray],
               world_init: WorldState, grid: GridMapper, bank: SpatialDirichletBank,
               trajectory_buffer: TrajectoryBuffer, scenario_state: ScenarioState,
               rng: np.random.Generator, output_dir: Path, sigma: float,
               vis_mode: str = "qmax") -> Dict[str, Any]:
    """运行单个episode。"""
    
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
            trajectory = generate_agent_trajectory(agent, horizon, grid_bounds=(-25.0, 25.0))
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
                next_pos = np.clip(next_pos, -9.0, 9.0)
                fallback_trajectory.append(next_pos)
                fallback_cells.append(grid.world_to_cell(tuple(next_pos)))
            agent_trajectories[agent_id] = fallback_trajectory
            agent_trajectory_cells[agent_id] = fallback_cells
    
    # 逐时刻执行和可视化
    frame_paths = []
    episode_stats = []
    
    for t in range(horizon):
        # 更新自车位置
        ego_world_xy = ego_trajectory[t]
        ego = EgoState(position_m=tuple(ego_world_xy), velocity_mps=(5.0, 0.0), yaw_rad=0.0)
        
        # 获取当前时刻环境智能体位置（从预生成的轨迹）
        current_agents = []
        
        for i, agent_init in enumerate(world_init.agents):
            agent_id = i + 1
            agent_world_xy = agent_trajectories[agent_id][t]
            
            # 用轨迹的相邻点估计当前速度与朝向，避免使用初始值导致方向错误
            if t < horizon - 1:
                nxt = agent_trajectories[agent_id][t + 1]
                vel_vec = (nxt - agent_world_xy)
            elif t > 0:
                prv = agent_trajectories[agent_id][t - 1]
                vel_vec = (agent_world_xy - prv)
            else:
                # 单点退化，使用初始速度
                vel_vec = np.array(agent_init.velocity_mps)
            
            vel_tuple = (float(vel_vec[0]), float(vel_vec[1]))
            heading_est = float(np.arctan2(vel_vec[1], vel_vec[0])) if (vel_vec[0]**2 + vel_vec[1]**2) > 1e-9 else float(agent_init.heading_rad)
            
            # 创建当前智能体状态（用估计的速度与朝向）
            current_agent = AgentState(
                agent_id=agent_init.agent_id,
                position_m=tuple(agent_world_xy),
                velocity_mps=vel_tuple,
                heading_rad=heading_est,
                agent_type=agent_init.agent_type
            )
            current_agents.append(current_agent)
        
        # 构建当前世界状态
        world_current = WorldState(time_s=float(t), ego=ego, agents=current_agents)
        
        # 计算每个智能体当前位置的下一时刻可达集
        current_reachable = {}
        for i, agent in enumerate(current_agents):
            agent_id = i + 1
            reachable = grid.successor_cells(agent, n_samples=50)
            current_reachable[agent_id] = reachable
        
        # 使用历史数据和当前观测更新Dirichlet分布
        for i, agent in enumerate(current_agents):
            agent_id = i + 1
            try:
                # 获取历史转移数据
                historical_transitions = trajectory_buffer.get_agent_historical_transitions(
                    scenario_state, agent_id, timestep=t
                )
                
                # 计算当前观测的下一步位置（如果不是最后一步）
                if t < horizon - 1:
                    next_pos = agent_trajectories[agent_id][t + 1]
                    next_cell_id = grid.world_to_cell(tuple(next_pos))
                    
                    # 获取当前位置的可达集
                    reachable = current_reachable[agent_id]
                    
                    if len(reachable) > 0:
                        # 创建基于可达集的软计数
                        w = np.zeros(grid.K, dtype=float)
                        
                        # 将历史转移数据加入软计数
                        for hist_cell in historical_transitions:
                            if hist_cell in reachable:
                                w[hist_cell] += 1.0
                        
                        # 将当前观测加入软计数
                        if next_cell_id in reachable:
                            w[next_cell_id] += 1.0
                        
                        # 归一化到可达集
                        if w.sum() > 0:
                            w = w / w.sum()
                        else:
                            # 如果没有历史数据且当前转移不在可达集内，使用均匀分布
                            for idx in reachable:
                                w[idx] = 1.0 / len(reachable)
                        
                        # 检查Alpha参数防止溢出
                        current_alpha_sum = bank.get_agent_alpha(agent_id).sum()
                        if current_alpha_sum > 10000:
                            lr = 0.1
                            print(f"    警告: Agent {agent_id} alpha过大({current_alpha_sum:.1f})")
                        else:
                            lr = 1.0
                            
                        # 更新Dirichlet分布
                        bank.update_with_softcount(agent_id, w, lr=lr)
                        
                        print(f"    Agent {agent_id}: 历史={len(historical_transitions)}, "
                              f"可达集={len(reachable)}, 下一步={next_cell_id}")
                
            except Exception as e:
                print(f"    错误: Agent {agent_id} 更新失败: {e}")
                continue
        
        # 计算当前“计数图”或概率图用于可视化
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
            c = bank.get_agent_counts(1, subtract_prior=True)
            p_plot = c / (np.max(c) + 1e-12)
        elif vis_mode == "counts-agent2":
            c = bank.get_agent_counts(2, subtract_prior=True)
            p_plot = c / (np.max(c) + 1e-12)
        elif vis_mode == "counts-avg":
            c1 = bank.get_agent_counts(1, subtract_prior=True)
            c2 = bank.get_agent_counts(2, subtract_prior=True)
            c = 0.5 * (c1 + c2)
            p_plot = c / (np.max(c) + 1e-12)
        else:
            p_plot = bank.conservative_qmax_union([1, 2])
        
        # 转换坐标用于可视化（转换到网格坐标系）
        ego_grid = grid.to_grid_frame(ego.position_m)
        agents_grid = []
        for agent in current_agents:
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
    
    # 将轨迹数据存储到buffer
    trajectory_data_list = []
    for i, agent in enumerate(world_init.agents):
        agent_id = i + 1
        if agent_id in agent_trajectory_cells:
            traj_data = AgentTrajectoryData(
                agent_id=agent_id,
                agent_type=agent.agent_type.value,
                init_position=agent.position_m,
                init_velocity=agent.velocity_mps,
                init_heading=agent.heading_rad,
                trajectory_cells=agent_trajectory_cells[agent_id]
            )
            trajectory_data_list.append(traj_data)
    
    trajectory_buffer.store_episode_trajectories(scenario_state, episode_id, trajectory_data_list)
    
    return {
        'episode_id': episode_id,
        'frame_paths': frame_paths,
        'gif_path': str(gif_path),
        'stats': episode_stats
    }


def main():
    parser = argparse.ArgumentParser(description="多次场景执行的概率热力图可视化")
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
            "counts-agent1", "counts-agent2", "counts-avg"
        ],
        default="qmax",
        help=(
            "可视化模式：qmax(保守并集上界)；pmean-* 为后验均值；"
            "counts-* 为计数(α-α_prior)归一化"
        )
    )
    
    args = parser.parse_args()
    
    print(f"=== 多场景贝叶斯学习可视化 ===")
    print(f"Episodes: {args.episodes}, Horizon: {args.horizon}")
    print(f"Ego mode: {args.ego_mode}, Sigma: {args.sigma}")
    print(f"Seed: {args.seed}")
    
    # 设置随机种子
    np.random.seed(args.seed)
    
    # 初始化 - 创建固定的世界网格，以第一帧自车位置为中心
    world_init = create_scenario()
    scenario_state = create_scenario_state(world_init)
    ego_start_pos = world_init.ego.position_m
    
    # 注意：若希望展示 50m×50m，将 size_m 调整为 50.0，并保持可视化extent一致
    grid_spec = GridSpec(size_m=50.0, cell_m=0.5, macro=True)
    grid = GridMapper(grid_spec, world_center=ego_start_pos)
    
    dirichlet_params = DirichletParams(alpha_in=30.0, alpha_out=1e-6, delta=0.05, cK=1.0)
    bank = SpatialDirichletBank(grid.K, dirichlet_params)
    
    # 初始化轨迹缓冲区
    trajectory_buffer = TrajectoryBuffer()
    
    ego_trajectory = generate_ego_trajectory(args.ego_mode, args.horizon)
    
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
                vis_mode=args.vis_mode
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
            print(f"Episode {e+1} 执行失败: {e}")
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
    buffer_stats = trajectory_buffer.get_stats()
    print(f"\n轨迹Buffer统计:")
    print(f"  场景数: {buffer_stats['total_scenarios']}")
    print(f"  Episode数: {buffer_stats['total_episodes']}")
    print(f"  总轨迹数: {buffer_stats['total_trajectories']}")


if __name__ == "__main__":
    main()
