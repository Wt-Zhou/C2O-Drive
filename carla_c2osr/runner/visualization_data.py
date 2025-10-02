"""
可视化数据准备模块

从 run_episode 中提取的可视化数据准备逻辑,职责单一,易于测试。
"""

from typing import Dict, List, Tuple
import numpy as np

from carla_c2osr.env.types import WorldState
from carla_c2osr.config import get_global_config
from carla_c2osr.runner.episode_context import EpisodeContext


def prepare_visualization_data(
    ctx: EpisodeContext,
    world_state: WorldState,
    timestep: int
) -> Dict:
    """准备可视化所需的所有数据(不渲染)

    Args:
        ctx: Episode运行上下文
        world_state: 当前世界状态
        timestep: 当前时间步

    Returns:
        可视化数据字典:
        {
            'p_plot': np.ndarray,  # 归一化的热力图数据
            'ego_grid': np.ndarray,  # 自车网格坐标
            'agents_grid': List[np.ndarray],  # 环境智能体网格坐标
            'multi_timestep_reachable': Dict[int, Dict[int, List[int]]],  # 多时间步可达集
            'historical_data_sets': Dict[int, Dict[int, List[int]]],  # 历史轨迹数据
            'title': str,  # 可视化标题
            'stats': Dict  # 统计信息
        }
    """
    # 1. 构造自车未来动作轨迹
    ego_action_trajectory = _extract_ego_action_trajectory(ctx, timestep)

    # 2. 获取当前状态
    current_ego_state = _extract_ego_state(world_state)
    current_agents_states = _extract_agents_states(world_state)

    # 3. 初始化可视化数据
    c = np.zeros(ctx.grid.spec.num_cells)

    # 获取verbose级别
    from carla_c2osr.config import get_global_config
    verbose = get_global_config().visualization.verbose_level

    if verbose >= 2:
        print(f"    === t={timestep+1}: Agent可达集 + 历史轨迹 + 自车轨迹 ===")

    # 4. 处理每个Agent
    config = get_global_config()
    multi_timestep_reachable = {}
    historical_data_sets = {}

    for i, agent in enumerate(world_state.agents):
        agent_id = i + 1

        # 4a. 计算Agent的多时间步可达集
        agent_multi_reachable = ctx.grid.multi_timestep_successor_cells(
            agent,
            horizon=len(ego_action_trajectory),
            dt=config.time.dt,
            n_samples=config.sampling.reachable_set_samples
        )

        if not agent_multi_reachable:
            if verbose >= 2:
                print(f"    Agent {agent_id} ({agent.agent_type.value}): 无法计算可达集")
            continue

        # 保存可达集
        multi_timestep_reachable[agent_id] = agent_multi_reachable

        total_reachable = sum(len(cells) for cells in agent_multi_reachable.values())
        if verbose >= 2:
            print(f"    Agent {agent_id} ({agent.agent_type.value}): 未来{len(ego_action_trajectory)}步可达集={total_reachable}个单元")

        # 4b. 将多时间步可达集添加到热力图数据
        for ts, reachable_cells in agent_multi_reachable.items():
            # 时间步越远,权重越低
            timestep_weight = 0.3 / (ts + 1)
            for cell in reachable_cells:
                if 0 <= cell < ctx.grid.spec.num_cells:
                    c[cell] += timestep_weight
            if verbose >= 2:
                print(f"      时间步{ts}: {len(reachable_cells)}个可达单元 (权重: {timestep_weight:.2f})")

        # 4c. 获取Agent的历史轨迹数据
        agent_historical_data = ctx.trajectory_buffer.get_agent_historical_transitions_strict_matching(
            agent_id=agent_id,
            current_ego_state=current_ego_state,
            current_agents_states=current_agents_states,
            ego_action_trajectory=ego_action_trajectory,
            ego_state_threshold=config.matching.ego_state_threshold,
            agents_state_threshold=config.matching.agents_state_threshold,
            ego_action_threshold=config.matching.ego_action_threshold
        )

        historical_data_sets[agent_id] = agent_historical_data

        # 4d. 统计历史数据
        total_historical = sum(len(cells) for cells in agent_historical_data.values())
        if verbose >= 2:
            print(f"      历史轨迹数据: {total_historical}个位置")

        # 4e. 统计有效历史数据
        if verbose >= 2:
            for ts, agent_cells in agent_historical_data.items():
                if len(agent_cells) > 0 and ts in agent_multi_reachable:
                    timestep_reachable = agent_multi_reachable[ts]
                    valid_historical_cells = [cell for cell in agent_cells if cell in timestep_reachable]

                    if valid_historical_cells:
                        print(f"        时间步{ts}: {len(agent_cells)}个历史位置 -> {len(valid_historical_cells)}个有效位置")
                    else:
                        print(f"        时间步{ts}: {len(agent_cells)}个历史位置 -> 0个有效位置（不在可达集内）")

    # 5. 将自车未来轨迹添加到热力图
    if verbose >= 2:
        print(f"    自车未来轨迹: {len(ego_action_trajectory)}步")
    for step_idx, ego_pos in enumerate(ego_action_trajectory):
        ego_cell = ctx.grid.world_to_cell(ego_pos)
        if 0 <= ego_cell < ctx.grid.spec.num_cells:
            c[ego_cell] += 1.0  # 自车轨迹用高权重
        if verbose >= 2:
            print(f"      步骤{step_idx}: 位置{ego_pos} -> cell {ego_cell}")

    # 6. 归一化热力图数据
    p_plot = c / (np.max(c) + 1e-12)

    # 7. 转换到网格坐标系
    ego_grid = ctx.grid.to_grid_frame(world_state.ego.position_m)
    agents_grid = []
    for agent in world_state.agents:
        agent_grid = ctx.grid.to_grid_frame(agent.position_m)
        agents_grid.append(np.array(agent_grid))

    # 8. 生成统计信息
    stats = _compute_visualization_stats(ctx, timestep, p_plot, multi_timestep_reachable)

    return {
        'p_plot': p_plot,
        'ego_grid': np.array(ego_grid),
        'agents_grid': agents_grid,
        'multi_timestep_reachable': multi_timestep_reachable,
        'historical_data_sets': historical_data_sets,
        'title': f"Episode {ctx.episode_id+1}, t={timestep+1}s: 可达集+历史轨迹+自车轨迹",
        'stats': stats
    }


def _extract_ego_action_trajectory(ctx: EpisodeContext, timestep: int) -> List[Tuple[float, float]]:
    """提取自车未来动作轨迹"""
    ego_action_trajectory = []
    for action_t in range(timestep, min(timestep + ctx.horizon, len(ctx.ego_trajectory))):
        ego_action_trajectory.append(tuple(ctx.ego_trajectory[action_t]))
    return ego_action_trajectory


def _extract_ego_state(world_state: WorldState) -> Tuple[float, float, float]:
    """提取自车状态"""
    return (
        world_state.ego.position_m[0],
        world_state.ego.position_m[1],
        world_state.ego.yaw_rad
    )


def _extract_agents_states(world_state: WorldState) -> List[Tuple]:
    """提取环境智能体状态"""
    current_agents_states = []
    for agent in world_state.agents:
        current_agents_states.append((
            agent.position_m[0],
            agent.position_m[1],
            agent.velocity_mps[0],
            agent.velocity_mps[1],
            agent.heading_rad,
            agent.agent_type.value
        ))
    return current_agents_states


def _compute_visualization_stats(
    ctx: EpisodeContext,
    timestep: int,
    p_plot: np.ndarray,
    multi_timestep_reachable: Dict
) -> Dict:
    """计算可视化统计信息"""
    # 动态获取所有已初始化的agent ID
    initialized_agent_ids = list(ctx.bank.agent_alphas.keys()) if hasattr(ctx.bank, 'agent_alphas') else []

    # 计算Alpha总和(兼容不同Bank类型)
    alpha_sum = 0.0
    for aid in initialized_agent_ids:
        if aid in ctx.bank.agent_alphas:
            alpha = ctx.bank.agent_alphas[aid]
            # 检查是否是多时间步Bank(Dict[int, np.ndarray])
            if isinstance(alpha, dict):
                alpha_sum += sum(a.sum() for a in alpha.values())
            # 单时间步Bank(np.ndarray)
            elif isinstance(alpha, np.ndarray):
                alpha_sum += alpha.sum()

    return {
        't': timestep + 1,
        'alpha_sum': alpha_sum,
        'qmax_max': float(np.max(p_plot)),
        'nz_cells': int(np.count_nonzero(p_plot > 1e-6)),
        'reachable_cells': {aid: sum(len(cells) for cells in reachable.values())
                          for aid, reachable in multi_timestep_reachable.items()}
    }
