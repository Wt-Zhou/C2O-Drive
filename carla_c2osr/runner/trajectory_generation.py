"""
轨迹生成辅助函数

从run_episode中提取的轨迹生成逻辑。
"""

from typing import Dict, List, Tuple
import numpy as np

from carla_c2osr.env.types import AgentState
from carla_c2osr.runner.episode_context import EpisodeContext


def generate_agent_trajectories(ctx: EpisodeContext) -> Tuple[Dict[int, List[np.ndarray]], Dict[int, List[int]]]:
    """生成所有环境智能体的轨迹

    Args:
        ctx: Episode运行上下文

    Returns:
        (agent_trajectories, agent_trajectory_cells)
        - agent_trajectories: {agent_id: [positions]}
        - agent_trajectory_cells: {agent_id: [cell_indices]}
    """
    agent_trajectories = {}
    agent_trajectory_cells = {}

    # 从配置读取随机种子确保轨迹可复现
    from carla_c2osr.config import get_global_config
    config = get_global_config()
    rng = np.random.default_rng(config.agent_trajectory.random_seed)

    for i, agent in enumerate(ctx.world_init.agents):
        agent_id = i + 1

        try:
            # 生成符合动力学约束的轨迹
            trajectory = _generate_single_agent_trajectory(agent, ctx, rng)
            agent_trajectories[agent_id] = trajectory

            # 将轨迹转换为网格单元ID
            trajectory_cells = _convert_trajectory_to_cells(trajectory, ctx)
            agent_trajectory_cells[agent_id] = trajectory_cells

            from carla_c2osr.config import get_global_config
            verbose = get_global_config().visualization.verbose_level
            if verbose >= 2:
                print(f"  Agent {agent_id} ({agent.agent_type.value}) 轨迹生成: {len(trajectory)} 步")

        except Exception as e:
            print(f"  警告: Agent {agent_id} 轨迹生成失败: {e}")

            # 使用简单的后备轨迹
            fallback_trajectory, fallback_cells = _generate_fallback_trajectory(agent, ctx)
            agent_trajectories[agent_id] = fallback_trajectory
            agent_trajectory_cells[agent_id] = fallback_cells

    return agent_trajectories, agent_trajectory_cells


def _generate_single_agent_trajectory(agent: AgentState, ctx: EpisodeContext, rng: np.random.Generator) -> List[np.ndarray]:
    """生成单个智能体的轨迹"""
    return ctx.trajectory_generator.generate_agent_trajectory(agent, ctx.horizon)


def _convert_trajectory_to_cells(trajectory: List[np.ndarray], ctx: EpisodeContext) -> List[int]:
    """将轨迹位置转换为网格单元ID"""
    trajectory_cells = []
    for pos in trajectory:
        cell_id = ctx.grid.world_to_cell(tuple(pos))
        trajectory_cells.append(cell_id)
    return trajectory_cells


def _generate_fallback_trajectory(agent: AgentState, ctx: EpisodeContext) -> Tuple[List[np.ndarray], List[int]]:
    """生成简单的后备轨迹（直线移动）"""
    fallback_trajectory = []
    fallback_cells = []
    start_pos = np.array(agent.position_m)
    grid_half_size = ctx.grid.size_m / 2.0

    for t in range(ctx.horizon):
        # 简单直线移动
        next_pos = start_pos + np.array([0.5 * t, 0.1 * t])
        next_pos = np.clip(next_pos, -grid_half_size, grid_half_size)

        fallback_trajectory.append(next_pos)
        fallback_cells.append(ctx.grid.world_to_cell(tuple(next_pos)))

    return fallback_trajectory, fallback_cells
