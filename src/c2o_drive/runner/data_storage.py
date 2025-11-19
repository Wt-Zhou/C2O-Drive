"""
数据存储辅助模块

从 run_episode 中提取的轨迹数据存储逻辑,职责单一,易于测试。
"""

from typing import Dict, List
from c2o_drive.algorithms.c2osr.trajectory_buffer import AgentTrajectoryData
from c2o_drive.runner.episode_context import EpisodeContext


def store_episode_data(
    ctx: EpisodeContext,
    agent_trajectory_cells: Dict[int, List[int]]
):
    """存储episode的轨迹数据到buffer

    Args:
        ctx: Episode运行上下文
        agent_trajectory_cells: 环境智能体轨迹(网格单元格式)
            {agent_id: [cell_indices]}
    """
    from c2o_drive.config import get_global_config
    verbose = get_global_config().visualization.verbose_level
    if verbose >= 2:
        print(f"  存储episode {ctx.episode_id} 轨迹数据到buffer...")

    # 按时间步组织数据
    timestep_scenarios = []

    for t in range(ctx.horizon):
        # 获取当前时刻的世界状态
        world_current = ctx.scenario_manager.create_world_state_from_trajectories(
            t, ctx.ego_trajectory, {}, ctx.world_init  # agent_trajectories暂时为空
        )

        # 创建当前时刻的场景状态
        current_scenario_state = ctx.scenario_manager.create_scenario_state(world_current)

        # 创建当前时刻的轨迹数据(只包含剩余步骤)
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
    ego_trajectory_tuples = [tuple(pos) for pos in ctx.ego_trajectory]
    ctx.trajectory_buffer.store_episode_trajectories_by_timestep(
        ctx.episode_id,
        timestep_scenarios,
        ego_trajectory_tuples
    )

    from c2o_drive.config import get_global_config
    verbose = get_global_config().visualization.verbose_level
    if verbose >= 2:
        print(f"  ✓ Episode {ctx.episode_id} 存储完成: {ctx.horizon}个时间步, {len(agent_trajectory_cells)}个环境智能体")
