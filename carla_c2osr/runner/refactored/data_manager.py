"""
数据存储管理模块
"""

from __future__ import annotations
from typing import Dict, List
import numpy as np

from carla_c2osr.agents.c2osr.trajectory_buffer import AgentTrajectoryData
from carla_c2osr.utils.simple_trajectory_generator import SimpleTrajectoryGenerator
from .episode_context import EpisodeContext


class DataManager:
    """数据存储管理"""

    def __init__(self, ctx: EpisodeContext):
        self.ctx = ctx

    def generate_agent_trajectories(self) -> tuple:
        """
        生成所有agent的轨迹

        Returns:
            (agent_trajectories, agent_trajectory_cells)
            - agent_trajectories: {agent_id: trajectory}
            - agent_trajectory_cells: {agent_id: trajectory_cells}
        """
        agent_trajectories = {}
        agent_trajectory_cells = {}

        # 从配置读取随机种子保证轨迹一致性
        # 注意：轨迹模式现在可以通过 config.agent_trajectory.mode 设置 ("dynamic", "straight", "stationary")
        from carla_c2osr.config import get_global_config
        config = get_global_config()
        rng = np.random.default_rng(config.agent_trajectory.random_seed)

        for i, agent in enumerate(self.ctx.world_init.agents):
            agent_id = i + 1
            try:
                # 生成符合动力学约束的轨迹
                trajectory = self.ctx.trajectory_generator.generate_agent_trajectory(
                    agent, self.ctx.horizon
                )
                agent_trajectories[agent_id] = trajectory

                # 转换为网格单元ID
                trajectory_cells = []
                for pos in trajectory:
                    cell_id = self.ctx.grid.world_to_cell(tuple(pos))
                    trajectory_cells.append(cell_id)
                agent_trajectory_cells[agent_id] = trajectory_cells

            except Exception as e:
                print(f"  警告: Agent {agent_id} 轨迹生成失败: {e}")
                # 使用简单的直线轨迹作为后备
                fallback_trajectory, fallback_cells = self._generate_fallback_trajectory(agent)
                agent_trajectories[agent_id] = fallback_trajectory
                agent_trajectory_cells[agent_id] = fallback_cells

        return agent_trajectories, agent_trajectory_cells

    def store_episode_trajectories(self,
                                   ego_trajectory: List[np.ndarray],
                                   agent_trajectories: Dict[int, List[np.ndarray]],
                                   agent_trajectory_cells: Dict[int, List[int]]):
        """
        将轨迹数据存储到buffer

        Args:
            ego_trajectory: 自车轨迹
            agent_trajectories: agent轨迹字典
            agent_trajectory_cells: agent轨迹单元ID字典
        """
        timestep_scenarios = []

        # 为每个时刻创建轨迹数据
        for t in range(self.ctx.horizon):
            # 获取当前时刻的世界状态
            world_current = self.ctx.scenario_manager.create_world_state_from_trajectories(
                t, ego_trajectory, agent_trajectories, self.ctx.world_init
            )

            # 创建当前时刻的场景状态
            current_scenario_state = self.ctx.scenario_manager.create_scenario_state(world_current)

            # 创建当前时刻的轨迹数据（只包含剩余轨迹）
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
        ego_trajectory_tuples = [tuple(pos) for pos in ego_trajectory]
        self.ctx.trajectory_buffer.store_episode_trajectories_by_timestep(
            self.ctx.episode_id, timestep_scenarios, ego_trajectory_tuples
        )

    def _generate_fallback_trajectory(self, agent) -> tuple:
        """生成简单的后备轨迹"""
        fallback_trajectory = []
        fallback_cells = []
        start_pos = np.array(agent.position_m)
        grid_half_size = self.ctx.grid.size_m / 2.0

        for t in range(self.ctx.horizon):
            next_pos = start_pos + np.array([0.5 * t, 0.1 * t])
            next_pos = np.clip(next_pos, -grid_half_size, grid_half_size)
            fallback_trajectory.append(next_pos)
            fallback_cells.append(self.ctx.grid.world_to_cell(tuple(next_pos)))

        return fallback_trajectory, fallback_cells
