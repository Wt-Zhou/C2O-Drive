"""
单时间步执行模块
"""

from __future__ import annotations
from typing import Dict, List, Any, Tuple
import numpy as np

from c2o_drive.algorithms.c2osr.q_value import QValueCalculator, QValueConfig
from c2o_drive.config import get_global_config
from c2o_drive.algorithms.c2osr.spatial_dirichlet import MultiTimestepSpatialDirichletBank, OptimizedMultiTimestepSpatialDirichletBank
from .episode_context import EpisodeContext


class TimestepExecutor:
    """单时间步执行逻辑"""

    def __init__(self, ctx: EpisodeContext):
        self.ctx = ctx

    def execute_all_timesteps(self,
                             ego_trajectory: List[np.ndarray],
                             agent_trajectories: Dict[int, List[np.ndarray]]) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        执行所有时间步

        Args:
            ego_trajectory: 自车轨迹
            agent_trajectories: agent轨迹字典

        Returns:
            (episode_stats, frame_paths)
        """
        episode_stats = []
        frame_paths = []

        for t in range(self.ctx.horizon):
            # 执行单个时间步
            stats, frame_path = self.execute_single_timestep(
                t, ego_trajectory, agent_trajectories
            )

            if stats:
                episode_stats.append(stats)
            if frame_path:
                frame_paths.append(frame_path)

        return episode_stats, frame_paths

    def execute_single_timestep(self,
                                t: int,
                                ego_trajectory: List[np.ndarray],
                                agent_trajectories: Dict[int, List[np.ndarray]]) -> Tuple[Dict[str, Any], str]:
        """
        执行单个时间步

        Args:
            t: 时间步索引
            ego_trajectory: 自车轨迹
            agent_trajectories: agent轨迹字典

        Returns:
            (stats_dict, frame_path)
        """
        # 1. 创建当前世界状态
        world_current = self.ctx.scenario_manager.create_world_state_from_trajectories(
            t, ego_trajectory, agent_trajectories, self.ctx.world_init
        )

        # 2. 计算当前可达集
        current_reachable, multi_timestep_reachable = self._compute_reachable_sets(world_current)

        # 3. 在第一个时间步计算Q值
        if t == 0:
            self._compute_and_track_q_values(t, ego_trajectory, world_current)

        # 4. 初始化Agent的Dirichlet分布
        self._initialize_agent_distributions(world_current, ego_trajectory)

        # 5. 准备可视化数据
        p_plot, multi_timestep_reachable, historical_data_sets = self._prepare_visualization_data(
            t, ego_trajectory, world_current
        )

        # 6. 渲染热力图
        frame_path = self._render_timestep(t, world_current, p_plot,
                                          multi_timestep_reachable, historical_data_sets)

        # 7. 收集统计信息
        stats = self._collect_statistics(t, p_plot, current_reachable)

        return stats, frame_path

    def _compute_reachable_sets(self, world_current):
        """计算当前和多时间步可达集"""
        config = get_global_config()
        current_reachable = {}
        multi_timestep_reachable = {}

        for i, agent in enumerate(world_current.agents):
            agent_id = i + 1
            # 单时间步可达集
            reachable = self.ctx.grid.successor_cells(
                agent, n_samples=config.sampling.reachable_set_samples_legacy
            )
            current_reachable[agent_id] = reachable

            # 多时间步可达集
            multi_reachable = self.ctx.grid.multi_timestep_successor_cells(
                agent,
                horizon=self.ctx.horizon,
                dt=config.time.dt,
                n_samples=config.sampling.reachable_set_samples
            )
            multi_timestep_reachable[agent_id] = multi_reachable

        return current_reachable, multi_timestep_reachable

    def _compute_and_track_q_values(self, t: int, ego_trajectory: List[np.ndarray], world_current):
        """计算并跟踪Q值（仅在t=0时）"""
        # 构造自车未来动作轨迹
        ego_action_trajectory = []
        for action_t in range(t, min(t + self.ctx.horizon, len(ego_trajectory))):
            ego_action_trajectory.append(tuple(ego_trajectory[action_t]))

        try:
            # 创建Q值配置和计算器
            q_config = QValueConfig.from_global_config()
            global_config = get_global_config()
            reward_config = global_config.reward
            q_calculator = QValueCalculator(q_config, reward_config)

            # 计算Q值
            q_values, detailed_info = q_calculator.compute_q_value(
                current_world_state=world_current,
                ego_action_trajectory=ego_action_trajectory,
                trajectory_buffer=self.ctx.trajectory_buffer,
                grid=self.ctx.grid,
                bank=self.ctx.bank,
                rng=self.ctx.rng,
                reference_path=self.ctx.reference_path
            )

            # 记录Q值分布
            if self.ctx.q_tracker is not None:
                avg_q_value = np.mean(q_values)
                q_distribution = detailed_info['reward_breakdown']['all_q_values']
                collision_rate = detailed_info['reward_breakdown']['collision_rate']
                self.ctx.q_tracker.add_episode_data(
                    episode_id=self.ctx.episode_id,
                    q_value=avg_q_value,
                    q_distribution=q_distribution,
                    collision_rate=collision_rate,
                    detailed_info=detailed_info
                )

            # 每5个episode生成分布可视化
            if t == 0 and self.ctx.episode_id % 5 == 0:
                self._visualize_distributions(q_calculator, world_current, ego_action_trajectory)

        except Exception as e:
            print(f"  警告: Q值计算失败: {e}")
            # 记录失败信息
            if self.ctx.q_tracker is not None:
                config = get_global_config()
                n_samples = config.sampling.q_value_samples
                self.ctx.q_tracker.add_episode_data(
                    episode_id=self.ctx.episode_id,
                    q_value=0.0,
                    q_distribution=[0.0] * n_samples,
                    collision_rate=0.0,
                    detailed_info={'error': str(e)}
                )

    def _visualize_distributions(self, q_calculator, world_current, ego_action_trajectory):
        """生成transition和Dirichlet分布可视化"""
        try:
            from c2o_drive.visualization.transition_visualizer import (
                visualize_transition_distributions, visualize_dirichlet_distributions
            )

            print(f"  生成分布可视化...")

            # 获取transition分布数据
            agent_transition_samples = q_calculator._build_agent_transition_distributions(
                world_current, ego_action_trajectory, self.ctx.trajectory_buffer,
                self.ctx.grid, self.ctx.bank, self.ctx.horizon
            )

            # 可视化transition分布
            visualize_transition_distributions(
                agent_transition_samples=agent_transition_samples,
                current_world_state=world_current,
                grid=self.ctx.grid,
                episode_idx=self.ctx.episode_id,
                output_dir=self.ctx.output_dir
            )

            # 可视化Dirichlet分布
            visualize_dirichlet_distributions(
                bank=self.ctx.bank,
                current_world_state=world_current,
                grid=self.ctx.grid,
                episode_idx=self.ctx.episode_id,
                output_dir=self.ctx.output_dir
            )

            print(f"  分布可视化完成")

        except Exception as e:
            print(f"  警告: 可视化生成失败: {e}")

    def _initialize_agent_distributions(self, world_current, ego_trajectory):
        """为可视化初始化Agent的Dirichlet分布"""
        config = get_global_config()

        for i, agent in enumerate(world_current.agents):
            agent_id = i + 1
            try:
                # 计算多时间步可达集
                agent_multi_reachable = self.ctx.grid.multi_timestep_successor_cells(
                    agent,
                    horizon=len(ego_trajectory),
                    dt=config.time.dt,
                    n_samples=config.sampling.reachable_set_samples
                )
                if agent_multi_reachable and agent_id not in self.ctx.bank.agent_alphas:
                    self.ctx.bank.init_agent(agent_id, agent_multi_reachable)
            except Exception as e:
                print(f"  警告: Agent {agent_id} 初始化失败: {e}")
                continue

    def _prepare_visualization_data(self, t: int, ego_trajectory: List[np.ndarray], world_current):
        """准备可视化数据"""
        config = get_global_config()

        # 1. 构造自车未来动作轨迹
        ego_action_trajectory = []
        for action_t in range(t, min(t + self.ctx.horizon, len(ego_trajectory))):
            ego_action_trajectory.append(tuple(ego_trajectory[action_t]))

        # 2. 获取当前状态
        current_ego_state = (
            world_current.ego.position_m[0],
            world_current.ego.position_m[1],
            world_current.ego.yaw_rad
        )
        current_agents_states = []
        for agent in world_current.agents:
            current_agents_states.append((
                agent.position_m[0], agent.position_m[1],
                agent.velocity_mps[0], agent.velocity_mps[1],
                agent.heading_rad, agent.agent_type.value
            ))

        # 3. 初始化可视化数据
        c = np.zeros(self.ctx.grid.spec.num_cells)

        # 4. 处理每个Agent
        multi_timestep_reachable = {}
        historical_data_sets = {}

        for i, agent in enumerate(world_current.agents):
            agent_id = i + 1

            # 计算多时间步可达集
            agent_multi_reachable = self.ctx.grid.multi_timestep_successor_cells(
                agent,
                horizon=len(ego_action_trajectory),
                dt=config.time.dt,
                n_samples=config.sampling.reachable_set_samples
            )

            if not agent_multi_reachable:
                continue

            multi_timestep_reachable[agent_id] = agent_multi_reachable

            # 添加到可视化（按时间步分权重）
            for timestep, reachable_cells in agent_multi_reachable.items():
                timestep_weight = 0.3 / (timestep + 1)
                for cell in reachable_cells:
                    if 0 <= cell < self.ctx.grid.spec.num_cells:
                        c[cell] += timestep_weight

            # 获取历史轨迹数据
            agent_historical_data = self.ctx.trajectory_buffer.get_agent_historical_transitions_strict_matching(
                agent_id=agent_id,
                current_ego_state=current_ego_state,
                current_agents_states=current_agents_states,
                ego_action_trajectory=ego_action_trajectory,
                ego_state_threshold=config.matching.ego_state_threshold,
                agents_state_threshold=config.matching.agents_state_threshold,
                ego_action_threshold=config.matching.ego_action_threshold
            )

            historical_data_sets[agent_id] = agent_historical_data

        # 5. 添加自车未来轨迹
        for step_idx, ego_pos in enumerate(ego_action_trajectory):
            ego_cell = self.ctx.grid.world_to_cell(ego_pos)
            if 0 <= ego_cell < self.ctx.grid.spec.num_cells:
                c[ego_cell] += 1.0

        # 6. 归一化
        p_plot = c / (np.max(c) + 1e-12)

        return p_plot, multi_timestep_reachable, historical_data_sets

    def _render_timestep(self, t: int, world_current, p_plot,
                        multi_timestep_reachable, historical_data_sets) -> str:
        """渲染时间步热力图"""
        from c2o_drive.visualization.vis import grid_heatmap

        # 转换坐标到网格坐标系
        ego_grid = self.ctx.grid.to_grid_frame(world_current.ego.position_m)
        agents_grid = []
        for agent in world_current.agents:
            agent_grid = self.ctx.grid.to_grid_frame(agent.position_m)
            agents_grid.append(np.array(agent_grid))

        # 渲染热力图
        ep_dir = self.ctx.get_episode_dir()
        frame_path = ep_dir / f"t_{t+1:02d}.png"
        title = f"Episode {self.ctx.episode_id+1}, t={t+1}s: 可达集+历史轨迹+自车轨迹"

        try:
            grid_heatmap(
                p_plot,
                self.ctx.grid.N,
                np.array(ego_grid),
                agents_grid,
                title,
                str(frame_path),
                self.ctx.grid.size_m,
                multi_timestep_reachable_sets=multi_timestep_reachable,
                historical_data_sets=historical_data_sets,
            )
            return str(frame_path)
        except Exception as e:
            print(f"  警告: 渲染失败 t={t+1}: {e}")
            return None

    def _collect_statistics(self, t: int, p_plot, current_reachable) -> Dict[str, Any]:
        """收集统计信息"""
        # 动态获取所有已初始化的agent ID
        initialized_agent_ids = list(self.ctx.bank.agent_alphas.keys()) if hasattr(self.ctx.bank, 'agent_alphas') else []

        # 计算Alpha总和和真实非零单元数（兼容不同的Bank类型）
        if isinstance(self.ctx.bank, (MultiTimestepSpatialDirichletBank, OptimizedMultiTimestepSpatialDirichletBank)):
            alpha_sum = 0.0
            bank_nonzero_cells = 0
            alpha_out_threshold = self.ctx.bank.params.alpha_out

            for aid in initialized_agent_ids:
                if aid in self.ctx.bank.agent_alphas:
                    for timestep, alpha in self.ctx.bank.agent_alphas[aid].items():
                        alpha_sum += alpha.sum()
                        # 统计超过先验值的单元（真正学到了知识）
                        bank_nonzero_cells += int(np.count_nonzero(alpha > alpha_out_threshold))
        else:
            # 对于旧版本的单时间步Bank
            alpha_sum = sum(self.ctx.bank.get_agent_alpha(aid).sum() for aid in initialized_agent_ids)
            # 简化统计
            bank_nonzero_cells = sum(int(np.count_nonzero(self.ctx.bank.get_agent_alpha(aid) > 1e-6))
                                    for aid in initialized_agent_ids)

        stats = {
            't': t + 1,
            'alpha_sum': alpha_sum,
            'qmax_max': float(np.max(p_plot)),
            'nz_cells': bank_nonzero_cells,  # 改为Bank真实非零单元统计
            'reachable_cells': {aid: len(current_reachable[aid]) for aid in current_reachable.keys()}
        }

        return stats
