"""
Q值计算模块 - 重新设计版本

基于历史转移数据和Dirichlet分布的Q值计算系统。
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from c2o_drive.core.types import EgoState, AgentState, WorldState, AgentType
from c2o_drive.algorithms.c2osr.grid_mapper import GridMapper
from c2o_drive.algorithms.c2osr.dirichlet import SpatialDirichletBank, DirichletParams, MultiTimestepSpatialDirichletBank, OptimizedMultiTimestepSpatialDirichletBank
from c2o_drive.algorithms.c2osr.trajectory_buffer import TrajectoryBuffer
from c2o_drive.config import get_global_config, RewardConfig
from c2o_drive.utils.collision import ShapeBasedCollisionDetector


@dataclass
class QValueConfig:
    """Q值计算配置

    默认从全局配置读取参数。如果需要自定义,请显式传入参数值。
    """
    horizon: Optional[int] = None  # 预测时间步长
    n_samples: Optional[int] = None  # Dirichlet采样数量
    dirichlet_alpha_in: Optional[float] = None  # 可达集内的先验强度
    dirichlet_alpha_out: Optional[float] = None  # 可达集外的先验强度
    learning_rate: Optional[float] = None  # 历史数据更新学习率
    q_selection_percentile: Optional[float] = None  # Q值选择百分位数（从全局配置读取）
    gamma: Optional[float] = None  # 折扣因子

    def __post_init__(self):
        """从全局配置读取未设置的参数"""
        global_config = get_global_config()
        if self.horizon is None:
            self.horizon = global_config.time.default_horizon
        if self.n_samples is None:
            self.n_samples = global_config.sampling.q_value_samples
        if self.dirichlet_alpha_in is None:
            self.dirichlet_alpha_in = global_config.dirichlet.alpha_in
        if self.dirichlet_alpha_out is None:
            self.dirichlet_alpha_out = global_config.dirichlet.alpha_out
        if self.learning_rate is None:
            self.learning_rate = global_config.dirichlet.learning_rate
        if self.q_selection_percentile is None:
            self.q_selection_percentile = global_config.c2osr.q_selection_percentile
        if self.gamma is None:
            self.gamma = global_config.c2osr.gamma

    @classmethod
    def from_global_config(cls):
        """从全局配置创建Q值配置（向后兼容方法）"""
        return cls()


class RewardCalculator:
    """模块化奖励计算器"""
    
    def __init__(self, config: RewardConfig):
        self.config = config
    
    def calculate_collision_reward(self, collision_occurred: bool) -> float:
        """计算碰撞奖励"""
        if collision_occurred:
            return self.config.collision_penalty
        return 0.0
    
    def calculate_comfort_reward(self, ego_trajectory: List[Tuple[float, float]], dt: Optional[float] = None) -> float:
        """计算舒适性奖励（基于加速度和急动）

        Args:
            ego_trajectory: 自车轨迹
            dt: 时间步长，默认从全局配置读取
        """
        if dt is None:
            dt = get_global_config().time.dt

        if len(ego_trajectory) < 3:
            return 0.0

        reward = 0.0

        # 计算加速度
        for i in range(1, len(ego_trajectory) - 1):
            v_prev = np.array(ego_trajectory[i]) - np.array(ego_trajectory[i-1])
            v_curr = np.array(ego_trajectory[i+1]) - np.array(ego_trajectory[i])

            accel = (v_curr - v_prev) / dt
            accel_magnitude = np.linalg.norm(accel)
            
            # 加速度惩罚
            if accel_magnitude > self.config.max_comfortable_accel:
                reward -= (accel_magnitude - self.config.max_comfortable_accel) * self.config.acceleration_penalty_weight
            
            # 急动惩罚（加速度变化率）
            if i > 1:
                v_prev_prev = np.array(ego_trajectory[i-1]) - np.array(ego_trajectory[i-2])
                accel_prev = (v_prev - v_prev_prev) / dt
                jerk = np.linalg.norm(accel - accel_prev) / dt
                reward -= jerk * self.config.jerk_penalty_weight
        
        return reward
    
    def calculate_efficiency_reward(self, ego_trajectory: List[Tuple[float, float]], dt: Optional[float] = None) -> float:
        """计算驾驶效率奖励（速度和前进距离）

        Args:
            ego_trajectory: 自车轨迹
            dt: 时间步长，默认从全局配置读取
        """
        if dt is None:
            dt = get_global_config().time.dt

        if len(ego_trajectory) < 2:
            return 0.0

        reward = 0.0
        total_distance = 0.0

        for i in range(1, len(ego_trajectory)):
            # 计算速度
            velocity = np.array(ego_trajectory[i]) - np.array(ego_trajectory[i-1])
            speed = np.linalg.norm(velocity) / dt
            
            # 速度奖励（鼓励接近目标速度）
            speed_reward = -abs(speed - self.config.target_speed) * self.config.speed_reward_weight
            reward += speed_reward
            
            # 前进距离奖励
            distance = np.linalg.norm(velocity)
            total_distance += distance
        
        # 前进距离奖励
        reward += total_distance * self.config.progress_reward_weight
        
        return reward
    
    def calculate_safety_reward(self, ego_trajectory: List[Tuple[float, float]], 
                              agent_trajectories: Dict[int, List[Tuple[float, float]]]) -> float:
        """计算安全距离奖励"""
        reward = 0.0
        
        for t in range(min(len(ego_trajectory), min(len(traj) for traj in agent_trajectories.values()) if agent_trajectories else len(ego_trajectory))):
            ego_pos = np.array(ego_trajectory[t])
            
            for agent_id, agent_traj in agent_trajectories.items():
                if t < len(agent_traj):
                    agent_pos = np.array(agent_traj[t])
                    distance = np.linalg.norm(ego_pos - agent_pos)
                    
                    if distance < self.config.safe_distance:
                        penalty = -(self.config.safe_distance - distance) * self.config.distance_penalty_weight
                        reward += penalty
        
        return reward
    
    def calculate_centerline_offset_reward(self, ego_trajectory: List[Tuple[float, float]],
                                          reference_path: Optional[List] = None) -> float:
        """计算中心线偏移奖励

        Args:
            ego_trajectory: 自车轨迹
            reference_path: 参考路径（中心线）

        Returns:
            中心线偏移奖励（负值为惩罚）
        """
        if reference_path is None or len(reference_path) == 0:
            return 0.0

        from c2o_drive.algorithms.c2osr.rewards import calculate_distance_to_path

        reward = 0.0
        for pos in ego_trajectory:
            offset = calculate_distance_to_path(pos, reference_path)
            # 使用配置中的权重，如果没有则使用默认值
            weight = getattr(self.config, 'centerline_offset_penalty_weight', 1.0)
            reward -= offset * weight

        return reward

    def calculate_total_reward(self, ego_trajectory: List[Tuple[float, float]],
                             agent_trajectories: Dict[int, List[Tuple[float, float]]],
                             collision_occurred: bool,
                             reference_path: Optional[List] = None) -> float:
        """计算总奖励"""
        collision_reward = self.calculate_collision_reward(collision_occurred)

        # 如果发生碰撞，直接返回碰撞惩罚
        if collision_occurred:
            return collision_reward

        comfort_reward = self.calculate_comfort_reward(ego_trajectory)
        efficiency_reward = self.calculate_efficiency_reward(ego_trajectory)
        safety_reward = self.calculate_safety_reward(ego_trajectory, agent_trajectories)
        centerline_reward = self.calculate_centerline_offset_reward(ego_trajectory, reference_path)

        return collision_reward + comfort_reward + efficiency_reward + safety_reward + centerline_reward


class QValueCalculator:
    """终极优化的Q值计算器 - 完全消除采样，纯期望计算"""
    
    def __init__(self, config: QValueConfig, reward_config: RewardConfig):
        self.config = config
        self.reward_config = reward_config
        self.collision_detector = ShapeBasedCollisionDetector()

        # 创建Dirichlet参数（从全局配置读取）
        from c2o_drive.config import get_global_config
        global_config = get_global_config()

        self.dirichlet_params = DirichletParams(
            alpha_in=config.dirichlet_alpha_in,
            alpha_out=config.dirichlet_alpha_out,
            delta=global_config.dirichlet.delta,
            cK=global_config.dirichlet.cK
        )
    
    def compute_q_value(self,
                       current_world_state: WorldState,
                       ego_action_trajectory: List[Tuple[float, float]],
                       trajectory_buffer: TrajectoryBuffer,
                       grid: GridMapper,
                       bank: Optional[MultiTimestepSpatialDirichletBank] = None,
                       rng: Optional[np.random.Generator] = None,
                       reference_path: Optional[List] = None) -> Tuple[List[float], Dict]:
        """终极优化的Q值计算 - 完全消除采样，纯期望计算

        Args:
            current_world_state: 当前世界状态
            ego_action_trajectory: 自车动作序列（未来horizon个位置）
            trajectory_buffer: 历史轨迹缓冲区
            grid: 网格映射器
            bank: 持久化的Dirichlet Bank（如果为None则创建临时Bank）
            rng: 随机数生成器（为了兼容性保留）
            reference_path: 参考路径（中心线），用于计算偏移惩罚

        Returns:
            (所有Q值列表, 详细信息字典)
        """
        # 将 reference_path 存储为实例变量，供内部方法使用
        self._reference_path = reference_path
        if rng is None:
            rng = np.random.default_rng()
        
        horizon = len(ego_action_trajectory)

        # CRITICAL FIX: Bank must be provided and persistent
        # Do NOT create temporary banks as they discard alpha updates
        if bank is None:
            raise ValueError(
                "Dirichlet bank must be provided. "
                "Cannot create temporary bank as it would discard all learned alpha values."
            )

        if not isinstance(bank, OptimizedMultiTimestepSpatialDirichletBank):
            raise TypeError(
                f"Expected OptimizedMultiTimestepSpatialDirichletBank, got {type(bank).__name__}. "
                "Ensure planner passes the persistent bank instance."
            )

        optimized_bank = bank
        
        # 获取verbose级别
        verbose = get_global_config().visualization.verbose_level

        if verbose >= 3:  # Only show in trace mode
            print(f"  [Q-Value] Starting optimized calculation")

        # 第1步:计算与agent完全无关的奖励(只计算一次!)
        agent_independent_reward = self._calculate_agent_independent_rewards(ego_action_trajectory)

        # 第2步:建立agent的transition分布
        agent_transition_samples = self._build_agent_transition_distributions(
            current_world_state, ego_action_trajectory, trajectory_buffer, grid, optimized_bank, horizon
        )

        # 第3步:直接计算期望的agent相关奖励(无采样!)
        q_values = []
        collision_probabilities = []  # 收集所有样本的期望碰撞概率
        
        for sample_idx in range(self.config.n_samples):
            # 获取该样本的transition分布
            sample_distributions = self._extract_sample_distributions(
                agent_transition_samples, sample_idx
            )
            
            # 直接计算期望的agent相关奖励（关键优化！）
            expected_collision_reward, expected_collision_prob = self._calculate_expected_collision_reward_directly(
                ego_action_trajectory, sample_distributions, grid, current_world_state
            )
            
            expected_safety_reward = self._calculate_expected_safety_reward_directly(
                ego_action_trajectory, sample_distributions, grid
            )
            
            # 组合最终Q值
            total_agent_dependent_reward = expected_collision_reward + expected_safety_reward
            final_q_value = agent_independent_reward + total_agent_dependent_reward
            q_values.append(final_q_value)
            collision_probabilities.append(expected_collision_prob)

        # 计算percentile对应的碰撞率（与percentile Q对应）
        if len(q_values) > 0 and len(collision_probabilities) > 0:
            # 转换为numpy数组
            q_values_array = np.array(q_values)
            collision_probs_array = np.array(collision_probabilities)

            # 对Q值排序，获取percentile对应的索引
            sorted_indices = np.argsort(q_values_array)
            percentile_position = int(self.config.q_selection_percentile * (len(q_values_array) - 1))
            percentile_index = sorted_indices[percentile_position]

            # 使用percentile Q对应的碰撞率
            percentile_collision_rate = float(collision_probs_array[percentile_index])

            # 同时保留平均碰撞率用于对比
            mean_collision_probability = float(np.mean(collision_probabilities))
        else:
            percentile_collision_rate = 0.0
            mean_collision_probability = 0.0

        # 统计信息
        detailed_info = {
            'calculation_method': '终极优化：纯期望计算，零采样',
            'agent_independent_reward': agent_independent_reward,
            'agent_dependent_rewards': [q - agent_independent_reward for q in q_values],
            'computational_savings': f'消除了 {self.config.n_samples} × trajectory_samples 次采样',
            'reward_breakdown': {
                'mean_q_value': np.mean(q_values),
                'q_value_std': np.std(q_values),
                'q_value_min': np.min(q_values),
                'q_value_max': np.max(q_values),
                'collision_rate': percentile_collision_rate,  # 改为使用percentile对应的碰撞率
                'mean_collision_rate': mean_collision_probability,  # 保留平均值用于对比
                'all_q_values': q_values,
                'collision_probabilities': collision_probabilities  # 保存所有样本的碰撞概率
            },
            'agent_info': {}
        }

        if verbose >= 3:  # Only show detailed stats in trace mode
            # Enhanced Q distribution statistics
            q_arr = np.array(q_values)
            p5 = np.percentile(q_arr, 5)
            p50 = np.percentile(q_arr, 50)
            p95 = np.percentile(q_arr, 95)
            print(f"  [Q-Value] Distribution: P5={p5:.3f}, P50={p50:.3f}, P95={p95:.3f}, "
                  f"Mean={np.mean(q_arr):.3f}, Std={np.std(q_arr):.3f}")
        
        return q_values, detailed_info
    
    def _calculate_agent_independent_rewards(self, ego_trajectory: List[Tuple[float, float]]) -> float:
        """计算与agent完全无关的奖励"""
        total_reward = 0.0
        dt = get_global_config().time.dt

        # 1. 舒适性奖励（基于自车加速度和急动）
        if len(ego_trajectory) >= 3:
            for i in range(1, len(ego_trajectory) - 1):
                v_prev = np.array(ego_trajectory[i]) - np.array(ego_trajectory[i-1])
                v_curr = np.array(ego_trajectory[i+1]) - np.array(ego_trajectory[i])
                
                accel = (v_curr - v_prev) / dt
                accel_magnitude = np.linalg.norm(accel)
                
                # 加速度惩罚
                if accel_magnitude > self.reward_config.max_comfortable_accel:
                    total_reward -= (accel_magnitude - self.reward_config.max_comfortable_accel) * self.reward_config.acceleration_penalty_weight
                
                # 急动惩罚
                if i > 1:
                    v_prev_prev = np.array(ego_trajectory[i-1]) - np.array(ego_trajectory[i-2])
                    accel_prev = (v_prev - v_prev_prev) / dt
                    jerk = np.linalg.norm(accel - accel_prev) / dt
                    total_reward -= jerk * self.reward_config.jerk_penalty_weight
        
        # 2. 速度奖励（基于自车速度）
        if len(ego_trajectory) >= 2:
            for i in range(1, len(ego_trajectory)):
                velocity = np.array(ego_trajectory[i]) - np.array(ego_trajectory[i-1])
                speed = np.linalg.norm(velocity) / dt
                speed_reward = -abs(speed - self.reward_config.target_speed) * self.reward_config.speed_reward_weight
                total_reward += speed_reward
        
        # 3. 进度奖励（基于前进距离）
        if len(ego_trajectory) >= 2:
            total_distance = 0.0
            for i in range(1, len(ego_trajectory)):
                velocity = np.array(ego_trajectory[i]) - np.array(ego_trajectory[i-1])
                distance = np.linalg.norm(velocity)
                total_distance += distance
            total_reward += total_distance * self.reward_config.progress_reward_weight

        # 4. 中心线偏移惩罚
        if hasattr(self, '_reference_path') and self._reference_path is not None:
            from c2o_drive.algorithms.c2osr.rewards import calculate_distance_to_path
            for pos in ego_trajectory:
                offset = calculate_distance_to_path(pos, self._reference_path)
                weight = getattr(self.reward_config, 'centerline_offset_penalty_weight', 1.0)
                total_reward -= offset * weight

        return total_reward
    
    def _calculate_expected_collision_reward_directly(self,
                                                    ego_trajectory: List[Tuple[float, float]],
                                                    agent_distributions: Dict[int, Dict[int, Tuple[List[int], np.ndarray]]],
                                                    grid: GridMapper,
                                                    current_world_state: WorldState) -> Tuple[float, float]:
        """直接计算期望碰撞奖励和期望碰撞概率 - 使用精确车辆形状碰撞检测 + Cell剪枝优化！

        Returns:
            (expected_reward, expected_collision_probability)
        """
        expected_reward = 0.0
        expected_collision_prob = 0.0  # 期望碰撞概率
        collision_count = 0  # 调试：碰撞计数

        # 🚀 优化1: 预计算ego轨迹占据的cell集合（用于剪枝）
        # 使用配置的剪枝半径（默认radius=10，约5米，覆盖车辆长度4.5m）
        ego_cells_set = set()
        for ego_pos in ego_trajectory:
            ego_cell = grid.world_to_cell(ego_pos)
            # 扩展到邻域（考虑车辆尺寸）
            neighbors = grid.get_neighbors(ego_cell, radius=self.reward_config.collision_check_cell_radius)
            ego_cells_set.update(neighbors)

        # 统计剪枝效果
        total_checks = 0
        pruned_checks = 0

        # 计算自车朝向序列（假设直行，实际应根据轨迹计算）
        ego_headings = []
        ego_initial_heading = current_world_state.ego.yaw_rad

        for i in range(len(ego_trajectory)):
            if i == 0:
                ego_headings.append(ego_initial_heading)
            else:
                # 根据轨迹计算朝向
                dx = ego_trajectory[i][0] - ego_trajectory[i-1][0]
                dy = ego_trajectory[i][1] - ego_trajectory[i-1][1]
                if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                    heading = np.arctan2(dy, dx)
                    ego_headings.append(heading)
                else:
                    ego_headings.append(ego_headings[-1])

        for timestep, ego_pos in enumerate(ego_trajectory):
            timestep_key = timestep + 1  # timestep从1开始
            
            if timestep >= len(ego_headings):
                continue
                
            for agent_id, distributions in agent_distributions.items():
                if timestep_key in distributions:
                    reachable_cells, probabilities = distributions[timestep_key]
                    
                    # 获取agent类型（从当前世界状态）
                    if agent_id <= len(current_world_state.agents):
                        agent_type = current_world_state.agents[agent_id - 1].agent_type
                    else:
                        agent_type = AgentType.VEHICLE  # 默认类型
                    
                    # 直接计算期望：E[碰撞奖励] = Σ P(位置i) × I(碰撞) × 惩罚
                    for cell_idx, prob in enumerate(probabilities):
                        if prob > 0:
                            cell = reachable_cells[cell_idx]
                            total_checks += 1

                            # 🚀 优化2: Cell剪枝 - 快速跳过不可能碰撞的cells
                            if cell not in ego_cells_set:
                                pruned_checks += 1
                                continue  # 不在ego附近，跳过精确检测！

                            cell_center = grid.index_to_xy_center(cell)
                            world_pos = grid.grid_to_world(np.array(cell_center))

                            # 使用精确的车辆形状碰撞检测
                            # 假设agent朝向与自车相同（简化处理）
                            agent_heading = ego_headings[timestep]

                            collision_occurred = self.collision_detector.check_point_collision(
                                ego_pos=ego_pos,
                                ego_heading=ego_headings[timestep],
                                agent_pos=tuple(world_pos),
                                agent_heading=agent_heading,
                                ego_type=AgentType.VEHICLE,
                                agent_type=agent_type
                            )

                            if collision_occurred:
                                # 直接累加期望值：概率 × 碰撞惩罚
                                collision_contribution = prob * self.reward_config.collision_penalty
                                expected_reward += collision_contribution
                                
                                # 累加期望碰撞概率：概率 × 1（碰撞发生）
                                expected_collision_prob += prob
                                collision_count += 1

        # 打印剪枝效果统计(仅在debug模式)
        if total_checks > 0:
            verbose = get_global_config().visualization.verbose_level
            if verbose >= 3:  # Only show in trace mode (verbose=3)
                prune_rate = (pruned_checks / total_checks) * 100
                actual_checks = total_checks - pruned_checks
                radius_m = self.reward_config.collision_check_cell_radius * self.grid_spec.cell_m
                print(f"  [Collision] Cell pruning: total={total_checks}, pruned={pruned_checks}, "
                      f"actual={actual_checks}, rate={prune_rate:.1f}%")

        return expected_reward, expected_collision_prob
    
    def _calculate_expected_safety_reward_directly(self,
                                                 ego_trajectory: List[Tuple[float, float]],
                                                 agent_distributions: Dict[int, Dict[int, Tuple[List[int], np.ndarray]]],
                                                 grid: GridMapper) -> float:
        """直接计算期望安全距离奖励 - 您的另一个核心洞察！"""
        expected_reward = 0.0
        
        for timestep, ego_pos in enumerate(ego_trajectory):
            timestep_key = timestep + 1
            
            for agent_id, distributions in agent_distributions.items():
                if timestep_key in distributions:
                    reachable_cells, probabilities = distributions[timestep_key]
                    
                    # 直接计算期望：E[安全奖励] = Σ P(位置i) × 安全奖励(距离i)
                    for cell_idx, prob in enumerate(probabilities):
                        if prob > 0:
                            cell = reachable_cells[cell_idx]
                            cell_center = grid.index_to_xy_center(cell)
                            world_pos = grid.grid_to_world(np.array(cell_center))
                            
                            distance = np.linalg.norm(np.array(ego_pos) - np.array(world_pos))
                            if distance < self.reward_config.safe_distance:
                                penalty = -(self.reward_config.safe_distance - distance) * self.reward_config.distance_penalty_weight
                                # 直接累加期望值：概率 × 安全距离惩罚
                                expected_reward += prob * penalty
        
        return expected_reward
    
    def _build_agent_transition_distributions(self, current_world_state, ego_action_trajectory, 
                                            trajectory_buffer, grid, bank, horizon):
        """建立agent的transition分布（复用之前的逻辑）"""
        current_ego_state = (
            current_world_state.ego.position_m[0], 
            current_world_state.ego.position_m[1], 
            current_world_state.ego.yaw_rad
        )
        current_agents_states = []
        for agent in current_world_state.agents:
            current_agents_states.append((
                agent.position_m[0], agent.position_m[1],
                agent.velocity_mps[0], agent.velocity_mps[1],
                agent.heading_rad, agent.agent_type.value
            ))
        # 重要：必须按agent_type排序，与存储时保持一致
        current_agents_states = sorted(current_agents_states, key=lambda x: x[5])
        
        agent_transition_samples = {}
        
        for i, agent in enumerate(current_world_state.agents):
            agent_id = i + 1
            
            config = get_global_config()
            reachable_sets = grid.multi_timestep_successor_cells(
                agent, horizon=horizon, dt=config.time.dt,
                n_samples=config.sampling.reachable_set_samples
            )
            if not reachable_sets:
                continue

            # CRITICAL FIX: Only initialize if agent doesn't exist in bank
            # This preserves accumulated alpha values across Q-value calculations
            if agent_id not in bank.agent_alphas:
                bank.init_agent(agent_id, reachable_sets)
                if config.visualization.verbose_level >= 2:
                    print(f"  [Bank Init] Agent {agent_id} initialized with {len(reachable_sets)} timesteps")
            # If agent already exists, keep existing alpha values (don't reset!)
            
            # 从全局配置读取匹配阈值
            config = get_global_config()
            historical_transitions_by_timestep = trajectory_buffer.get_agent_historical_transitions_strict_matching(
                agent_id=agent_id,
                current_ego_state=current_ego_state,
                current_agents_states=current_agents_states,
                ego_action_trajectory=ego_action_trajectory,
                ego_state_threshold=config.matching.ego_state_threshold,
                agents_state_threshold=config.matching.agents_state_threshold,
                ego_action_threshold=config.matching.ego_action_threshold,
                debug=(config.visualization.verbose_level >= 2)  # Enable debug logging when verbose
            )

            # Log matching statistics
            total_matched = sum(len(cells) for cells in historical_transitions_by_timestep.values())
            matched_timesteps = len(historical_transitions_by_timestep)

            # Update Dirichlet bank with matched historical data
            update_count = 0
            for timestep, historical_cells in historical_transitions_by_timestep.items():
                if len(historical_cells) > 0 and timestep in reachable_sets:
                    bank.update_with_softcount(
                        agent_id, timestep, historical_cells,
                        lr=self.config.learning_rate
                    )
                    update_count += 1

            # Log update statistics with matching density (only if verbose)
            if get_global_config().visualization.verbose_level >= 1:
                buffer_size = len(trajectory_buffer)
                # Calculate matching density
                total_reachable = sum(len(cells) for cells in reachable_sets.values())
                match_density = (total_matched / total_reachable * 100) if total_reachable > 0 else 0.0
                print(f"  [Agent {agent_id}] Matched {total_matched}/{total_reachable} cells ({match_density:.1f}%) "
                      f"across {matched_timesteps} timesteps, Updated {update_count} timesteps, Buffer size: {buffer_size}")
            
            transition_distributions = bank.sample_transition_distributions(
                agent_id, n_samples=self.config.n_samples
            )

            # CRITICAL FIX: Use bank's stored reachable_sets to match distribution dimensions
            # distributions are sampled from bank.agent_alphas (using bank's dimensions)
            # so reachable_sets must also come from bank to ensure dimension consistency
            agent_transition_samples[agent_id] = {
                'distributions': transition_distributions,
                'reachable_sets': bank.agent_reachable_sets[agent_id]  # Use bank's stored sets
            }
        
        return agent_transition_samples
    
    def _extract_sample_distributions(self, agent_transition_samples, sample_idx):
        """提取指定样本的transition分布"""
        sample_distributions = {}
        
        for agent_id, transition_info in agent_transition_samples.items():
            distributions = transition_info['distributions']
            reachable_sets = transition_info['reachable_sets']
            
            agent_distributions = {}
            for timestep in distributions:
                if sample_idx < len(distributions[timestep]):
                    reachable_cells = reachable_sets[timestep]
                    probabilities = distributions[timestep][sample_idx]
                    agent_distributions[timestep] = (reachable_cells, probabilities)
            
            sample_distributions[agent_id] = agent_distributions
        
        return sample_distributions
