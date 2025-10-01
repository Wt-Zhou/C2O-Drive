"""
Q值计算模块 - 重新设计版本

基于历史转移数据和Dirichlet分布的Q值计算系统。
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from carla_c2osr.env.types import EgoState, AgentState, WorldState, AgentType
from carla_c2osr.agents.c2osr.grid import GridMapper
from carla_c2osr.agents.c2osr.spatial_dirichlet import SpatialDirichletBank, DirichletParams, MultiTimestepSpatialDirichletBank, OptimizedMultiTimestepSpatialDirichletBank
from carla_c2osr.agents.c2osr.trajectory_buffer import TrajectoryBuffer
from carla_c2osr.config import get_global_config
from carla_c2osr.evaluation.collision_detector import CollisionDetector


@dataclass
class RewardConfig:
    """奖励函数配置"""
    # 碰撞相关
    collision_penalty: float = -100.0
    collision_threshold: float = 0.1  # 碰撞概率阈值
    
    # 舒适性相关
    acceleration_penalty_weight: float = 0.1
    jerk_penalty_weight: float = 0.05
    max_comfortable_accel: float = 2.0  # m/s²
    
    # 驾驶效率相关
    speed_reward_weight: float = 1.0
    target_speed: float = 5.0  # m/s
    progress_reward_weight: float = 2.0
    
    # 安全距离相关
    safe_distance: float = 3.0  # m
    distance_penalty_weight: float = 2.0


@dataclass
class QValueConfig:
    """Q值计算配置"""
    horizon: int = 8  # 预测时间步长
    n_samples: int = 10  # Dirichlet采样数量
    dirichlet_alpha_in: float = 50.0  # 可达集内的先验强度
    dirichlet_alpha_out: float = 1e-6  # 可达集外的先验强度
    learning_rate: float = 1.0  # 历史数据更新学习率
    
    @classmethod
    def from_global_config(cls):
        """从全局配置创建Q值配置"""
        global_config = get_global_config()
        return cls(
            horizon=global_config.time.default_horizon,
            n_samples=global_config.sampling.q_value_samples,
            dirichlet_alpha_in=global_config.dirichlet.alpha_in,
            dirichlet_alpha_out=global_config.dirichlet.alpha_out,
            learning_rate=global_config.dirichlet.learning_rate
        )


class RewardCalculator:
    """模块化奖励计算器"""
    
    def __init__(self, config: RewardConfig):
        self.config = config
    
    def calculate_collision_reward(self, collision_occurred: bool) -> float:
        """计算碰撞奖励"""
        if collision_occurred:
            return self.config.collision_penalty
        return 0.0
    
    def calculate_comfort_reward(self, ego_trajectory: List[Tuple[float, float]], dt: float = 1.0) -> float:
        """计算舒适性奖励（基于加速度和急动）"""
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
    
    def calculate_efficiency_reward(self, ego_trajectory: List[Tuple[float, float]], dt: float = 1.0) -> float:
        """计算驾驶效率奖励（速度和前进距离）"""
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
    
    def calculate_total_reward(self, ego_trajectory: List[Tuple[float, float]],
                             agent_trajectories: Dict[int, List[Tuple[float, float]]],
                             collision_occurred: bool) -> float:
        """计算总奖励"""
        collision_reward = self.calculate_collision_reward(collision_occurred)
        
        # 如果发生碰撞，直接返回碰撞惩罚
        if collision_occurred:
            return collision_reward
        
        comfort_reward = self.calculate_comfort_reward(ego_trajectory)
        efficiency_reward = self.calculate_efficiency_reward(ego_trajectory)
        safety_reward = self.calculate_safety_reward(ego_trajectory, agent_trajectories)
        
        return collision_reward + comfort_reward + efficiency_reward + safety_reward


class QValueCalculator:
    """终极优化的Q值计算器 - 完全消除采样，纯期望计算"""
    
    def __init__(self, config: QValueConfig, reward_config: RewardConfig):
        self.config = config
        self.reward_config = reward_config
        self.collision_detector = CollisionDetector()
        
        # 创建Dirichlet参数
        self.dirichlet_params = DirichletParams(
            alpha_in=config.dirichlet_alpha_in,
            alpha_out=config.dirichlet_alpha_out,
            delta=0.05,  # 95% 置信度
            cK=1.0
        )
    
    def compute_q_value(self, 
                       current_world_state: WorldState,
                       ego_action_trajectory: List[Tuple[float, float]],
                       trajectory_buffer: TrajectoryBuffer,
                       grid: GridMapper,
                       bank: Optional[MultiTimestepSpatialDirichletBank] = None,
                       rng: Optional[np.random.Generator] = None) -> Tuple[List[float], Dict]:
        """终极优化的Q值计算 - 完全消除采样，纯期望计算
        
        Args:
            current_world_state: 当前世界状态
            ego_action_trajectory: 自车动作序列（未来horizon个位置）
            trajectory_buffer: 历史轨迹缓冲区
            grid: 网格映射器
            bank: 持久化的Dirichlet Bank（如果为None则创建临时Bank）
            rng: 随机数生成器（为了兼容性保留）
            
        Returns:
            (所有Q值列表, 详细信息字典)
        """
        if rng is None:
            rng = np.random.default_rng()
        
        horizon = len(ego_action_trajectory)
        
        # 使用终极优化版本的Bank
        if bank is None or not isinstance(bank, OptimizedMultiTimestepSpatialDirichletBank):
            # 创建优化版本的Bank
            optimized_bank = OptimizedMultiTimestepSpatialDirichletBank(
                grid.spec.num_cells, 
                self.dirichlet_params, 
                horizon=horizon
            )
        else:
            optimized_bank = bank
        
        print(f"  🚀 === 终极优化Q值计算开始 ===")
        
        # 第1步：计算与agent完全无关的奖励（只计算一次！）
        print(f"  📊 第1步: 计算agent无关奖励（固定值）")
        agent_independent_reward = self._calculate_agent_independent_rewards(ego_action_trajectory)
        print(f"    ✅ Agent无关奖励: {agent_independent_reward:.3f}")
        
        # 第2步：建立agent的transition分布
        print(f"  📈 第2步: 建立agent transition分布")
        agent_transition_samples = self._build_agent_transition_distributions(
            current_world_state, ego_action_trajectory, trajectory_buffer, grid, optimized_bank, horizon
        )
        
        # 第3步：直接计算期望的agent相关奖励（无采样！）
        print(f"  🎯 第3步: 直接计算期望agent相关奖励（零采样）")
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
            
            # if sample_idx < 3:  # 只打印前几个样本
            #     print(f"    样本{sample_idx+1}: 碰撞={expected_collision_reward:.3f}, "
            #           f"安全={expected_safety_reward:.3f}, 总Q={final_q_value:.3f}")
        
        # 计算平均期望碰撞概率作为碰撞率
        mean_collision_probability = np.mean(collision_probabilities) if collision_probabilities else 0.0
        
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
                'collision_rate': mean_collision_probability,  # 使用期望碰撞概率
                'all_q_values': q_values,
                'collision_probabilities': collision_probabilities  # 保存所有样本的碰撞概率
            },
            'agent_info': {}
        }
        
        print(f"  🎉 === Q值计算完成 ===")
        print(f"  方法: {detailed_info['calculation_method']}")
        print(f"  基础奖励: {agent_independent_reward:.3f} (固定)")
        print(f"  可变奖励范围: [{np.min(detailed_info['agent_dependent_rewards']):.3f}, "
              f"{np.max(detailed_info['agent_dependent_rewards']):.3f}]")
        print(f"  最终Q值: 均值={np.mean(q_values):.3f}, 标准差={np.std(q_values):.3f}")
        print(f"  期望碰撞概率: {mean_collision_probability:.3f}")
        print(f"  计算优化: {detailed_info['computational_savings']}")
        
        return q_values, detailed_info
    
    def _calculate_agent_independent_rewards(self, ego_trajectory: List[Tuple[float, float]]) -> float:
        """计算与agent完全无关的奖励"""
        total_reward = 0.0
        dt = 1.0  # 时间步长
        
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
        
        return total_reward
    
    def _calculate_expected_collision_reward_directly(self,
                                                    ego_trajectory: List[Tuple[float, float]],
                                                    agent_distributions: Dict[int, Dict[int, Tuple[List[int], np.ndarray]]],
                                                    grid: GridMapper,
                                                    current_world_state: WorldState) -> Tuple[float, float]:
        """直接计算期望碰撞奖励和期望碰撞概率 - 使用精确车辆形状碰撞检测！
        
        Returns:
            (expected_reward, expected_collision_probability)
        """
        expected_reward = 0.0
        expected_collision_prob = 0.0  # 期望碰撞概率
        collision_count = 0  # 调试：碰撞计数
        
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
                                
                                # 调试：打印碰撞信息（仅前几个）
                                if timestep < 2 and cell_idx < 3:
                                    print(f"    🚨 精确碰撞检测: t{timestep} agent{agent_id} 位置={world_pos}, 概率={prob:.3f}, 惩罚={collision_contribution:.3f}")
        
        # 调试：打印总结信息
        if collision_count > 0:
            print(f"  🚨 总精确碰撞检测: {collision_count}次, 期望惩罚={expected_reward:.3f}, 期望碰撞概率={expected_collision_prob:.3f}")
        else:
            print(f"  ✅ 无精确碰撞检测, 期望惩罚={expected_reward:.3f}, 期望碰撞概率={expected_collision_prob:.3f}")
        
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
            
            bank.init_agent(agent_id, reachable_sets)
            
            historical_transitions_by_timestep = trajectory_buffer.get_agent_historical_transitions_strict_matching(
                agent_id=agent_id,
                current_ego_state=current_ego_state,
                current_agents_states=current_agents_states,
                ego_action_trajectory=ego_action_trajectory,
                ego_state_threshold=5.0,
                agents_state_threshold=5.0,
                ego_action_threshold=5.0
            )
            
            for timestep, historical_cells in historical_transitions_by_timestep.items():
                if len(historical_cells) > 0 and timestep in reachable_sets:
                    bank.update_with_softcount(
                        agent_id, timestep, historical_cells, 
                        lr=self.config.learning_rate
                    )
            
            transition_distributions = bank.sample_transition_distributions(
                agent_id, n_samples=self.config.n_samples
            )
            agent_transition_samples[agent_id] = {
                'distributions': transition_distributions,
                'reachable_sets': reachable_sets
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
