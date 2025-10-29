#!/usr/bin/env python3
"""
简化的轨迹生成器

为智能体生成符合动力学约束的轨迹。
"""

import numpy as np
import math
from typing import List, Tuple, Optional
from carla_c2osr.env.types import AgentState, AgentType, AgentDynamicsParams


class SimpleTrajectoryGenerator:
    """简化的轨迹生成器，生成符合动力学约束的轨迹。"""

    def __init__(self, grid_bounds: Tuple[float, float] = (-9.0, 9.0)):
        """初始化轨迹生成器。

        Args:
            grid_bounds: 网格边界，防止智能体移动到网格外
        """
        self.grid_bounds = grid_bounds

    def generate_agent_trajectory(self, agent: AgentState, horizon: int, dt: Optional[float] = None, mode: Optional[str] = None) -> List[np.ndarray]:
        """生成智能体轨迹。

        Args:
            agent: 智能体状态
            horizon: 轨迹长度
            dt: 时间步长，默认从全局配置读取
            mode: 轨迹生成模式，默认从全局配置读取
                  - "stochastic": 基于意图的随机轨迹（默认）
                  - "straight": 匀速直线运动
                  - "stationary": 静止不动

        Returns:
            轨迹位置列表
        """
        # 从全局配置读取默认参数
        if dt is None or mode is None:
            from carla_c2osr.config import get_global_config
            config = get_global_config()
            if dt is None:
                dt = config.time.dt
            if mode is None:
                mode = config.agent_trajectory.mode

        # 根据模式选择生成方法
        if mode == "stationary":
            return self._generate_stationary_trajectory(agent, horizon)
        elif mode == "straight":
            return self._generate_straight_trajectory(agent, horizon, dt)
        elif mode == "stochastic":
            return self._generate_stochastic_trajectory(agent, horizon, dt)
        else:
            raise ValueError(f"Unknown trajectory mode: {mode}")

    def _generate_stochastic_trajectory(self, agent: AgentState, horizon: int, dt: float) -> List[np.ndarray]:
        """基于意图的随机轨迹生成

        预定义轨迹类型及概率：
        - 加速直行: 40%
        - 减速直行: 60%

        Args:
            agent: 智能体状态
            horizon: 轨迹长度
            dt: 时间步长

        Returns:
            轨迹位置列表
        """
        # 随机选择意图
        intention = np.random.choice(
            ['accelerate', 'decelerate'],
            p=[0.4, 0.6]  # 概率分布：40%加速，60%减速
        )

        if intention == 'accelerate':
            return self._generate_accelerate_trajectory(agent, horizon, dt)
        else:  # decelerate
            return self._generate_decelerate_trajectory(agent, horizon, dt)

    def _generate_accelerate_trajectory(self, agent: AgentState, horizon: int, dt: float) -> List[np.ndarray]:
        """生成加速直行轨迹

        Args:
            agent: 智能体状态
            horizon: 轨迹长度
            dt: 时间步长

        Returns:
            轨迹位置列表
        """
        trajectory = []
        current_pos = np.array(agent.position_m, dtype=float)
        velocity = np.array(agent.velocity_mps, dtype=float)
        velocity_norm = np.linalg.norm(velocity)

        if velocity_norm < 1e-6:
            # 如果初始速度为0，使用默认方向
            velocity = np.array([1.0, 0.0])
            velocity_norm = 1.0

        # 加速参数
        accel_per_step = 0.5  # m/s 每个时间步增加的速度
        max_speed = 8.0  # 最大速度限制 m/s

        for t in range(horizon):
            # 逐步加速：速度系数随时间线性增长
            speed_increase = accel_per_step * t
            current_speed = min(velocity_norm + speed_increase, max_speed)

            # 计算下一位置（保持原方向）
            next_pos = current_pos + (velocity / velocity_norm) * current_speed * dt

            # 边界检查
            if self._is_within_bounds(next_pos):
                current_pos = next_pos

            trajectory.append(current_pos.copy())

        return trajectory

    def _generate_decelerate_trajectory(self, agent: AgentState, horizon: int, dt: float) -> List[np.ndarray]:
        """生成减速直行轨迹

        Args:
            agent: 智能体状态
            horizon: 轨迹长度
            dt: 时间步长

        Returns:
            轨迹位置列表
        """
        trajectory = []
        current_pos = np.array(agent.position_m, dtype=float)
        velocity = np.array(agent.velocity_mps, dtype=float)
        velocity_norm = np.linalg.norm(velocity)

        if velocity_norm < 1e-6:
            # 如果初始速度为0，使用默认方向
            velocity = np.array([1.0, 0.0])
            velocity_norm = 1.0

        # 减速参数
        decel_per_step = 0.3  # m/s 每个时间步减少的速度
        min_speed = 0.3  # 最低速度（不完全停止）

        for t in range(horizon):
            # 逐步减速：速度系数随时间线性降低
            speed_decrease = decel_per_step * t
            current_speed = max(velocity_norm - speed_decrease, min_speed)

            # 计算下一位置（保持原方向）
            next_pos = current_pos + (velocity / velocity_norm) * current_speed * dt

            # 边界检查
            if self._is_within_bounds(next_pos):
                current_pos = next_pos

            trajectory.append(current_pos.copy())

        return trajectory

    def _is_within_bounds(self, pos: np.ndarray) -> bool:
        """检查位置是否在网格边界内

        Args:
            pos: 位置坐标

        Returns:
            是否在边界内
        """
        min_bound, max_bound = self.grid_bounds
        return min_bound <= pos[0] <= max_bound and min_bound <= pos[1] <= max_bound

    def _generate_stationary_trajectory(self, agent: AgentState, horizon: int) -> List[np.ndarray]:
        """生成静止轨迹（agent保持在初始位置）。

        Args:
            agent: 智能体状态
            horizon: 轨迹长度

        Returns:
            轨迹位置列表（所有位置相同）
        """
        trajectory = []
        position = np.array(agent.position_m, dtype=float)
        for _ in range(horizon):
            trajectory.append(position.copy())
        return trajectory

    def _generate_straight_trajectory(self, agent: AgentState, horizon: int, dt: float) -> List[np.ndarray]:
        """生成匀速直线轨迹。

        Args:
            agent: 智能体状态
            horizon: 轨迹长度
            dt: 时间步长

        Returns:
            轨迹位置列表
        """
        trajectory = []
        current_pos = np.array(agent.position_m, dtype=float)
        velocity = np.array(agent.velocity_mps, dtype=float)

        min_bound, max_bound = self.grid_bounds

        for _ in range(horizon):
            # 匀速直线运动
            next_pos = current_pos + velocity * dt

            # 边界处理：如果超出边界，停止移动
            if next_pos[0] < min_bound or next_pos[0] > max_bound or \
               next_pos[1] < min_bound or next_pos[1] > max_bound:
                # 保持在当前位置
                next_pos = current_pos.copy()

            current_pos = next_pos
            trajectory.append(current_pos.copy())

        return trajectory

    
    def generate_ego_trajectory(self, 
                               ego_mode: str, 
                               horizon: int, 
                               ego_speed: float = 3.0) -> List[np.ndarray]:
        """生成自车固定轨迹。
        
        Args:
            ego_mode: 轨迹模式 ("straight", "fixed-traj")
            horizon: 轨迹长度
            ego_speed: 自车速度
            
        Returns:
            轨迹列表
        """
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
    
    def estimate_agent_state_from_trajectory(self, 
                                           trajectory: List[np.ndarray], 
                                           t: int,
                                           agent_init: AgentState) -> AgentState:
        """从轨迹估计智能体在时刻t的状态。
        
        Args:
            trajectory: 轨迹列表
            t: 时刻索引
            agent_init: 初始智能体状态
            
        Returns:
            估计的智能体状态
        """
        if t >= len(trajectory):
            raise ValueError(f"时刻 {t} 超出轨迹长度 {len(trajectory)}")
        
        agent_world_xy = trajectory[t]
        
        # 用轨迹的相邻点估计当前速度与朝向
        horizon = len(trajectory)
        if t < horizon - 1:
            nxt = trajectory[t + 1]
            vel_vec = (nxt - agent_world_xy)
        elif t > 0:
            prv = trajectory[t - 1]
            vel_vec = (agent_world_xy - prv)
        else:
            # 单点退化，使用初始速度
            vel_vec = np.array(agent_init.velocity_mps)
        
        vel_tuple = (float(vel_vec[0]), float(vel_vec[1]))
        heading_est = float(np.arctan2(vel_vec[1], vel_vec[0])) if (vel_vec[0]**2 + vel_vec[1]**2) > 1e-9 else float(agent_init.heading_rad)
        
        # 创建当前智能体状态
        current_agent = AgentState(
            agent_id=agent_init.agent_id,
            position_m=tuple(agent_world_xy),
            velocity_mps=vel_tuple,
            heading_rad=heading_est,
            agent_type=agent_init.agent_type
        )
        
        return current_agent
