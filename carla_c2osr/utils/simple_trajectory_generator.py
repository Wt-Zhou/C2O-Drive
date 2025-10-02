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
                  - "dynamic": 随机动力学模型（默认）
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
        elif mode == "dynamic":
            return self._generate_dynamic_trajectory(agent, horizon, dt)
        else:
            raise ValueError(f"Unknown trajectory mode: {mode}")

    def _generate_dynamic_trajectory(self, agent: AgentState, horizon: int, dt: float) -> List[np.ndarray]:
        """生成动态轨迹（原有的随机动力学模型）。

        Args:
            agent: 智能体状态
            horizon: 轨迹长度
            dt: 时间步长

        Returns:
            轨迹位置列表
        """
        # 获取动力学参数
        dynamics = AgentDynamicsParams.for_agent_type(agent.agent_type)
        
        trajectory = []
        current_pos = np.array(agent.position_m, dtype=float)
        current_vel = np.array(agent.velocity_mps, dtype=float)
        current_heading = agent.heading_rad
        current_speed = math.sqrt(current_vel[0]**2 + current_vel[1]**2)
        
        min_bound, max_bound = self.grid_bounds
        
        for t in range(horizon):
            if agent.agent_type == AgentType.PEDESTRIAN:
                # 行人：随机游走，偶尔改变方向
                if t % 4 == 0:
                    angle_change = np.random.uniform(-np.pi/3, np.pi/3)  # ±60度
                    current_heading += angle_change
                
                # 速度变化：偶尔加速或减速
                if t % 4 == 0:
                    speed_change = np.random.uniform(-0.2, 0.2)
                    current_speed = np.clip(current_speed + speed_change, 0.3, dynamics.max_speed_mps)
                
                # 计算下一位置
                next_x = current_pos[0] + current_speed * math.cos(current_heading) * dt
                next_y = current_pos[1] + current_speed * math.sin(current_heading) * dt
                
            else:  # 车辆类型
                # 复用车辆动力学模型逻辑
                next_pos, next_speed, next_heading = self._vehicle_dynamics_step(
                    current_pos, current_speed, current_heading, dynamics, dt, t
                )
                next_x, next_y = next_pos
                current_speed = next_speed
                current_heading = next_heading
            
            # 边界处理：如果超出边界，停止移动
            if next_x < min_bound or next_x > max_bound or next_y < min_bound or next_y > max_bound:
                # 保持在当前位置
                next_x = current_pos[0]
                next_y = current_pos[1]
            
            current_pos = np.array([next_x, next_y])
            trajectory.append(current_pos.copy())

        return trajectory

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

    def _vehicle_dynamics_step(self, pos: np.ndarray, speed: float, heading: float,
                              dynamics: AgentDynamicsParams, dt: float, timestep: int) -> Tuple[np.ndarray, float, float]:
        """车辆动力学一步积分，复用现有的自行车模型逻辑。
        
        Args:
            pos: 当前位置
            speed: 当前速度
            heading: 当前朝向
            dynamics: 动力学参数
            dt: 时间步长
            timestep: 当前时间步
            
        Returns:
            (下一位置, 下一速度, 下一朝向)
        """
        # 添加方向扰动（每6步轻微调整）
        if timestep % 6 == 0:
            # 根据阿克曼约束计算最大转向角
            if dynamics.wheelbase_m > 0:
                # 根据轴距和最大偏航角速度计算最大转向角
                max_steer_rad = math.atan(dynamics.max_yaw_rate_rps * dynamics.wheelbase_m / max(speed, 0.1))
                max_steer_rad = min(max_steer_rad, math.pi / 12)  # 限制最大 15 度
            else:
                max_steer_rad = dynamics.max_yaw_rate_rps * dt
            
            # 添加轻微的随机转向扰动
            steer_angle = np.random.uniform(-max_steer_rad * 0.3, max_steer_rad * 0.3)
            
            # 自行车模型计算偏航角速度
            if dynamics.wheelbase_m > 0:
                yaw_rate = speed * math.tan(steer_angle) / dynamics.wheelbase_m
            else:
                yaw_rate = steer_angle / dt
            
            yaw_rate = np.clip(yaw_rate, -dynamics.max_yaw_rate_rps, dynamics.max_yaw_rate_rps)
            heading += yaw_rate * dt
        else:
            yaw_rate = 0.0
        
        # 速度调整（每6步轻微调整）
        if timestep % 6 == 0:
            # 偏向保持当前速度，偶尔轻微调整
            if np.random.random() < 0.7:  # 70% 概率保持或轻微加速
                accel = np.random.uniform(-0.5, dynamics.max_accel_mps2 * 0.2)  # 减少加速度范围
            else:  # 30% 概率减速
                accel = np.random.uniform(-dynamics.max_decel_mps2 * 0.2, 0)  # 减少减速度范围
            
            # 确保加速度不超过限制
            accel = np.clip(accel, -dynamics.max_decel_mps2, dynamics.max_accel_mps2)
            speed = max(0, speed + accel * dt)
            speed = min(speed, dynamics.max_speed_mps)
        
        # 位置积分（使用平均航向角，提高精度）
        avg_heading = heading - 0.5 * yaw_rate * dt  # 使用半步前的航向角
        next_x = pos[0] + speed * math.cos(avg_heading) * dt
        next_y = pos[1] + speed * math.sin(avg_heading) * dt
        
        return np.array([next_x, next_y]), speed, heading
    
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
