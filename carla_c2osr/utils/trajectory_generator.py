"""
轨迹生成模块

提供智能体轨迹生成和动力学约束功能。
"""

from __future__ import annotations
import numpy as np
import math
from typing import List, Tuple
from carla_c2osr.env.types import AgentState, AgentType, AgentDynamicsParams


class TrajectoryGenerator:
    """轨迹生成器"""
    
    def __init__(self, grid_bounds: Tuple[float, float] = (-9.0, 9.0)):
        """
        Args:
            grid_bounds: 网格边界，防止智能体移动到网格外
        """
        self.grid_bounds = grid_bounds
    
    def generate_agent_trajectory(self, 
                                 agent: AgentState, 
                                 horizon: int, 
                                 dt: float = 1.0) -> List[np.ndarray]:
        """为智能体生成符合动力学约束的固定轨迹。
        
        Args:
            agent: 智能体状态
            horizon: 轨迹长度（秒）
            dt: 时间步长
            
        Returns:
            轨迹列表，每个元素为世界坐标np.ndarray([x, y])
        """
        trajectory = []
        current_pos = np.array(agent.position_m, dtype=float)
        current_vel = np.array(agent.velocity_mps, dtype=float)
        current_heading = agent.heading_rad
        current_speed = math.sqrt(current_vel[0]**2 + current_vel[1]**2)
        
        # 获取动力学参数
        dynamics = AgentDynamicsParams.for_agent_type(agent.agent_type)
        min_bound, max_bound = self.grid_bounds
        
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
                # 使用更平滑的转向策略
                # if t % 12 == 0:  # 每12秒轻微调整（进一步减少频率）
                #     # 非常轻微的转向（阿克曼约束）
                #     max_steer = math.atan(dynamics.max_yaw_rate_rps * dynamics.wheelbase_m / max(current_speed, 0.1))
                #     max_steer = min(max_steer, math.pi / 48)  # 最大3.75度（进一步减小）
                #     steer_change = np.random.uniform(-max_steer, max_steer)
                #     yaw_rate = current_speed * math.tan(steer_change) / dynamics.wheelbase_m
                #     current_heading += yaw_rate * dt
                    
                #     # 速度调整 - 更小的加速度变化
                #     accel_change = np.random.uniform(-0.2, 0.5)  # 进一步减小加速度范围
                #     current_speed = np.clip(current_speed + accel_change * dt, 0.5, dynamics.max_speed_mps)
                
                # 计算下一位置
                next_x = current_pos[0] + current_speed * math.cos(current_heading) * dt
                next_y = current_pos[1] + current_speed * math.sin(current_heading) * dt
            
            # 边界检查：确保智能体不会移动到网格外
            next_x = np.clip(next_x, min_bound, max_bound)
            next_y = np.clip(next_y, min_bound, max_bound)
            
            # 更平滑的边界处理：使用更大的缓冲区和更小的调整步长
            boundary_margin = 4.0  # 增大边界缓冲区
            max_heading_adjustment = 0.05  # 减小最大调整步长（约3度）
            
            if next_x <= min_bound + boundary_margin or next_x >= max_bound - boundary_margin:
                # 逐渐调整朝向，避免突然反弹
                target_heading = math.pi - current_heading if next_x <= min_bound + boundary_margin else -current_heading
                heading_diff = target_heading - current_heading
                # 标准化角度差到[-pi, pi]
                while heading_diff > math.pi:
                    heading_diff -= 2 * math.pi
                while heading_diff < -math.pi:
                    heading_diff += 2 * math.pi
                # 更平滑的调整
                current_heading += np.clip(heading_diff, -max_heading_adjustment, max_heading_adjustment)
            
            if next_y <= min_bound + boundary_margin or next_y >= max_bound - boundary_margin:
                # 逐渐调整朝向，避免突然反弹
                target_heading = -current_heading if next_y <= min_bound + boundary_margin else math.pi - current_heading
                heading_diff = target_heading - current_heading
                # 标准化角度差到[-pi, pi]
                while heading_diff > math.pi:
                    heading_diff -= 2 * math.pi
                while heading_diff < -math.pi:
                    heading_diff += 2 * math.pi
                # 更平滑的调整
                current_heading += np.clip(heading_diff, -max_heading_adjustment, max_heading_adjustment)
            
            current_pos = np.array([next_x, next_y])
            trajectory.append(current_pos.copy())
        
        return trajectory
    
    def generate_ego_trajectory(self, 
                               ego_mode: str, 
                               horizon: int, 
                               ego_speed: float = 5.0) -> List[np.ndarray]:
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
