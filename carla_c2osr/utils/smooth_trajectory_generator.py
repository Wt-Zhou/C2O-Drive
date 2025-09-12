"""
平滑轨迹生成器

提供更简单、更平滑的智能体轨迹生成功能。
"""

from __future__ import annotations
import numpy as np
import math
from typing import List, Tuple
from carla_c2osr.env.types import AgentState, AgentType, AgentDynamicsParams


class SmoothTrajectoryGenerator:
    """平滑轨迹生成器"""
    
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
        """为智能体生成符合动力学约束的平滑轨迹。
        
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
        
        # 初始化目标朝向（与初始速度方向一致）
        target_heading = current_heading
        
        for t in range(horizon):
            if agent.agent_type == AgentType.PEDESTRIAN:
                # 行人：简单的随机游走，但更平滑
                if t % 5 == 0:  # 每5秒改变一次目标方向
                    # 在当前朝向基础上小幅调整
                    heading_change = np.random.uniform(-np.pi/6, np.pi/6)  # ±30度
                    target_heading += heading_change
                
                # 平滑朝向调整
                heading_diff = target_heading - current_heading
                # 标准化角度差
                while heading_diff > np.pi:
                    heading_diff -= 2 * np.pi
                while heading_diff < -np.pi:
                    heading_diff += 2 * np.pi
                
                # 平滑调整朝向（每次最多调整0.1弧度，约6度）
                current_heading += np.clip(heading_diff, -0.1, 0.1)
                
                # 速度变化
                if t % 3 == 0:
                    speed_change = np.random.uniform(-0.3, 0.3)
                    current_speed = np.clip(current_speed + speed_change, 0.2, dynamics.max_speed_mps)
                
                # 计算下一位置
                next_x = current_pos[0] + current_speed * math.cos(current_heading) * dt
                next_y = current_pos[1] + current_speed * math.sin(current_heading) * dt
                
            else:  # 车辆类型
                # 车辆：非常平滑的运动，几乎直线行驶
                if t % 15 == 0:  # 每15秒才轻微调整（非常少）
                    # 非常轻微的转向
                    heading_change = np.random.uniform(-np.pi/36, np.pi/36)  # ±5度
                    target_heading += heading_change
                
                # 平滑朝向调整
                heading_diff = target_heading - current_heading
                # 标准化角度差
                while heading_diff > np.pi:
                    heading_diff -= 2 * np.pi
                while heading_diff < -np.pi:
                    heading_diff += 2 * np.pi
                
                # 非常平滑的朝向调整（每次最多调整0.02弧度，约1度）
                current_heading += np.clip(heading_diff, -0.02, 0.02)
                
                # 速度保持相对稳定
                if t % 10 == 0:
                    speed_change = np.random.uniform(-0.1, 0.2)
                    current_speed = np.clip(current_speed + speed_change, 0.8, dynamics.max_speed_mps)
                
                # 计算下一位置
                next_x = current_pos[0] + current_speed * math.cos(current_heading) * dt
                next_y = current_pos[1] + current_speed * math.sin(current_heading) * dt
            
            # 边界处理：如果接近边界，逐渐调整朝向
            boundary_margin = 3.0
            if next_x <= min_bound + boundary_margin or next_x >= max_bound - boundary_margin:
                # 朝向中心方向
                center_x = (min_bound + max_bound) / 2
                target_heading = math.atan2(center_x - next_x, 0.1)  # 避免除零
                heading_diff = target_heading - current_heading
                while heading_diff > np.pi:
                    heading_diff -= 2 * np.pi
                while heading_diff < -np.pi:
                    heading_diff += 2 * np.pi
                current_heading += np.clip(heading_diff, -0.05, 0.05)
            
            if next_y <= min_bound + boundary_margin or next_y >= max_bound - boundary_margin:
                # 朝向中心方向
                center_y = (min_bound + max_bound) / 2
                target_heading = math.atan2(0.1, center_y - next_y)  # 避免除零
                heading_diff = target_heading - current_heading
                while heading_diff > np.pi:
                    heading_diff -= 2 * np.pi
                while heading_diff < -np.pi:
                    heading_diff += 2 * np.pi
                current_heading += np.clip(heading_diff, -0.05, 0.05)
            
            # 确保位置在边界内
            next_x = np.clip(next_x, min_bound, max_bound)
            next_y = np.clip(next_y, min_bound, max_bound)
            
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

