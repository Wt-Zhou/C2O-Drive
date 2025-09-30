"""
精确的车辆形状碰撞检测模块
"""

import numpy as np
import math
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from carla_c2osr.env.types import AgentType, AgentDynamicsParams


@dataclass
class VehicleShape:
    """车辆形状定义"""
    center: Tuple[float, float]  # 中心位置
    length: float               # 长度
    width: float                # 宽度
    heading: float              # 朝向角度 (弧度)
    
    def get_corners(self) -> List[Tuple[float, float]]:
        """获取车辆四个角点的世界坐标"""
        half_length = self.length / 2
        half_width = self.width / 2
        
        # 在车辆本地坐标系中的四个角点
        local_corners = [
            (-half_length, -half_width),  # 后左
            (-half_length, half_width),   # 后右
            (half_length, half_width),    # 前右
            (half_length, -half_width)    # 前左
        ]
        
        # 转换到世界坐标系
        cos_h = math.cos(self.heading)
        sin_h = math.sin(self.heading)
        
        world_corners = []
        for local_x, local_y in local_corners:
            # 旋转
            world_x = cos_h * local_x - sin_h * local_y
            world_y = sin_h * local_x + cos_h * local_y
            
            # 平移
            world_x += self.center[0]
            world_y += self.center[1]
            
            world_corners.append((world_x, world_y))
        
        return world_corners


class CollisionDetector:
    """精确的碰撞检测器"""
    
    def __init__(self):
        """初始化碰撞检测器"""
        pass
    
    def check_trajectory_collision(self, 
                                 ego_trajectory: List[Tuple[float, float]],
                                 ego_headings: List[float],
                                 agent_trajectories: Dict[int, List[Tuple[float, float]]],
                                 agent_headings: Dict[int, List[float]],
                                 agent_types: Dict[int, AgentType],
                                 ego_type: AgentType = AgentType.VEHICLE) -> bool:
        """
        检查轨迹是否发生碰撞
        
        Args:
            ego_trajectory: 自车轨迹 [(x, y), ...]
            ego_headings: 自车朝向 [heading_rad, ...]
            agent_trajectories: 其他车辆轨迹 {agent_id: [(x, y), ...]}
            agent_headings: 其他车辆朝向 {agent_id: [heading_rad, ...]}
            agent_types: 车辆类型 {agent_id: AgentType}
            ego_type: 自车类型
            
        Returns:
            bool: 是否发生碰撞
        """
        if not agent_trajectories:
            return False
        
        # 获取自车动力学参数
        ego_dynamics = AgentDynamicsParams.for_agent_type(ego_type)
        
        # 确定检查的时间步数
        max_length = len(ego_trajectory)
        if agent_trajectories:
            max_length = min(max_length, 
                           min(len(traj) for traj in agent_trajectories.values()))
        
        # 逐时间步检查碰撞
        for t in range(max_length):
            if t >= len(ego_headings):
                continue
                
            # 创建自车形状
            ego_shape = VehicleShape(
                center=ego_trajectory[t],
                length=ego_dynamics.length_m,
                width=ego_dynamics.width_m,
                heading=ego_headings[t]
            )
            
            # 检查与每个agent的碰撞
            for agent_id, agent_traj in agent_trajectories.items():
                if t >= len(agent_traj) or agent_id not in agent_types:
                    continue
                
                if t >= len(agent_headings.get(agent_id, [])):
                    continue
                
                # 获取agent动力学参数
                agent_dynamics = AgentDynamicsParams.for_agent_type(agent_types[agent_id])
                
                # 创建agent形状
                agent_shape = VehicleShape(
                    center=agent_traj[t],
                    length=agent_dynamics.length_m,
                    width=agent_dynamics.width_m,
                    heading=agent_headings[agent_id][t]
                )
                
                # 检查两个形状是否碰撞
                if self._check_obb_collision(ego_shape, agent_shape):
                    return True
        
        return False
    
    def _check_obb_collision(self, shape1: VehicleShape, shape2: VehicleShape) -> bool:
        """
        使用分离轴定理(SAT)检查两个有向边界框(OBB)是否碰撞
        
        Args:
            shape1: 第一个车辆形状
            shape2: 第二个车辆形状
            
        Returns:
            bool: 是否发生碰撞
        """
        # 获取两个形状的角点
        corners1 = shape1.get_corners()
        corners2 = shape2.get_corners()
        
        # 获取分离轴（每个矩形的两个边的法向量）
        axes = []
        
        # shape1的分离轴
        for i in range(4):
            p1 = np.array(corners1[i])
            p2 = np.array(corners1[(i + 1) % 4])
            edge = p2 - p1
            # 法向量（垂直于边）
            normal = np.array([-edge[1], edge[0]])
            # 归一化
            if np.linalg.norm(normal) > 0:
                normal = normal / np.linalg.norm(normal)
                axes.append(normal)
        
        # shape2的分离轴
        for i in range(4):
            p1 = np.array(corners2[i])
            p2 = np.array(corners2[(i + 1) % 4])
            edge = p2 - p1
            # 法向量（垂直于边）
            normal = np.array([-edge[1], edge[0]])
            # 归一化
            if np.linalg.norm(normal) > 0:
                normal = normal / np.linalg.norm(normal)
                axes.append(normal)
        
        # 对每个分离轴进行投影测试
        for axis in axes:
            # 投影shape1的所有角点到轴上
            proj1 = [np.dot(corner, axis) for corner in corners1]
            min1, max1 = min(proj1), max(proj1)
            
            # 投影shape2的所有角点到轴上
            proj2 = [np.dot(corner, axis) for corner in corners2]
            min2, max2 = min(proj2), max(proj2)
            
            # 检查投影是否有重叠
            if max1 < min2 or max2 < min1:
                # 在这个轴上没有重叠，说明没有碰撞
                return False
        
        # 所有轴上都有重叠，说明发生碰撞
        return True
    
    def check_point_collision(self, 
                            ego_pos: Tuple[float, float],
                            ego_heading: float,
                            agent_pos: Tuple[float, float], 
                            agent_heading: float,
                            ego_type: AgentType = AgentType.VEHICLE,
                            agent_type: AgentType = AgentType.VEHICLE) -> bool:
        """
        检查两个车辆在单个时间点是否碰撞
        
        Args:
            ego_pos: 自车位置
            ego_heading: 自车朝向
            agent_pos: 其他车辆位置
            agent_heading: 其他车辆朝向
            ego_type: 自车类型
            agent_type: 其他车辆类型
            
        Returns:
            bool: 是否发生碰撞
        """
        # 获取动力学参数
        ego_dynamics = AgentDynamicsParams.for_agent_type(ego_type)
        agent_dynamics = AgentDynamicsParams.for_agent_type(agent_type)
        
        # 创建车辆形状
        ego_shape = VehicleShape(
            center=ego_pos,
            length=ego_dynamics.length_m,
            width=ego_dynamics.width_m,
            heading=ego_heading
        )
        
        agent_shape = VehicleShape(
            center=agent_pos,
            length=agent_dynamics.length_m,
            width=agent_dynamics.width_m,
            heading=agent_heading
        )
        
        # 检查碰撞
        return self._check_obb_collision(ego_shape, agent_shape)
