from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
from enum import Enum


Vector2 = Tuple[float, float]


class AgentType(Enum):
    """智能体类型枚举。"""
    PEDESTRIAN = "pedestrian"
    VEHICLE = "vehicle"
    BICYCLE = "bicycle"
    MOTORCYCLE = "motorcycle"
    OBSTACLE = "obstacle"  # 静态障碍物（锥桶、箱子等）


@dataclass
class AgentDynamicsParams:
    """智能体动力学参数。
    
    根据不同类型提供相应的运动约束参数。
    """
    # 通用参数
    max_speed_mps: float          # 最大速度 m/s
    max_accel_mps2: float         # 最大加速度 m/s²
    max_decel_mps2: float         # 最大减速度 m/s² (正值)
    
    # 转向参数
    max_yaw_rate_rps: float       # 最大偏航角速度 rad/s
    wheelbase_m: float            # 轴距 m (车辆类型用于自行车模型)
    
    # 尺寸参数 (用于碰撞检测)
    length_m: float               # 长度 m
    width_m: float                # 宽度 m
    
    @classmethod
    def for_agent_type(cls, agent_type: AgentType) -> 'AgentDynamicsParams':
        """根据智能体类型返回默认动力学参数。"""
        if agent_type == AgentType.PEDESTRIAN:
            return cls(
                max_speed_mps=2.0,       # 步行速度
                max_accel_mps2=1.0,      # 行人加速度
                max_decel_mps2=2.0,      # 行人减速度
                max_yaw_rate_rps=3.14,   # 行人可快速转向
                wheelbase_m=0.0,         # 行人无轴距
                length_m=0.6,            # 行人肩宽
                width_m=0.4              # 行人厚度
            )
        elif agent_type == AgentType.BICYCLE:
            return cls(
                max_speed_mps=8.0,       # 自行车速度
                max_accel_mps2=2.0,      # 自行车加速度
                max_decel_mps2=4.0,      # 自行车刹车
                max_yaw_rate_rps=1.57,   # 自行车转向
                wheelbase_m=1.1,         # 自行车轴距
                length_m=1.8,            # 自行车长度
                width_m=0.6              # 自行车宽度
            )
        elif agent_type == AgentType.MOTORCYCLE:
            return cls(
                max_speed_mps=20.0,      # 摩托车速度
                max_accel_mps2=4.0,      # 摩托车加速度
                max_decel_mps2=8.0,      # 摩托车刹车
                max_yaw_rate_rps=1.0,    # 摩托车转向
                wheelbase_m=1.4,         # 摩托车轴距
                length_m=2.2,            # 摩托车长度
                width_m=0.8              # 摩托车宽度
            )
        elif agent_type == AgentType.VEHICLE:
            return cls(
                max_speed_mps=15.0,      # 城市车辆速度
                max_accel_mps2=3.0,      # 车辆加速度
                max_decel_mps2=6.0,      # 车辆刹车
                max_yaw_rate_rps=0.5,    # 车辆转向 (受阿克曼约束)
                wheelbase_m=2.7,         # 典型轿车轴距
                length_m=4.5,            # 车辆长度
                width_m=1.8              # 车辆宽度
            )
        elif agent_type == AgentType.OBSTACLE:
            return cls(
                max_speed_mps=0.0,       # 静态障碍物不移动
                max_accel_mps2=0.0,
                max_decel_mps2=0.0,
                max_yaw_rate_rps=0.0,
                wheelbase_m=0.0,
                length_m=0.5,            # 锥桶/箱子尺寸
                width_m=0.5
            )
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")


@dataclass
class AgentState:
    """环境中其他交通参与者状态。

    Attributes:
        agent_id: 参与者唯一标识。
        position_m: (x, y) 位置，单位米。
        velocity_mps: (vx, vy) 速度，单位米/秒。
        heading_rad: 航向角，单位弧度。
        agent_type: 参与者类型枚举。
    """
    agent_id: str
    position_m: Vector2
    velocity_mps: Vector2
    heading_rad: float
    agent_type: AgentType
    
@dataclass
class EgoState:
    """自车状态。"""
    position_m: Vector2
    velocity_mps: Vector2
    yaw_rad: float


@dataclass
class EgoControl:
    """自车控制。值域限制在 [0,1] 或 [-1,1]，具体由下游 wrapper 处理。"""
    throttle: float
    steer: float
    brake: float


@dataclass
class WorldState:
    """世界状态快照。"""
    time_s: float
    ego: EgoState
    agents: List[AgentState]


@dataclass
class Trajectory:
    """自车规划轨迹（占位）。"""
    states: List[EgoState]
    controls: List[EgoControl]


@dataclass
class OccupancyGrid:
    """占据概率栅格（局部）。"""
    width_cells: int
    height_cells: int
    cell_size_m: float
    origin_m: Vector2
    data: List[float]  # 行优先扁平化，范围 [0,1]
