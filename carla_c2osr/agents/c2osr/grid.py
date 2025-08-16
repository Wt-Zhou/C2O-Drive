from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List
import numpy as np
import math

from carla_c2osr.env.types import AgentState, WorldState, EgoState, AgentType, AgentDynamicsParams


@dataclass
class GridSpec:
    size_m: float
    cell_m: float
    macro: bool

    @property
    def num_cells(self) -> int:
        side = int(self.size_m / self.cell_m)
        return side * side

    @property
    def side_cells(self) -> int:
        return int(self.size_m / self.cell_m)


class GridMapper:
    """将世界坐标映射到固定世界栅格索引，提供坐标转换与运动学可达性分析。"""

    def __init__(self, spec: GridSpec, world_center: tuple[float, float] = (0.0, 0.0)) -> None:
        self.spec = spec
        self.side = int(spec.size_m / spec.cell_m)
        assert self.side > 0, "Grid side must be positive"
        # 固定的世界坐标中心点
        self.world_center = world_center

    def to_grid_frame(self, xy_world: Tuple[float, float]) -> Tuple[float, float]:
        """将世界坐标转换为固定网格坐标系。
        
        Args:
            xy_world: 世界坐标 (x, y)
            
        Returns:
            网格坐标 (x_grid, y_grid)，网格中心在world_center
        """
        world_x, world_y = xy_world
        center_x, center_y = self.world_center
        
        x_grid = world_x - center_x
        y_grid = world_y - center_y
        
        return (x_grid, y_grid)

    def xy_to_index(self, xy_grid: Tuple[float, float]) -> int:
        """将网格坐标转换为网格索引。
        
        Args:
            xy_grid: 网格坐标 (x, y)
            
        Returns:
            网格单元索引，超出边界时 clamp 到边界
        """
        half = self.spec.size_m / 2.0
        x, y = xy_grid
        
        ix = int((x + half) / self.spec.cell_m)
        iy = int((y + half) / self.spec.cell_m)
        
        ix = max(0, min(self.side - 1, ix))
        iy = max(0, min(self.side - 1, iy))
        
        return iy * self.side + ix

    def index_to_xy_center(self, idx: int) -> Tuple[float, float]:
        """将网格索引转换为单元中心的网格坐标。
        
        Args:
            idx: 网格单元索引
            
        Returns:
            单元中心的网格坐标 (x, y)
        """
        assert 0 <= idx < self.spec.num_cells, f"Index {idx} out of range [0, {self.spec.num_cells})"
        
        iy = idx // self.side
        ix = idx % self.side
        
        half = self.spec.size_m / 2.0
        x = ix * self.spec.cell_m - half + self.spec.cell_m / 2.0
        y = iy * self.spec.cell_m - half + self.spec.cell_m / 2.0
        
        return (x, y)

    def world_to_cell(self, position_m: Tuple[float, float]) -> int:
        """将世界坐标直接映射到网格索引。
        
        Args:
            position_m: 世界坐标 (x, y)
            
        Returns:
            网格单元索引
        """
        grid_xy = self.to_grid_frame(position_m)
        return self.xy_to_index(grid_xy)

    def successor_cells(self, agent: AgentState, dt: float = 1.0, 
                       n_samples: int = 100) -> List[int]:
        """计算智能体在下一时刻的一步可达网格单元集合。
        
        根据智能体类型使用相应的动力学模型：
        - 行人: 质点模型，可任意方向加速
        - 车辆: 自行车模型，考虑阿克曼转向约束
        - 自行车/摩托车: 自行车模型，转向能力更强
        
        Args:
            agent: 智能体当前状态
            dt: 时间步长，默认 1.0 秒
            n_samples: 采样数量，默认 100
            
        Returns:
            可达网格单元索引列表（去重）
        """
        # 获取智能体动力学参数
        dynamics = AgentDynamicsParams.for_agent_type(agent.agent_type)
        
        current_pos = agent.position_m
        current_vel = agent.velocity_mps
        current_heading = agent.heading_rad
        current_speed = math.sqrt(current_vel[0]**2 + current_vel[1]**2)
        
        reachable_indices = set()
        
        if agent.agent_type == AgentType.PEDESTRIAN:
            # 行人模型：质点运动学，可任意方向加速/减速
            reachable_indices.update(
                self._pedestrian_successors(current_pos, current_vel, dynamics, dt, n_samples)
            )
        else:
            # 车辆类型：自行车模型
            reachable_indices.update(
                self._vehicle_successors(current_pos, current_speed, current_heading, dynamics, dt, n_samples)
            )
        
        # 添加当前位置（静止/保持情况）
        current_grid = self.to_grid_frame(current_pos)
        current_idx = self.xy_to_index(current_grid)
        reachable_indices.add(current_idx)
        
        return list(reachable_indices)

    def _pedestrian_successors(self, pos: Tuple[float, float], vel: Tuple[float, float], 
                              dynamics: AgentDynamicsParams, dt: float, 
                              n_samples: int) -> List[int]:
        """行人可达集：质点模型，任意方向加速。"""
        reachable_indices = set()
        
        for _ in range(n_samples):
            # 采样加速度方向和大小
            accel_angle = np.random.uniform(0, 2 * np.pi)
            # 加速或减速
            if np.random.random() < 0.7:  # 70% 概率加速
                accel_mag = np.random.uniform(0, dynamics.max_accel_mps2)
            else:  # 30% 概率减速
                accel_mag = np.random.uniform(-dynamics.max_decel_mps2, 0)
            
            a_x = accel_mag * np.cos(accel_angle)
            a_y = accel_mag * np.sin(accel_angle)
            
            # 运动学积分
            new_vx = vel[0] + a_x * dt
            new_vy = vel[1] + a_y * dt
            
            # 速度限制
            new_speed = math.sqrt(new_vx**2 + new_vy**2)
            if new_speed > dynamics.max_speed_mps:
                scale = dynamics.max_speed_mps / new_speed
                new_vx *= scale
                new_vy *= scale
            
            # 位置积分
            next_x = pos[0] + vel[0] * dt + 0.5 * a_x * dt * dt
            next_y = pos[1] + vel[1] * dt + 0.5 * a_y * dt * dt
            
            # 转换到网格索引
            xy_grid = self.to_grid_frame((next_x, next_y))
            idx = self.xy_to_index(xy_grid)
            reachable_indices.add(idx)
        
        return list(reachable_indices)

    def _vehicle_successors(self, pos: Tuple[float, float], speed: float, heading: float,
                           dynamics: AgentDynamicsParams, dt: float,
                           n_samples: int) -> List[int]:
        """车辆可达集：自行车模型，考虑阿克曼转向约束。"""
        reachable_indices = set()
        
        for _ in range(n_samples):
            # 采样控制输入
            # 油门/刹车：[-max_decel, +max_accel]
            if np.random.random() < 0.6:  # 60% 概率加速或保持
                accel = np.random.uniform(-0.5, dynamics.max_accel_mps2)
            else:  # 40% 概率减速
                accel = np.random.uniform(-dynamics.max_decel_mps2, 0)
            
            # 转向角：受阿克曼约束限制
            if dynamics.wheelbase_m > 0:
                # 根据轴距和最大偏航角速度计算最大转向角
                max_steer_rad = math.atan(dynamics.max_yaw_rate_rps * dynamics.wheelbase_m / max(speed, 0.1))
                max_steer_rad = min(max_steer_rad, math.pi / 6)  # 限制最大 30 度
            else:
                max_steer_rad = dynamics.max_yaw_rate_rps * dt  # 无轴距情况（摩托/自行车）
            
            steer_angle = np.random.uniform(-max_steer_rad, max_steer_rad)
            
            # 自行车模型积分
            next_speed = max(0, speed + accel * dt)
            next_speed = min(next_speed, dynamics.max_speed_mps)
            
            if dynamics.wheelbase_m > 0:
                # 阿克曼转向模型
                yaw_rate = next_speed * math.tan(steer_angle) / dynamics.wheelbase_m
            else:
                # 直接偏航角速度控制（摩托/自行车）
                yaw_rate = steer_angle / dt
            
            yaw_rate = np.clip(yaw_rate, -dynamics.max_yaw_rate_rps, dynamics.max_yaw_rate_rps)
            
            next_heading = heading + yaw_rate * dt
            
            # 位置积分（使用平均航向角）
            avg_heading = heading + 0.5 * yaw_rate * dt
            next_x = pos[0] + next_speed * math.cos(avg_heading) * dt
            next_y = pos[1] + next_speed * math.sin(avg_heading) * dt
            
            # 转换到网格索引
            xy_grid = self.to_grid_frame((next_x, next_y))
            idx = self.xy_to_index(xy_grid)
            reachable_indices.add(idx)
        
        return list(reachable_indices)

    def soft_bin(self, xy_world: Tuple[float, float], sigma: float = 0.4) -> np.ndarray:
        """计算位置观测的软分箱权重，返回 K 维归一化权重向量。
        
        使用高斯核在所有网格单元上分配权重：
        w(g) ∝ exp(-||pos - center(g)||^2 / (2 * sigma^2))
        
        Args:
            xy_world: 观测位置的世界坐标
            sigma: 高斯核标准差，默认 0.4 米
            
        Returns:
            形状为 (K,) 的权重数组，满足 sum(w) = 1
        """
        xy_grid = self.to_grid_frame(xy_world)
        weights = np.zeros(self.spec.num_cells, dtype=float)
        
        for idx in range(self.spec.num_cells):
            center_x, center_y = self.index_to_xy_center(idx)
            dist_sq = (xy_grid[0] - center_x) ** 2 + (xy_grid[1] - center_y) ** 2
            weights[idx] = np.exp(-dist_sq / (2 * sigma * sigma))
        
        # 归一化
        total_weight = weights.sum()
        if total_weight > 1e-12:
            weights /= total_weight
        else:
            # 退化情况：均匀分布
            weights.fill(1.0 / self.spec.num_cells)
            
        return weights

    def ego_occupancy(self, ego_state: EgoState, inflate_m: float = 0.8) -> List[int]:
        """计算自车占用的网格单元列表。
        
        使用圆形近似自车占用区域。
        
        Args:
            ego_state: 自车状态
            inflate_m: 膨胀半径，默认 0.8 米
            
        Returns:
            自车占用的网格单元索引列表
        """
        # 自车在局部坐标系原点
        ego_local = (0.0, 0.0)
        occupied_indices = []
        
        for idx in range(self.spec.num_cells):
            center_x, center_y = self.index_to_xy_center(idx)
            dist = math.sqrt((center_x - ego_local[0]) ** 2 + (center_y - ego_local[1]) ** 2)
            
            if dist <= inflate_m:
                occupied_indices.append(idx)
        
        return occupied_indices

    @property
    def K(self) -> int:
        """网格总单元数"""
        return self.spec.num_cells
    
    @property 
    def N(self) -> int:
        """网格边长"""
        return self.side
    
    @property
    def size_m(self) -> float:
        """网格物理尺寸（米）"""
        return self.spec.size_m

    def index_grid(self) -> np.ndarray:
        """返回所有网格单元中心的局部坐标。
        
        Returns:
            形状(K, 2)的数组，每行为一个单元的(x, y)局部坐标
        """
        centers = np.zeros((self.spec.num_cells, 2), dtype=float)
        for idx in range(self.spec.num_cells):
            centers[idx] = self.index_to_xy_center(idx)
        return centers

    def grid_to_world(self, xy_grid: np.ndarray) -> np.ndarray:
        """将网格坐标转换为世界坐标。
        
        Args:
            xy_grid: 网格坐标，形状为(2,) 或 (N, 2)
            
        Returns:
            世界坐标，与输入形状相同
        """
        center_x, center_y = self.world_center
        
        if xy_grid.ndim == 1:
            return np.array([xy_grid[0] + center_x, xy_grid[1] + center_y])
        else:
            world_coords = xy_grid.copy()
            world_coords[:, 0] += center_x
            world_coords[:, 1] += center_y
            return world_coords
