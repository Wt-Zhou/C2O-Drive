from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
import numpy as np
import math

from carla_c2osr.env.types import AgentState, WorldState, EgoState, AgentType, AgentDynamicsParams

# Numba加速（可选依赖）
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # 创建一个空装饰器用于兼容
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


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

    def get_neighbors(self, cell_idx: int, radius: int = 1) -> List[int]:
        """获取指定cell的邻域cells（用于碰撞检测剪枝）。

        Args:
            cell_idx: 中心cell索引
            radius: 邻域半径（cell数），默认1表示8邻域

        Returns:
            邻域cell索引列表（包含中心cell）
        """
        if cell_idx < 0 or cell_idx >= self.spec.num_cells:
            return []

        # 转换为2D坐标
        center_iy = cell_idx // self.side
        center_ix = cell_idx % self.side

        neighbors = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx = center_ix + dx
                ny = center_iy + dy

                # 边界检查
                if 0 <= nx < self.side and 0 <= ny < self.side:
                    neighbor_idx = ny * self.side + nx
                    neighbors.append(neighbor_idx)

        return neighbors

    def successor_cells(self, agent: AgentState, dt: Optional[float] = None,
                       n_samples: Optional[int] = None) -> List[int]:
        """计算智能体在下一时刻的一步可达网格单元集合。

        根据智能体类型使用相应的动力学模型：
        - 行人: 质点模型，可任意方向加速
        - 车辆: 自行车模型，考虑阿克曼转向约束
        - 自行车/摩托车: 自行车模型，转向能力更强

        Args:
            agent: 智能体当前状态
            dt: 时间步长，默认从全局配置读取
            n_samples: 采样数量，默认从全局配置读取

        Returns:
            可达网格单元索引列表（去重）
        """
        # 从全局配置读取默认参数
        if dt is None or n_samples is None:
            from carla_c2osr.config import get_global_config
            config = get_global_config()
            if dt is None:
                dt = config.time.dt
            if n_samples is None:
                n_samples = config.sampling.reachable_set_samples_legacy

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
    
    def multi_timestep_successor_cells(self, agent: AgentState, horizon: Optional[int] = None,
                                     dt: Optional[float] = None, n_samples: Optional[int] = None,
                                     use_numba: bool = True) -> Dict[int, List[int]]:
        """计算智能体在未来多个时刻的可达网格单元集合。

        Args:
            agent: 智能体当前状态
            horizon: 预测时间步数，默认从全局配置读取
            dt: 时间步长，默认从全局配置读取
            n_samples: 采样数量，默认从全局配置读取
            use_numba: 是否使用numba加速版本（默认True，如果numba不可用则自动降级）

        Returns:
            {timestep: [reachable_cell_indices]} 每个时刻的可达集
        """
        # 如果numba可用且use_numba=True，使用优化版本
        if use_numba and NUMBA_AVAILABLE:
            return self._multi_timestep_successor_cells_numba(agent, horizon, dt, n_samples)
        else:
            # 使用原始版本
            return self._multi_timestep_successor_cells_original(agent, horizon, dt, n_samples)

    def _multi_timestep_successor_cells_original(self, agent: AgentState, horizon: Optional[int] = None,
                                     dt: Optional[float] = None, n_samples: Optional[int] = None) -> Dict[int, List[int]]:
        """原始版本的多时间步可达集计算（保留用于兼容性和fallback）

        Args:
            agent: 智能体当前状态
            horizon: 预测时间步数，默认从全局配置读取
            dt: 时间步长，默认从全局配置读取
            n_samples: 采样数量，默认从全局配置读取

        Returns:
            {timestep: [reachable_cell_indices]} 每个时刻的可达集
        """
        # 从全局配置读取默认参数
        if horizon is None or dt is None or n_samples is None:
            from carla_c2osr.config import get_global_config
            config = get_global_config()
            if horizon is None:
                horizon = config.time.default_horizon
            if dt is None:
                dt = config.time.dt
            if n_samples is None:
                n_samples = config.sampling.reachable_set_samples

        # 获取智能体动力学参数
        dynamics = AgentDynamicsParams.for_agent_type(agent.agent_type)
        
        current_pos = agent.position_m
        current_vel = agent.velocity_mps
        current_heading = agent.heading_rad
        current_speed = math.sqrt(current_vel[0]**2 + current_vel[1]**2)
        
        # 为每个时间步计算可达集
        reachable_sets = {}
        
        for timestep in range(1, horizon + 1):
            reachable_indices = set()
            
            # 多次采样以获得更精确的可达集
            for _ in range(n_samples):
                # 模拟从当前状态到指定时间步的轨迹
                final_pos = self._simulate_trajectory_to_timestep(
                    current_pos, current_vel, current_heading, current_speed,
                    dynamics, agent.agent_type, dt, timestep
                )
                
                if final_pos is not None:
                    # 转换到网格索引
                    xy_grid = self.to_grid_frame(final_pos)
                    idx = self.xy_to_index(xy_grid)
                    reachable_indices.add(idx)
            
            # 添加当前位置（静止情况）
            current_grid = self.to_grid_frame(current_pos)
            current_idx = self.xy_to_index(current_grid)
            reachable_indices.add(current_idx)
            
            reachable_sets[timestep] = list(reachable_indices)
        
        return reachable_sets
    
    def _simulate_trajectory_to_timestep(self, start_pos: Tuple[float, float],
                                       start_vel: Tuple[float, float], 
                                       start_heading: float,
                                       start_speed: float,
                                       dynamics: AgentDynamicsParams,
                                       agent_type: AgentType,
                                       dt: float,
                                       target_timestep: int) -> Optional[Tuple[float, float]]:
        """模拟智能体轨迹到指定时间步"""
        pos = np.array(start_pos)
        vel = np.array(start_vel)
        heading = start_heading
        speed = start_speed
        
        for step in range(target_timestep):
            # 采样控制输入并执行一步
            if agent_type == AgentType.PEDESTRIAN:
                pos, vel, heading, speed = self._pedestrian_dynamics_step(
                    pos, vel, heading, speed, dynamics, dt
                )
            else:
                pos, vel, heading, speed = self._vehicle_dynamics_step(
                    pos, vel, heading, speed, dynamics, dt
                )
        
        return tuple(pos)
    
    def _pedestrian_dynamics_step(self, pos, vel, heading, speed, dynamics, dt):
        """行人动力学单步模拟"""
        # 采样加速度方向和大小
        accel_angle = np.random.uniform(0, 2 * np.pi)
        
        if np.random.random() < 0.7:  # 70% 概率加速
            accel_mag = np.random.uniform(0, dynamics.max_accel_mps2)
        else:  # 30% 概率减速
            accel_mag = np.random.uniform(-dynamics.max_decel_mps2, 0)
        
        # 计算加速度分量
        accel_x = accel_mag * math.cos(accel_angle)
        accel_y = accel_mag * math.sin(accel_angle)
        
        # 更新速度
        new_vel_x = vel[0] + accel_x * dt
        new_vel_y = vel[1] + accel_y * dt
        
        # 限制速度
        new_speed = math.sqrt(new_vel_x**2 + new_vel_y**2)
        if new_speed > dynamics.max_speed_mps:
            scale = dynamics.max_speed_mps / new_speed
            new_vel_x *= scale
            new_vel_y *= scale
            new_speed = dynamics.max_speed_mps
        
        # 更新位置
        new_pos_x = pos[0] + new_vel_x * dt
        new_pos_y = pos[1] + new_vel_y * dt
        
        new_heading = math.atan2(new_vel_y, new_vel_x)
        
        return np.array([new_pos_x, new_pos_y]), np.array([new_vel_x, new_vel_y]), new_heading, new_speed
    
    def _vehicle_dynamics_step(self, pos, vel, heading, speed, dynamics, dt):
        """车辆动力学单步模拟"""
        # 采样控制输入
        if np.random.random() < 0.6:  # 60% 概率加速或保持
            accel = np.random.uniform(-0.5, dynamics.max_accel_mps2)
        else:  # 40% 概率减速
            accel = np.random.uniform(-dynamics.max_decel_mps2, 0)
        
        # 转向角
        if dynamics.wheelbase_m > 0:
            max_steer_rad = math.atan(dynamics.max_yaw_rate_rps * dynamics.wheelbase_m / max(speed, 0.1))
            max_steer_rad = min(max_steer_rad, math.pi / 6)  # 限制最大 30 度
        else:
            max_steer_rad = dynamics.max_yaw_rate_rps * dt
        
        steer_angle = np.random.uniform(-max_steer_rad, max_steer_rad)
        
        # 自行车模型积分
        new_speed = max(0, speed + accel * dt)
        new_speed = min(new_speed, dynamics.max_speed_mps)
        
        if dynamics.wheelbase_m > 0:
            yaw_rate = new_speed * math.tan(steer_angle) / dynamics.wheelbase_m
        else:
            yaw_rate = steer_angle / dt
        
        yaw_rate = np.clip(yaw_rate, -dynamics.max_yaw_rate_rps, dynamics.max_yaw_rate_rps)
        new_heading = heading + yaw_rate * dt
        
        # 位置积分
        avg_heading = heading + 0.5 * yaw_rate * dt
        new_pos_x = pos[0] + new_speed * math.cos(avg_heading) * dt
        new_pos_y = pos[1] + new_speed * math.sin(avg_heading) * dt

        new_vel_x = new_speed * math.cos(new_heading)
        new_vel_y = new_speed * math.sin(new_heading)

        return np.array([new_pos_x, new_pos_y]), np.array([new_vel_x, new_vel_y]), new_heading, new_speed

    def _multi_timestep_successor_cells_numba(self, agent: AgentState, horizon: Optional[int] = None,
                                            dt: Optional[float] = None, n_samples: Optional[int] = None) -> Dict[int, List[int]]:
        """Numba优化版本的多时间步可达集计算（向量化批量处理）

        Args:
            agent: 智能体当前状态
            horizon: 预测时间步数，默认从全局配置读取
            dt: 时间步长，默认从全局配置读取
            n_samples: 采样数量，默认从全局配置读取

        Returns:
            {timestep: [reachable_cell_indices]} 每个时刻的可达集
        """
        # 从全局配置读取默认参数
        if horizon is None or dt is None or n_samples is None:
            from carla_c2osr.config import get_global_config
            config = get_global_config()
            if horizon is None:
                horizon = config.time.default_horizon
            if dt is None:
                dt = config.time.dt
            if n_samples is None:
                n_samples = config.sampling.reachable_set_samples

        # 获取智能体动力学参数
        dynamics = AgentDynamicsParams.for_agent_type(agent.agent_type)

        # 提取初始状态
        start_pos_x, start_pos_y = agent.position_m
        start_vel_x, start_vel_y = agent.velocity_mps
        start_heading = agent.heading_rad
        start_speed = math.sqrt(start_vel_x**2 + start_vel_y**2)

        # 确定是否是行人
        is_pedestrian = (agent.agent_type == AgentType.PEDESTRIAN)

        # 生成随机种子
        random_seed = np.random.randint(0, 1000000)

        # 计算所有时间步的可达集
        reachable_sets = {}

        for timestep in range(1, horizon + 1):
            # 使用numba批量模拟到该时间步
            final_pos_x, final_pos_y = _numba_simulate_trajectories_to_timestep(
                start_pos_x, start_pos_y,
                start_vel_x, start_vel_y,
                start_heading, start_speed,
                dynamics.max_accel_mps2, dynamics.max_decel_mps2, dynamics.max_speed_mps,
                dynamics.max_yaw_rate_rps, dynamics.wheelbase_m,
                dt, timestep, n_samples,
                is_pedestrian, random_seed + timestep
            )

            # 批量转换坐标到网格索引
            indices = _numba_xy_to_indices(
                final_pos_x, final_pos_y,
                self.world_center[0], self.world_center[1],
                self.spec.size_m, self.spec.cell_m, self.side
            )

            # 去重并添加当前位置
            reachable_indices = set(indices.tolist())
            current_grid = self.to_grid_frame(agent.position_m)
            current_idx = self.xy_to_index(current_grid)
            reachable_indices.add(current_idx)

            reachable_sets[timestep] = list(reachable_indices)

        return reachable_sets

    def get_neighbors(self, cell_idx: int, radius: int = 1) -> List[int]:
        """获取网格单元的邻域

        Args:
            cell_idx: 中心单元索引
            radius: 邻域半径（单元数）

        Returns:
            邻域单元索引列表（包括中心单元）
        """
        neighbors = []
        iy_center = cell_idx // self.side
        ix_center = cell_idx % self.side

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                iy = iy_center + dy
                ix = ix_center + dx

                # 边界检查
                if 0 <= iy < self.side and 0 <= ix < self.side:
                    neighbors.append(iy * self.side + ix)

        return neighbors


# ==================== Numba优化版本可达集计算 ====================

@jit(nopython=True, cache=True)
def _numba_pedestrian_dynamics_vectorized(
    pos_x: np.ndarray, pos_y: np.ndarray,
    vel_x: np.ndarray, vel_y: np.ndarray,
    heading: np.ndarray, speed: np.ndarray,
    max_accel: float, max_decel: float, max_speed: float,
    dt: float, n_samples: int, random_seed: int
) -> tuple:
    """Numba优化的行人动力学批量计算（单步）

    Args:
        pos_x, pos_y: 位置数组 (n_samples,)
        vel_x, vel_y: 速度数组 (n_samples,)
        heading, speed: 朝向和速度数组 (n_samples,)
        max_accel, max_decel, max_speed: 动力学参数
        dt: 时间步长
        n_samples: 样本数量
        random_seed: 随机种子

    Returns:
        (new_pos_x, new_pos_y, new_vel_x, new_vel_y, new_heading, new_speed)
    """
    np.random.seed(random_seed)

    # 采样加速度方向和大小
    accel_angles = np.random.uniform(0, 2 * np.pi, n_samples)
    accel_choice = np.random.uniform(0, 1, n_samples)

    # 70% 加速，30% 减速
    accel_mags = np.zeros(n_samples)
    for i in range(n_samples):
        if accel_choice[i] < 0.7:
            accel_mags[i] = np.random.uniform(0, max_accel)
        else:
            accel_mags[i] = np.random.uniform(-max_decel, 0)

    # 计算加速度分量
    accel_x = accel_mags * np.cos(accel_angles)
    accel_y = accel_mags * np.sin(accel_angles)

    # 更新速度
    new_vel_x = vel_x + accel_x * dt
    new_vel_y = vel_y + accel_y * dt

    # 限制速度
    new_speeds = np.sqrt(new_vel_x**2 + new_vel_y**2)
    for i in range(n_samples):
        if new_speeds[i] > max_speed:
            scale = max_speed / new_speeds[i]
            new_vel_x[i] *= scale
            new_vel_y[i] *= scale
            new_speeds[i] = max_speed

    # 更新位置
    new_pos_x = pos_x + new_vel_x * dt
    new_pos_y = pos_y + new_vel_y * dt

    # 更新朝向
    new_heading = np.arctan2(new_vel_y, new_vel_x)

    return new_pos_x, new_pos_y, new_vel_x, new_vel_y, new_heading, new_speeds


@jit(nopython=True, cache=True)
def _numba_vehicle_dynamics_vectorized(
    pos_x: np.ndarray, pos_y: np.ndarray,
    vel_x: np.ndarray, vel_y: np.ndarray,
    heading: np.ndarray, speed: np.ndarray,
    max_accel: float, max_decel: float, max_speed: float,
    max_yaw_rate: float, wheelbase: float,
    dt: float, n_samples: int, random_seed: int
) -> tuple:
    """Numba优化的车辆动力学批量计算（单步）

    Args:
        pos_x, pos_y: 位置数组 (n_samples,)
        vel_x, vel_y: 速度数组 (n_samples,)
        heading, speed: 朝向和速度数组 (n_samples,)
        max_accel, max_decel, max_speed: 动力学参数
        max_yaw_rate: 最大偏航角速度
        wheelbase: 轴距
        dt: 时间步长
        n_samples: 样本数量
        random_seed: 随机种子

    Returns:
        (new_pos_x, new_pos_y, new_vel_x, new_vel_y, new_heading, new_speed)
    """
    np.random.seed(random_seed)

    # 采样加速度
    accel_choice = np.random.uniform(0, 1, n_samples)
    accels = np.zeros(n_samples)
    for i in range(n_samples):
        if accel_choice[i] < 0.6:
            accels[i] = np.random.uniform(-0.5, max_accel)
        else:
            accels[i] = np.random.uniform(-max_decel, 0)

    # 计算新速度
    new_speeds = speed + accels * dt
    for i in range(n_samples):
        new_speeds[i] = max(0.0, min(new_speeds[i], max_speed))

    # 采样转向角
    steer_angles = np.zeros(n_samples)
    for i in range(n_samples):
        if wheelbase > 0:
            max_steer = np.arctan(max_yaw_rate * wheelbase / max(speed[i], 0.1))
            max_steer = min(max_steer, np.pi / 6)
        else:
            max_steer = max_yaw_rate * dt
        steer_angles[i] = np.random.uniform(-max_steer, max_steer)

    # 计算偏航角速度
    yaw_rates = np.zeros(n_samples)
    for i in range(n_samples):
        if wheelbase > 0:
            yaw_rates[i] = new_speeds[i] * np.tan(steer_angles[i]) / wheelbase
        else:
            yaw_rates[i] = steer_angles[i] / dt
        # numba不支持np.clip标量，使用min/max
        yaw_rates[i] = max(-max_yaw_rate, min(yaw_rates[i], max_yaw_rate))

    # 更新朝向
    new_heading = heading + yaw_rates * dt

    # 更新位置
    avg_heading = heading + 0.5 * yaw_rates * dt
    new_pos_x = pos_x + new_speeds * np.cos(avg_heading) * dt
    new_pos_y = pos_y + new_speeds * np.sin(avg_heading) * dt

    # 更新速度分量
    new_vel_x = new_speeds * np.cos(new_heading)
    new_vel_y = new_speeds * np.sin(new_heading)

    return new_pos_x, new_pos_y, new_vel_x, new_vel_y, new_heading, new_speeds


@jit(nopython=True, cache=True)
def _numba_simulate_trajectories_to_timestep(
    start_pos_x: float, start_pos_y: float,
    start_vel_x: float, start_vel_y: float,
    start_heading: float, start_speed: float,
    max_accel: float, max_decel: float, max_speed: float,
    max_yaw_rate: float, wheelbase: float,
    dt: float, target_timestep: int, n_samples: int,
    is_pedestrian: bool, random_seed: int
) -> tuple:
    """Numba优化的批量轨迹模拟到指定时间步

    Returns:
        (final_pos_x, final_pos_y): 形状为(n_samples,)的数组
    """
    # 初始化状态数组
    pos_x = np.full(n_samples, start_pos_x, dtype=np.float64)
    pos_y = np.full(n_samples, start_pos_y, dtype=np.float64)
    vel_x = np.full(n_samples, start_vel_x, dtype=np.float64)
    vel_y = np.full(n_samples, start_vel_y, dtype=np.float64)
    heading = np.full(n_samples, start_heading, dtype=np.float64)
    speed = np.full(n_samples, start_speed, dtype=np.float64)

    # 逐步模拟到目标时间步
    for step in range(target_timestep):
        # 每步使用不同的随机种子
        step_seed = random_seed + step * 1000

        if is_pedestrian:
            pos_x, pos_y, vel_x, vel_y, heading, speed = _numba_pedestrian_dynamics_vectorized(
                pos_x, pos_y, vel_x, vel_y, heading, speed,
                max_accel, max_decel, max_speed, dt, n_samples, step_seed
            )
        else:
            pos_x, pos_y, vel_x, vel_y, heading, speed = _numba_vehicle_dynamics_vectorized(
                pos_x, pos_y, vel_x, vel_y, heading, speed,
                max_accel, max_decel, max_speed, max_yaw_rate, wheelbase,
                dt, n_samples, step_seed
            )

    return pos_x, pos_y


@jit(nopython=True, cache=True)
def _numba_xy_to_indices(
    pos_x: np.ndarray, pos_y: np.ndarray,
    world_center_x: float, world_center_y: float,
    grid_size_m: float, cell_size_m: float, grid_side: int
) -> np.ndarray:
    """Numba优化的批量坐标到网格索引转换

    Args:
        pos_x, pos_y: 位置数组
        world_center_x, world_center_y: 网格世界中心
        grid_size_m: 网格尺寸
        cell_size_m: 单元尺寸
        grid_side: 网格边长

    Returns:
        indices: 网格索引数组 (n_samples,)
    """
    n_samples = len(pos_x)
    indices = np.zeros(n_samples, dtype=np.int32)
    half = grid_size_m / 2.0

    for i in range(n_samples):
        # 转换到网格坐标
        x_grid = pos_x[i] - world_center_x
        y_grid = pos_y[i] - world_center_y

        # 转换到索引
        ix = int((x_grid + half) / cell_size_m)
        iy = int((y_grid + half) / cell_size_m)

        # Clamp到边界
        ix = max(0, min(grid_side - 1, ix))
        iy = max(0, min(grid_side - 1, iy))

        indices[i] = iy * grid_side + ix

    return indices