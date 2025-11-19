"""
Apollo风格的Lattice轨迹规划器

参考Apollo的lattice planner设计：
- 从自车当前状态出发
- 采样终点状态（横向偏移 + 纵向距离）
- 使用五次多项式生成平滑轨迹
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class LatticeTrajectory:
    """Lattice轨迹"""
    waypoints: List[Tuple[float, float]]  # 轨迹点列表
    lateral_offset: float  # 终点横向偏移量
    target_speed: float    # 目标速度
    trajectory_id: int     # 轨迹ID


class QuinticPolynomial:
    """五次多项式轨迹生成器"""

    def __init__(self, x0: float, v0: float, a0: float,
                 x1: float, v1: float, a1: float, T: float):
        """
        生成从(x0, v0, a0)到(x1, v1, a1)的五次多项式

        Args:
            x0, v0, a0: 起点位置、速度、加速度
            x1, v1, a1: 终点位置、速度、加速度
            T: 总时间
        """
        # 构建系数矩阵求解 x(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
        A = np.array([
            [0, 0, 0, 0, 0, 1],
            [T**5, T**4, T**3, T**2, T, 1],
            [0, 0, 0, 0, 1, 0],
            [5*T**4, 4*T**3, 3*T**2, 2*T, 1, 0],
            [0, 0, 0, 2, 0, 0],
            [20*T**3, 12*T**2, 6*T, 2, 0, 0]
        ])

        b = np.array([x0, x1, v0, v1, a0, a1])
        self.coeffs = np.linalg.solve(A, b)

    def calc_point(self, t: float) -> float:
        """计算t时刻的位置"""
        return (self.coeffs[0] * t**5 + self.coeffs[1] * t**4 +
                self.coeffs[2] * t**3 + self.coeffs[3] * t**2 +
                self.coeffs[4] * t + self.coeffs[5])

    def calc_velocity(self, t: float) -> float:
        """计算t时刻的速度"""
        return (5 * self.coeffs[0] * t**4 + 4 * self.coeffs[1] * t**3 +
                3 * self.coeffs[2] * t**2 + 2 * self.coeffs[3] * t +
                self.coeffs[4])


class LatticePlanner:
    """Apollo风格的Lattice规划器"""

    def __init__(self, lateral_offsets: List[float], speed_variations: List[float],
                 num_trajectories: int = 10):
        """
        Args:
            lateral_offsets: 终点横向偏移量列表（米）
            speed_variations: 目标速度列表（m/s）
            num_trajectories: 目标轨迹数量
        """
        self.lateral_offsets = lateral_offsets
        self.speed_variations = speed_variations
        self.num_trajectories = num_trajectories

    def generate_trajectories(self,
                            reference_path: List[Tuple[float, float]],
                            horizon: int,
                            dt: float = 1.0,
                            ego_state: Optional[Tuple[float, float, float]] = None) -> List[LatticeTrajectory]:
        """生成lattice轨迹集合（Apollo风格）

        Args:
            reference_path: 参考路径（用于确定行驶方向）
            horizon: 预测时间步数
            dt: 时间步长（秒）
            ego_state: 自车状态 (x, y, heading)，如果为None则使用reference_path起点

        Returns:
            候选轨迹列表
        """
        # 确定起点
        if ego_state is not None:
            start_x, start_y, heading = ego_state
        else:
            start_x, start_y = reference_path[0]
            # 从reference path计算初始朝向
            if len(reference_path) >= 2:
                dx = reference_path[1][0] - reference_path[0][0]
                dy = reference_path[1][1] - reference_path[0][1]
                heading = np.arctan2(dy, dx)
            else:
                heading = 0.0

        trajectories = []
        trajectory_id = 0

        # 生成所有lateral × speed组合
        for lateral_offset in self.lateral_offsets:
            for speed in self.speed_variations:
                if trajectory_id >= self.num_trajectories:
                    break

                # 生成单条轨迹
                trajectory = self._generate_frenet_trajectory(
                    start_x, start_y, heading,
                    lateral_offset, speed,
                    horizon, dt
                )

                trajectories.append(LatticeTrajectory(
                    waypoints=trajectory,
                    lateral_offset=lateral_offset,
                    target_speed=speed,
                    trajectory_id=trajectory_id
                ))

                trajectory_id += 1

            if trajectory_id >= self.num_trajectories:
                break

        return trajectories

    def _generate_frenet_trajectory(self,
                                   start_x: float,
                                   start_y: float,
                                   heading: float,
                                   lateral_offset: float,
                                   target_speed: float,
                                   horizon: int,
                                   dt: float) -> List[Tuple[float, float]]:
        """使用Frenet坐标系生成单条轨迹

        Args:
            start_x, start_y: 起点位置
            heading: 初始朝向
            lateral_offset: 终点横向偏移量
            target_speed: 目标速度
            horizon: 时间步数
            dt: 时间步长

        Returns:
            轨迹点列表
        """
        total_time = horizon * dt

        # Frenet坐标系：s为纵向（沿heading方向），l为横向（垂直于heading）
        # 纵向运动（s方向）
        s0, v_s0, a_s0 = 0.0, target_speed * 0.5, 0.0  # 起点：位置0，速度为目标速度的一半
        s1 = target_speed * total_time  # 终点：根据目标速度计算
        v_s1, a_s1 = target_speed, 0.0  # 终点：达到目标速度，加速度为0

        # 横向运动（l方向）
        l0, v_l0, a_l0 = 0.0, 0.0, 0.0  # 起点：无横向偏移
        l1 = lateral_offset  # 终点：目标横向偏移
        v_l1, a_l1 = 0.0, 0.0  # 终点：横向速度和加速度为0

        # 生成五次多项式
        lon_poly = QuinticPolynomial(s0, v_s0, a_s0, s1, v_s1, a_s1, total_time)
        lat_poly = QuinticPolynomial(l0, v_l0, a_l0, l1, v_l1, a_l1, total_time)

        # 采样轨迹点
        trajectory = []
        for i in range(horizon):
            t = i * dt

            # Frenet坐标
            s = lon_poly.calc_point(t)
            l = lat_poly.calc_point(t)

            # 转换到笛卡尔坐标
            # 假设沿着heading方向前进
            x = start_x + s * np.cos(heading) - l * np.sin(heading)
            y = start_y + s * np.sin(heading) + l * np.cos(heading)

            trajectory.append((x, y))

        return trajectory

    @classmethod
    def from_config(cls, config) -> 'LatticePlanner':
        """从全局配置创建LatticePlanner

        Args:
            config: GlobalConfig实例

        Returns:
            LatticePlanner实例
        """
        return cls(
            lateral_offsets=config.lattice.lateral_offsets,
            speed_variations=config.lattice.speed_variations,
            num_trajectories=config.lattice.num_trajectories
        )
