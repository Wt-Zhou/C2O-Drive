"""
轨迹生成与Q值评估模块
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
import numpy as np

from carla_c2osr.evaluation.q_value_calculator import QValueCalculator, QValueConfig
from carla_c2osr.config import get_global_config
from .episode_context import EpisodeContext


def get_percentile_q_value(q_values: np.ndarray, percentile: float) -> float:
    """
    根据百分位数获取Q值

    Args:
        q_values: Q值数组
        percentile: 百分位数 [0.0, 1.0]

    Returns:
        百分位对应的Q值（使用线性插值）
    """
    if len(q_values) == 0:
        return 0.0

    if len(q_values) == 1:
        return float(q_values[0])

    # 排序Q值
    sorted_q = np.sort(q_values)
    n = len(sorted_q)

    # 计算精确位置（0-based index）
    position = percentile * (n - 1)

    # 下界和上界索引
    lower_idx = int(np.floor(position))
    upper_idx = int(np.ceil(position))

    # 如果正好是整数位置
    if lower_idx == upper_idx:
        return float(sorted_q[lower_idx])

    # 线性插值
    weight = position - lower_idx
    return float(sorted_q[lower_idx] * (1 - weight) + sorted_q[upper_idx] * weight)


class TrajectoryEvaluator:
    """轨迹生成、评估和选择"""

    def __init__(self, ctx: EpisodeContext):
        self.ctx = ctx

    def generate_and_evaluate_trajectories(self) -> List[Dict[str, Any]]:
        """
        生成候选轨迹并计算每条轨迹的Q值

        Returns:
            trajectory_q_values: 包含每条轨迹的Q值统计信息
        """
        # 生成候选轨迹
        candidate_trajectories = self._generate_candidate_trajectories()
        print(f"  生成 {len(candidate_trajectories)} 条候选轨迹")

        # 为每条候选轨迹计算Q值
        trajectory_q_values = []

        for traj in candidate_trajectories:
            q_info = self._evaluate_single_trajectory(traj)
            if q_info is not None:
                trajectory_q_values.append(q_info)

        return trajectory_q_values

    def select_optimal_trajectory(self, trajectory_q_values: List[Dict[str, Any]]) -> tuple:
        """
        根据百分位Q值选择最优轨迹

        Args:
            trajectory_q_values: 轨迹Q值列表

        Returns:
            (ego_trajectory, selected_trajectory_info)
        """
        if not trajectory_q_values:
            print(f"  警告: 没有有效轨迹，使用reference path")
            return self.ctx.reference_path, None

        # 选择percentile_q最大的轨迹
        best_trajectory = max(trajectory_q_values, key=lambda x: x['percentile_q'])
        ego_trajectory = best_trajectory['trajectory']
        selected_trajectory_info = best_trajectory

        # 打印选择结果
        q_config = QValueConfig.from_global_config()
        percentile = q_config.q_selection_percentile

        print(f"  选中轨迹{best_trajectory['trajectory_id']}: "
              f"偏移={best_trajectory['lateral_offset']:.1f}m, "
              f"速度={best_trajectory['target_speed']:.1f}m/s")
        print(f"    Min_Q={best_trajectory['min_q']:.2f}, "
              f"Mean_Q={best_trajectory['mean_q']:.2f}, "
              f"P{int(percentile*100)}_Q={best_trajectory['percentile_q']:.2f}, "
              f"碰撞率={best_trajectory['collision_rate']:.3f}")

        return ego_trajectory, selected_trajectory_info

    def _generate_candidate_trajectories(self):
        """生成候选轨迹"""
        config = get_global_config()

        # 从world_init获取自车当前状态
        ego_state = (
            self.ctx.world_init.ego.position_m[0],
            self.ctx.world_init.ego.position_m[1],
            self.ctx.world_init.ego.yaw_rad
        )

        candidate_trajectories = self.ctx.lattice_planner.generate_trajectories(
            reference_path=self.ctx.reference_path,
            horizon=self.ctx.horizon,
            dt=config.time.dt,
            ego_state=ego_state
        )

        return candidate_trajectories

    def _evaluate_single_trajectory(self, traj) -> Optional[Dict[str, Any]]:
        """
        评估单条轨迹的Q值

        Args:
            traj: 轨迹对象

        Returns:
            包含Q值统计信息的字典，失败返回None
        """
        # 构造自车动作轨迹
        ego_action_trajectory = traj.waypoints

        # 创建Q值配置和计算器
        q_config = QValueConfig.from_global_config()
        config = get_global_config()
        reward_config = config.reward
        q_calculator = QValueCalculator(q_config, reward_config)

        # 计算Q值
        try:
            q_values, detailed_info = q_calculator.compute_q_value(
                current_world_state=self.ctx.world_init,
                ego_action_trajectory=ego_action_trajectory,
                trajectory_buffer=self.ctx.trajectory_buffer,
                grid=self.ctx.grid,
                bank=self.ctx.bank,
                rng=self.ctx.rng,
                reference_path=self.ctx.reference_path
            )

            # 计算Q值统计指标
            min_q = np.min(q_values)
            mean_q = np.mean(q_values)
            percentile_q = get_percentile_q_value(q_values, q_config.q_selection_percentile)

            return {
                'trajectory_id': traj.trajectory_id,
                'lateral_offset': traj.lateral_offset,
                'target_speed': traj.target_speed,
                'min_q': min_q,
                'mean_q': mean_q,
                'percentile_q': percentile_q,
                'q_values': q_values,
                'collision_rate': detailed_info['reward_breakdown']['collision_rate'],
                'trajectory': traj.waypoints
            }

        except Exception as e:
            print(f"  警告: 轨迹{traj.trajectory_id}计算失败: {e}")
            return None
