from __future__ import annotations
from typing import List
import numpy as np
from c2o_drive.algorithms.c2osr.grid import GridMapper


def sample_transition_row(alpha: np.ndarray, reachable: List[int], 
                         rng: np.random.Generator, c_draw: float = 50.0) -> np.ndarray:
    """从Dirichlet分布采样一个离散转移概率行。
    
    Args:
        alpha: 当前智能体的Dirichlet参数向量，形状(K,)
        reachable: 可达单元索引列表
        rng: 随机数生成器
        c_draw: Dirichlet采样浓度参数
        
    Returns:
        长度K的概率行向量，只在reachable位置非零，其余为0
    """
    assert len(reachable) > 0, "Reachable set cannot be empty"
    assert all(0 <= idx < len(alpha) for idx in reachable), "Reachable indices out of range"
    
    # 提取可达位置的alpha值并归一化
    alpha_reachable = alpha[reachable]
    alpha_normalized = alpha_reachable / alpha_reachable.sum()
    
    # 从Dirichlet(c_draw * alpha_normalized)采样
    dirichlet_params = c_draw * alpha_normalized
    sampled_probs = rng.dirichlet(dirichlet_params)
    
    # 构造完整的K维行向量
    T_row = np.zeros(len(alpha), dtype=float)
    T_row[reachable] = sampled_probs
    
    assert abs(T_row.sum() - 1.0) < 1e-6, f"T_row sum {T_row.sum()} != 1.0"
    return T_row


def rollout_once(start_xy: np.ndarray, yaw: float, grid: GridMapper, 
                T_row: np.ndarray, H: int = 8) -> List[np.ndarray]:
    """用固定转移概率滚动生成H步轨迹。
    
    Args:
        start_xy: 起始位置的世界坐标 (x, y)
        yaw: 起始朝向（未使用，预留）
        grid: 网格映射器
        T_row: 转移概率行向量，形状(K,)
        H: 滚动步数
        
    Returns:
        长度H的轨迹，每个元素为世界坐标np.ndarray([x, y])
    """
    assert abs(T_row.sum() - 1.0) < 1e-6, f"T_row must sum to 1, got {T_row.sum()}"
    assert H > 0, "Horizon must be positive"
    
    trajectory = []
    current_world_xy = np.array(start_xy, dtype=float)
    
    # 简化的mock world用于坐标转换（假设自车在原点）
    from c2o_drive.environments.carla.types import EgoState, WorldState
    mock_ego = EgoState(position_m=(0.0, 0.0), velocity_mps=(0.0, 0.0), yaw_rad=0.0)
    mock_world = WorldState(time_s=0.0, ego=mock_ego, agents=[])
    
    for step in range(H):
        # 将当前世界坐标转换为网格坐标
        current_grid = grid.to_grid_frame(tuple(current_world_xy))
        current_idx = grid.xy_to_index(current_grid)
        
        # 根据转移概率采样下一个单元
        next_idx = np.random.choice(len(T_row), p=T_row)
        
        # 获取下一单元的中心网格坐标
        next_grid = grid.index_to_xy_center(next_idx)
        
        # 转换回世界坐标
        next_world_xy = grid.grid_to_world(np.array(next_grid))
        
        trajectory.append(next_world_xy.copy())
        current_world_xy = next_world_xy
    
    return trajectory
