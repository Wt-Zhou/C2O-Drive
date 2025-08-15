from __future__ import annotations
from typing import List
import numpy as np
import math


def union_risk(probabilities: List[float]) -> float:
    """并集界: 1 - ∏(1-p)。"""
    prod = 1.0
    for p in probabilities:
        prod *= (1.0 - max(0.0, min(1.0, p)))
    return 1.0 - prod


def independent_risk(probabilities: List[float]) -> float:
    """独立近似下的同一公式，与 union_risk 等价占位。"""
    return union_risk(probabilities)


def compose_union_singlelayer(q_list_per_agent: List[np.ndarray]) -> np.ndarray:
    """单层多智能体占据概率的并集界合成。
    
    对每个网格单元 g，计算多智能体的并集占据概率：
    q_union(g) = 1 - ∏ᵢ (1 - qᵢ(g))
    
    并确保结果 clip 到 [0, 1]。
    
    Args:
        q_list_per_agent: 智能体占据概率数组列表，每个形状 (K,)
        
    Returns:
        形状 (K,) 的并集占据概率数组，值域 [0, 1]
    """
    if not q_list_per_agent:
        raise ValueError("Agent list cannot be empty")
    
    K = q_list_per_agent[0].shape[0]
    assert all(q.shape == (K,) for q in q_list_per_agent), "All arrays must have same shape"
    
    # 初始化为 "无占据" (概率为 1)
    prob_no_occupation = np.ones(K, dtype=float)
    
    for q in q_list_per_agent:
        # 确保概率在 [0,1] 范围内
        q_clipped = np.clip(q, 0.0, 1.0)
        # 累积 "无占据" 概率
        prob_no_occupation *= (1.0 - q_clipped)
    
    # 并集占据概率 = 1 - "无占据"概率
    q_union = 1.0 - prob_no_occupation
    
    return np.clip(q_union, 0.0, 1.0)


def trajectory_risk_singlelayer(q_max: np.ndarray, ego_cells: List[int]) -> float:
    """计算自车轨迹在单层占据图上的碰撞风险。
    
    使用简化的加性风险模型：
    risk = min(1, Σ_{g∈ego_cells} q_max(g))
    
    Args:
        q_max: 形状 (K,) 的占据概率数组
        ego_cells: 自车占用的网格单元索引列表
        
    Returns:
        轨迹碰撞风险，值域 [0, 1]
    """
    assert q_max.ndim == 1, "q_max must be 1-dimensional"
    K = q_max.shape[0]
    
    if not ego_cells:
        return 0.0
    
    assert all(0 <= idx < K for idx in ego_cells), "Ego cell indices out of range"
    
    # 累积自车占用区域的风险
    total_risk = 0.0
    for idx in ego_cells:
        total_risk += q_max[idx]
    
    return min(1.0, total_risk)
