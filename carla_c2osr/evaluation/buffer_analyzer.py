"""
缓冲区分析模块

提供轨迹缓冲区计数计算和分析功能。
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List
from carla_c2osr.agents.c2osr.trajectory_buffer import TrajectoryBuffer, ScenarioState
from carla_c2osr.agents.c2osr.grid import GridMapper


class BufferAnalyzer:
    """缓冲区分析器"""
    
    def __init__(self, trajectory_buffer: TrajectoryBuffer):
        """
        Args:
            trajectory_buffer: 轨迹缓冲区
        """
        self.trajectory_buffer = trajectory_buffer
    
    def calculate_buffer_counts(self, 
                               scenario_state: ScenarioState,
                               agent_ids: List[int], 
                               timestep: int, 
                               grid: GridMapper) -> Dict[int, np.ndarray]:
        """从Trajectory Buffer计算每个智能体的计数向量。
        
        Args:
            scenario_state: 场景状态
            agent_ids: 智能体ID列表
            timestep: 时间步
            grid: 网格映射器
            
        Returns:
            字典 {agent_id: count_vector}，每个count_vector是K维向量
        """
        counts = {}
        for agent_id in agent_ids:
            # 获取历史转移数据
            historical_transitions = self.trajectory_buffer.get_agent_historical_transitions(
                scenario_state, agent_id, timestep=timestep
            )
            
            # 创建计数向量
            count_vector = np.zeros(grid.K, dtype=float)
            for cell_id in historical_transitions:
                count_vector[cell_id] += 1.0
                
            counts[agent_id] = count_vector
        
        return counts
    
    def calculate_fuzzy_buffer_counts(self, 
                                    scenario_state: ScenarioState,
                                    agent_ids: List[int], 
                                    timestep: int, 
                                    grid: GridMapper,
                                    position_threshold: float = 3.0,
                                    velocity_threshold: float = 2.0,
                                    heading_threshold: float = 0.8) -> Dict[int, np.ndarray]:
        """从Trajectory Buffer计算每个智能体的模糊匹配计数向量。
        
        Args:
            scenario_state: 场景状态
            agent_ids: 智能体ID列表
            timestep: 时间步
            grid: 网格映射器
            position_threshold: 位置相似度阈值（米）
            velocity_threshold: 速度相似度阈值（米/秒）
            heading_threshold: 朝向相似度阈值（弧度）
            
        Returns:
            字典 {agent_id: count_vector}，每个count_vector是K维向量
        """
        counts = {}
        for agent_id in agent_ids:
            # 获取模糊匹配的历史转移数据
            historical_transitions = self.trajectory_buffer.get_agent_fuzzy_historical_transitions(
                scenario_state, agent_id, timestep=timestep,
                position_threshold=position_threshold,
                velocity_threshold=velocity_threshold,
                heading_threshold=heading_threshold
            )
            
            # 创建计数向量
            count_vector = np.zeros(grid.K, dtype=float)
            for cell_id in historical_transitions:
                count_vector[cell_id] += 1.0
                
            counts[agent_id] = count_vector
        
        return counts
    
    def get_buffer_stats(self) -> Dict[str, int]:
        """获取缓冲区统计信息。
        
        Returns:
            缓冲区统计字典
        """
        return self.trajectory_buffer.get_stats()

