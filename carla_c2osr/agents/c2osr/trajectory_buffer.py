from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass, field
import hashlib
import json


@dataclass
class AgentTrajectoryData:
    """单个智能体的轨迹数据"""
    agent_id: int
    agent_type: str
    init_position: Tuple[float, float]
    init_velocity: Tuple[float, float]
    init_heading: float
    trajectory_cells: List[int]  # 轨迹中每个时刻的网格单元ID
    
    
@dataclass 
class ScenarioState:
    """场景初始状态"""
    ego_position: Tuple[float, float]
    ego_velocity: Tuple[float, float]
    ego_heading: float
    agents_states: List[Tuple[float, float, float, float, float, str]]  # (x,y,vx,vy,heading,type)
    
    def to_hash(self) -> str:
        """生成场景状态的哈希值用作索引"""
        # 将状态转换为字符串并生成哈希
        state_str = json.dumps({
            'ego': [self.ego_position, self.ego_velocity, self.ego_heading],
            'agents': self.agents_states
        }, sort_keys=True)
        return hashlib.md5(state_str.encode()).hexdigest()


class TrajectoryBuffer:
    """高效的轨迹数据存储和检索系统"""
    
    def __init__(self):
        # 主存储：scenario_hash -> episode_id -> List[AgentTrajectoryData]
        self._buffer: Dict[str, Dict[int, List[AgentTrajectoryData]]] = {}
        # 索引：scenario_hash -> agent_id -> List[episode_ids]
        self._agent_index: Dict[str, Dict[int, List[int]]] = {}
        
    def store_episode_trajectories(self, scenario_state: ScenarioState, episode_id: int, 
                                 trajectories_data: List[AgentTrajectoryData]) -> None:
        """存储一个episode的所有智能体轨迹数据"""
        scenario_hash = scenario_state.to_hash()
        
        # 初始化存储结构
        if scenario_hash not in self._buffer:
            self._buffer[scenario_hash] = {}
            self._agent_index[scenario_hash] = {}
            
        # 存储轨迹数据
        self._buffer[scenario_hash][episode_id] = trajectories_data
        
        # 更新索引
        for traj_data in trajectories_data:
            agent_id = traj_data.agent_id
            if agent_id not in self._agent_index[scenario_hash]:
                self._agent_index[scenario_hash][agent_id] = []
            self._agent_index[scenario_hash][agent_id].append(episode_id)
    
    def get_agent_historical_transitions(self, scenario_state: ScenarioState, agent_id: int, 
                                       timestep: int = 0) -> List[int]:
        """获取指定智能体在指定时刻的历史转移数据
        
        Args:
            scenario_state: 场景初始状态
            agent_id: 智能体ID
            timestep: 时刻步数（0表示第一步转移）
            
        Returns:
            历史中该智能体在该时刻转移到的网格单元ID列表
        """
        scenario_hash = scenario_state.to_hash()
        
        if scenario_hash not in self._buffer:
            return []
            
        if agent_id not in self._agent_index[scenario_hash]:
            return []
            
        transitions = []
        for episode_id in self._agent_index[scenario_hash][agent_id]:
            if episode_id in self._buffer[scenario_hash]:
                # 找到该智能体在该episode的轨迹数据
                for traj_data in self._buffer[scenario_hash][episode_id]:
                    if traj_data.agent_id == agent_id:
                        if timestep < len(traj_data.trajectory_cells):
                            transitions.append(traj_data.trajectory_cells[timestep])
                        break
        
        return transitions
    
    def get_agent_all_historical_transitions(self, scenario_state: ScenarioState, 
                                           agent_id: int) -> Dict[int, List[int]]:
        """获取智能体所有时刻的历史转移数据
        
        Returns:
            Dict[timestep, List[cell_ids]]
        """
        scenario_hash = scenario_state.to_hash()
        
        if scenario_hash not in self._buffer:
            return {}
            
        if agent_id not in self._agent_index[scenario_hash]:
            return {}
            
        all_transitions = {}
        for episode_id in self._agent_index[scenario_hash][agent_id]:
            if episode_id in self._buffer[scenario_hash]:
                for traj_data in self._buffer[scenario_hash][episode_id]:
                    if traj_data.agent_id == agent_id:
                        for t, cell_id in enumerate(traj_data.trajectory_cells):
                            if t not in all_transitions:
                                all_transitions[t] = []
                            all_transitions[t].append(cell_id)
                        break
        
        return all_transitions
    
    def get_episode_count(self, scenario_state: ScenarioState) -> int:
        """获取指定场景的episode数量"""
        scenario_hash = scenario_state.to_hash()
        if scenario_hash not in self._buffer:
            return 0
        return len(self._buffer[scenario_hash])
    
    def get_agent_episode_count(self, scenario_state: ScenarioState, agent_id: int) -> int:
        """获取指定智能体在指定场景下的episode数量"""
        scenario_hash = scenario_state.to_hash()
        if scenario_hash not in self._agent_index:
            return 0
        if agent_id not in self._agent_index[scenario_hash]:
            return 0
        return len(self._agent_index[scenario_hash][agent_id])
    
    def clear(self) -> None:
        """清空buffer"""
        self._buffer.clear()
        self._agent_index.clear()
    
    def get_stats(self) -> Dict:
        """获取buffer统计信息"""
        total_scenarios = len(self._buffer)
        total_episodes = sum(len(episodes) for episodes in self._buffer.values())
        total_trajectories = sum(
            len(trajectories) 
            for episodes in self._buffer.values() 
            for trajectories in episodes.values()
        )
        
        return {
            'total_scenarios': total_scenarios,
            'total_episodes': total_episodes,
            'total_trajectories': total_trajectories
        }
