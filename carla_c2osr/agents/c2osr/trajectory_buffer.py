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
        """存储一个episode的所有智能体轨迹数据（基于初始状态）"""
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

    def store_episode_trajectories_by_timestep(self, episode_id: int, 
                                             timestep_scenarios: List[Tuple[ScenarioState, List[AgentTrajectoryData]]]) -> None:
        """按时间步存储episode轨迹数据（每个时刻的状态都独立存储）
        
        Args:
            episode_id: episode ID
            timestep_scenarios: 列表，每个元素为(timestep_scenario_state, trajectories_data)
        """
        for timestep, (scenario_state, trajectories_data) in enumerate(timestep_scenarios):
            scenario_hash = scenario_state.to_hash()
            
            # 初始化存储结构
            if scenario_hash not in self._buffer:
                self._buffer[scenario_hash] = {}
                self._agent_index[scenario_hash] = {}
            
            # 为每个时刻创建唯一的episode_id
            timestep_episode_id = f"{episode_id}_t{timestep}"
            
            # 存储轨迹数据
            self._buffer[scenario_hash][timestep_episode_id] = trajectories_data
            
            # 更新索引
            for traj_data in trajectories_data:
                agent_id = traj_data.agent_id
                if agent_id not in self._agent_index[scenario_hash]:
                    self._agent_index[scenario_hash][agent_id] = []
                self._agent_index[scenario_hash][agent_id].append(timestep_episode_id)
    
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

    def get_agent_fuzzy_historical_transitions(self, scenario_state: ScenarioState, agent_id: int,
                                             timestep: int = 0, 
                                             position_threshold: float = 2.0,
                                             velocity_threshold: float = 1.0,
                                             heading_threshold: float = 0.5) -> List[int]:
        """获取指定智能体在相似状态下的历史转移数据（模糊匹配）
        
        Args:
            scenario_state: 当前场景状态
            agent_id: 智能体ID
            timestep: 时刻步数（0表示第一步转移）
            position_threshold: 位置相似度阈值（米）
            velocity_threshold: 速度相似度阈值（米/秒）
            heading_threshold: 朝向相似度阈值（弧度）
            
        Returns:
            历史中该智能体在相似状态下该时刻转移到的网格单元ID列表
        """
        transitions = []
        
        # 获取当前智能体的状态
        current_agent_state = None
        for i, agent_state in enumerate(scenario_state.agents_states):
            if i + 1 == agent_id:  # agent_id从1开始
                current_agent_state = agent_state
                break
        
        if current_agent_state is None:
            return transitions
        
        current_pos = np.array(current_agent_state[:2])  # x, y
        current_vel = np.array(current_agent_state[2:4])  # vx, vy
        current_heading = current_agent_state[4]
        current_type = current_agent_state[5]
        
        # 遍历所有存储的场景
        for scenario_hash, episodes in self._buffer.items():
            if agent_id not in self._agent_index.get(scenario_hash, {}):
                continue
                
            for episode_id in self._agent_index[scenario_hash][agent_id]:
                if episode_id not in episodes:
                    continue
                    
                # 找到该智能体在该episode的轨迹数据
                for traj_data in episodes[episode_id]:
                    if traj_data.agent_id == agent_id and traj_data.agent_type == current_type:
                        # 计算状态相似度
                        pos_diff = np.linalg.norm(np.array(traj_data.init_position) - current_pos)
                        vel_diff = np.linalg.norm(np.array(traj_data.init_velocity) - current_vel)
                        heading_diff = abs(traj_data.init_heading - current_heading)
                        
                        # 处理角度差异的周期性
                        heading_diff = min(heading_diff, 2 * np.pi - heading_diff)
                        
                        # 检查是否满足相似度阈值
                        if (pos_diff <= position_threshold and 
                            vel_diff <= velocity_threshold and 
                            heading_diff <= heading_threshold):
                            
                            if timestep < len(traj_data.trajectory_cells):
                                transitions.append(traj_data.trajectory_cells[timestep])
                        break
        
        return transitions

    def get_agent_fuzzy_historical_transitions_weighted(self, scenario_state: ScenarioState, agent_id: int,
                                                      timestep: int = 0,
                                                      position_weight: float = 1.0,
                                                      velocity_weight: float = 0.5,
                                                      heading_weight: float = 0.3,
                                                      similarity_threshold: float = 0.7) -> List[Tuple[int, float]]:
        """获取指定智能体在相似状态下的历史转移数据（带权重的模糊匹配）
        
        Args:
            scenario_state: 当前场景状态
            agent_id: 智能体ID
            timestep: 时刻步数（0表示第一步转移）
            position_weight: 位置权重
            velocity_weight: 速度权重
            heading_weight: 朝向权重
            similarity_threshold: 相似度阈值（0-1）
            
        Returns:
            历史中该智能体在相似状态下该时刻转移到的网格单元ID列表，每个元素为(cell_id, similarity_score)
        """
        weighted_transitions = []
        
        # 获取当前智能体的状态
        current_agent_state = None
        for i, agent_state in enumerate(scenario_state.agents_states):
            if i + 1 == agent_id:  # agent_id从1开始
                current_agent_state = agent_state
                break
        
        if current_agent_state is None:
            return weighted_transitions
        
        current_pos = np.array(current_agent_state[:2])  # x, y
        current_vel = np.array(current_agent_state[2:4])  # vx, vy
        current_heading = current_agent_state[4]
        current_type = current_agent_state[5]
        
        # 遍历所有存储的场景
        for scenario_hash, episodes in self._buffer.items():
            if agent_id not in self._agent_index.get(scenario_hash, {}):
                continue
                
            for episode_id in self._agent_index[scenario_hash][agent_id]:
                if episode_id not in episodes:
                    continue
                    
                # 找到该智能体在该episode的轨迹数据
                for traj_data in episodes[episode_id]:
                    if traj_data.agent_id == agent_id and traj_data.agent_type == current_type:
                        # 计算状态相似度
                        pos_diff = np.linalg.norm(np.array(traj_data.init_position) - current_pos)
                        vel_diff = np.linalg.norm(np.array(traj_data.init_velocity) - current_vel)
                        heading_diff = abs(traj_data.init_heading - current_heading)
                        
                        # 处理角度差异的周期性
                        heading_diff = min(heading_diff, 2 * np.pi - heading_diff)
                        
                        # 计算归一化的相似度分数
                        pos_similarity = max(0, 1 - pos_diff / 5.0)  # 5米作为最大差异
                        vel_similarity = max(0, 1 - vel_diff / 3.0)  # 3m/s作为最大差异
                        heading_similarity = max(0, 1 - heading_diff / np.pi)  # π弧度作为最大差异
                        
                        # 计算加权相似度
                        total_similarity = (position_weight * pos_similarity + 
                                          velocity_weight * vel_similarity + 
                                          heading_weight * heading_similarity) / (position_weight + velocity_weight + heading_weight)
                        
                        # 检查是否满足相似度阈值
                        if total_similarity >= similarity_threshold:
                            if timestep < len(traj_data.trajectory_cells):
                                weighted_transitions.append((traj_data.trajectory_cells[timestep], total_similarity))
                        break
        
        return weighted_transitions
    
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

