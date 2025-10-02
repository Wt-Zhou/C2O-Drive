from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Set
import numpy as np
from dataclasses import dataclass, field
import hashlib
import json
from scipy.optimize import linear_sum_assignment


@dataclass
class MDPStateAction:
    """MDP状态-动作对（用于匹配）"""
    # 状态S: 环境完整状态
    ego_position: Tuple[float, float]
    ego_velocity: Tuple[float, float] 
    ego_heading: float
    agents_states: List[Tuple[float, float, float, float, float, str]]  # 按agent_id排序
    
    # 动作A: 自车动作序列
    ego_action_trajectory: List[Tuple[float, float]]  # horizon长度


@dataclass  
class AgentEpisodeData:
    """agent的单个episode数据"""
    episode_id: int
    agent_id: int
    agent_type: str
    initial_mdp: MDPStateAction
    agent_trajectory_cells: List[int]  # horizon长度的agent轨迹
    
    # 预计算的哈希值（用于快速匹配）
    agent_count_hash: int
    ego_spatial_hash: int


@dataclass 
class ScenarioState:
    """场景初始状态（保持向后兼容）"""
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


@dataclass
class AgentTrajectoryData:
    """单个智能体的轨迹数据（保持向后兼容）"""
    agent_id: int
    agent_type: str
    init_position: Tuple[float, float]
    init_velocity: Tuple[float, float]
    init_heading: float
    trajectory_cells: List[int]  # 轨迹中每个时刻的网格单元ID


class HighPerformanceTrajectoryBuffer:
    """高性能MDP状态-动作匹配的轨迹缓冲区"""

    def __init__(self, horizon: Optional[int] = None,
                 spatial_resolution: Optional[float] = None,
                 ego_action_resolution: Optional[float] = None):
        # 从全局配置读取参数
        from carla_c2osr.config import get_global_config
        config = get_global_config()

        if horizon is None:
            horizon = config.time.default_horizon
        if spatial_resolution is None:
            spatial_resolution = config.matching.spatial_resolution
        if ego_action_resolution is None:
            ego_action_resolution = config.matching.ego_action_resolution

        self.horizon = horizon

        # 主存储：agent_id -> List[AgentEpisodeData]
        self._agent_episodes: Dict[int, List[AgentEpisodeData]] = {}

        # 一级索引：按agent数量快速过滤
        self._agent_count_index: Dict[int, Set[int]] = {}  # agent_count -> set(episode_ids)

        # 二级索引：按自车初始位置空间分区
        self._ego_spatial_index: Dict[Tuple[int, int], Set[int]] = {}  # (grid_x, grid_y) -> set(episode_ids)

        # 三级索引：按自车动作轨迹特征（可选，用于极端性能要求）
        self._ego_action_index: Dict[str, Set[int]] = {}  # action_signature -> set(episode_ids)

        # 索引参数
        self.spatial_resolution = spatial_resolution
        self.ego_action_resolution = ego_action_resolution
        
        # 全局episode计数器
        self._episode_counter = 0
        
        # episode_id到AgentEpisodeData的映射（用于快速查找）
        self._episode_lookup: Dict[int, AgentEpisodeData] = {}

    def _get_spatial_key(self, position: Tuple[float, float]) -> Tuple[int, int]:
        """将位置映射到空间网格key"""
        x, y = position
        return (int(x // self.spatial_resolution), int(y // self.spatial_resolution))
    
    def _get_action_signature(self, ego_action: List[Tuple[float, float]]) -> str:
        """生成动作轨迹的特征签名"""
        # 简化为起点、终点、中点的组合
        if len(ego_action) == 0:
            return "empty"
        start = ego_action[0]
        end = ego_action[-1] 
        mid = ego_action[len(ego_action)//2] if len(ego_action) > 1 else start
        
        # 量化到分辨率网格
        start_key = self._get_spatial_key(start)
        end_key = self._get_spatial_key(end)
        mid_key = self._get_spatial_key(mid)
        
        return f"{start_key}_{mid_key}_{end_key}"

    def store_agent_episode(self, agent_id: int, agent_type: str,
                           initial_ego_state: Tuple[float, float, float],
                           initial_agents_states: List[Tuple[float, float, float, float, float, str]],
                           ego_action_trajectory: List[Tuple[float, float]],
                           agent_trajectory_cells: List[int],
                           episode_id: Optional[int] = None) -> None:
        """存储agent episode数据并更新所有索引"""
        
        if episode_id is None:
            episode_id = self._episode_counter
            self._episode_counter += 1
        
        # 创建episode数据
        mdp_state_action = MDPStateAction(
            ego_position=initial_ego_state[:2],
            ego_velocity=(0.0, 0.0),  # 简化处理，可以后续扩展
            ego_heading=initial_ego_state[2] if len(initial_ego_state) == 3 else 0.0,
            agents_states=sorted(initial_agents_states, key=lambda x: x[5]),  # 按类型排序保证一致性
            ego_action_trajectory=ego_action_trajectory
        )
        
        episode_data = AgentEpisodeData(
            episode_id=episode_id,
            agent_id=agent_id,
            agent_type=agent_type,
            initial_mdp=mdp_state_action,
            agent_trajectory_cells=agent_trajectory_cells,
            agent_count_hash=len(initial_agents_states),
            ego_spatial_hash=hash(self._get_spatial_key(initial_ego_state[:2]))
        )
        
        # 存储到主数据结构
        if agent_id not in self._agent_episodes:
            self._agent_episodes[agent_id] = []
        self._agent_episodes[agent_id].append(episode_data)
        
        # 存储到episode查找表
        self._episode_lookup[episode_id] = episode_data
        
        # 更新所有索引
        self._update_indices(episode_data)

    def _update_indices(self, episode_data: AgentEpisodeData):
        """更新所有索引结构"""
        episode_id = episode_data.episode_id
        
        # 更新agent数量索引
        agent_count = episode_data.agent_count_hash
        if agent_count not in self._agent_count_index:
            self._agent_count_index[agent_count] = set()
        self._agent_count_index[agent_count].add(episode_id)
        
        # 更新空间索引
        spatial_key = self._get_spatial_key(episode_data.initial_mdp.ego_position)
        if spatial_key not in self._ego_spatial_index:
            self._ego_spatial_index[spatial_key] = set()
        self._ego_spatial_index[spatial_key].add(episode_id)
        
        # 更新动作索引
        action_sig = self._get_action_signature(episode_data.initial_mdp.ego_action_trajectory)
        if action_sig not in self._ego_action_index:
            self._ego_action_index[action_sig] = set()
        self._ego_action_index[action_sig].add(episode_id)

    def get_agent_historical_transitions_strict_matching(
        self,
        agent_id: int,
        current_ego_state: Tuple[float, float, float],  # (x, y, heading)
        current_agents_states: List[Tuple[float, float, float, float, float, str]],
        ego_action_trajectory: List[Tuple[float, float]],

        # 匹配阈值
        ego_state_threshold: float = 3.0,
        agents_state_threshold: float = 3.0,
        ego_action_threshold: float = 2.0,

        # 调试选项
        debug: bool = False

    ) -> Dict[int, List[int]]:  # {timestep: [agent_cells]}
        """严格匹配的历史轨迹检索"""

        results: Dict[int, List[int]] = {t: [] for t in range(1, self.horizon + 1)}

        if agent_id not in self._agent_episodes:
            if debug:
                print(f"  [Debug] Agent {agent_id} 没有历史数据")
            return results

        # 第一步：多级索引快速过滤候选episodes
        candidate_episodes = self._filter_candidates_with_index(
            len(current_agents_states),
            current_ego_state[:2],
            ego_action_trajectory,
            agent_id
        )

        if debug:
            print(f"  [Debug] Agent {agent_id}: 索引过滤后有 {len(candidate_episodes)} 个候选episodes")
        
        # 第二步：对候选episodes进行严格匹配
        total_matched_timesteps = 0
        for episode_data in candidate_episodes:
            # 严格匹配MDP状态-动作
            matched_timesteps = self._strict_match_mdp(
                episode_data,
                current_ego_state,
                current_agents_states,
                ego_action_trajectory,
                ego_state_threshold,
                agents_state_threshold,
                ego_action_threshold
            )

            # 将匹配成功的时间步数据加入结果
            for t in matched_timesteps:
                # t从1开始,需要转换为agent_trajectory_cells的索引(从0开始)
                cell_idx = t - 1
                if cell_idx < len(episode_data.agent_trajectory_cells):
                    results[t].append(episode_data.agent_trajectory_cells[cell_idx])
                    total_matched_timesteps += 1

        if debug:
            total_cells = sum(len(cells) for cells in results.values())
            print(f"  [Debug] Agent {agent_id}: 匹配成功 {total_matched_timesteps} 个时间步, 总计 {total_cells} 个历史cells")

        return results

    def _filter_candidates_with_index(self, agent_count, ego_pos, ego_action, agent_id):
        """使用多级索引快速过滤候选episodes"""
        
        # 一级过滤：agent数量必须完全匹配
        if agent_count not in self._agent_count_index:
            return []
        candidate_episode_ids = self._agent_count_index[agent_count].copy()
        
        # 二级过滤：自车初始位置空间邻近
        ego_spatial_key = self._get_spatial_key(ego_pos)
        spatial_candidates = set()
        
        # 搜索当前网格及其邻近网格
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                neighbor_key = (ego_spatial_key[0] + dx, ego_spatial_key[1] + dy)
                if neighbor_key in self._ego_spatial_index:
                    spatial_candidates.update(self._ego_spatial_index[neighbor_key])
        
        candidate_episode_ids &= spatial_candidates
        
        # 三级过滤：自车动作特征匹配（可选）
        # 注意：只在索引中有该签名时才过滤，否则跳过此级过滤避免丢失数据
        if self._ego_action_index:
            action_sig = self._get_action_signature(ego_action)
            if action_sig in self._ego_action_index:
                candidate_episode_ids &= self._ego_action_index[action_sig]
            # 如果action_sig不在索引中，保留所有候选（不过滤）
        
        # 根据episode_id获取实际的episode数据（只返回指定agent的episodes）
        candidates = []
        if agent_id in self._agent_episodes:
            for episode in self._agent_episodes[agent_id]:
                if episode.episode_id in candidate_episode_ids:
                    candidates.append(episode)
        
        return candidates

    def _strict_match_mdp(self, episode_data, current_ego_state, current_agents, 
                         ego_action, ego_thresh, agents_thresh, action_thresh):
        """严格的MDP状态-动作匹配，返回匹配成功的时间步列表"""
        
        matched_timesteps = []
        historical_mdp = episode_data.initial_mdp
        
        # 1. 自车初始状态匹配
        ego_pos_diff = np.linalg.norm(np.array(current_ego_state[:2]) - 
                                     np.array(historical_mdp.ego_position))
        
        if ego_pos_diff > ego_thresh:
            return matched_timesteps
        
        # 2. 环境agents状态严格匹配（相同数量 + 位置匹配）
        if len(current_agents) != len(historical_mdp.agents_states):
            return matched_timesteps
        
        if not self._match_all_agents(current_agents, historical_mdp.agents_states, agents_thresh):
            return matched_timesteps
        
        # 3. 逐时间步自车动作匹配
        for t in range(min(len(ego_action), len(historical_mdp.ego_action_trajectory))):
            action_diff = np.linalg.norm(
                np.array(ego_action[t]) -
                np.array(historical_mdp.ego_action_trajectory[t])
            )

            if action_diff <= action_thresh:
                matched_timesteps.append(t + 1)  # timestep从1开始,与reachable_sets对齐
        
        return matched_timesteps

    def _match_all_agents(self, current_agents, historical_agents, threshold):
        """严格的环境agents匹配 - 使用匈牙利算法保证全局最优"""
        if len(current_agents) != len(historical_agents):
            return False
        
        # 构建距离矩阵
        n = len(current_agents)
        if n == 0:
            return True
            
        distance_matrix = np.zeros((n, n))
        
        for i, curr_agent in enumerate(current_agents):
            for j, hist_agent in enumerate(historical_agents):
                # 检查agent类型是否匹配
                if curr_agent[5] != hist_agent[5]:  # agent_type不匹配
                    distance_matrix[i][j] = float('inf')
                    continue
                    
                # 计算位置距离
                pos_diff = np.linalg.norm(np.array(curr_agent[:2]) - np.array(hist_agent[:2]))
                vel_diff = np.linalg.norm(np.array(curr_agent[2:4]) - np.array(hist_agent[2:4]))
                
                distance_matrix[i][j] = pos_diff + vel_diff
        
        # 使用匈牙利算法找最优匹配
        try:
            row_indices, col_indices = linear_sum_assignment(distance_matrix)
            
            # 检查所有匹配是否都在阈值内
            for i, j in zip(row_indices, col_indices):
                if distance_matrix[i][j] > threshold:
                    return False
            
            return True
        except:
            # 如果匈牙利算法失败，使用简单的贪心匹配
            return self._greedy_match_agents(current_agents, historical_agents, threshold)

    def _greedy_match_agents(self, current_agents, historical_agents, threshold):
        """贪心agent匹配作为备选方案"""
        used_historical = set()
        
        for curr_agent in current_agents:
            best_match = None
            best_distance = float('inf')
            
            for j, hist_agent in enumerate(historical_agents):
                if j in used_historical:
                    continue
                    
                # 检查类型匹配
                if curr_agent[5] != hist_agent[5]:
                    continue
                
                # 计算距离
                pos_diff = np.linalg.norm(np.array(curr_agent[:2]) - np.array(hist_agent[:2]))
                vel_diff = np.linalg.norm(np.array(curr_agent[2:4]) - np.array(hist_agent[2:4]))
                distance = pos_diff + vel_diff
                
                if distance < best_distance:
                    best_distance = distance
                    best_match = j
            
            if best_match is None or best_distance > threshold:
                return False
                
            used_historical.add(best_match)
        
        return True

    # ===== 向后兼容接口 =====
    
    def get_agent_fuzzy_historical_transitions(self, scenario_state: ScenarioState, agent_id: int,
                                             timestep: int = 0, 
                                             position_threshold: float = 2.0,
                                             velocity_threshold: float = 1.0,
                                             heading_threshold: float = 0.5) -> List[int]:
        """向后兼容的模糊匹配接口"""
        # 使用简化的匹配逻辑，只匹配单个时间步
        current_ego_state = (scenario_state.ego_position[0], scenario_state.ego_position[1], scenario_state.ego_heading)
        
        # 创建简化的ego action（只有当前位置）
        ego_action = [scenario_state.ego_position]
        
        # 调用新的严格匹配接口，但使用宽松阈值
        results = self.get_agent_historical_transitions_strict_matching(
            agent_id=agent_id,
            current_ego_state=current_ego_state,
            current_agents_states=scenario_state.agents_states,
            ego_action_trajectory=ego_action,
            ego_state_threshold=position_threshold,
            agents_state_threshold=position_threshold,
            ego_action_threshold=position_threshold
        )
        
        # 返回指定timestep的结果
        return results.get(timestep, [])

    def store_episode_trajectories(self, scenario_state: ScenarioState, episode_id: int, 
                                 trajectories_data: List[AgentTrajectoryData]) -> None:
        """向后兼容的存储接口"""
        # 为每个agent存储轨迹数据
        for traj_data in trajectories_data:
            # 创建简化的ego action（假设自车保持当前位置）
            ego_action = [scenario_state.ego_position] * len(traj_data.trajectory_cells)
            
            # 创建initial ego state
            initial_ego_state = (scenario_state.ego_position[0], scenario_state.ego_position[1], scenario_state.ego_heading)
            
            self.store_agent_episode(
                agent_id=traj_data.agent_id,
                agent_type=traj_data.agent_type,
                initial_ego_state=initial_ego_state,
                initial_agents_states=scenario_state.agents_states,
                ego_action_trajectory=ego_action,
                agent_trajectory_cells=traj_data.trajectory_cells,
                episode_id=episode_id
            )

    def store_episode_trajectories_by_timestep(self, episode_id: int,
                                             timestep_scenarios: List[Tuple[ScenarioState, List[AgentTrajectoryData]]],
                                             ego_trajectory: List[Tuple[float, float]] = None) -> None:
        """向后兼容的按时间步存储接口（支持数据增强倍数）"""
        # 从全局配置读取存储倍数
        from carla_c2osr.config import get_global_config
        config = get_global_config()
        storage_multiplier = config.matching.trajectory_storage_multiplier

        for timestep, (scenario_state, trajectories_data) in enumerate(timestep_scenarios):
            # 为每个agent存储轨迹数据，使用timestep作为episode_id的一部分
            for traj_data in trajectories_data:
                # 创建与查询时一致的自车动作轨迹
                if ego_trajectory is not None:
                    # 使用传入的自车轨迹，从当前时间步开始的未来轨迹
                    ego_action = []
                    for action_t in range(timestep, min(timestep + len(traj_data.trajectory_cells), len(ego_trajectory))):
                        ego_action.append(tuple(ego_trajectory[action_t]))
                else:
                    # 回退到简化的当前位置
                    ego_action = [scenario_state.ego_position] * len(traj_data.trajectory_cells)

                # 创建initial ego state
                initial_ego_state = (scenario_state.ego_position[0], scenario_state.ego_position[1], scenario_state.ego_heading)

                # 根据存储倍数，重复存储多次
                for rep_idx in range(storage_multiplier):
                    # 使用episode_id + timestep + rep_idx创建唯一标识
                    unique_episode_id = f"{episode_id}_t{timestep}_r{rep_idx}"

                    self.store_agent_episode(
                        agent_id=traj_data.agent_id,
                        agent_type=traj_data.agent_type,
                        initial_ego_state=initial_ego_state,
                        initial_agents_states=scenario_state.agents_states,
                        ego_action_trajectory=ego_action,
                        agent_trajectory_cells=traj_data.trajectory_cells,
                        episode_id=hash(unique_episode_id)  # 转换为整数ID
                    )

    def clear(self) -> None:
        """清空buffer"""
        self._agent_episodes.clear()
        self._agent_count_index.clear()
        self._ego_spatial_index.clear()
        self._ego_action_index.clear()
        self._episode_lookup.clear()
        self._episode_counter = 0

    def get_stats(self) -> Dict:
        """获取buffer统计信息"""
        total_episodes = len(self._episode_lookup)
        total_agents = len(self._agent_episodes)
        total_agent_episodes = sum(len(episodes) for episodes in self._agent_episodes.values())

        return {
            'total_episodes': total_episodes,
            'total_agents': total_agents,
            'total_agent_episodes': total_agent_episodes,
            'agent_count_index_size': len(self._agent_count_index),
            'spatial_index_size': len(self._ego_spatial_index),
            'action_index_size': len(self._ego_action_index)
        }

    def to_dict(self) -> Dict:
        """序列化buffer状态为字典

        Returns:
            包含所有内部状态的字典
        """
        # 序列化AgentEpisodeData
        agent_episodes_serialized = {}
        for agent_id, episodes in self._agent_episodes.items():
            agent_episodes_serialized[str(agent_id)] = [
                {
                    'episode_id': ep.episode_id,
                    'agent_id': ep.agent_id,
                    'agent_type': ep.agent_type,
                    'initial_mdp': {
                        'ego_position': ep.initial_mdp.ego_position,
                        'ego_velocity': ep.initial_mdp.ego_velocity,
                        'ego_heading': ep.initial_mdp.ego_heading,
                        'agents_states': ep.initial_mdp.agents_states,
                        'ego_action_trajectory': ep.initial_mdp.ego_action_trajectory
                    },
                    'agent_trajectory_cells': ep.agent_trajectory_cells,
                    'agent_count_hash': ep.agent_count_hash,
                    'ego_spatial_hash': ep.ego_spatial_hash
                }
                for ep in episodes
            ]

        # 序列化索引
        agent_count_index_serialized = {
            str(k): list(v) for k, v in self._agent_count_index.items()
        }

        ego_spatial_index_serialized = {
            f"{k[0]}_{k[1]}": list(v) for k, v in self._ego_spatial_index.items()
        }

        ego_action_index_serialized = {
            k: list(v) for k, v in self._ego_action_index.items()
        }

        episode_lookup_serialized = {
            str(episode_id): {
                'episode_id': ep.episode_id,
                'agent_id': ep.agent_id,
                'agent_type': ep.agent_type,
                'initial_mdp': {
                    'ego_position': ep.initial_mdp.ego_position,
                    'ego_velocity': ep.initial_mdp.ego_velocity,
                    'ego_heading': ep.initial_mdp.ego_heading,
                    'agents_states': ep.initial_mdp.agents_states,
                    'ego_action_trajectory': ep.initial_mdp.ego_action_trajectory
                },
                'agent_trajectory_cells': ep.agent_trajectory_cells,
                'agent_count_hash': ep.agent_count_hash,
                'ego_spatial_hash': ep.ego_spatial_hash
            }
            for episode_id, ep in self._episode_lookup.items()
        }

        return {
            'horizon': self.horizon,
            'spatial_resolution': self.spatial_resolution,
            'ego_action_resolution': self.ego_action_resolution,
            'episode_counter': self._episode_counter,
            'agent_episodes': agent_episodes_serialized,
            'agent_count_index': agent_count_index_serialized,
            'ego_spatial_index': ego_spatial_index_serialized,
            'ego_action_index': ego_action_index_serialized,
            'episode_lookup': episode_lookup_serialized
        }

    @staticmethod
    def from_dict(data: Dict) -> 'HighPerformanceTrajectoryBuffer':
        """从字典恢复buffer状态

        Args:
            data: 序列化的字典

        Returns:
            恢复的TrajectoryBuffer实例
        """
        # 创建buffer实例
        buffer = HighPerformanceTrajectoryBuffer(
            horizon=data['horizon'],
            spatial_resolution=data['spatial_resolution'],
            ego_action_resolution=data['ego_action_resolution']
        )

        # 恢复episode计数器
        buffer._episode_counter = data['episode_counter']

        # 恢复agent_episodes
        for agent_id_str, episodes in data['agent_episodes'].items():
            agent_id = int(agent_id_str)
            buffer._agent_episodes[agent_id] = []

            for ep_data in episodes:
                mdp = MDPStateAction(
                    ego_position=tuple(ep_data['initial_mdp']['ego_position']),
                    ego_velocity=tuple(ep_data['initial_mdp']['ego_velocity']),
                    ego_heading=ep_data['initial_mdp']['ego_heading'],
                    agents_states=[tuple(s) for s in ep_data['initial_mdp']['agents_states']],
                    ego_action_trajectory=[tuple(pos) for pos in ep_data['initial_mdp']['ego_action_trajectory']]
                )

                episode_data = AgentEpisodeData(
                    episode_id=ep_data['episode_id'],
                    agent_id=ep_data['agent_id'],
                    agent_type=ep_data['agent_type'],
                    initial_mdp=mdp,
                    agent_trajectory_cells=ep_data['agent_trajectory_cells'],
                    agent_count_hash=ep_data['agent_count_hash'],
                    ego_spatial_hash=ep_data['ego_spatial_hash']
                )

                buffer._agent_episodes[agent_id].append(episode_data)

        # 恢复索引
        buffer._agent_count_index = {
            int(k): set(v) for k, v in data['agent_count_index'].items()
        }

        buffer._ego_spatial_index = {
            tuple(map(int, k.split('_'))): set(v)
            for k, v in data['ego_spatial_index'].items()
        }

        buffer._ego_action_index = {
            k: set(v) for k, v in data['ego_action_index'].items()
        }

        # 恢复episode_lookup
        for episode_id_str, ep_data in data['episode_lookup'].items():
            episode_id = int(episode_id_str)

            mdp = MDPStateAction(
                ego_position=tuple(ep_data['initial_mdp']['ego_position']),
                ego_velocity=tuple(ep_data['initial_mdp']['ego_velocity']),
                ego_heading=ep_data['initial_mdp']['ego_heading'],
                agents_states=[tuple(s) for s in ep_data['initial_mdp']['agents_states']],
                ego_action_trajectory=[tuple(pos) for pos in ep_data['initial_mdp']['ego_action_trajectory']]
            )

            episode_data = AgentEpisodeData(
                episode_id=ep_data['episode_id'],
                agent_id=ep_data['agent_id'],
                agent_type=ep_data['agent_type'],
                initial_mdp=mdp,
                agent_trajectory_cells=ep_data['agent_trajectory_cells'],
                agent_count_hash=ep_data['agent_count_hash'],
                ego_spatial_hash=ep_data['ego_spatial_hash']
            )

            buffer._episode_lookup[episode_id] = episode_data

        return buffer


# 为了向后兼容，将新的HighPerformanceTrajectoryBuffer作为默认的TrajectoryBuffer
TrajectoryBuffer = HighPerformanceTrajectoryBuffer