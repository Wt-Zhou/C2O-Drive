"""
场景管理模块

提供场景创建和状态转换功能。
"""

from __future__ import annotations
from typing import List, Dict, Any
from carla_c2osr.env.types import AgentState, EgoState, WorldState, AgentType
from carla_c2osr.agents.c2osr.trajectory_buffer import ScenarioState
import numpy as np


class ScenarioManager:
    """场景管理器"""
    
    def __init__(self, grid_size_m: float = 20.0):
        """
        Args:
            grid_size_m: 网格大小
        """
        self.grid_size_m = grid_size_m
    
    def create_scenario(self) -> WorldState:
        """创建固定的mock场景。"""
        ego = EgoState(position_m=(0.0, 0.0), velocity_mps=(2.0, 0.0), yaw_rad=0.0)
        
        # 确保智能体位置在网格范围内 [-grid_size_m/2, grid_size_m/2] = [-10, 10]
        # 修改智能体位置，让它们更接近自车轨迹，增加碰撞可能性
        agent1 = AgentState(
            agent_id="vehicle-1",
            position_m=(10.0, 3.0),  # 更接近自车
            velocity_mps=(2.0, 0.0),
            heading_rad=0,
            agent_type=AgentType.VEHICLE
        )
        
        agent2 = AgentState(
            agent_id="pedestrian-1", 
            position_m=(3.0, -25.0),  
            velocity_mps=(0.8, 0.3),
            heading_rad=0.3,
            agent_type=AgentType.PEDESTRIAN
        )
        
        # 测试动态agent数量：可以选择只创建一个agent
        # return WorldState(time_s=0.0, ego=ego, agents=[agent1, agent2])  # 两个agent
        return WorldState(time_s=0.0, ego=ego, agents=[agent1])  # 只有一个agent
        # return WorldState(time_s=0.0, ego=ego, agents=[])  # 没有agent
    
    def create_scenario_state(self, world: WorldState) -> ScenarioState:
        """从WorldState创建ScenarioState用于buffer索引"""
        agents_states = []
        for agent in world.agents:
            agents_states.append((
                agent.position_m[0], agent.position_m[1],
                agent.velocity_mps[0], agent.velocity_mps[1], 
                agent.heading_rad, agent.agent_type.value
            ))
        
        return ScenarioState(
            ego_position=world.ego.position_m,
            ego_velocity=world.ego.velocity_mps,
            ego_heading=world.ego.yaw_rad,
            agents_states=agents_states
        )
    
    def create_world_state_from_trajectories(self, 
                                           t: int, 
                                           ego_trajectory: List, 
                                           agent_trajectories: Dict[int, List],
                                           world_init: WorldState) -> WorldState:
        """从轨迹创建世界状态。
        
        Args:
            t: 时刻索引
            ego_trajectory: 自车轨迹
            agent_trajectories: 智能体轨迹字典
            world_init: 初始世界状态
            
        Returns:
            当前时刻的世界状态
        """
        # 更新自车位置
        ego_world_xy = ego_trajectory[t]
        ego = EgoState(position_m=tuple(ego_world_xy), velocity_mps=(5.0, 0.0), yaw_rad=0.0)
        
        # 获取当前时刻环境智能体位置（从预生成的轨迹）
        current_agents = []
        
        for i, agent_init in enumerate(world_init.agents):
            agent_id = i + 1
            if agent_id in agent_trajectories and t < len(agent_trajectories[agent_id]):
                agent_world_xy = agent_trajectories[agent_id][t]
                
                # 用轨迹的相邻点估计当前速度与朝向，避免使用初始值导致方向错误
                horizon = len(agent_trajectories[agent_id])
                if t < horizon - 1:
                    nxt = agent_trajectories[agent_id][t + 1]
                    vel_vec = (nxt - agent_world_xy)
                elif t > 0:
                    prv = agent_trajectories[agent_id][t - 1]
                    vel_vec = (agent_world_xy - prv)
                else:
                    # 单点退化，使用初始速度
                    vel_vec = np.array(agent_init.velocity_mps)
                
                vel_tuple = (float(vel_vec[0]), float(vel_vec[1]))
                heading_est = float(np.arctan2(vel_vec[1], vel_vec[0])) if (vel_vec[0]**2 + vel_vec[1]**2) > 1e-9 else float(agent_init.heading_rad)
                
                # 创建当前智能体状态（用估计的速度与朝向）
                current_agent = AgentState(
                    agent_id=agent_init.agent_id,
                    position_m=tuple(agent_world_xy),
                    velocity_mps=vel_tuple,
                    heading_rad=heading_est,
                    agent_type=agent_init.agent_type
                )
                current_agents.append(current_agent)
            else:
                # 如果轨迹不存在，使用初始状态
                current_agents.append(agent_init)
        
        # 构建当前世界状态
        return WorldState(time_s=float(t), ego=ego, agents=current_agents)
