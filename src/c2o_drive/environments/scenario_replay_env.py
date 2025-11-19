"""ScenarioManager 的 Gymnasium 环境封装

将 ScenarioManager 创建的静态场景转换为符合 Gym 标准的交互式环境。
"""

from __future__ import annotations
from typing import Dict, Any, Tuple, Optional, List
import numpy as np

from c2o_drive.core.environment import DrivingEnvironment, StepResult, Box
from c2o_drive.core.types import WorldState, EgoState, AgentState, EgoControl
from c2o_drive.environments.virtual.scenario_manager import ScenarioManager
from c2o_drive.environments.rewards import RewardFunction, create_default_reward


class ScenarioReplayEnvironment(DrivingEnvironment[WorldState, EgoControl]):
    """将 ScenarioManager 场景封装为 Gym 环境

    提供标准的 reset/step 接口，使得原本用于静态轨迹回放的场景
    可以与新架构的规划器进行交互式测试。
    """

    def __init__(
        self,
        scenario_manager: Optional[ScenarioManager] = None,
        reference_path_mode: str = "straight",
        dt: float = 0.5,
        max_episode_steps: int = 100,
        reward_fn: Optional[RewardFunction] = None,
        horizon: int = 10,
    ):
        """初始化场景回放环境

        Args:
            scenario_manager: 场景管理器（如为None则创建默认）
            reference_path_mode: 参考路径模式 ("straight", "curve", "s_curve")
            dt: 时间步长（秒）
            max_episode_steps: 最大步数
            reward_fn: 奖励函数（可选）
            horizon: 规划时域
        """
        self.scenario_manager = scenario_manager or ScenarioManager()
        self.reference_path_mode = reference_path_mode
        self.dt = dt
        self.max_episode_steps = max_episode_steps
        self.reward_fn = reward_fn or create_default_reward()
        self.horizon = horizon

        # 环境状态
        self._current_state: Optional[WorldState] = None
        self._initial_state: Optional[WorldState] = None
        self._reference_path: Optional[List] = None
        self._step_count = 0
        self._episode_reward = 0.0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[WorldState, Dict[str, Any]]:
        """重置环境并开始新 episode

        Args:
            seed: 随机种子
            options: 额外选项，可包含：
                - 'reference_path_mode': 覆盖参考路径模式
                - 'horizon': 覆盖规划时域

        Returns:
            (初始 WorldState, info 字典)
        """
        if seed is not None:
            np.random.seed(seed)

        options = options or {}

        # 创建场景
        self._current_state = self.scenario_manager.create_scenario()
        self._initial_state = self._current_state

        # 生成参考路径
        path_mode = options.get('reference_path_mode', self.reference_path_mode)
        horizon = options.get('horizon', self.horizon)

        self._reference_path = self.scenario_manager.generate_reference_path(
            mode=path_mode,
            horizon=horizon,
            ego_start=self._current_state.ego.position_m
        )

        self._step_count = 0
        self._episode_reward = 0.0

        info = {
            'reference_path': self._reference_path,
            'reference_path_mode': path_mode,
            'episode': 0,
        }

        return self._current_state, info

    def step(self, action: EgoControl) -> StepResult[WorldState]:
        """执行一步仿真

        Args:
            action: 控制动作

        Returns:
            StepResult 包含下一个状态、奖励等
        """
        if self._current_state is None:
            raise RuntimeError("Must call reset() before step()")

        # 1. 根据 action 更新 ego 状态
        next_ego = self._update_ego_state(self._current_state.ego, action)

        # 2. 更新 agent 状态（使用简单的常速度模型）
        next_agents = self._update_agents_states(self._current_state.agents)

        # 3. 创建新的 WorldState
        next_state = WorldState(
            ego=next_ego,
            agents=next_agents,
            time_s=self._current_state.time_s + self.dt,
        )

        # 4. 计算奖励
        reward = self._calculate_reward(
            self._current_state,
            action,
            next_state,
        )

        # 5. 检测终止条件
        terminated = self._check_collision(next_state)
        truncated = (self._step_count >= self.max_episode_steps - 1)

        # 6. 更新状态
        self._current_state = next_state
        self._step_count += 1
        self._episode_reward += reward

        info = {
            'collision': terminated,
            'step': self._step_count,
            'episode_reward': self._episode_reward,
            'reference_path': self._reference_path,
        }

        return StepResult(
            observation=next_state,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def step_to_waypoint(
        self,
        waypoint: Tuple[float, float]
    ) -> StepResult[WorldState]:
        """直接跟踪指定的世界坐标 waypoint。

        在虚拟环境中用于绕过控制器,直接验证规划轨迹。
        """
        if self._current_state is None:
            raise RuntimeError("Must call reset() before step_to_waypoint()")

        current_ego = self._current_state.ego
        current_pos = np.array(current_ego.position_m)
        target_pos = np.array(waypoint, dtype=float)
        delta = target_pos - current_pos
        distance = np.linalg.norm(delta)

        if self.dt > 0:
            velocity = delta / self.dt
        else:
            velocity = np.zeros_like(delta)

        if distance > 1e-6:
            yaw = float(np.arctan2(delta[1], delta[0]))
        else:
            yaw = current_ego.yaw_rad

        next_ego = EgoState(
            position_m=(float(target_pos[0]), float(target_pos[1])),
            velocity_mps=(float(velocity[0]), float(velocity[1])),
            yaw_rad=yaw,
        )

        next_agents = self._update_agents_states(self._current_state.agents)
        next_state = WorldState(
            ego=next_ego,
            agents=next_agents,
            time_s=self._current_state.time_s + self.dt,
        )

        reward = self._calculate_reward(
            self._current_state,
            EgoControl(throttle=0.0, steer=0.0, brake=0.0),
            next_state,
        )

        terminated = self._check_collision(next_state)
        truncated = (self._step_count >= self.max_episode_steps - 1)

        self._current_state = next_state
        self._step_count += 1
        self._episode_reward += reward

        info = {
            'collision': terminated,
            'step': self._step_count,
            'episode_reward': self._episode_reward,
            'reference_path': self._reference_path,
        }

        return StepResult(
            observation=next_state,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def _update_ego_state(
        self,
        ego: EgoState,
        action: EgoControl,
    ) -> EgoState:
        """根据控制动作更新 ego 状态

        使用简化的运动学模型：
        - throttle/brake 控制速度变化
        - steer 控制航向角变化
        """
        # 提取当前状态
        x, y = ego.position_m
        vx, vy = ego.velocity_mps
        yaw = ego.yaw_rad

        # 计算速度大小和方向
        v = np.sqrt(vx**2 + vy**2)

        # 根据 throttle/brake 更新速度
        if action.throttle > 0:
            v += action.throttle * 5.0 * self.dt  # 加速度约5 m/s²
        if action.brake > 0:
            v -= action.brake * 5.0 * self.dt  # 制动加速度约-5 m/s²

        # 限制速度范围
        v = np.clip(v, 0.0, 20.0)

        # 根据 steer 更新航向角
        # 简化的自行车模型
        max_steer_rate = 0.5  # rad/s
        yaw += action.steer * max_steer_rate * self.dt

        # 归一化航向角到 [-π, π]
        yaw = np.arctan2(np.sin(yaw), np.cos(yaw))

        # 更新位置
        vx = v * np.cos(yaw)
        vy = v * np.sin(yaw)
        x += vx * self.dt
        y += vy * self.dt

        return EgoState(
            position_m=(x, y),
            velocity_mps=(vx, vy),
            yaw_rad=yaw,
        )

    def _update_agents_states(
        self,
        agents: List[AgentState],
    ) -> List[AgentState]:
        """更新 agents 状态（使用常速度模型）"""
        updated_agents = []

        for agent in agents:
            x, y = agent.position_m
            vx, vy = agent.velocity_mps

            # 常速度运动
            x += vx * self.dt
            y += vy * self.dt

            updated_agent = AgentState(
                agent_id=agent.agent_id,
                position_m=(x, y),
                velocity_mps=(vx, vy),
                heading_rad=agent.heading_rad,
                agent_type=agent.agent_type,
            )
            updated_agents.append(updated_agent)

        return updated_agents

    def _calculate_reward(
        self,
        state: WorldState,
        action: EgoControl,
        next_state: WorldState,
    ) -> float:
        """计算奖励"""
        return self.reward_fn.compute(state, action, next_state, {})

    def _check_collision(self, state: WorldState) -> bool:
        """检测是否发生碰撞

        使用简单的圆形碰撞检测
        """
        ego_pos = np.array(state.ego.position_m)

        # Ego 的半径（简化为圆形，假设车辆为 4.5m x 2.0m）
        ego_radius = 2.5  # 大约的半径

        collision_threshold = 3.5  # 简化圆形碰撞半径之和

        for agent in state.agents:
            agent_pos = np.array(agent.position_m)
            distance = np.linalg.norm(ego_pos - agent_pos)

            if distance < collision_threshold:
                return True

        return False

    @property
    def observation_space(self):
        """观测空间（WorldState 无固定 shape，返回 None）"""
        return None  # WorldState 是结构化数据

    @property
    def action_space(self):
        """动作空间"""
        # EgoControl: throttle[0,1], steer[-1,1], brake[0,1]
        return Box(
            low=np.array([0.0, -1.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            shape=(3,),
        )

    def render(self, mode: str = 'human'):
        """渲染（暂不实现）"""
        pass

    def close(self):
        """关闭环境"""
        pass

    def get_reference_path(self) -> Optional[List]:
        """获取当前的参考路径"""
        return self._reference_path


__all__ = ['ScenarioReplayEnvironment']
