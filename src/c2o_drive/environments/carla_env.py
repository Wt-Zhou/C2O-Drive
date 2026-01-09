"""CARLA仿真环境的Gymnasium封装

将CARLA仿真器封装为标准Gym环境接口，使其可以与算法无缝集成。
"""

from __future__ import annotations
from typing import Dict, Any, Tuple, Optional, List, TYPE_CHECKING
import numpy as np

from c2o_drive.core.environment import DrivingEnvironment, StepResult, Box
from c2o_drive.environments.carla.types import WorldState, EgoControl
from c2o_drive.environments.rewards import RewardFunction, create_default_reward
from c2o_drive.environments.carla.scenarios import (
    CarlaScenarioLibrary,
    ScenarioDefinition,
)

if TYPE_CHECKING:
    from c2o_drive.environments.carla.carla_scenario_1 import CarlaSimulator


class CarlaEnvironment(DrivingEnvironment[WorldState, EgoControl]):
    """CARLA仿真环境的Gym封装

    封装CarlaSimulator，提供标准的reset/step接口。
    """

    def __init__(
        self,
        host: str = 'localhost',
        port: int = 2000,
        town: str = 'Town03',
        dt: float = 0.1,
        max_episode_steps: int = 500,
        reward_fn: Optional[RewardFunction] = None,
        num_vehicles: int = 10,
        num_pedestrians: int = 5,
        no_rendering: bool = False,
        sim_dt: float = 0.1,
    ):
        """初始化CARLA环境

        Args:
            host: CARLA服务器地址
            port: CARLA服务器端口
            town: 地图名称
            dt: 时间步长
            max_episode_steps: 最大步数
            reward_fn: 奖励函数（可选）
            num_vehicles: 环境车辆数
            num_pedestrians: 行人数
        """
        self.host = host
        self.port = port
        self.town = town
        self.dt = dt
        self.max_episode_steps = max_episode_steps
        self.reward_fn = reward_fn or create_default_reward()
        self.num_vehicles = num_vehicles
        self.num_pedestrians = num_pedestrians
        self.no_rendering = no_rendering
        self.sim_dt = sim_dt
        self._substeps = max(1, int(round(self.dt / self.sim_dt)))
        self._substeps = max(1, self._substeps)

        # 初始化CARLA仿真器
        self.simulator: Optional[CarlaSimulator] = None
        self._current_state: Optional[WorldState] = None
        self._step_count = 0
        self._episode_reward = 0.0

        # 轨迹记录
        self._episode_trajectory: List[Dict[str, Any]] = []
        self._previous_action: Optional[EgoControl] = None

    def _ensure_connected(self):
        """确保与CARLA服务器连接"""
        if self.simulator is None:
            # Lazy import to avoid loading CARLA dependencies at module import time
            from c2o_drive.environments.carla.simulator import CarlaSimulator
            self.simulator = CarlaSimulator(
                host=self.host,
                port=self.port,
                town=self.town,
                dt=self.sim_dt,
                no_rendering=self.no_rendering,
            )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[WorldState, Dict[str, Any]]:
        """重置环境并开始新episode

        Args:
            seed: 随机种子
            options: 额外选项，可包含：
                - 'scenario_config': 场景配置
                - 'reference_path': 参考路径

        Returns:
            (初始WorldState, info字典)
        """
        if seed is not None:
            np.random.seed(seed)

        # 确保连接
        self._ensure_connected()

        # 创建场景（create_scenario内部会自动调用cleanup）
        options = options or {}
        scenario_config = options.get('scenario_config', {})
        scenario_def = scenario_config.get('scenario')
        scenario_name = scenario_config.get('scenario_name')

        if isinstance(scenario_def, str):
            scenario_def = CarlaScenarioLibrary.get_scenario(scenario_def)
        if scenario_def is None and scenario_name:
            scenario_def = CarlaScenarioLibrary.get_scenario(scenario_name)

        if scenario_def is not None:
            ego_spawn = CarlaScenarioLibrary.spawn_to_transform(scenario_def.ego_spawn)
            agent_spawns = [
                CarlaScenarioLibrary.spawn_to_transform(spawn)
                for spawn in scenario_def.agent_spawns
            ]
            autopilot = scenario_def.autopilot
        else:
            # 使用默认 scenario 而不是空场景
            scenario_def = CarlaScenarioLibrary.get_scenario('s4_wrong_way')
            ego_spawn = CarlaScenarioLibrary.spawn_to_transform(scenario_def.ego_spawn)
            agent_spawns = [
                CarlaScenarioLibrary.spawn_to_transform(spawn)
                for spawn in scenario_def.agent_spawns
            ]
            autopilot = scenario_def.autopilot

        # 使用simulator创建场景
        # 传递scenario的metadata以支持不同类型的agent（如自行车）
        metadata = scenario_def.metadata if scenario_def is not None else None
        self._current_state = self.simulator.create_scenario(
            ego_spawn=ego_spawn,
            agent_spawns=agent_spawns,
            agent_autopilot=autopilot,
            metadata=metadata,
        )

        self._step_count = 0
        self._episode_reward = 0.0
        self._episode_trajectory = []  # 清空轨迹记录
        self._previous_action = None

        # 存储预定义的agent轨迹（如果有）
        self._agent_trajectories = None
        if scenario_def is not None and scenario_def.metadata is not None:
            self._agent_trajectories = scenario_def.metadata.get('agent_trajectories')

        reference_path = options.get('reference_path')
        if reference_path is None and scenario_def is not None:
            reference_path = CarlaScenarioLibrary.get_reference_path(
                scenario_def,
                horizon=self.max_episode_steps,
                dt=self.dt,
            )

        info = {
            'town': self.town,
            'episode': 0,
            'reference_path': reference_path,
            'scenario': scenario_def.name if scenario_def else 'default',
        }

        return self._current_state, info

    def step(self, action: EgoControl) -> StepResult[WorldState]:
        """执行一步仿真

        Args:
            action: 控制动作

        Returns:
            StepResult包含下一个状态、奖励等
        """
        if self._current_state is None:
            raise RuntimeError("Must call reset() before step()")

        # 执行动作，考虑CARLA内部更小的时间步
        next_state = self._current_state
        collision_occurred = False

        # 如果有预定义的agent轨迹，执行它们
        if self._agent_trajectories is not None:
            for agent_idx, trajectory in self._agent_trajectories.items():
                if agent_idx < len(self.simulator.env_vehicles) and self._step_count < len(trajectory) - 1:
                    # 获取当前和下一个位置
                    current_pos = trajectory[self._step_count]
                    next_pos = trajectory[self._step_count + 1]

                    # 为该车辆设置目标速度向量
                    vehicle = self.simulator.env_vehicles[agent_idx]
                    dx = next_pos[0] - current_pos[0]
                    dy = next_pos[1] - current_pos[1]

                    # 计算速度（基于时间步长）
                    vx = dx / self.dt
                    vy = dy / self.dt

                    # 计算朝向
                    import math
                    if abs(dx) > 0.01 or abs(dy) > 0.01:
                        target_yaw = math.degrees(math.atan2(dy, dx))

                        # 设置朝向和速度
                        try:
                            from carla import Vector3D, Transform, Rotation
                            # 更新朝向
                            current_transform = vehicle.get_transform()
                            new_rotation = Rotation(
                                pitch=current_transform.rotation.pitch,
                                yaw=target_yaw,
                                roll=current_transform.rotation.roll
                            )
                            vehicle.set_transform(Transform(current_transform.location, new_rotation))

                            # 设置速度
                            velocity_vector = Vector3D(x=vx, y=vy, z=0)
                            vehicle.set_target_velocity(velocity_vector)
                        except Exception as e:
                            pass

        for _ in range(self._substeps):
            next_state = self.simulator.step(action)
            if self._check_collision(next_state):
                collision_occurred = True
                break

        # 计算奖励
        reward = self._calculate_reward(
            self._current_state,
            action,
            next_state,
        )

        # 检测终止条件
        terminated = collision_occurred or self._check_collision(next_state)
        truncated = (self._step_count >= self.max_episode_steps - 1)

        # 计算动力学信息（用于奖励和info）
        acceleration = self._calculate_acceleration(action, next_state)
        jerk = self._calculate_jerk(action)

        # 获取碰撞传感器状态
        collision_sensor_triggered = self.simulator.is_collision_occurred() if self.simulator else False

        # 更新状态
        self._current_state = next_state
        self._step_count += 1
        self._episode_reward += reward

        # 记录轨迹
        self._episode_trajectory.append({
            'step': self._step_count - 1,
            'state': next_state,
            'action': action,
            'reward': reward,
            'terminated': terminated,
            'truncated': truncated,
            'acceleration': acceleration,
            'jerk': jerk,
        })
        self._previous_action = action

        # 完善info字典
        info = {
            'collision': terminated,
            'collision_sensor': collision_sensor_triggered,
            'step': self._step_count,
            'episode_reward': self._episode_reward,
            'acceleration': acceleration,
            'jerk': jerk,
        }

        return StepResult(
            observation=next_state,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )


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

        优先使用CARLA碰撞传感器数据，辅以距离检测作为备份。
        根据agent类型使用不同的碰撞阈值。
        """
        # 优先使用CARLA碰撞传感器
        if self.simulator and self.simulator.is_collision_occurred():
            return True

        # 备份：简单距离检测（根据agent类型使用不同阈值）
        ego_pos = np.array(state.ego.position_m)

        # 导入AgentType枚举
        from c2o_drive.core.types import AgentType

        for agent in state.agents:
            agent_pos = np.array(agent.position_m)
            distance = np.linalg.norm(ego_pos - agent_pos)

            # 根据agent类型设置不同的碰撞阈值
            if agent.agent_type == AgentType.BICYCLE:
                collision_threshold = 2.0  # 自行车：较小的碰撞距离
            elif agent.agent_type == AgentType.PEDESTRIAN:
                collision_threshold = 1.5  # 行人：最小的碰撞距离
            elif agent.agent_type == AgentType.MOTORCYCLE:
                collision_threshold = 2.5  # 摩托车：中等碰撞距离
            else:  # VEHICLE
                collision_threshold = 3.5  # 汽车：较大的碰撞距离

            if distance < collision_threshold:
                return True

        return False

    def _calculate_acceleration(self, action: EgoControl, next_state: WorldState) -> float:
        """计算加速度（简化估计）

        使用油门和刹车来估计加速度大小。
        """
        max_accel = 3.0  # m/s^2
        max_decel = 6.0  # m/s^2

        if action.throttle > 0:
            return action.throttle * max_accel
        elif action.brake > 0:
            return -action.brake * max_decel
        return 0.0

    def _calculate_jerk(self, action: EgoControl) -> float:
        """计算急动度（加速度变化率）

        简化为相邻两个动作之间的加速度差异。
        """
        if self._previous_action is None:
            return 0.0

        current_accel = self._calculate_acceleration(action, self._current_state)
        previous_accel = self._calculate_acceleration(self._previous_action, self._current_state)

        # 急动度 = 加速度变化 / 时间步长
        jerk = abs(current_accel - previous_accel) / self.dt
        return jerk

    def get_episode_trajectory(self) -> List[Dict[str, Any]]:
        """获取当前episode的完整轨迹记录

        Returns:
            轨迹记录列表，每个元素包含：
            - step: 步数
            - state: WorldState
            - action: EgoControl
            - reward: 奖励值
            - terminated: 是否终止
            - truncated: 是否截断
            - acceleration: 加速度
            - jerk: 急动度
        """
        return self._episode_trajectory

    @property
    def observation_space(self):
        """观测空间（WorldState无固定shape，返回None）"""
        return None  # WorldState是结构化数据

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
        """渲染（CARLA自带渲染）"""
        pass

    def visualize_trajectory(
        self,
        save_path: Optional[str] = None,
        show_agents: bool = True,
        show_rewards: bool = True,
    ):
        """可视化episode轨迹

        生成matplotlib图表，显示ego和agents的轨迹。

        Args:
            save_path: 保存路径（可选），如果为None则显示图表
            show_agents: 是否显示agent轨迹
            show_rewards: 是否显示奖励曲线
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
        except ImportError:
            print("警告: matplotlib未安装，无法生成可视化")
            return

        if len(self._episode_trajectory) == 0:
            print("警告: 没有轨迹数据可供可视化")
            return

        # 创建子图
        if show_rewards:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))

        # 提取轨迹数据
        ego_positions = []
        agent_trajectories = {}  # {agent_id: [(x, y), ...]}
        rewards = []
        collision_step = None

        for i, record in enumerate(self._episode_trajectory):
            state = record['state']

            # Ego轨迹
            ego_positions.append(state.ego.position_m)

            # Agent轨迹
            for agent in state.agents:
                if agent.agent_id not in agent_trajectories:
                    agent_trajectories[agent.agent_id] = []
                agent_trajectories[agent.agent_id].append(agent.position_m)

            # 奖励
            rewards.append(record['reward'])

            # 碰撞检测
            if record['terminated'] and collision_step is None:
                collision_step = i

        # 绘制轨迹图
        ego_positions = np.array(ego_positions)
        ego_plot = ego_positions.copy()
        ego_plot[:, 1] *= -1.0
        ax1.plot(ego_plot[:, 0], ego_plot[:, 1],
                'b-o', linewidth=2, markersize=4, label='Ego Vehicle')

        # 标记起点和终点
        ax1.plot(ego_plot[0, 0], ego_plot[0, 1],
                'go', markersize=12, label='Start')
        ax1.plot(ego_plot[-1, 0], ego_plot[-1, 1],
                'rs' if collision_step is not None else 'r*',
                markersize=12, label='End (Collision)' if collision_step else 'End')

        # 绘制agent轨迹
        if show_agents:
            colors = ['orange', 'purple', 'cyan', 'magenta', 'yellow', 'pink']
            for idx, (agent_id, positions) in enumerate(agent_trajectories.items()):
                positions = np.array(positions)
                positions[:, 1] *= -1.0
                color = colors[idx % len(colors)]
                ax1.plot(positions[:, 0], positions[:, 1],
                        '--', color=color, linewidth=1.5, alpha=0.7,
                        label=f'Agent {agent_id}')

        ax1.set_xlabel('X Position (m)', fontsize=12)
        ax1.set_ylabel('Y Position (m)', fontsize=12)
        ax1.set_title('Vehicle Trajectories', fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')

        # 绘制奖励曲线
        if show_rewards and len(rewards) > 0:
            steps = np.arange(len(rewards))
            ax2.plot(steps, rewards, 'g-', linewidth=2, label='Step Reward')
            ax2.plot(steps, np.cumsum(rewards), 'b--', linewidth=2, label='Cumulative Reward')

            if collision_step is not None:
                ax2.axvline(x=collision_step, color='r', linestyle='--',
                           linewidth=2, label=f'Collision (step {collision_step})')

            ax2.set_xlabel('Step', fontsize=12)
            ax2.set_ylabel('Reward', fontsize=12)
            ax2.set_title('Reward Progression', fontsize=14, fontweight='bold')
            ax2.legend(loc='best')
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存或显示
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"可视化已保存: {save_path}")
        else:
            plt.show()

        plt.close(fig)

    def close(self):
        """关闭环境"""
        if self.simulator is not None:
            # CarlaSimulator会自动清理
            self.simulator = None


# Backwards compatible alias used throughout the repo
CarlaEnv = CarlaEnvironment

__all__ = ['CarlaEnvironment', 'CarlaEnv']
