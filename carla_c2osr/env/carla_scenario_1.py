"""
CARLA仿真器接口

简洁的CARLA接口，用于场景执行和状态获取。
与现有的WorldState/AgentState类型完全兼容。
"""

import glob
import os
import sys
import math
from typing import List, Tuple, Optional, Dict

# CARLA路径设置
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from carla import Transform, Location, Rotation, VehicleControl, Vector3D

# 导入项目类型
from carla_c2osr.env.types import AgentState, EgoState, WorldState, AgentType


class CarlaSimulator:
    """CARLA仿真器接口

    提供简洁的API用于：
    1. 场景创建和初始化
    2. 获取WorldState（与replay_openloop_refactored.py兼容）
    3. 执行自车轨迹
    4. 碰撞检测

    使用示例:
        sim = CarlaSimulator(town="Town03")

        # 创建场景
        ego_spawn = Transform(Location(x=5.5, y=-90, z=0.5), Rotation(yaw=-90))
        agent_spawns = [
            Transform(Location(x=10, y=-100, z=0.5), Rotation(yaw=0)),
            Transform(Location(x=3, y=-95, z=0.5), Rotation(yaw=90))
        ]
        world_state = sim.create_scenario(ego_spawn, agent_spawns)

        # 执行轨迹
        trajectory = [(5.5, -90), (5.5, -95), (5.5, -100)]
        states = sim.execute_trajectory(trajectory, horizon=3)

        # 清理
        sim.cleanup()
    """

    def __init__(self,
                 host: str = "localhost",
                 port: int = 2000,
                 town: str = "Town03",
                 dt: float = 0.1,
                 no_rendering: bool = False):
        """初始化CARLA仿真器连接

        Args:
            host: CARLA服务器地址
            port: CARLA服务器端口
            town: 地图名称（Town01-Town10）
            dt: 仿真时间步长（秒）
            no_rendering: 是否禁用渲染
        """
        # 连接CARLA
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()

        # 加载地图
        if self.world.get_map().name != f'Carla/Maps/{town}':
            self.world = self.client.load_world(town)
            self.world.unload_map_layer(carla.MapLayer.StreetLights)
            self.world.unload_map_layer(carla.MapLayer.Buildings)

        # 设置天气
        self.world.set_weather(carla.WeatherParameters(
            cloudiness=50,
            precipitation=10.0,
            sun_altitude_angle=30.0
        ))

        # 配置仿真设置
        settings = self.world.get_settings()
        settings.no_rendering_mode = no_rendering
        settings.fixed_delta_seconds = dt
        settings.synchronous_mode = True
        self.world.apply_settings(settings)

        # 设置交通管理器
        self.tm = self.client.get_trafficmanager(8000)
        self.tm.set_random_device_seed(0)

        # 释放所有交通灯
        self._free_traffic_lights()

        # 状态变量
        self.dt = dt
        self.ego_vehicle = None
        self.ego_collision_sensor = None
        self.ego_collision_occurred = False
        self.env_vehicles = []
        self.current_time = 0.0

        # 相机设置
        self.spectator = self.world.get_spectator()
        self.camera_height = 60.0  # 相机高度（米）
        self.camera_pitch = -90.0  # 俯视角（度）

        # 车辆蓝图
        self.ego_bp = self.world.get_blueprint_library().find('vehicle.lincoln.mkz_2020')
        self.ego_bp.set_attribute('color', '0,0,255')  # 蓝色自车
        self.ego_bp.set_attribute('role_name', 'hero')

        self.env_bp = self.world.get_blueprint_library().find('vehicle.audi.tt')
        self.env_bp.set_attribute('color', '255,0,0')  # 红色环境车辆
        self.env_bp.set_attribute('role_name', 'autopilot')

        self.collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')

        print(f"✅ CARLA仿真器已连接: {town}, dt={dt}s")

    def _free_traffic_lights(self):
        """释放所有交通灯（设置为绿灯）"""
        traffic_lights = self.world.get_actors().filter('*traffic_light*')
        for tl in traffic_lights:
            tl.set_green_time(999.0)
            tl.set_red_time(0.0)

    def _on_collision(self, event):
        """碰撞回调函数"""
        self.ego_collision_occurred = True
        print(f"⚠️ 碰撞检测: 自车与 {event.other_actor.type_id} 发生碰撞")

    def _update_camera(self, follow_ego: bool = True):
        """更新相机位置，聚焦自车俯视图

        Args:
            follow_ego: 是否跟随自车
        """
        if not follow_ego or self.ego_vehicle is None:
            return

        # 获取自车位置
        ego_location = self.ego_vehicle.get_location()

        # 设置相机到自车正上方
        camera_location = Location(
            x=ego_location.x,
            y=ego_location.y,
            z=ego_location.z + self.camera_height
        )

        camera_rotation = Rotation(
            pitch=self.camera_pitch,
            yaw=0.0,
            roll=0.0
        )

        camera_transform = Transform(camera_location, camera_rotation)
        self.spectator.set_transform(camera_transform)

    def set_camera_view(self, height: float = 60.0, pitch: float = -90.0):
        """设置相机视角参数

        Args:
            height: 相机高度（米）
            pitch: 俯仰角（度），-90为正俯视
        """
        self.camera_height = height
        self.camera_pitch = pitch
        self._update_camera()

    def create_scenario(self,
                       ego_spawn: Transform,
                       agent_spawns: List[Transform] = None,
                       agent_autopilot: bool = False) -> WorldState:
        """创建场景并返回初始WorldState

        Args:
            ego_spawn: 自车生成位置
            agent_spawns: 环境车辆生成位置列表
            agent_autopilot: 环境车辆是否启用自动驾驶

        Returns:
            初始WorldState
        """
        # 清理现有车辆
        self.cleanup()

        # 生成自车
        self.ego_vehicle = self.world.spawn_actor(self.ego_bp, ego_spawn)
        self.ego_collision_sensor = self.world.spawn_actor(
            self.collision_bp,
            Transform(),
            self.ego_vehicle,
            carla.AttachmentType.Rigid
        )
        self.ego_collision_sensor.listen(self._on_collision)
        self.ego_collision_occurred = False

        # 生成环境车辆
        self.env_vehicles = []
        if agent_spawns:
            for i, spawn in enumerate(agent_spawns):
                try:
                    vehicle = self.world.spawn_actor(self.env_bp, spawn)
                    self.env_vehicles.append(vehicle)

                    # 设置初始速度（朝向车辆前方）
                    import math
                    yaw_rad = math.radians(spawn.rotation.yaw)
                    initial_velocity = Vector3D(
                        x=2.0 * math.cos(yaw_rad),  # 初始速度2m/s
                        y=2.0 * math.sin(yaw_rad),
                        z=0
                    )
                    vehicle.set_target_velocity(initial_velocity)

                    # 配置交通管理器
                    if agent_autopilot:
                        vehicle.set_autopilot(True, self.tm.get_port())
                        self.tm.ignore_signs_percentage(vehicle, 100)
                        self.tm.ignore_lights_percentage(vehicle, 100)

                    print(f"  - 环境车辆{i+1}: 位置=({spawn.location.x:.1f}, {spawn.location.y:.1f}), "
                          f"朝向={spawn.rotation.yaw:.1f}°, 速度=2m/s")
                except RuntimeError as e:
                    print(f"⚠️ 车辆生成失败: {e}")

        # 前进一个时间步
        self.world.tick()
        self.current_time = 0.0

        # 更新相机到自车俯视图
        self._update_camera()

        # 返回初始WorldState
        world_state = self.get_world_state()
        print(f"✅ 场景已创建: 自车 + {len(self.env_vehicles)} 环境车辆")
        print(f"📷 相机已聚焦到自车俯视图 (高度={self.camera_height}m)")
        return world_state

    def get_world_state(self) -> WorldState:
        """获取当前WorldState（与现有代码兼容）

        Returns:
            当前WorldState，包含自车和所有环境车辆状态
        """
        if self.ego_vehicle is None:
            raise RuntimeError("场景未初始化，请先调用create_scenario()")

        # 获取自车状态
        ego_loc = self.ego_vehicle.get_location()
        ego_vel = self.ego_vehicle.get_velocity()
        ego_rot = self.ego_vehicle.get_transform().rotation

        ego_state = EgoState(
            position_m=(ego_loc.x, ego_loc.y),
            velocity_mps=(ego_vel.x, ego_vel.y),
            yaw_rad=math.radians(ego_rot.yaw)
        )

        # 获取环境车辆状态
        agents = []
        for i, vehicle in enumerate(self.env_vehicles):
            if vehicle.is_alive:
                v_loc = vehicle.get_location()
                v_vel = vehicle.get_velocity()
                v_rot = vehicle.get_transform().rotation

                agent_state = AgentState(
                    agent_id=f"vehicle-{i+1}",
                    position_m=(v_loc.x, v_loc.y),
                    velocity_mps=(v_vel.x, v_vel.y),
                    heading_rad=math.radians(v_rot.yaw),
                    agent_type=AgentType.VEHICLE
                )
                agents.append(agent_state)

        return WorldState(
            time_s=self.current_time,
            ego=ego_state,
            agents=agents
        )

    def step(self, ego_control: VehicleControl = None) -> WorldState:
        """执行一个仿真步

        Args:
            ego_control: 自车控制指令（可选，默认保持当前状态）

        Returns:
            新的WorldState
        """
        if self.ego_vehicle is None:
            raise RuntimeError("场景未初始化，请先调用create_scenario()")

        # 应用自车控制
        if ego_control is not None:
            self.ego_vehicle.apply_control(ego_control)

        # 前进一个时间步
        self.world.tick()
        self.current_time += self.dt

        # 更新相机跟随自车
        self._update_camera()

        return self.get_world_state()

    def execute_trajectory(self,
                          ego_trajectory: List[Tuple[float, float]],
                          horizon: int,
                          velocity: float = 5.0,
                          smooth: bool = True) -> List[WorldState]:
        """执行自车轨迹并返回WorldState序列

        Args:
            ego_trajectory: 自车轨迹 [(x0, y0), (x1, y1), ...]
            horizon: 执行步数
            velocity: 目标速度（m/s）
            smooth: 是否使用平滑控制（True=速度控制，False=传送）

        Returns:
            WorldState序列
        """
        if self.ego_vehicle is None:
            raise RuntimeError("场景未初始化，请先调用create_scenario()")

        world_states = []

        if smooth:
            # 平滑控制模式：使用速度向量跟踪轨迹
            for t in range(min(horizon, len(ego_trajectory))):
                current_loc = self.ego_vehicle.get_location()
                current_pos = (current_loc.x, current_loc.y)
                target_pos = ego_trajectory[t]

                # 计算朝向目标的速度向量
                dx = target_pos[0] - current_pos[0]
                dy = target_pos[1] - current_pos[1]
                distance = math.sqrt(dx**2 + dy**2)

                if distance > 0.1:  # 如果距离目标足够远
                    # 归一化方向向量并乘以目标速度
                    vx = (dx / distance) * velocity
                    vy = (dy / distance) * velocity

                    # 计算朝向
                    target_yaw = math.degrees(math.atan2(dy, dx))

                    # 设置车辆朝向（只设置朝向，不传送位置）
                    current_transform = self.ego_vehicle.get_transform()
                    new_rotation = Rotation(
                        pitch=current_transform.rotation.pitch,
                        yaw=target_yaw,
                        roll=current_transform.rotation.roll
                    )
                    self.ego_vehicle.set_transform(Transform(current_transform.location, new_rotation))

                    # 设置目标速度
                    velocity_vector = Vector3D(x=vx, y=vy, z=0)
                    self.ego_vehicle.set_target_velocity(velocity_vector)
                else:
                    # 已到达目标点，停止
                    self.ego_vehicle.set_target_velocity(Vector3D(x=0, y=0, z=0))

                # 前进一个时间步
                self.world.tick()
                self.current_time += self.dt

                # 更新相机跟随自车
                self._update_camera()

                # 记录WorldState
                world_states.append(self.get_world_state())
        else:
            # 传送模式：直接设置位置（快速但不平滑）
            for t in range(min(horizon, len(ego_trajectory))):
                target_pos = ego_trajectory[t]
                current_loc = self.ego_vehicle.get_location()

                # 计算朝向
                if t < len(ego_trajectory) - 1:
                    next_pos = ego_trajectory[t + 1]
                    dx = next_pos[0] - target_pos[0]
                    dy = next_pos[1] - target_pos[1]
                    target_yaw = math.degrees(math.atan2(dy, dx))
                else:
                    target_yaw = self.ego_vehicle.get_transform().rotation.yaw

                # 传送自车到目标位置
                new_transform = Transform(
                    Location(x=target_pos[0], y=target_pos[1], z=current_loc.z),
                    Rotation(yaw=target_yaw)
                )
                self.ego_vehicle.set_transform(new_transform)

                # 设置速度
                velocity_vector = Vector3D(
                    x=velocity * math.cos(math.radians(target_yaw)),
                    y=velocity * math.sin(math.radians(target_yaw)),
                    z=0
                )
                self.ego_vehicle.set_target_velocity(velocity_vector)

                # 前进一个时间步
                self.world.tick()
                self.current_time += self.dt

                # 更新相机跟随自车
                self._update_camera()

                # 记录WorldState
                world_states.append(self.get_world_state())

        return world_states

    def is_collision_occurred(self) -> bool:
        """检查是否发生碰撞"""
        return self.ego_collision_occurred

    def cleanup(self):
        """清理所有车辆和传感器"""
        # 清理碰撞传感器
        if self.ego_collision_sensor is not None:
            try:
                if self.ego_collision_sensor.is_alive:
                    self.ego_collision_sensor.destroy()
            except RuntimeError:
                pass  # Actor已被销毁，忽略错误
            except Exception as e:
                print(f"⚠️ 清理碰撞传感器时出错: {e}")
            finally:
                self.ego_collision_sensor = None

        # 清理自车
        if self.ego_vehicle is not None:
            try:
                if self.ego_vehicle.is_alive:
                    self.ego_vehicle.destroy()
            except RuntimeError:
                pass  # Actor已被销毁，忽略错误
            except Exception as e:
                print(f"⚠️ 清理自车时出错: {e}")
            finally:
                self.ego_vehicle = None

        # 清理环境车辆
        for vehicle in self.env_vehicles:
            try:
                if vehicle.is_alive:
                    vehicle.destroy()
            except RuntimeError:
                pass  # Actor已被销毁，忽略错误
            except Exception as e:
                print(f"⚠️ 清理环境车辆时出错: {e}")
        self.env_vehicles = []

        # 清理所有其他车辆（防止泄漏）
        try:
            actors = self.world.get_actors().filter('vehicle*')
            for actor in actors:
                try:
                    # 只清理非自车角色的车辆
                    if actor.is_alive and actor.attributes.get('role_name') != 'hero':
                        actor.destroy()
                except RuntimeError:
                    pass  # Actor已被销毁，忽略错误
        except Exception as e:
            print(f"⚠️ 清理残留车辆时出错: {e}")

    def set_vehicle_trajectory(self, vehicle_index: int, trajectory: List[Tuple[float, float]],
                               velocity: float = 5.0, smooth: bool = True):
        """为指定环境车辆设置轨迹执行

        Args:
            vehicle_index: 环境车辆索引（0开始）
            trajectory: 轨迹点列表 [(x0, y0), (x1, y1), ...]
            velocity: 目标速度（m/s）
            smooth: 是否平滑控制
        """
        if vehicle_index >= len(self.env_vehicles):
            raise IndexError(f"车辆索引{vehicle_index}超出范围（共{len(self.env_vehicles)}辆）")

        vehicle = self.env_vehicles[vehicle_index]

        for target_pos in trajectory:
            current_loc = vehicle.get_location()
            current_pos = (current_loc.x, current_loc.y)

            # 计算朝向目标的速度向量
            dx = target_pos[0] - current_pos[0]
            dy = target_pos[1] - current_pos[1]
            distance = math.sqrt(dx**2 + dy**2)

            if distance > 0.1:
                # 归一化并设置速度
                vx = (dx / distance) * velocity
                vy = (dy / distance) * velocity
                target_yaw = math.degrees(math.atan2(dy, dx))

                # 设置朝向
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

                # 执行一步
                self.world.tick()
                self.current_time += self.dt

    def execute_multi_vehicle_trajectories(self,
                                          ego_trajectory: List[Tuple[float, float]],
                                          agent_trajectories: Dict[int, List[Tuple[float, float]]],
                                          horizon: int,
                                          ego_velocity: float = 5.0,
                                          agent_velocities: Dict[int, float] = None,
                                          smooth: bool = True) -> List[WorldState]:
        """同时执行自车和环境车辆轨迹

        Args:
            ego_trajectory: 自车轨迹
            agent_trajectories: 环境车辆轨迹字典 {vehicle_index: trajectory}
            horizon: 执行步数
            ego_velocity: 自车目标速度
            agent_velocities: 环境车辆速度字典 {vehicle_index: velocity}，默认5.0
            smooth: 是否平滑控制

        Returns:
            WorldState序列
        """
        if self.ego_vehicle is None:
            raise RuntimeError("场景未初始化，请先调用create_scenario()")

        if agent_velocities is None:
            agent_velocities = {i: 5.0 for i in agent_trajectories.keys()}

        world_states = []

        for t in range(horizon):
            # 控制自车
            if t < len(ego_trajectory):
                ego_target = ego_trajectory[t]
                ego_current = self.ego_vehicle.get_location()

                dx = ego_target[0] - ego_current.x
                dy = ego_target[1] - ego_current.y
                distance = math.sqrt(dx**2 + dy**2)

                if distance > 0.1:
                    vx = (dx / distance) * ego_velocity
                    vy = (dy / distance) * ego_velocity
                    target_yaw = math.degrees(math.atan2(dy, dx))

                    # 设置自车朝向
                    ego_transform = self.ego_vehicle.get_transform()
                    new_rotation = Rotation(
                        pitch=ego_transform.rotation.pitch,
                        yaw=target_yaw,
                        roll=ego_transform.rotation.roll
                    )
                    self.ego_vehicle.set_transform(Transform(ego_transform.location, new_rotation))

                    # 设置自车速度
                    self.ego_vehicle.set_target_velocity(Vector3D(x=vx, y=vy, z=0))

            # 控制环境车辆
            for vehicle_idx, trajectory in agent_trajectories.items():
                if vehicle_idx < len(self.env_vehicles) and t < len(trajectory):
                    vehicle = self.env_vehicles[vehicle_idx]
                    target_pos = trajectory[t]
                    current_loc = vehicle.get_location()

                    dx = target_pos[0] - current_loc.x
                    dy = target_pos[1] - current_loc.y
                    distance = math.sqrt(dx**2 + dy**2)

                    if distance > 0.1:
                        velocity = agent_velocities.get(vehicle_idx, 5.0)
                        vx = (dx / distance) * velocity
                        vy = (dy / distance) * velocity
                        target_yaw = math.degrees(math.atan2(dy, dx))

                        # 设置车辆朝向
                        vehicle_transform = vehicle.get_transform()
                        new_rotation = Rotation(
                            pitch=vehicle_transform.rotation.pitch,
                            yaw=target_yaw,
                            roll=vehicle_transform.rotation.roll
                        )
                        vehicle.set_transform(Transform(vehicle_transform.location, new_rotation))

                        # 设置速度
                        vehicle.set_target_velocity(Vector3D(x=vx, y=vy, z=0))

            # 执行一步仿真
            self.world.tick()
            self.current_time += self.dt

            # 更新相机
            self._update_camera()

            # 记录状态
            world_states.append(self.get_world_state())

        return world_states

    def __del__(self):
        """析构函数，确保资源清理"""
        self.cleanup()


# ============================================================================
# 便捷函数
# ============================================================================

def carla_transform_from_position(x: float, y: float, z: float = 0.5,
                                 yaw: float = 0.0) -> Transform:
    """从位置和朝向创建CARLA Transform

    Args:
        x, y: 位置（米）
        z: 高度（米），默认0.5
        yaw: 朝向（度），默认0

    Returns:
        CARLA Transform
    """
    return Transform(
        Location(x=x, y=y, z=z),
        Rotation(yaw=yaw)
    )


def world_state_to_carla_spawns(world_state: WorldState) -> Tuple[Transform, List[Transform]]:
    """将WorldState转换为CARLA生成点

    Args:
        world_state: WorldState对象

    Returns:
        (ego_spawn, agent_spawns)
    """
    # 自车生成点
    ego_spawn = carla_transform_from_position(
        world_state.ego.position_m[0],
        world_state.ego.position_m[1],
        yaw=math.degrees(world_state.ego.yaw_rad)
    )

    # 环境车辆生成点
    agent_spawns = []
    for agent in world_state.agents:
        agent_spawn = carla_transform_from_position(
            agent.position_m[0],
            agent.position_m[1],
            yaw=math.degrees(agent.heading_rad)
        )
        agent_spawns.append(agent_spawn)

    return ego_spawn, agent_spawns


def generate_oncoming_trajectory(start_x: float, start_y: float,
                                 end_y: float,
                                 horizon: int,
                                 lateral_offset_range: Tuple[float, float] = (-2.0, 2.0),
                                 seed: int = None) -> List[Tuple[float, float]]:
    """生成逆行车辆的随机轨迹

    逆行车从远处向自车方向驶来，带有随机横向偏移。

    Args:
        start_x: 起始x坐标
        start_y: 起始y坐标（远处）
        end_y: 结束y坐标（接近自车）
        horizon: 轨迹点数
        lateral_offset_range: 横向偏移范围（米）
        seed: 随机种子

    Returns:
        轨迹点列表 [(x0, y0), (x1, y1), ...]
    """
    import numpy as np

    if seed is not None:
        np.random.seed(seed)

    trajectory = []

    # 基础纵向位置（匀速接近）
    y_positions = np.linspace(start_y, end_y, horizon)

    # 随机横向偏移（平滑变化）
    # 使用正弦波 + 随机噪声
    t = np.linspace(0, 2*np.pi, horizon)
    base_offset = np.sin(t) * (lateral_offset_range[1] - lateral_offset_range[0]) / 4
    noise = np.random.normal(0, 0.3, horizon)  # 随机噪声
    lateral_offsets = base_offset + noise

    # 裁剪到范围内
    lateral_offsets = np.clip(lateral_offsets,
                              lateral_offset_range[0],
                              lateral_offset_range[1])

    # 生成轨迹
    for i in range(horizon):
        x = start_x + lateral_offsets[i]
        y = y_positions[i]
        trajectory.append((x, y))

    return trajectory


def generate_straight_trajectory(start_x: float, start_y: float,
                                 direction_yaw: float,
                                 distance: float,
                                 horizon: int) -> List[Tuple[float, float]]:
    """生成直线轨迹

    Args:
        start_x: 起始x坐标
        start_y: 起始y坐标
        direction_yaw: 方向角度（度）
        distance: 总距离（米）
        horizon: 轨迹点数

    Returns:
        轨迹点列表
    """
    import numpy as np

    trajectory = []
    yaw_rad = math.radians(direction_yaw)

    for i in range(horizon):
        progress = i / max(horizon - 1, 1)
        traveled = progress * distance

        x = start_x + traveled * math.cos(yaw_rad)
        y = start_y + traveled * math.sin(yaw_rad)
        trajectory.append((x, y))

    return trajectory


# ============================================================================
# 示例用法
# ============================================================================

if __name__ == "__main__":
    # 创建仿真器
    sim = CarlaSimulator(town="Town03", dt=0.1)

    # 定义场景
    ego_spawn = carla_transform_from_position(x=5.5, y=-90, yaw=-90)
    agent_spawns = [
        carla_transform_from_position(x=10, y=-100, yaw=0),
        carla_transform_from_position(x=3, y=-95, yaw=90)
    ]

    # 创建场景
    initial_state = sim.create_scenario(ego_spawn, agent_spawns)
    print(f"初始状态: {len(initial_state.agents)} 环境车辆")
    print(f"自车位置: {initial_state.ego.position_m}")

    # 定义轨迹
    trajectory = [
        (5.5, -90),
        (5.5, -95),
        (5.5, -100),
        (5.5, -105),
        (5.5, -110)
    ]

    # 执行轨迹
    print("\n执行轨迹...")
    states = sim.execute_trajectory(trajectory, horizon=5, velocity=5.0)

    for i, state in enumerate(states):
        print(f"t={i}: 自车={state.ego.position_m}, "
              f"碰撞={sim.is_collision_occurred()}, "
              f"环境车辆数={len(state.agents)}")

    # 清理
    sim.cleanup()
    print("\n✅ 示例执行完成")
