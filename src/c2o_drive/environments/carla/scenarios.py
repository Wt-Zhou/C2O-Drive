"""CARLA 场景库

包含 S1-S5 场景定义，用于 C2OSR 算法测试。
各场景的具体坐标和参数需根据实际地图配置。"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

try:
    from carla import Transform, Location, Rotation
except ImportError:
    # Fallback for type hints when CARLA is not available
    Transform = Any
    Location = Any
    Rotation = Any


@dataclass
class ScenarioDefinition:
    """场景定义"""

    name: str
    description: str
    town: str  # CARLA地图名称（Town01-Town10）
    ego_spawn: Tuple[float, float, float, float]  # (x, y, z, yaw)
    agent_spawns: List[Tuple[float, float, float, float]]  # List of (x, y, z, yaw)
    reference_path_mode: str = "straight"  # straight, curve, s_curve
    autopilot: bool = False
    difficulty: str = "medium"  # easy, medium, hard
    metadata: Optional[Dict[str, Any]] = None  # 额外参数（速度/角度范围等）


class CarlaScenarioLibrary:
    """CARLA 场景库（包含 S1-S5 场景）"""

    _SCENARIOS: Dict[str, ScenarioDefinition] = {}
    _ALIASES: Dict[str, str] = {
        "s1": "s1_scenario",
        "s2": "s2_scenario",
        "s3": "s3_scenario",
        "s4": "s4_pedestrian_crossing",
        "s5": "s5_scenario",
    }

    @staticmethod
    def get_scenario(name: str) -> ScenarioDefinition:
        """根据名称获取场景定义

        Args:
            name: 场景名称（支持别名，如 "s4" → "s4_wrong_way"）

        Returns:
            ScenarioDefinition对象

        Raises:
            ValueError: 如果场景名称不存在
        """
        # 检查是否是别名
        if name in CarlaScenarioLibrary._ALIASES:
            name = CarlaScenarioLibrary._ALIASES[name]

        if not CarlaScenarioLibrary._SCENARIOS:
            CarlaScenarioLibrary._SCENARIOS = {
                "s1_scenario": CarlaScenarioLibrary.s1_scenario(),
                "s2_scenario": CarlaScenarioLibrary.s2_scenario(),
                "s3_scenario": CarlaScenarioLibrary.s3_scenario(),
                "s4_pedestrian_crossing": CarlaScenarioLibrary.s4_pedestrian_crossing(),
                "s5_scenario": CarlaScenarioLibrary.s5_scenario(),
            }

        if name not in CarlaScenarioLibrary._SCENARIOS:
            raise ValueError(
                f"Unknown scenario: {name}. "
                f"Available: {list(CarlaScenarioLibrary._SCENARIOS.keys())} "
                f"(aliases: {list(CarlaScenarioLibrary._ALIASES.keys())})"
            )

        return CarlaScenarioLibrary._SCENARIOS[name]

    @staticmethod
    def list_scenarios() -> List[str]:
        """列出所有可用场景"""
        if not CarlaScenarioLibrary._SCENARIOS:
            CarlaScenarioLibrary.get_scenario("s1_scenario")
        return list(CarlaScenarioLibrary._SCENARIOS.keys())

    @staticmethod
    def spawn_to_transform(spawn: Tuple[float, float, float, float]) -> Transform:
        """将 (x, y, z, yaw) spawn 转换为 CARLA Transform。"""
        if Transform is None:
            raise ImportError(
                "CARLA Transform 未导入，无法创建场景。请确认 CARLA PythonAPI 已正确安装。"
            )
        x, y, z, yaw = spawn
        return Transform(Location(x=x, y=y, z=z), Rotation(yaw=yaw))

    @staticmethod
    def s1_scenario() -> ScenarioDefinition:
        """S1: 环境车逆行场景

        场景描述：对向逆行车辆切入本车道

        参数说明：
        - 自车沿 Town03 东向车道 (-90°) 南行
        - 对向车辆在 (12.8, -123.0) 处以约 100° 角逆行切入该车道
        - metadata 提供速度/入射角范围，便于上层随机化
        """
        ego_spawn = (5.5, -90.0, 0.5, -90.0)
        wrong_way_spawn = (12.8, -123.0, 1.0, 125.0)

        metadata = {
            "source": "TestScenario_Town03_Waymo_long_tail.py",
            "agent_types": ["vehicle"],  # 逆行车辆
            "agent_speed_range_mps": (5.0, 10.0),   # 初速度 3~8 m/s
            "entry_angle_range_deg": (-20.0, 20.0),  # 以 100° 为中心的偏差
        }

        return ScenarioDefinition(
            name="s1_scenario",
            description="环境车逆行场景",
            town="Town03",
            ego_spawn=ego_spawn,
            agent_spawns=[wrong_way_spawn],
            reference_path_mode="straight",
            autopilot=False,
            difficulty="hard",
            metadata=metadata,
        )

    @staticmethod
    def s2_scenario() -> ScenarioDefinition:
        """S2: 右侧车辆变道切入场景

        场景描述：环境车从右侧自行车道/非机动车道变道切入自车前方

        参数说明：
        - 自车沿 Town03 东向车道南行（与s1相同位置）
        - Cut-in车辆在右侧后方，会切入自车车道
        - 包含1-2辆背景车辆增加场景复杂度

        坐标系统：
        - X轴正方向：东侧（自车右侧）
        - Y轴负方向：南侧（自车前方）
        - Yaw=-90°：朝南行驶
        """
        # 自车位置（与s1相同）
        ego_spawn = (5.5, -90.0, 0.5, -90.0)

        # Agent 1: 右侧后方的cut-in车辆
        cut_in_vehicle = (10.5, -85.0, 0.5, -90.0)  # 右侧5米，后方10米

        # # Agent 2: 前方背景车辆
        # front_vehicle = (5.5, -110.0, 0.5, -90.0)   # 前方20米

        # Agent 3: 后方背景车辆
        rear_vehicle = (5.5, -60.0, 0.5, -90.0)     # 后方30米

        metadata = {
            "source": "user_defined",
            "agent_types": ["vehicle", "vehicle"],  # 切入车辆、后方车辆
            "agent_speed_range_mps": (5.0, 10.0),
            "cut_in_direction": "right",  # 标记cut-in方向
            "agent_trajectories": {
                0: [  # 第一辆车（cut-in车辆）的轨迹
                    (10.5, -85),   # 起始位置（右侧后方）
                    (10.5, -90.0),   # 向前行驶
                    (9.5, -95.0),    # 开始切入
                    (8.0, -100.0),    # 继续切入
                    (6.5, -105.0),   # 接近自车道
                    (5.5, -110.0),   # 完成切入到自车道
                    (5.5, -115.0),   # 在自车道继续行驶
                    (5.5, -120.0),   # 继续前进
                ]
            }
        }

        return ScenarioDefinition(
            name="s2_scenario",
            description="右侧车辆变道切入场景",
            town="Town03",
            ego_spawn=ego_spawn,
            # agent_spawns=[cut_in_vehicle, front_vehicle, rear_vehicle],
            agent_spawns=[cut_in_vehicle, rear_vehicle],
            reference_path_mode="straight",
            autopilot=False,
            difficulty="medium",
            metadata=metadata,
        )

    @staticmethod
    def s3_scenario() -> ScenarioDefinition:
        """S3: 右侧自行车波浪行驶场景

        场景描述：自车驾驶中，右侧自行车道有一辆自行车进行波浪线行驶

        参数说明：
        - 自车沿 Town03 东向车道南行（与s1/s2相同位置）
        - 自行车在右前方自行车道，进行波浪线（左右摆动）行驶
        - 包含1-2辆背景车辆模拟正常交通流

        坐标系统：
        - X轴正方向：东侧（自车右侧）
        - Y轴负方向：南侧（自车前方）
        - Yaw=-90°：朝南行驶

        注意：
        - 自行车的波浪运动需要在 simulator 控制层实现
        - spawn 仅设置初始位置，运动模式通过 metadata 标记
        """
        # 自车位置（与s1/s2相同）
        ego_spawn = (9.5, -90.0, 0.5, -90.0)

        # Agent 1: 右前方波浪行驶的自行车
        bicycle = (11.5, -100.0, 0.5, -90.0)  # 右侧6米（自行车道），前方10米

        # Agent 2: 前方背景车辆
        front_vehicle = (5.5, -100.0, 0.5, -90.0)  # 前方30米

        # Agent 3: 后方背景车辆
        rear_vehicle = (5.5, -90.0, 0.5, -90.0)    # 后方20米

        metadata = {
            "source": "user_defined",
            "agent_types": ["bicycle", "vehicle", "vehicle"],  # 类型：自行车、汽车、汽车
            "agent_blueprints": ["vehicle.bh.crossbike", None, None],  # CARLA blueprint IDs
            "vehicle_types": ["bicycle", "car", "car"],  # 车辆类型
            "agent_categories": ["bicycle", "vehicle", "vehicle"],  # 分类
            "agent_speed_range_mps": (1, 2),  # 自行车速度较慢
            "motion_pattern": "wave",  # 标记波浪运动模式
            "wave_amplitude": 1.5,     # 波浪幅度（米）
            "wave_frequency": 0.3,     # 波浪频率（Hz）
            "agent_trajectories": {
                0: [  # 自行车的波浪轨迹（1.5 m/s速度，每步0.15米）
                    (11.5, -100.00),    # 起始位置
                    (11.5, -100.15),    # 直行
                    (11.4, -100.30),    # 开始向左
                    (11.3, -100.45),
                    (11.1, -100.60),    # 向左摆动
                    (10.9, -100.75),
                    (10.7, -100.90),    # 最左侧
                    (10.6, -101.05),
                    (10.5, -101.20),    # 接近主车道
                    (10.6, -101.35),    # 开始向右
                    (10.7, -101.50),
                    (10.9, -101.65),
                    (11.1, -101.80),    # 回到中心
                    (11.3, -101.95),
                    (11.5, -102.10),    # 中心位置
                    (11.7, -102.25),    # 开始向右
                    (11.9, -102.40),
                    (12.1, -102.55),    # 向右摆动
                    (12.3, -102.70),
                    (12.5, -102.85),    # 最右侧
                    (12.6, -103.00),
                    (12.7, -103.15),    # 远离主车道
                    (12.6, -103.30),    # 开始向左
                    (12.5, -103.45),
                    (12.3, -103.60),
                    (12.1, -103.75),    # 回到中心
                    (11.9, -103.90),
                    (11.7, -104.05),
                    (11.5, -104.20),    # 中心位置
                    (11.3, -104.35),    # 再次向左
                    (11.1, -104.50),
                    (10.9, -104.65),    # 向左摆动
                    (10.7, -104.80),
                    (10.5, -104.95),    # 最左侧
                    (10.6, -105.10),    # 开始向右
                    (10.7, -105.25),
                    (10.9, -105.40),
                    (11.1, -105.55),
                    (11.3, -105.70),
                    (11.5, -105.85),    # 回到中心
                    (11.5, -106.00),    # 继续前进
                ]
            }
        }

        return ScenarioDefinition(
            name="s3_scenario",
            description="右侧自行车波浪行驶场景",
            town="Town03",
            ego_spawn=ego_spawn,
            agent_spawns=[bicycle, front_vehicle, rear_vehicle],
            reference_path_mode="straight",
            autopilot=False,
            difficulty="medium",
            metadata=metadata,
        )

    @staticmethod
    def s4_pedestrian_crossing() -> ScenarioDefinition:
        """S4: 行人横穿突然减速场景

        场景描述：自车前进时，在人行横道处一个行人从右侧横穿马路，但突然减速并停在马路中间

        参数说明：
        - 自车沿 Town03 东向车道南行（与s3相同位置）
        - 行人在人行横道（自车前方27米）从右侧横穿马路（从东向西）
        - 行人在接近自车道时突然减速，并停在道路中间约2秒，然后缓慢继续前进

        坐标系统：
        - X轴正方向：东侧（自车右侧）
        - Y轴负方向：南侧（自车前方）
        - Yaw=180°：朝西（行人横穿方向，反向）

        注意：
        - 行人的减速/停止行为需要在 simulator 控制层实现
        - spawn 仅设置初始位置，运动模式通过 metadata 标记
        """
        # 自车位置（与s1/s2/s3相同）
        ego_spawn = (9.5, -100.0, 0.5, -90.0)

        # 行人：右侧人行道，准备横穿（人行横道位置）
        pedestrian = (13.5, -127.0, 0.5, 180.0)  # 右侧4米，前方27米（人行横道），朝西

        metadata = {
            "source": "user_defined",
            "agent_types": ["walker"],  # 行人
            "agent_blueprints": ["walker.pedestrian.0001"],  # CARLA行人blueprint
            "agent_speed_range_mps": (1.2, 1.5),  # 正常步行速度
            "motion_pattern": "crossing_decelerate",  # 横穿减速模式
            "deceleration_position_x": 9.0,  # 在道路中间减速（主车道位置）
            "target_speed_after_decel": 0.0,  # 减速后速度（0=停止）
            "deceleration_rate": 2.0,  # 减速度 m/s²
            "agent_trajectories": {
                0: [  # 行人横穿轨迹（反向：从右向左，正常速度1.3 m/s → 减速 → 停止）
                    # 第一阶段：正常速度横穿（1.3 m/s，每步0.13米，从右侧向左）
                    (13.5, -127.00),    # 起始位置（右侧人行道，人行横道）
                    (13.37, -127.00),   # 开始横穿
                    (13.24, -127.00),
                    (13.11, -127.00),
                    (12.98, -127.00),
                    (12.85, -127.00),
                    (12.72, -127.00),
                    (12.59, -127.00),
                    (12.46, -127.00),
                    (12.33, -127.00),
                    (12.20, -127.00),
                    (12.07, -127.00),
                    (11.94, -127.00),
                    (11.81, -127.00),
                    (11.68, -127.00),
                    (11.55, -127.00),
                    (11.42, -127.00),
                    (11.29, -127.00),
                    (11.16, -127.00),
                    (11.03, -127.00),
                    (10.90, -127.00),
                    (10.77, -127.00),
                    (10.64, -127.00),
                    (10.51, -127.00),   # 接近道路边缘

                    # 第二阶段：开始减速（速度逐渐降低）
                    (10.40, -127.00),   # 减速开始，步幅变小
                    (10.30, -127.00),   # 0.10米/步
                    (10.22, -127.00),   # 0.08米/步
                    (10.16, -127.00),   # 0.06米/步
                    (10.11, -127.00),   # 0.05米/步
                    (10.07, -127.00),   # 0.04米/步
                    (10.04, -127.00),   # 0.03米/步
                    (10.02, -127.00),   # 0.02米/步
                    (10.00, -127.00),   # 0.02米/步，几乎停止

                    # 第三阶段：在道路中间停止（保持位置约2秒）
                    (10.00, -127.00),   # 停止在自车道中心附近
                    (10.00, -127.00),
                    (10.00, -127.00),
                    (10.00, -127.00),
                    (10.00, -127.00),
                    (10.00, -127.00),
                    (10.00, -127.00),
                    (10.00, -127.00),
                    (10.00, -127.00),
                    (10.00, -127.00),
                    (10.00, -127.00),
                    (10.00, -127.00),
                    (10.00, -127.00),
                    (10.00, -127.00),
                    (10.00, -127.00),
                    (10.00, -127.00),
                    (10.00, -127.00),
                    (10.00, -127.00),
                    (10.00, -127.00),
                    (10.00, -127.00),   # 停止约2秒（20步）

                    # 第四阶段：犹豫后缓慢继续前进（小步移动，继续向左）
                    (9.98, -127.00),   # 缓慢重新开始
                    (9.95, -127.00),
                    (9.92, -127.00),
                    (9.88, -127.00),
                    (9.84, -127.00),
                    (9.79, -127.00),
                    (9.74, -127.00),
                    (9.68, -127.00),
                    (9.62, -127.00),
                    (9.55, -127.00),   # 经过自车道中心
                    (9.48, -127.00),   # 继续缓慢前进
                    (9.40, -127.00),
                    (9.32, -127.00),
                    (9.23, -127.00),
                    (9.14, -127.00),
                    (9.04, -127.00),
                    (8.94, -127.00),
                    (8.83, -127.00),
                    (8.72, -127.00),
                    (8.60, -127.00),   # 离开主车道（到达左侧）
                ]
            }
        }

        return ScenarioDefinition(
            name="s4_pedestrian_crossing",
            description="行人横穿突然减速场景",
            town="Town03",
            ego_spawn=ego_spawn,
            agent_spawns=[pedestrian],
            reference_path_mode="straight",
            autopilot=False,
            difficulty="high",  # 行人不可预测性高
            metadata=metadata,
        )

    @staticmethod
    def s5_scenario() -> ScenarioDefinition:
        """S5: 施工路段绕行场景

        场景描述：自车在南北向车道上北行，前方遇到施工路段有锥桶和箱子阻碍车道，
                  需要绕行到对向车道，此时对向有车辆驶来

        参数说明：
        - 自车位置：(30, -150, 0.5, 0°) 朝北行驶
        - 对向车辆：(50, -160, 0.5, 180°) 朝南驶来
        - 施工区域：(40, -150) 附近，使用锥桶和箱子等静态障碍物阻挡车道

        坐标系统：
        - X轴正方向：东侧
        - Y轴负方向：南侧
        - Yaw=0°：朝北（自车方向）
        - Yaw=180°：朝南（对向车辆方向）
        """
        # 自车位置（南北车道，朝北行驶）
        ego_spawn = (20.0, -134.0, 0.5, 0.0)

        # 对向车辆（从东向西驶来）
        oncoming_vehicle = (70.0, -137.5, 9, 180.0)

        # 施工区域障碍物（锥桶和箱子阻挡自车道）
        # 放在自车前方X=40附近（自车和对向车之间），形成施工区域
        cone1 = (40.0, -135.0, 5, 0.0)   # 锥桶1（施工区域起点）
        cone2 = (40.5, -133.0, 5, 0.0)   # 锥桶2
        cone3 = (41.0, -133.0, 5, 0.0)   # 锥桶3
        box1 = (40.5, -134.0, 5, 0.0)    # 箱子1（施工区域中央）
        box2 = (41.0, -133.5, 5, 0.0)    # 箱子2
        cone4 = (41.5, -134.0, 5, 0.0)   # 锥桶4（施工区域末端）
        cone5 = (42.0, -133.0, 5, 0.0)   # 锥桶5

        # 所有agent spawns：对向车辆 + 静态障碍物
        agent_spawns = [
            oncoming_vehicle,  # 索引0：对向车辆
            cone1,  # 索引1-7：静态障碍物
            cone2,
            cone3,
            box1,
            box2,
            cone4,
            cone5,
        ]

        metadata = {
            "source": "user_defined",
            "agent_types": ["vehicle", "obstacle", "obstacle", "obstacle", "obstacle", "obstacle", "obstacle", "obstacle"],
            "agent_blueprints": [
                "vehicle.audi.tt",              # 对向车辆
                "static.prop.trafficcone02",    # 锥桶
                "static.prop.trafficcone02",
                "static.prop.trafficcone02",
                "static.prop.box03",            # 箱子
                "static.prop.box02",
                "static.prop.trafficcone02",
                "static.prop.trafficcone02",
            ],
            "agent_speed_range_mps": (1.5, 1.5),  # 对向车辆速度1.5 m/s（匀速直线）
            "camera_position": (40.0, -150.0),  # S5场景的固定相机位置
            # S5不使用轨迹控制，对向车辆通过velocity控制实现匀速直线行驶
            # 静态障碍物不需要移动
        }

        return ScenarioDefinition(
            name="s5_scenario",
            description="施工路段绕行场景",
            town="Town03",
            ego_spawn=ego_spawn,
            agent_spawns=agent_spawns,
            reference_path_mode="straight",
            autopilot=False,
            difficulty="medium",
            metadata=metadata,
        )

    @staticmethod
    def to_carla_transform(spawn: Tuple[float, float, float, float]) -> Transform:
        """将spawn元组转换为CARLA Transform对象

        Args:
            spawn: (x, y, z, yaw)元组

        Returns:
            CARLA Transform对象
        """
        x, y, z, yaw = spawn
        return Transform(
            Location(x=x, y=y, z=z),
            Rotation(yaw=yaw)
        )

    @staticmethod
    def get_reference_path(
        scenario: ScenarioDefinition,
        horizon: int = 10,
        dt: float = 0.5,
    ) -> List[np.ndarray]:
        """根据场景生成参考路径

        Args:
            scenario: 场景定义
            horizon: 预测时域
            dt: 时间步长

        Returns:
            参考路径点列表
        """
        ego_x, ego_y, _, ego_yaw = scenario.ego_spawn

        if scenario.reference_path_mode == "straight":
            # 直线路径
            path = []
            yaw_rad = np.radians(ego_yaw)
            for i in range(horizon):
                distance = (i + 1) * dt * 5.0  # 假设速度5 m/s
                point = np.array([
                    ego_x + distance * np.cos(yaw_rad),
                    ego_y + distance * np.sin(yaw_rad)
                ])
                path.append(point)
            return path

        elif scenario.reference_path_mode == "curve":
            # 曲线路径（右转）
            path = []
            yaw_rad = np.radians(ego_yaw)
            radius = 20.0
            for i in range(horizon):
                angle = (i + 1) * 0.1  # 弯道角度
                x = ego_x + radius * (np.sin(angle) * np.cos(yaw_rad) - (1 - np.cos(angle)) * np.sin(yaw_rad))
                y = ego_y + radius * (np.sin(angle) * np.sin(yaw_rad) + (1 - np.cos(angle)) * np.cos(yaw_rad))
                path.append(np.array([x, y]))
            return path

        else:  # s_curve
            # S型曲线
            path = []
            yaw_rad = np.radians(ego_yaw)
            for i in range(horizon):
                t = (i + 1) * dt * 5.0
                lateral_offset = 5.0 * np.sin(t / 10.0)  # 正弦偏移
                x = ego_x + t * np.cos(yaw_rad) + lateral_offset * np.sin(yaw_rad)
                y = ego_y + t * np.sin(yaw_rad) - lateral_offset * np.cos(yaw_rad)
                path.append(np.array([x, y]))
            return path


# 便捷函数
def get_scenario(name: str) -> ScenarioDefinition:
    """获取场景定义（便捷函数）"""
    return CarlaScenarioLibrary.get_scenario(name)


def list_scenarios() -> List[str]:
    """列出所有场景（便捷函数）"""
    return CarlaScenarioLibrary.list_scenarios()


__all__ = [
    'ScenarioDefinition',
    'CarlaScenarioLibrary',
    'get_scenario',
    'list_scenarios',
]
