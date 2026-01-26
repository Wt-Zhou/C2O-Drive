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
        ego_spawn = (5.5, -93.0, 0.5, -90.0)
        wrong_way_spawn = (12.8, -120.0, 1.0, 125.0)

        metadata = {
            "source": "TestScenario_Town03_Waymo_long_tail.py",
            "agent_types": ["vehicle"],  # 逆行车辆
            "agent_speed_range_mps": (8.0, 14.0),   # 初速度 3~8 m/s
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
        ego_spawn = (9, -90.0, 0.5, -90.0)

        # Agent 1: 右侧后方的cut-in车辆
        # cut_in_vehicle = (10.5, -85.0, 0.5, -90.0)  # 右侧5米，后方10米
        cut_in_vehicle = (5, -85.0, 0.5, -90.0)  # 右侧5米，后方10米

        # # Agent 2: 前方背景车辆
        # front_vehicle = (5.5, -110.0, 0.5, -90.0)   # 前方20米

        # Agent 3: 后方背景车辆
        rear_vehicle = (9, -60.0, 0.5, -90.0)     # 后方30米

        metadata = {
            "source": "user_defined",
            "agent_types": ["vehicle", "vehicle"],  # 切入车辆、后方车辆
            "agent_speed_range_mps": (5.0, 10.0),
            "cut_in_direction": "right",  # 标记cut-in方向
            "agent_trajectories": {
                0: [  # 第一辆车（cut-in车辆）的轨迹
                    (5, -85.0),   # 起始位置（右侧后方）
                    (5, -90.0),   # 向前行驶
                    (6, -95.0),    # 开始切入
                    (7.5, -100.0),    # 继续切入
                    (9, -105.0),   # 接近自车道
                    (9, -110.0),   # 完成切入到自车道
                    (9, -115.0),   # 在自车道继续行驶
                    (9, -120.0),   # 继续前进
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
        ego_spawn = (9.5, -95.0, 0.5, -90.0)

        # Agent 1: 右前方波浪行驶的自行车
        bicycle = (11.5, -100.0, 0.5, -90.0)  # 右侧6米（自行车道），前方10米

        # # Agent 2: 前方背景车辆
        # front_vehicle = (5.5, -100.0, 0.5, -90.0)  # 前方30米

        # # Agent 3: 后方背景车辆
        # rear_vehicle = (5.5, -90.0, 0.5, -90.0)    # 后方20米

        # 原配置（3个agents：自行车+2辆车）
        # metadata = {
        #     "source": "user_defined",
        #     "agent_types": ["bicycle", "vehicle", "vehicle"],
        #     "agent_blueprints": ["vehicle.bh.crossbike", None, None],
        #     "vehicle_types": ["bicycle", "car", "car"],
        #     "agent_categories": ["bicycle", "vehicle", "vehicle"],
        #     "agent_speed_range_mps": (1, 2),
        #     "motion_pattern": "wave",
        #     "wave_amplitude": 1.5,
        #     "wave_frequency": 0.3,

        # 简化配置（只有1个agent：自行车）
        metadata = {
            "source": "user_defined",
            "agent_types": ["bicycle"],  # 只有自行车
            "agent_blueprints": ["vehicle.bh.crossbike"],  # CARLA blueprint IDs
            "vehicle_types": ["bicycle"],  # 车辆类型
            "agent_categories": ["bicycle"],  # 分类
            "agent_speed_range_mps": (1, 2),  # 自行车速度较慢
            "motion_pattern": "wave",  # 标记波浪运动模式
            "wave_amplitude": 1.5,     # 波浪幅度（米）
            "wave_frequency": 0.3,     # 波浪频率（Hz）
            "agent_trajectories": {
                0: [  # 自行车的大幅波浪轨迹（每次摆动1.5米，适配9步episode）
                    (11.5, -100.0),     # 起始位置（中心）
                    (10.0, -102.0),     # 向左摆1.5米（最左侧，接近主车道）
                    (11.5, -104.0),     # 回到中心
                    (13.0, -106.0),     # 向右摆1.5米（最右侧，远离主车道）
                    (11.5, -108.0),     # 回到中心
                    (10.0, -110.0),     # 再次向左摆1.5米
                    (11.5, -112.0),     # 回到中心
                    (13.0, -114.0),     # 再次向右摆1.5米
                    (11.5, -116.0),     # 回到中心
                    (11.5, -118.0),     # 中心位置继续前进
                ]
            }
        }

        return ScenarioDefinition(
            name="s3_scenario",
            description="右侧自行车波浪行驶场景",
            town="Town03",
            ego_spawn=ego_spawn,
            agent_spawns=[bicycle],  # 原来是 [bicycle, front_vehicle, rear_vehicle]
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
        ego_spawn = (9.5, -115.0, 0.5, -90.0)

        # 行人：右侧人行道，准备横穿（人行横道位置）
        pedestrian = (13.5, -127.0, 0.5, 180.0)  # 右侧4米，前方27米（人行横道），朝西

        metadata = {
            "source": "user_defined",
            "agent_types": ["walker"],  # 行人
            "agent_blueprints": ["walker.pedestrian.0014"],  # CARLA行人blueprint（换更大体型的模型）
            "agent_speed_range_mps": (1.8, 2.25),  # 1.5倍步行速度
            "motion_pattern": "crossing_decelerate",  # 横穿减速模式
            "deceleration_position_x": 9.0,  # 在道路中间减速（主车道位置）
            "target_speed_after_decel": 0.0,  # 减速后速度（0=停止）
            "deceleration_rate": 3.0,  # 减速度 m/s²（1.5倍）
            "agent_trajectories": {
                0: [  # 行人横穿轨迹（大幅加快：每步0.5米）
                    # 第一阶段：快速横穿（每步0.5米，从右侧向左）
                    (13.5, -127.00),    # 起始位置（右侧人行道）
                    (13.0, -127.00),    # 步长0.5米
                    (12.5, -127.00),
                    (12.0, -127.00),
                    (11.5, -127.00),
                    (11.0, -127.00),
                    (10.5, -127.00),    # 接近道路边缘

                    # 第二阶段：短暂停顿（减少到3步）
                    (10.3, -127.00),    # 减速
                    (10.0, -127.00),    # 到达停止位置
                    (10.0, -127.00),    # 停顿1步
                    (10.0, -127.00),    # 停顿2步

                    # 第三阶段：继续快速前进（每步0.4米）
                    (9.6, -127.00),     # 重新开始，步长0.4米
                    (9.2, -127.00),
                    (8.8, -127.00),
                    (8.4, -127.00),
                    (8.0, -127.00),     # 离开主车道
                    (7.6, -127.00),
                    (7.2, -127.00),
                    (6.8, -127.00),     # 到达左侧人行道
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
        ego_spawn = (-50, -135.5, 0.5, 0.0)

        # 对向车辆（从东向西驶来）

        oncoming_vehicle = (-20, -138.5, 6, 180.0)

        # 施工区域障碍物（锥桶和箱子阻挡自车道）
        # 放在自车前方X=40附近（自车和对向车之间），形成施工区域
        cone1 = (-30, -139.0, 0.5, 0.0)   # 锥桶1（施工区域起点）
        cone2 = (-30.5, -138.0, 0.5, 0.0)   # 锥桶2
        cone3 = (-31.5, -138.0, 0.5, 0.0)   # 锥桶3
        box1 = (-31.5, -139.0, 0.5, 0.0)    # 箱子1（施工区域中央）
        box2 = (-32.5, -138.5, 0.5, 0.0)    # 箱子2
        cone4 = (-32.5, -138, 0.5, 0.0)   # 锥桶4（施工区域末端）
        cone5 = (-33.5, -138.0, 0.5, 0.0)   # 锥桶5

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
            "agent_speed_range_mps": (7.5, 7.5),  # 对向车辆速度7.5 m/s (5倍加速)
            "camera_position": (-40.0, -150.0),  # S5场景的固定相机位置（更新到新坐标区域）
            "agent_trajectories": {
                0: [  # 对向车辆绕行轨迹（速度7.5m/s，每步约1.0米，5倍加速）
                    # 对向车道直行
                    (-20.0, -138.5), #(-21.0, -138.5), 
                    (-22.0, -138.5), #(-23.0, -138.5), 
                    (-24.0, -138.5),
                    # 开始变道
                    #(-25.0, -138.0), 
                    (-26.0, -137.0), #(-27.0, -136.0), 
                    (-28.0, -135.5),
                    # 完成变道，进入自车车道
                    #(-29.0, -135.0), 
                    (-30.0, -135.0), (-31.0, -135.0), (-32.0, -135.0), (-33.0, -135.0),
                    # 自车车道继续直行
                    (-34.0, -135.0), (-35.0, -135.0), (-36.0, -135.0), (-37.0, -135.0), (-38.0, -135.0),
                    (-39.0, -135.0), (-40.0, -135.0), (-41.0, -135.0), (-42.0, -135.0), (-43.0, -135.0),
                    (-44.0, -135.0), (-45.0, -135.0), (-46.0, -135.0), (-47.0, -135.0), (-48.0, -135.0),
                    (-49.0, -135.0), (-50.0, -135.0), (-51.0, -135.0), (-52.0, -135.0), (-53.0, -135.0),
                    (-54.0, -135.0), (-55.0, -135.0), (-56.0, -135.0), (-57.0, -135.0), (-58.0, -135.0),
                    (-59.0, -135.0), (-60.0, -135.0), (-61.0, -135.0), (-62.0, -135.0), (-63.0, -135.0),
                    (-64.0, -135.0), (-65.0, -135.0), (-66.0, -135.0), (-67.0, -135.0), (-68.0, -135.0),
                    (-69.0, -135.0), (-70.0, -135.0), (-71.0, -135.0), (-72.0, -135.0), (-73.0, -135.0),
                    (-74.0, -135.0), (-75.0, -135.0), (-76.0, -135.0), (-77.0, -135.0), (-78.0, -135.0),
                    (-79.0, -135.0), (-80.0, -135.0),
                ]
                # 静态障碍物不需要轨迹
            }
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
