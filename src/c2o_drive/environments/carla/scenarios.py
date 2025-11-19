"""CARLA 场景库（精简版）

当前仅保留真实项目中需要的 S4 Wrong-way vehicle 场景，该场景来源于
``bak/TestScenario_Town03_Waymo_long_tail.py``，用于模拟对向逆行车辆进入
自车车道的长尾风险。"""

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
    ego_spawn: Tuple[float, float, float, float]  # (x, y, z, yaw)
    agent_spawns: List[Tuple[float, float, float, float]]  # List of (x, y, z, yaw)
    reference_path_mode: str = "straight"  # straight, curve, s_curve
    autopilot: bool = False
    difficulty: str = "medium"  # easy, medium, hard
    metadata: Optional[Dict[str, Any]] = None  # 额外参数（速度/角度范围等）


class CarlaScenarioLibrary:
    """CARLA 场景库（仅包含 S4 Wrong-way vehicle）"""

    _SCENARIOS: Dict[str, ScenarioDefinition] = {}

    @staticmethod
    def get_scenario(name: str) -> ScenarioDefinition:
        """根据名称获取场景定义

        Args:
            name: 场景名称

        Returns:
            ScenarioDefinition对象

        Raises:
            ValueError: 如果场景名称不存在
        """
        if not CarlaScenarioLibrary._SCENARIOS:
            CarlaScenarioLibrary._SCENARIOS = {
                "s4_wrong_way": CarlaScenarioLibrary.wrong_way_vehicle(),
            }

        if name not in CarlaScenarioLibrary._SCENARIOS:
            raise ValueError(
                f"Unknown scenario: {name}. "
                f"Available: {list(CarlaScenarioLibrary._SCENARIOS.keys())}"
            )

        return CarlaScenarioLibrary._SCENARIOS[name]

    @staticmethod
    def list_scenarios() -> List[str]:
        """列出所有可用场景"""
        if not CarlaScenarioLibrary._SCENARIOS:
            CarlaScenarioLibrary.get_scenario("s4_wrong_way")
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
    def wrong_way_vehicle() -> ScenarioDefinition:
        """S4 Wrong-way vehicle 场景定义。

        - 参考 ``TestScenario_Town03_Waymo_long_tail`` 的坐标系；
        - 自车沿 Town03 东向车道 (-90°) 南行；
        - 对向车辆在 (12.8, -123.0) 处以约 100° 角逆行切入该车道；
        - metadata 提供速度/入射角范围，便于上层随机化。
        """

        ego_spawn = (5.5, -90.0, 0.5, -90.0)
        wrong_way_spawn = (12.8, -123.0, 1.0, 100.0)

        metadata = {
            "source": "TestScenario_Town03_Waymo_long_tail.py",
            "agent_speed_range_mps": (3.0, 8.0),   # 初速度 3~8 m/s
            "entry_angle_range_deg": (-20.0, 20.0),  # 以 100° 为中心的偏差
            "recommended_town": "Town03",
        }

        return ScenarioDefinition(
            name="s4_wrong_way",
            description="Wrong-way vehicle: 对向逆行车辆切入本车道",
            ego_spawn=ego_spawn,
            agent_spawns=[wrong_way_spawn],
            reference_path_mode="straight",
            autopilot=False,
            difficulty="hard",
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
