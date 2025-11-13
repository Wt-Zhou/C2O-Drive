"""
CARLA场景库

提供预定义的测试场景，包括：
- 对向碰撞风险场景
- 变道超车场景
- 路口场景
- 多车交互场景
"""

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


class CarlaScenarioLibrary:
    """CARLA场景库

    提供各种预定义场景用于测试和评估。
    """

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
        scenarios = {
            "oncoming_easy": CarlaScenarioLibrary.oncoming_collision_easy(),
            "oncoming_medium": CarlaScenarioLibrary.oncoming_collision_medium(),
            "oncoming_hard": CarlaScenarioLibrary.oncoming_collision_hard(),
            "lane_change_left": CarlaScenarioLibrary.lane_change_left(),
            "lane_change_right": CarlaScenarioLibrary.lane_change_right(),
            "overtake": CarlaScenarioLibrary.overtake_scenario(),
            "intersection": CarlaScenarioLibrary.intersection_scenario(),
            "multi_agent": CarlaScenarioLibrary.multi_agent_scenario(),
            "highway": CarlaScenarioLibrary.highway_scenario(),
        }

        if name not in scenarios:
            raise ValueError(f"Unknown scenario: {name}. Available: {list(scenarios.keys())}")

        return scenarios[name]

    @staticmethod
    def list_scenarios() -> List[str]:
        """列出所有可用场景"""
        return [
            "oncoming_easy",
            "oncoming_medium",
            "oncoming_hard",
            "lane_change_left",
            "lane_change_right",
            "overtake",
            "intersection",
            "multi_agent",
            "highway",
        ]

    @staticmethod
    def oncoming_collision_easy() -> ScenarioDefinition:
        """对向碰撞场景 - 简单难度

        自车与对向车距离较远，有充足时间反应。
        """
        return ScenarioDefinition(
            name="oncoming_easy",
            description="对向车距离较远，有充足反应时间",
            ego_spawn=(5.5, -70.0, 0.5, -90.0),  # 自车朝北
            agent_spawns=[
                (5.5, -120.0, 0.5, 90.0),  # 对向车距离50m
            ],
            reference_path_mode="straight",
            autopilot=False,
            difficulty="easy",
        )

    @staticmethod
    def oncoming_collision_medium() -> ScenarioDefinition:
        """对向碰撞场景 - 中等难度

        对向车距离适中，需要及时决策避让。
        """
        return ScenarioDefinition(
            name="oncoming_medium",
            description="对向车距离适中，需要及时避让",
            ego_spawn=(5.5, -80.0, 0.5, -90.0),
            agent_spawns=[
                (5.5, -110.0, 0.5, 90.0),  # 对向车距离30m
            ],
            reference_path_mode="straight",
            autopilot=False,
            difficulty="medium",
        )

    @staticmethod
    def oncoming_collision_hard() -> ScenarioDefinition:
        """对向碰撞场景 - 困难难度

        对向车距离很近，需要快速反应和精确控制。
        """
        return ScenarioDefinition(
            name="oncoming_hard",
            description="对向车距离很近，需要快速反应",
            ego_spawn=(5.5, -85.0, 0.5, -90.0),
            agent_spawns=[
                (5.5, -105.0, 0.5, 90.0),  # 对向车距离20m
            ],
            reference_path_mode="straight",
            autopilot=False,
            difficulty="hard",
        )

    @staticmethod
    def lane_change_left() -> ScenarioDefinition:
        """左变道场景

        前方有慢车，需要向左变道超车。
        """
        return ScenarioDefinition(
            name="lane_change_left",
            description="前方慢车，向左变道超车",
            ego_spawn=(5.5, -70.0, 0.5, -90.0),
            agent_spawns=[
                (5.5, -90.0, 0.5, -90.0),  # 前方慢车
            ],
            reference_path_mode="straight",
            autopilot=False,
            difficulty="medium",
        )

    @staticmethod
    def lane_change_right() -> ScenarioDefinition:
        """右变道场景

        前方有慢车，需要向右变道超车。
        """
        return ScenarioDefinition(
            name="lane_change_right",
            description="前方慢车，向右变道超车",
            ego_spawn=(5.5, -70.0, 0.5, -90.0),
            agent_spawns=[
                (5.5, -90.0, 0.5, -90.0),  # 前方慢车
            ],
            reference_path_mode="straight",
            autopilot=False,
            difficulty="medium",
        )

    @staticmethod
    def overtake_scenario() -> ScenarioDefinition:
        """超车场景

        前方慢车，左侧有对向来车，需要寻找合适时机超车。
        """
        return ScenarioDefinition(
            name="overtake",
            description="前方慢车，左侧对向车，需寻找超车时机",
            ego_spawn=(5.5, -70.0, 0.5, -90.0),
            agent_spawns=[
                (5.5, -85.0, 0.5, -90.0),  # 前方慢车
                (2.0, -120.0, 0.5, 90.0),   # 左侧对向车
            ],
            reference_path_mode="straight",
            autopilot=False,
            difficulty="hard",
        )

    @staticmethod
    def intersection_scenario() -> ScenarioDefinition:
        """路口场景

        十字路口，需要避让横向来车。
        """
        return ScenarioDefinition(
            name="intersection",
            description="十字路口，避让横向来车",
            ego_spawn=(5.5, -70.0, 0.5, -90.0),
            agent_spawns=[
                (15.0, -95.0, 0.5, 180.0),  # 横向来车（从右侧）
                (-5.0, -95.0, 0.5, 0.0),    # 横向来车（从左侧）
            ],
            reference_path_mode="straight",
            autopilot=False,
            difficulty="hard",
        )

    @staticmethod
    def multi_agent_scenario() -> ScenarioDefinition:
        """多车交互场景

        多辆车同时存在，需要处理复杂的多智能体交互。
        """
        return ScenarioDefinition(
            name="multi_agent",
            description="多车交互，复杂决策",
            ego_spawn=(5.5, -70.0, 0.5, -90.0),
            agent_spawns=[
                (5.5, -90.0, 0.5, -90.0),   # 前方车
                (5.5, -110.0, 0.5, 90.0),   # 对向车
                (2.0, -80.0, 0.5, -90.0),   # 左侧车
                (9.0, -100.0, 0.5, 90.0),   # 右侧对向车
            ],
            reference_path_mode="straight",
            autopilot=False,
            difficulty="hard",
        )

    @staticmethod
    def highway_scenario() -> ScenarioDefinition:
        """高速公路场景

        高速行驶，多车道，需要高效决策。
        """
        return ScenarioDefinition(
            name="highway",
            description="高速公路，多车道行驶",
            ego_spawn=(5.5, -70.0, 0.5, -90.0),
            agent_spawns=[
                (5.5, -95.0, 0.5, -90.0),   # 同车道前车
                (2.0, -85.0, 0.5, -90.0),   # 左车道车辆
                (9.0, -100.0, 0.5, -90.0),  # 右车道车辆
                (2.0, -120.0, 0.5, 90.0),   # 对向车
            ],
            reference_path_mode="straight",
            autopilot=True,  # 使用自动驾驶模拟真实交通
            difficulty="medium",
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
