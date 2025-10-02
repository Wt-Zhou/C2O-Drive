"""
Episode运行上下文

封装episode运行所需的所有组件和配置,避免参数传递爆炸。
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path
import numpy as np

from carla_c2osr.env.types import WorldState
from carla_c2osr.agents.c2osr.grid import GridMapper
from carla_c2osr.agents.c2osr.spatial_dirichlet import SpatialDirichletBank
from carla_c2osr.agents.c2osr.trajectory_buffer import TrajectoryBuffer, ScenarioState
from carla_c2osr.evaluation.q_evaluator import QEvaluator
from carla_c2osr.evaluation.buffer_analyzer import BufferAnalyzer
from carla_c2osr.evaluation.q_distribution_tracker import QDistributionTracker
from carla_c2osr.utils.simple_trajectory_generator import SimpleTrajectoryGenerator
from carla_c2osr.env.scenario_manager import ScenarioManager
from carla_c2osr.config import get_global_config


@dataclass
class EpisodeContext:
    """Episode运行上下文 - 封装所有运行时依赖

    优势:
    - 避免函数参数列表过长 (从15个参数减少到1个)
    - 集中管理配置和状态
    - 便于添加新的组件而不修改函数签名
    """
    # Episode基本信息
    episode_id: int
    horizon: int
    ego_trajectory: list
    world_init: WorldState

    # 核心组件
    grid: GridMapper
    bank: SpatialDirichletBank
    trajectory_buffer: TrajectoryBuffer
    scenario_state: ScenarioState

    # 工具组件
    trajectory_generator: SimpleTrajectoryGenerator
    scenario_manager: ScenarioManager
    q_evaluator: Optional[QEvaluator] = None
    buffer_analyzer: Optional[BufferAnalyzer] = None
    q_tracker: Optional[QDistributionTracker] = None

    # 运行参数
    rng: Optional[np.random.Generator] = None
    output_dir: Optional[Path] = None
    sigma: float = 0.5

    # 配置缓存
    _config: Any = None

    def __post_init__(self):
        """初始化后处理"""
        if self._config is None:
            self._config = get_global_config()

        if self.output_dir is None:
            self.output_dir = Path("outputs/replay_experiment")

    @property
    def config(self):
        """快捷访问全局配置"""
        return self._config

    def should_visualize(self) -> bool:
        """判断是否需要生成可视化

        可以根据episode_id、配置等灵活控制
        """
        # 每5个episode生成一次可视化
        return self.episode_id % 5 == 0

    def get_episode_output_dir(self) -> Path:
        """获取当前episode的输出目录"""
        ep_dir = self.output_dir / f"ep_{self.episode_id:02d}"
        ep_dir.mkdir(exist_ok=True, parents=True)
        return ep_dir
