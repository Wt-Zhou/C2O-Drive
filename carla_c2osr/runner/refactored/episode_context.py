"""
Episode上下文管理 - 封装episode执行所需的所有组件和参数
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import numpy as np

from carla_c2osr.env.types import WorldState
from carla_c2osr.agents.c2osr.grid import GridMapper
from carla_c2osr.agents.c2osr.spatial_dirichlet import SpatialDirichletBank
from carla_c2osr.agents.c2osr.trajectory_buffer import TrajectoryBuffer, ScenarioState
from carla_c2osr.evaluation.q_evaluator import QEvaluator
from carla_c2osr.evaluation.buffer_analyzer import BufferAnalyzer
from carla_c2osr.evaluation.q_distribution_tracker import QDistributionTracker
from carla_c2osr.utils.simple_trajectory_generator import SimpleTrajectoryGenerator
from carla_c2osr.utils.lattice_planner import LatticePlanner
from carla_c2osr.env.scenario_manager import ScenarioManager


@dataclass
class EpisodeContext:
    """
    封装Episode执行所需的所有上下文数据

    将原来run_episode函数的15个参数整合为一个结构化对象，
    提高代码可读性和可维护性。
    """
    # 基本参数
    episode_id: int
    horizon: int
    reference_path: List[np.ndarray]
    world_init: WorldState
    scenario_state: ScenarioState

    # 核心组件
    grid: GridMapper
    bank: SpatialDirichletBank
    trajectory_buffer: TrajectoryBuffer

    # 规划与评估
    lattice_planner: LatticePlanner
    q_evaluator: QEvaluator
    trajectory_generator: SimpleTrajectoryGenerator
    scenario_manager: ScenarioManager
    buffer_analyzer: BufferAnalyzer
    q_tracker: Optional[QDistributionTracker]

    # 随机数生成器和配置
    rng: np.random.Generator
    sigma: float
    output_dir: Path

    @classmethod
    def create(cls,
               episode_id: int,
               horizon: int,
               reference_path: List[np.ndarray],
               world_init: WorldState,
               grid: GridMapper,
               bank: SpatialDirichletBank,
               trajectory_buffer: TrajectoryBuffer,
               scenario_state: ScenarioState,
               rng: np.random.Generator,
               output_dir: Path,
               sigma: float,
               lattice_planner: Optional[LatticePlanner] = None,
               q_evaluator: Optional[QEvaluator] = None,
               trajectory_generator: Optional[SimpleTrajectoryGenerator] = None,
               scenario_manager: Optional[ScenarioManager] = None,
               buffer_analyzer: Optional[BufferAnalyzer] = None,
               q_tracker: Optional[QDistributionTracker] = None) -> EpisodeContext:
        """创建EpisodeContext，使用默认值初始化可选组件"""

        # 初始化可选组件
        if q_evaluator is None:
            q_evaluator = QEvaluator()
        if scenario_manager is None:
            scenario_manager = ScenarioManager()
        if buffer_analyzer is None:
            buffer_analyzer = BufferAnalyzer(trajectory_buffer)
        if lattice_planner is None:
            from carla_c2osr.config import get_global_config
            lattice_planner = LatticePlanner.from_config(get_global_config())

        return cls(
            episode_id=episode_id,
            horizon=horizon,
            reference_path=reference_path,
            world_init=world_init,
            scenario_state=scenario_state,
            grid=grid,
            bank=bank,
            trajectory_buffer=trajectory_buffer,
            lattice_planner=lattice_planner,
            q_evaluator=q_evaluator,
            trajectory_generator=trajectory_generator,
            scenario_manager=scenario_manager,
            buffer_analyzer=buffer_analyzer,
            q_tracker=q_tracker,
            rng=rng,
            sigma=sigma,
            output_dir=output_dir
        )

    def get_episode_dir(self) -> Path:
        """获取episode输出目录"""
        ep_dir = self.output_dir / f"ep_{self.episode_id:02d}"
        ep_dir.mkdir(exist_ok=True)
        return ep_dir
