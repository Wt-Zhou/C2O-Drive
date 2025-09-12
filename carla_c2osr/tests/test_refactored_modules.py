"""
测试重构后的模块功能

验证重构后的模块是否保持了原有功能。
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# 添加项目根目录到路径
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from carla_c2osr.env.types import AgentState, EgoState, WorldState, AgentType
from carla_c2osr.evaluation.rewards import RewardCalculator, CollisionDetector
from carla_c2osr.evaluation.q_evaluator import QEvaluator
from carla_c2osr.evaluation.buffer_analyzer import BufferAnalyzer
from carla_c2osr.utils.trajectory_generator import TrajectoryGenerator
from carla_c2osr.utils.scenario_manager import ScenarioManager
from carla_c2osr.agents.c2osr.trajectory_buffer import ScenarioState


class TestRewardCalculator:
    """测试奖励计算器"""
    
    def test_reward_calculator_initialization(self):
        """测试奖励计算器初始化"""
        calculator = RewardCalculator()
        assert calculator.collision_penalty == -100.0
        assert calculator.target_speed == 5.0
        assert calculator.safe_distance == 3.0
    
    def test_collision_reward(self):
        """测试碰撞奖励计算"""
        calculator = RewardCalculator()
        ego_state = EgoState(position_m=(0.0, 0.0), velocity_mps=(5.0, 0.0), yaw_rad=0.0)
        ego_next_state = EgoState(position_m=(5.0, 0.0), velocity_mps=(5.0, 0.0), yaw_rad=0.0)
        agent_state = AgentState(
            agent_id="test", position_m=(1.0, 1.0), velocity_mps=(1.0, 1.0),
            heading_rad=0.0, agent_type=AgentType.VEHICLE
        )
        agent_next_state = agent_state
        
        # 碰撞情况
        reward = calculator.calculate_reward(
            ego_state, ego_next_state, agent_state, agent_next_state, collision=True
        )
        assert reward == calculator.collision_penalty
    
    def test_normal_reward(self):
        """测试正常情况下的奖励计算"""
        calculator = RewardCalculator()
        ego_state = EgoState(position_m=(0.0, 0.0), velocity_mps=(5.0, 0.0), yaw_rad=0.0)
        ego_next_state = EgoState(position_m=(5.0, 0.0), velocity_mps=(5.0, 0.0), yaw_rad=0.0)
        agent_state = AgentState(
            agent_id="test", position_m=(10.0, 10.0), velocity_mps=(1.0, 1.0),
            heading_rad=0.0, agent_type=AgentType.VEHICLE
        )
        agent_next_state = agent_state
        
        # 无碰撞情况
        reward = calculator.calculate_reward(
            ego_state, ego_next_state, agent_state, agent_next_state, collision=False
        )
        assert reward > calculator.collision_penalty  # 应该大于碰撞惩罚


class TestCollisionDetector:
    """测试碰撞检测器"""
    
    def test_collision_detector_initialization(self):
        """测试碰撞检测器初始化"""
        detector = CollisionDetector()
        assert detector.collision_threshold == 0.1
    
    def test_collision_detection(self):
        """测试碰撞检测"""
        detector = CollisionDetector(collision_threshold=0.1)
        
        # 碰撞情况
        collision = detector.check_collision(
            agent_cell=5, ego_trajectory_cells=[5, 6, 7], agent_probability=0.5
        )
        assert collision is True
        
        # 无碰撞情况
        collision = detector.check_collision(
            agent_cell=5, ego_trajectory_cells=[6, 7, 8], agent_probability=0.5
        )
        assert collision is False
        
        # 概率低于阈值
        collision = detector.check_collision(
            agent_cell=5, ego_trajectory_cells=[5, 6, 7], agent_probability=0.05
        )
        assert collision is False
    
    def test_collision_probability_calculation(self):
        """测试碰撞概率计算"""
        detector = CollisionDetector(collision_threshold=0.1)
        
        reachable_probs = np.array([0.2, 0.3, 0.1, 0.4])
        reachable = [1, 2, 3, 4]
        overlap_cells = {2, 4}
        
        collision_prob, collision_count = detector.calculate_collision_probability(
            reachable_probs, reachable, overlap_cells
        )
        
        assert collision_prob == 0.7  # 0.3 + 0.4
        assert collision_count == 2


class TestTrajectoryGenerator:
    """测试轨迹生成器"""
    
    def test_trajectory_generator_initialization(self):
        """测试轨迹生成器初始化"""
        generator = TrajectoryGenerator()
        assert generator.grid_bounds == (-9.0, 9.0)
    
    def test_ego_trajectory_generation(self):
        """测试自车轨迹生成"""
        generator = TrajectoryGenerator()
        
        # 直线轨迹
        trajectory = generator.generate_ego_trajectory("straight", horizon=5, ego_speed=5.0)
        assert len(trajectory) == 5
        assert trajectory[0][0] == 5.0  # 第一秒x位置
        assert trajectory[0][1] == 0.0  # y位置为0
        
        # 固定轨迹
        trajectory = generator.generate_ego_trajectory("fixed-traj", horizon=3, ego_speed=3.0)
        assert len(trajectory) == 3
        
        # 无效模式
        with pytest.raises(ValueError):
            generator.generate_ego_trajectory("invalid", horizon=5)


class TestScenarioManager:
    """测试场景管理器"""
    
    def test_scenario_manager_initialization(self):
        """测试场景管理器初始化"""
        manager = ScenarioManager()
        assert manager.grid_size_m == 20.0
    
    def test_scenario_creation(self):
        """测试场景创建"""
        manager = ScenarioManager()
        world = manager.create_scenario()
        
        assert isinstance(world, WorldState)
        assert len(world.agents) == 2
        assert world.agents[0].agent_type == AgentType.VEHICLE
        assert world.agents[1].agent_type == AgentType.PEDESTRIAN
    
    def test_scenario_state_creation(self):
        """测试场景状态创建"""
        manager = ScenarioManager()
        world = manager.create_scenario()
        scenario_state = manager.create_scenario_state(world)
        
        assert isinstance(scenario_state, ScenarioState)
        assert scenario_state.ego_position == (0.0, 0.0)
        assert len(scenario_state.agents_states) == 2


class TestQEvaluator:
    """测试Q值评估器"""
    
    def test_q_evaluator_initialization(self):
        """测试Q值评估器初始化"""
        evaluator = QEvaluator()
        assert evaluator.default_n_samples == 10
        assert isinstance(evaluator.reward_calculator, RewardCalculator)
        assert isinstance(evaluator.collision_detector, CollisionDetector)
    
    def test_sample_agent_transitions(self):
        """测试智能体转移采样"""
        evaluator = QEvaluator()
        
        # 这里需要mock SpatialDirichletBank，简化测试
        # 实际测试中需要完整的依赖注入
        pass


class TestBufferAnalyzer:
    """测试缓冲区分析器"""
    
    def test_buffer_analyzer_initialization(self):
        """测试缓冲区分析器初始化"""
        # 需要mock TrajectoryBuffer
        pass


def test_module_imports():
    """测试所有模块都能正确导入"""
    # 这个测试确保所有重构的模块都能正确导入
    assert True  # 如果到这里说明导入成功


if __name__ == "__main__":
    pytest.main([__file__])
