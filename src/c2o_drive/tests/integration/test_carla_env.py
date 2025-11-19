"""CARLA环境单元测试

测试CarlaEnvironment的基本功能（不需要CARLA服务器）
"""

import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from c2o_drive.environments.carlaironments.carla_env import CarlaEnvironment
from c2o_drive.environments.carla.types import WorldState, EgoState, AgentState, AgentType, EgoControl


def create_mock_world_state():
    """创建模拟WorldState"""
    ego = EgoState(
        position_m=(0.0, 0.0),
        velocity_mps=(5.0, 0.0),
        heading_rad=0.0,
        length_m=4.5,
        width_m=2.0,
    )

    agents = [
        AgentState(
            agent_id="agent_1",
            position_m=(20.0, 0.0),
            velocity_mps=(3.0, 0.0),
            heading_rad=0.0,
            agent_type=AgentType.VEHICLE,
        ),
    ]

    return WorldState(ego=ego, agents=agents, time_s=0.0)


def test_carla_env_initialization():
    """测试环境初始化"""
    print("\n测试1: 环境初始化")

    env = CarlaEnvironment(
        host='localhost',
        port=2000,
        town='Town03',
        dt=0.1,
        max_episode_steps=500,
    )

    assert env.host == 'localhost'
    assert env.port == 2000
    assert env.town == 'Town03'
    assert env.dt == 0.1
    assert env.max_episode_steps == 500
    assert env.simulator is None  # 未连接

    print("  ✓ 环境参数初始化正确")
    print("  ✓ 仿真器未自动连接")


def test_carla_env_action_space():
    """测试动作空间定义"""
    print("\n测试2: 动作空间")

    env = CarlaEnvironment()
    action_space = env.action_space

    assert action_space is not None
    assert hasattr(action_space, 'low')
    assert hasattr(action_space, 'high')
    assert action_space.shape == (3,)

    # 检查范围
    assert np.allclose(action_space.low, [0.0, -1.0, 0.0])
    assert np.allclose(action_space.high, [1.0, 1.0, 1.0])

    print("  ✓ 动作空间维度正确: (3,)")
    print("  ✓ 动作范围正确: throttle[0,1], steer[-1,1], brake[0,1]")


@patch('c2o_drive.env.carla_scenario_1.CarlaSimulator')
def test_carla_env_reset(mock_simulator_class):
    """测试reset方法（使用mock）"""
    print("\n测试3: Reset方法")

    # 创建mock仿真器
    mock_simulator = MagicMock()
    mock_simulator_class.return_value = mock_simulator

    # Mock create_scenario返回初始状态
    initial_state = create_mock_world_state()
    mock_simulator.create_scenario.return_value = initial_state

    # 创建环境并reset
    env = CarlaEnvironment(
        host='localhost',
        port=2000,
        town='Town03',
    )

    state, info = env.reset(seed=42)

    # 验证
    assert env.simulator is not None
    assert state == initial_state
    assert 'town' in info
    assert info['town'] == 'Town03'

    # 验证仿真器创建
    mock_simulator_class.assert_called_once_with(
        host='localhost',
        port=2000,
        town='Town03',
    )

    # 验证场景创建
    mock_simulator.create_scenario.assert_called_once()

    print("  ✓ 仿真器成功创建")
    print("  ✓ 场景成功创建")
    print("  ✓ 返回初始状态和info")


@patch('c2o_drive.env.carla_scenario_1.CarlaSimulator')
def test_carla_env_step(mock_simulator_class):
    """测试step方法（使用mock）"""
    print("\n测试4: Step方法")

    # 创建mock仿真器
    mock_simulator = MagicMock()
    mock_simulator_class.return_value = mock_simulator

    initial_state = create_mock_world_state()
    next_state = create_mock_world_state()
    next_state.ego.position_m = (1.0, 0.0)  # 移动了

    mock_simulator.create_scenario.return_value = initial_state
    mock_simulator.step.return_value = next_state

    # 创建环境
    env = CarlaEnvironment(max_episode_steps=10)
    env.reset()

    # 执行动作
    action = EgoControl(throttle=0.5, steer=0.0, brake=0.0)
    step_result = env.step(action)

    # 验证
    assert step_result.observation == next_state
    assert isinstance(step_result.reward, float)
    assert isinstance(step_result.terminated, bool)
    assert isinstance(step_result.truncated, bool)
    assert 'collision' in step_result.info
    assert 'step' in step_result.info

    # 验证simulator.step被调用
    mock_simulator.step.assert_called_once_with(action, dt=0.1)

    print("  ✓ Step执行成功")
    print("  ✓ 返回正确的StepResult")
    print("  ✓ 调用simulator.step")


@patch('c2o_drive.env.carla_scenario_1.CarlaSimulator')
def test_carla_env_collision_detection(mock_simulator_class):
    """测试碰撞检测"""
    print("\n测试5: 碰撞检测")

    # 创建mock仿真器
    mock_simulator = MagicMock()
    mock_simulator_class.return_value = mock_simulator

    initial_state = create_mock_world_state()
    mock_simulator.create_scenario.return_value = initial_state

    # 创建碰撞状态（ego和agent距离<2米）
    collision_state = create_mock_world_state()
    collision_state.ego.position_m = (0.0, 0.0)
    collision_state.agents[0].position_m = (1.5, 0.0)  # 距离1.5米

    mock_simulator.step.return_value = collision_state

    # 创建环境
    env = CarlaEnvironment()
    env.reset()

    # 执行动作
    action = EgoControl(throttle=0.5, steer=0.0, brake=0.0)
    step_result = env.step(action)

    # 验证碰撞被检测
    assert step_result.terminated == True
    assert step_result.info['collision'] == True

    print("  ✓ 碰撞成功检测 (距离 < 2.0m)")


@patch('c2o_drive.env.carla_scenario_1.CarlaSimulator')
def test_carla_env_truncation(mock_simulator_class):
    """测试episode超时截断"""
    print("\n测试6: Episode截断")

    # 创建mock仿真器
    mock_simulator = MagicMock()
    mock_simulator_class.return_value = mock_simulator

    initial_state = create_mock_world_state()
    next_state = create_mock_world_state()

    mock_simulator.create_scenario.return_value = initial_state
    mock_simulator.step.return_value = next_state

    # 创建环境，最大步数=5
    env = CarlaEnvironment(max_episode_steps=5)
    env.reset()

    # 执行4步，不应该truncate
    action = EgoControl(throttle=0.5, steer=0.0, brake=0.0)
    for i in range(4):
        step_result = env.step(action)
        assert step_result.truncated == False, f"Step {i+1} should not truncate"

    # 第5步，应该truncate
    step_result = env.step(action)
    assert step_result.truncated == True

    print("  ✓ Episode在max_steps后正确截断")


def test_carla_env_reward_computation():
    """测试奖励计算"""
    print("\n测试7: 奖励计算")

    from c2o_drive.environments.carlaironments.rewards import create_default_reward

    # 创建自定义奖励函数
    reward_fn = create_default_reward()

    env = CarlaEnvironment(reward_fn=reward_fn)

    # 验证奖励函数已设置
    assert env.reward_fn == reward_fn

    print("  ✓ 奖励函数成功设置")


def test_carla_env_close():
    """测试环境关闭"""
    print("\n测试8: 环境关闭")

    env = CarlaEnvironment()

    # 关闭环境
    env.close()

    # 验证simulator被清理
    assert env.simulator is None

    print("  ✓ 环境成功关闭")


def run_all_tests():
    """运行所有测试"""
    print("=" * 70)
    print(" CARLA环境单元测试")
    print("=" * 70)

    tests = [
        test_carla_env_initialization,
        test_carla_env_action_space,
        test_carla_env_reset,
        test_carla_env_step,
        test_carla_env_collision_detection,
        test_carla_env_truncation,
        test_carla_env_reward_computation,
        test_carla_env_close,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"\n  ✗ 测试失败: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    print(f" 测试完成: {passed} passed, {failed} failed")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
