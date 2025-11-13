#!/usr/bin/env python3
"""测试配置参数修复是否正常工作"""

from carla_c2osr.algorithms.c2osr.config import (
    C2OSRPlannerConfig,
    RewardWeightsConfig,
    QValueConfig,
)
from carla_c2osr.algorithms.c2osr.planner import C2OSRPlanner

def test_reward_weights_config():
    """测试 RewardWeightsConfig 扩展"""
    print("测试 RewardWeightsConfig...")
    config = RewardWeightsConfig()

    # 检查所有字段是否存在
    assert hasattr(config, 'collision_penalty')
    assert hasattr(config, 'collision_threshold')
    assert hasattr(config, 'collision_check_cell_radius')
    assert hasattr(config, 'comfort_weight')
    assert hasattr(config, 'efficiency_weight')
    assert hasattr(config, 'safety_weight')
    assert hasattr(config, 'max_accel_penalty')
    assert hasattr(config, 'max_jerk_penalty')
    assert hasattr(config, 'acceleration_penalty_weight')
    assert hasattr(config, 'jerk_penalty_weight')
    assert hasattr(config, 'max_comfortable_accel')
    assert hasattr(config, 'speed_reward_weight')
    assert hasattr(config, 'target_speed')
    assert hasattr(config, 'progress_reward_weight')
    assert hasattr(config, 'safe_distance')
    assert hasattr(config, 'distance_penalty_weight')
    assert hasattr(config, 'centerline_offset_penalty_weight')

    print(f"  ✓ collision_penalty = {config.collision_penalty}")
    print(f"  ✓ comfort_weight = {config.comfort_weight}")
    print(f"  ✓ efficiency_weight = {config.efficiency_weight}")
    print(f"  ✓ safety_weight = {config.safety_weight}")
    print(f"  ✓ 所有 17 个字段均存在")

def test_gamma_parameter():
    """测试 gamma 参数"""
    print("\n测试 gamma 参数...")
    config = QValueConfig()

    assert hasattr(config, 'gamma')
    print(f"  ✓ QValueConfig.gamma = {config.gamma}")

def test_buffer_capacity():
    """测试 buffer_capacity 参数"""
    print("\n测试 buffer_capacity 参数...")
    config = C2OSRPlannerConfig()

    assert hasattr(config, 'buffer_capacity')
    assert config.buffer_capacity == 10000
    print(f"  ✓ buffer_capacity = {config.buffer_capacity}")

def test_min_buffer_size():
    """测试 min_buffer_size 参数"""
    print("\n测试 min_buffer_size 参数...")
    config = C2OSRPlannerConfig()

    assert hasattr(config, 'min_buffer_size')
    assert config.min_buffer_size == 10
    print(f"  ✓ min_buffer_size = {config.min_buffer_size}")

def test_trajectory_storage_multiplier():
    """测试 trajectory_storage_multiplier 参数"""
    print("\n测试 trajectory_storage_multiplier 参数...")
    config = C2OSRPlannerConfig()

    assert hasattr(config, 'trajectory_storage_multiplier')
    assert config.trajectory_storage_multiplier == 100  # 用户改成了100
    print(f"  ✓ trajectory_storage_multiplier = {config.trajectory_storage_multiplier}")

def test_planner_initialization():
    """测试 Planner 初始化"""
    print("\n测试 Planner 初始化...")

    config = C2OSRPlannerConfig(
        buffer_capacity=500,
        min_buffer_size=5,
        trajectory_storage_multiplier=50,
    )

    # 初始化 planner（这会测试所有参数是否正确传递）
    try:
        planner = C2OSRPlanner(config)

        # 检查 trajectory buffer 的容量
        assert planner.trajectory_buffer.capacity == 500
        print(f"  ✓ TrajectoryBuffer.capacity = {planner.trajectory_buffer.capacity}")

        # 检查 storage_multiplier
        assert planner.trajectory_buffer.storage_multiplier == 50
        print(f"  ✓ TrajectoryBuffer.storage_multiplier = {planner.trajectory_buffer.storage_multiplier}")

        # 检查 min_buffer_size
        assert planner.config.min_buffer_size == 5
        print(f"  ✓ min_buffer_size = {planner.config.min_buffer_size}")

        # 检查 reward_calculator 使用了正确的 config
        assert planner.reward_calculator.config == config.reward_weights
        print(f"  ✓ RewardCalculator 使用了正确的 config")

        # 检查 q_value_calculator 的 gamma
        assert planner.q_value_calculator.config.gamma == config.q_value.gamma
        print(f"  ✓ QValueCalculator.gamma = {planner.q_value_calculator.config.gamma}")

        print("  ✓ Planner 初始化成功，所有参数正确传递")

    except Exception as e:
        print(f"  ✗ Planner 初始化失败: {e}")
        raise

def test_buffer_fifo():
    """测试 buffer FIFO 淘汰机制"""
    print("\n测试 buffer FIFO 淘汰机制...")

    config = C2OSRPlannerConfig(buffer_capacity=3)
    planner = C2OSRPlanner(config)

    # 添加4个episode，应该触发淘汰
    for i in range(4):
        planner.trajectory_buffer.store_agent_episode(
            agent_id=i,
            agent_type='VEHICLE',
            initial_ego_state=(float(i), 0.0, 0.0),
            initial_agents_states=[],
            ego_action_trajectory=[(float(i), 0.0)],
            agent_trajectory_cells=[0],
        )

    # Buffer 大小应该是 3（容量限制）
    assert len(planner.trajectory_buffer) == 3
    print(f"  ✓ Buffer 大小正确: {len(planner.trajectory_buffer)} == 3")

    # 最老的 episode (id=0) 应该被淘汰
    assert 0 not in planner.trajectory_buffer._episode_lookup
    assert 3 in planner.trajectory_buffer._episode_lookup
    print(f"  ✓ FIFO 淘汰机制正常工作")

if __name__ == '__main__':
    print("=" * 60)
    print("配置参数修复测试")
    print("=" * 60)

    try:
        test_reward_weights_config()
        test_gamma_parameter()
        test_buffer_capacity()
        test_min_buffer_size()
        test_trajectory_storage_multiplier()
        test_planner_initialization()
        test_buffer_fifo()

        print("\n" + "=" * 60)
        print("✓ 所有测试通过！")
        print("=" * 60)

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"✗ 测试失败: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        exit(1)
