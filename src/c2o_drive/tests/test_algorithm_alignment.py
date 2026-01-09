#!/usr/bin/env python3
"""
综合测试：验证所有算法的配置对齐和动态适应

测试内容：
1. GlobalConfig动态num_trajectories
2. 所有算法从GlobalConfig读取lattice参数
3. 动态action_dim适应
4. C2OSR不受影响
5. UnifiedStateEncoder工作正常
"""

import numpy as np
from c2o_drive.config import get_global_config
from c2o_drive.algorithms.ppo import PPOConfig
from c2o_drive.algorithms.sac import SACConfig
from c2o_drive.algorithms.rainbow_dqn.config import RainbowDQNConfig
from c2o_drive.algorithms.rcrl.config import RCRLPlannerConfig
from c2o_drive.algorithms.c2osr import C2OSRPlannerConfig
from c2o_drive.utils.state_encoder import UnifiedStateEncoder
from c2o_drive.core.types import EgoState, AgentState, WorldState, AgentType


def test_global_config_dynamic_lattice():
    """测试1: GlobalConfig动态num_trajectories"""
    print("\n" + "="*70)
    print("测试1: GlobalConfig动态num_trajectories")
    print("="*70)

    gc = get_global_config()

    # 默认配置
    print(f"\n默认配置:")
    print(f"  lateral_offsets: {gc.lattice.lateral_offsets}")
    print(f"  speed_variations: {gc.lattice.speed_variations}")
    print(f"  num_trajectories: {gc.lattice.num_trajectories}")
    assert gc.lattice.num_trajectories == 5, "默认应为5"

    # 修改speed_variations
    gc.lattice.speed_variations = [4.0, 6.0, 8.0]
    print(f"\n修改speed_variations后:")
    print(f"  speed_variations: {gc.lattice.speed_variations}")
    print(f"  num_trajectories: {gc.lattice.num_trajectories}")
    assert gc.lattice.num_trajectories == 15, "应自动更新为15"

    # 恢复默认
    gc.lattice.speed_variations = [4.0]

    print("\n✓ 测试1通过：GlobalConfig动态num_trajectories正常")


def test_algorithm_configs_from_global():
    """测试2: 所有算法从GlobalConfig读取lattice参数"""
    print("\n" + "="*70)
    print("测试2: 所有算法从GlobalConfig读取lattice")
    print("="*70)

    gc = get_global_config()
    gc.lattice.speed_variations = [4.0, 6.0]  # 修改为2个速度

    configs = {
        'PPO': PPOConfig.from_global_config(),
        'SAC': SACConfig.from_global_config(),
        'RainbowDQN': RainbowDQNConfig.from_global_config(),
        'RCRL': RCRLPlannerConfig.from_global_config(),
    }

    print(f"\nGlobalConfig: lateral={len(gc.lattice.lateral_offsets)}, speed={len(gc.lattice.speed_variations)}")
    print(f"  num_trajectories = {gc.lattice.num_trajectories}")

    for name, config in configs.items():
        print(f"\n{name}:")
        print(f"  lateral_offsets: {config.lattice.lateral_offsets}")
        print(f"  speed_variations: {config.lattice.speed_variations if hasattr(config.lattice, 'speed_variations') else config.lattice.target_speeds}")
        print(f"  num_trajectories: {config.lattice.num_trajectories}")

        # 验证lattice参数对齐
        assert config.lattice.lateral_offsets == gc.lattice.lateral_offsets
        if name == 'RCRL':
            assert config.lattice.target_speeds == gc.lattice.speed_variations
        else:
            assert config.lattice.speed_variations == gc.lattice.speed_variations
        assert config.lattice.num_trajectories == 10  # 5 lateral × 2 speeds

    # 恢复默认
    gc.lattice.speed_variations = [4.0]

    print("\n✓ 测试2通过：所有算法从GlobalConfig读取lattice正常")


def test_dynamic_action_dim():
    """测试3: 动态action_dim适应"""
    print("\n" + "="*70)
    print("测试3: 动态action_dim适应")
    print("="*70)

    # PPO
    ppo_config = PPOConfig()
    print(f"\nPPO:")
    print(f"  默认 action_dim: {ppo_config.action_dim}")
    assert ppo_config.action_dim == 5

    # 修改lattice配置
    ppo_config.lattice.speed_variations = [4.0, 6.0, 8.0]
    print(f"  修改speed后 action_dim: {ppo_config.action_dim}")
    assert ppo_config.action_dim == 15

    # SAC
    sac_config = SACConfig()
    print(f"\nSAC:")
    print(f"  默认 action_dim: {sac_config.action_dim}")
    assert sac_config.action_dim == 5

    sac_config.lattice.speed_variations = [4.0, 6.0]
    print(f"  修改speed后 action_dim: {sac_config.action_dim}")
    assert sac_config.action_dim == 10

    # RainbowDQN
    rainbow_config = RainbowDQNConfig()
    print(f"\nRainbowDQN:")
    print(f"  默认 num_trajectories: {rainbow_config.lattice.num_trajectories}")
    assert rainbow_config.lattice.num_trajectories == 15  # 默认3个速度

    rainbow_config.lattice.speed_variations = [4.0]
    print(f"  修改speed后 num_trajectories: {rainbow_config.lattice.num_trajectories}")
    assert rainbow_config.lattice.num_trajectories == 5

    # RCRL
    rcrl_config = RCRLPlannerConfig()
    print(f"\nRCRL:")
    print(f"  默认 num_trajectories: {rcrl_config.lattice.num_trajectories}")
    print(f"  get_n_actions: {rcrl_config.get_n_actions()}")

    print("\n✓ 测试3通过：动态action_dim适应正常")


def test_c2osr_unaffected():
    """测试4: C2OSR不受影响"""
    print("\n" + "="*70)
    print("测试4: C2OSR不受影响")
    print("="*70)

    from c2o_drive.algorithms.c2osr import C2OSRPlanner

    config = C2OSRPlannerConfig()
    print(f"\nC2OSR配置:")
    print(f"  lattice.lateral_offsets: {config.lattice.lateral_offsets}")
    print(f"  lattice.speed_variations: {config.lattice.speed_variations}")
    print(f"  lattice.num_trajectories: {config.lattice.num_trajectories}")

    # 初始化planner
    planner = C2OSRPlanner(config)
    print(f"\nC2OSR Planner:")
    print(f"  lattice_planner.num_trajectories: {planner.lattice_planner.num_trajectories}")

    print("\n✓ 测试4通过：C2OSR工作正常，未受影响")


def test_unified_state_encoder():
    """测试5: UnifiedStateEncoder工作正常"""
    print("\n" + "="*70)
    print("测试5: UnifiedStateEncoder")
    print("="*70)

    encoder = UnifiedStateEncoder()

    # 创建测试状态
    ego = EgoState(
        position_m=(10.0, 5.0),
        velocity_mps=(5.0, 0.0),
        yaw_rad=0.0,
    )

    agent1 = AgentState(
        agent_id='agent1',
        position_m=(20.0, 6.0),
        velocity_mps=(4.0, 0.5),
        heading_rad=0.1,
        agent_type=AgentType.VEHICLE,
    )

    world_state = WorldState(
        time_s=0.0,
        ego=ego,
        agents=[agent1],
    )

    # 编码
    features = encoder.encode(world_state)

    print(f"\n编码结果:")
    print(f"  shape: {features.shape}")
    print(f"  dtype: {features.dtype}")
    print(f"  特征信息: {encoder.feature_info}")

    assert features.shape == (128,)
    assert features.dtype == np.float32

    # 批量编码
    features_batch = encoder.encode_batch([world_state, world_state])
    assert features_batch.shape == (2, 128)

    print("\n✓ 测试5通过：UnifiedStateEncoder工作正常")


def test_reward_config_completeness():
    """测试6: RewardConfig参数完整性"""
    print("\n" + "="*70)
    print("测试6: RewardConfig参数完整性")
    print("="*70)

    gc = get_global_config()
    rc = gc.reward

    required_params = [
        'collision_penalty',
        'target_speed',
        'safe_distance',
        'gamma',
        'max_deviation',
        'time_penalty',
    ]

    print(f"\n检查必要参数:")
    for param in required_params:
        value = getattr(rc, param)
        print(f"  {param}: {value}")
        assert value is not None, f"{param} 不应为None"

    print("\n✓ 测试6通过：RewardConfig参数完整")


def main():
    """运行所有测试"""
    print("\n" + "="*70)
    print("算法对齐和动态适应综合测试")
    print("="*70)

    try:
        test_global_config_dynamic_lattice()
        test_algorithm_configs_from_global()
        test_dynamic_action_dim()
        test_c2osr_unaffected()
        test_unified_state_encoder()
        test_reward_config_completeness()

        print("\n" + "="*70)
        print("✓✓✓ 所有测试通过！ ✓✓✓")
        print("="*70)
        print("\n总结:")
        print("  ✓ GlobalConfig动态num_trajectories")
        print("  ✓ 所有算法从GlobalConfig读取lattice")
        print("  ✓ 动态action_dim适应")
        print("  ✓ C2OSR不受影响")
        print("  ✓ UnifiedStateEncoder工作正常")
        print("  ✓ RewardConfig参数完整")
        print("\n算法对齐和统一完成！")
        print("="*70 + "\n")

    except AssertionError as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
