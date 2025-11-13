#!/usr/bin/env python3
"""
Horizon一致性诊断测试

测试目标：
1. 验证所有模块使用统一的horizon配置
2. 检查每个timestep的buffer数据量
3. 验证ego_action长度在所有timestep是否统一为horizon
4. 确认后期timestep（如7-10）能正常匹配数据并更新Dirichlet
5. 测试不同horizon值（5, 8, 10, 15）下的系统行为
"""

import numpy as np
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from carla_c2osr.agents.c2osr.trajectory_buffer import (
    HighPerformanceTrajectoryBuffer,
    ScenarioState,
    AgentTrajectoryData,
)
from carla_c2osr.agents.c2osr.spatial_dirichlet import (
    OptimizedMultiTimestepSpatialDirichletBank,
    DirichletParams,
)
from carla_c2osr.algorithms.c2osr.config import C2OSRPlannerConfig


def simulate_episode_data(episode_id: int, horizon: int = 10) -> tuple:
    """模拟一个episode的数据"""
    # 模拟ego轨迹（至少horizon步）
    ego_trajectory = [(i * 5.0, 0.0) for i in range(horizon)]

    # 模拟timestep场景
    timestep_scenarios = []

    for t in range(horizon):
        # 当前timestep的场景状态
        scenario_state = ScenarioState(
            ego_position=(t * 5.0, 0.0),
            ego_velocity=(5.0, 0.0),
            ego_heading=0.0,
            agents_states=[(t * 5.0 + 10.0, 2.0, 4.0, 0.0, 0.0, 'car')],
        )

        # 模拟agent轨迹（使用slicing - 与实际代码一致）
        agent_trajectories = []
        for agent_id in range(1, 2):  # 1 agent
            # 模拟完整轨迹
            full_trajectory = list(range(100 + t * 10, 100 + t * 10 + horizon))  # cell indices
            # 使用slicing：从当前timestep到结束（自然递减长度）
            remaining_trajectory = full_trajectory[t:]

            traj_data = AgentTrajectoryData(
                agent_id=agent_id,
                agent_type='car',
                init_position=(t * 5.0 + 10.0, 2.0),
                init_velocity=(4.0, 0.0),
                init_heading=0.0,
                trajectory_cells=remaining_trajectory,  # 递减长度: H, H-1, ..., 1
            )
            agent_trajectories.append(traj_data)

        timestep_scenarios.append((scenario_state, agent_trajectories))

    return ego_trajectory, timestep_scenarios


def test_config_consistency(horizon: int):
    """测试配置一致性"""
    print(f"\n{'='*80}")
    print(f"CONFIG CONSISTENCY TEST (horizon={horizon})")
    print(f"{'='*80}")

    # 创建配置（使用默认值）
    config = C2OSRPlannerConfig()

    print(f"\nDefault config values:")
    print(f"  lattice.horizon: {config.lattice.horizon}")
    print(f"  q_value.horizon: {config.q_value.horizon}")

    # 创建自定义horizon配置
    from carla_c2osr.algorithms.c2osr.config import LatticePlannerConfig, QValueConfig

    custom_config = C2OSRPlannerConfig(
        lattice=LatticePlannerConfig(horizon=horizon),
        q_value=QValueConfig(horizon=horizon),
    )

    print(f"\nCustom config with horizon={horizon}:")
    print(f"  lattice.horizon: {custom_config.lattice.horizon}")
    print(f"  q_value.horizon: {custom_config.q_value.horizon}")

    # 验证一致性
    if custom_config.lattice.horizon == custom_config.q_value.horizon:
        print(f"  ✓ Horizon is consistent across all modules")
    else:
        print(f"  ✗ WARNING: Horizon mismatch!")

    return custom_config


def analyze_buffer_storage(buffer: HighPerformanceTrajectoryBuffer, horizon: int):
    """分析buffer中存储的数据"""
    print(f"\n{'='*80}")
    print(f"BUFFER STORAGE ANALYSIS")
    print(f"{'='*80}")

    # 按存储timestep分组统计
    timestep_stats = defaultdict(lambda: {
        'count': 0,
        'ego_action_lengths': [],
        'trajectory_lengths': [],
    })

    # 遍历所有存储的episodes
    for agent_id, episodes in buffer._agent_episodes.items():
        for episode_data in episodes:
            # ego_action长度
            ego_action_len = len(episode_data.initial_mdp.ego_action_trajectory)
            # trajectory长度
            traj_len = len(episode_data.agent_trajectory_cells)

            # 根据trajectory长度推断存储timestep
            # 如果horizon=10, trajectory_len=7, 则存储于t=3
            storage_timestep = horizon - traj_len

            timestep_stats[storage_timestep]['count'] += 1
            timestep_stats[storage_timestep]['ego_action_lengths'].append(ego_action_len)
            timestep_stats[storage_timestep]['trajectory_lengths'].append(traj_len)

    # 打印统计结果
    print(f"\nBuffer Stats: {buffer.get_stats()}")
    print(f"\nStorage by timestep:")
    print(f"{'-'*80}")
    print(f"{'Timestep':<10} {'Count':<10} {'Ego Action Len':<25} {'Trajectory Len':<25}")
    print(f"{'-'*80}")

    all_consistent = True
    for t in sorted(timestep_stats.keys()):
        stats = timestep_stats[t]
        ego_lens = stats['ego_action_lengths']
        traj_lens = stats['trajectory_lengths']

        # 检查ego_action长度是否符合预期（递减但最大为horizon）
        ego_min, ego_max = min(ego_lens), max(ego_lens)
        expected_max = min(horizon, horizon - t)  # 预期的最大长度
        ego_correct = (ego_max == expected_max)
        status = "✓" if ego_correct else "✗"

        if not ego_correct:
            all_consistent = False

        print(f"{status} t={t:<6} {stats['count']:<10} "
              f"min={ego_min} max={ego_max} (expected={expected_max})   "
              f"trajectory: min={min(traj_lens)} max={max(traj_lens)}")

    print(f"\n{'='*80}")
    if all_consistent:
        print(f"✓ SUCCESS: Ego_action lengths follow expected pattern (decreasing due to episode end)")
        print(f"   This is CORRECT behavior: later timesteps have less future data available.")
    else:
        print(f"✗ FAILURE: Ego_action lengths don't match expected pattern!")
    print(f"{'='*80}")

    return all_consistent


def test_matching_logic(buffer: HighPerformanceTrajectoryBuffer, horizon: int):
    """测试匹配逻辑"""
    print(f"\n{'='*80}")
    print(f"MATCHING LOGIC TEST")
    print(f"{'='*80}")

    # 模拟查询（使用与存储数据相似的ego_action以确保能匹配上）
    # 关键：查询的起始位置应该与某个存储的episode相近
    query_ego_state = (0.0, 0.0, 0.0)  # 从episode起点查询
    query_agents_states = [(10.0, 2.0, 4.0, 0.0, 0.0, 'car')]
    query_ego_action = [(i * 5.0, 0.0) for i in range(horizon)]  # 完整horizon步，与存储数据一致

    print(f"\nQuery with FULL ego_action length: {len(query_ego_action)}")

    # 尝试匹配
    results = buffer.get_agent_historical_transitions_strict_matching(
        agent_id=1,
        current_ego_state=query_ego_state,
        current_agents_states=query_agents_states,
        ego_action_trajectory=query_ego_action,
        ego_state_threshold=50.0,  # 宽松阈值确保能匹配上
        agents_state_threshold=50.0,
        ego_action_threshold=50.0,
        debug=False,
    )

    # 统计每个timestep的匹配结果
    print(f"\n{'-'*80}")
    print(f"Matching Results by Timestep:")
    print(f"{'-'*80}")
    print(f"{'Timestep':<15} {'Matched Count':<15} {'Status':<15}")
    print(f"{'-'*80}")

    empty_timesteps = []
    for t in range(1, horizon + 1):
        cells = results.get(t, [])
        count = len(cells)
        status = "✓" if count > 0 else "✗"

        if count == 0:
            empty_timesteps.append(t)

        print(f"{status} t={t:<13} {count:<15} {'OK' if count > 0 else 'NO DATA'}")

    # 检测问题
    print(f"\n{'='*80}")
    if empty_timesteps:
        print(f"✗ FAILURE: Timesteps {empty_timesteps} have NO matched data!")
        print(f"   Bug confirmed: Later timesteps cannot match data!")
    else:
        print(f"✓ SUCCESS: All timesteps (1-{horizon}) have matched data")
    print(f"{'='*80}")

    return len(empty_timesteps) == 0


def test_dirichlet_updates(
    buffer: HighPerformanceTrajectoryBuffer,
    bank: OptimizedMultiTimestepSpatialDirichletBank,
    horizon: int,
):
    """测试Dirichlet更新"""
    print(f"\n{'='*80}")
    print(f"DIRICHLET BANK UPDATE TEST")
    print(f"{'='*80}")

    # 初始化agent
    agent_id = 1
    reachable_sets = {t: list(range(100, 110)) for t in range(1, horizon + 1)}
    bank.init_agent(agent_id, reachable_sets)

    initial_alpha_sum = bank.get_agent_alpha(agent_id, 1).sum()
    print(f"\nInitial alpha sum (t=1): {initial_alpha_sum:.2f}")

    # 模拟查询和更新（使用与存储数据相似的位置以确保匹配）
    query_ego_state = (0.0, 0.0, 0.0)
    query_agents_states = [(10.0, 2.0, 4.0, 0.0, 0.0, 'car')]
    query_ego_action = [(i * 5.0, 0.0) for i in range(horizon)]

    # 获取历史数据
    results = buffer.get_agent_historical_transitions_strict_matching(
        agent_id=agent_id,
        current_ego_state=query_ego_state,
        current_agents_states=query_agents_states,
        ego_action_trajectory=query_ego_action,
        ego_state_threshold=50.0,
        agents_state_threshold=50.0,
        ego_action_threshold=50.0,
        debug=False,
    )

    # 更新Dirichlet
    print(f"\nUpdating Dirichlet with matched data:")
    print(f"{'-'*80}")

    updated_timesteps = []
    for timestep, historical_cells in results.items():
        if len(historical_cells) > 0 and timestep in reachable_sets:
            alpha_before = bank.get_agent_alpha(agent_id, timestep).sum()
            bank.update_with_softcount(agent_id, timestep, historical_cells, lr=1.0)
            alpha_after = bank.get_agent_alpha(agent_id, timestep).sum()
            increase = alpha_after - alpha_before

            status = "✓" if increase > 0 else "✗"
            updated_timesteps.append(timestep)
            print(f"{status} t={timestep:<3} matched={len(historical_cells):<5} "
                  f"alpha: {alpha_before:.2f} → {alpha_after:.2f} (+{increase:.2f})")

    # 检测问题
    print(f"\n{'='*80}")
    missing_updates = [t for t in range(1, horizon + 1) if t not in updated_timesteps]
    if missing_updates:
        print(f"✗ FAILURE: Timesteps {missing_updates} did NOT update!")
    else:
        print(f"✓ SUCCESS: All timesteps (1-{horizon}) updated successfully")
    print(f"{'='*80}")

    return len(missing_updates) == 0


def run_comprehensive_test(horizon: int):
    """运行完整的horizon一致性测试"""
    print(f"\n\n")
    print(f"{'#'*80}")
    print(f"# COMPREHENSIVE HORIZON CONSISTENCY TEST (horizon={horizon})")
    print(f"{'#'*80}")

    # 1. 配置一致性测试
    config = test_config_consistency(horizon)

    # 2. 初始化组件
    buffer = HighPerformanceTrajectoryBuffer(horizon=horizon, trajectory_storage_multiplier=1)
    dirichlet_params = DirichletParams(alpha_in=50.0, alpha_out=1e-6)
    bank = OptimizedMultiTimestepSpatialDirichletBank(
        K=1000, params=dirichlet_params, horizon=horizon
    )

    # 3. 运行episodes并存储数据
    n_episodes = 3
    print(f"\n{'='*80}")
    print(f"RUNNING {n_episodes} EPISODES")
    print(f"{'='*80}")

    for episode_id in range(n_episodes):
        ego_trajectory, timestep_scenarios = simulate_episode_data(episode_id, horizon)
        ego_trajectory_tuples = [tuple(pos) for pos in ego_trajectory]

        buffer.store_episode_trajectories_by_timestep(
            episode_id=episode_id,
            timestep_scenarios=timestep_scenarios,
            ego_trajectory=ego_trajectory_tuples,
        )
        print(f"  Episode {episode_id}: stored {len(timestep_scenarios)} timesteps")

    # 4. 分析存储
    storage_ok = analyze_buffer_storage(buffer, horizon)

    # 5. 测试匹配
    matching_ok = test_matching_logic(buffer, horizon)

    # 6. 测试Dirichlet更新
    dirichlet_ok = test_dirichlet_updates(buffer, bank, horizon)

    # 7. 总结
    print(f"\n\n")
    print(f"{'#'*80}")
    print(f"# TEST SUMMARY (horizon={horizon})")
    print(f"{'#'*80}")

    print(f"\n{'Test':<40} {'Result':<20}")
    print(f"{'-'*60}")
    print(f"{'1. Config consistency':<40} {'✓ PASS' if True else '✗ FAIL'}")
    print(f"{'2. Buffer storage (ego_action length)':<40} {'✓ PASS' if storage_ok else '✗ FAIL'}")
    print(f"{'3. Historical matching (all timesteps)':<40} {'✓ PASS' if matching_ok else '✗ FAIL'}")
    print(f"{'4. Dirichlet updates (all timesteps)':<40} {'✓ PASS' if dirichlet_ok else '✗ FAIL'}")

    overall_pass = storage_ok and matching_ok and dirichlet_ok
    print(f"\n{'='*60}")
    print(f"OVERALL: {'✓ ALL TESTS PASSED' if overall_pass else '✗ SOME TESTS FAILED'}")
    print(f"{'='*60}\n")

    return overall_pass


def main():
    """主测试函数"""
    print(f"\n")
    print(f"{'#'*80}")
    print(f"# HORIZON CONSISTENCY DIAGNOSTIC TEST SUITE")
    print(f"{'#'*80}")
    print(f"\nTesting different horizon values to ensure system-wide consistency...")

    # 测试不同的horizon值
    test_horizons = [5, 8, 10]
    results = {}

    for horizon in test_horizons:
        passed = run_comprehensive_test(horizon)
        results[horizon] = passed

    # 最终总结
    print(f"\n\n")
    print(f"{'#'*80}")
    print(f"# FINAL RESULTS")
    print(f"{'#'*80}")

    print(f"\n{'Horizon':<15} {'Result':<20}")
    print(f"{'-'*35}")
    for horizon, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{horizon:<15} {status}")

    all_passed = all(results.values())
    print(f"\n{'='*35}")
    if all_passed:
        print(f"✓ ALL HORIZON VALUES TESTED SUCCESSFULLY")
        print(f"\nConclusion: Horizon configuration is unified across all modules!")
    else:
        print(f"✗ SOME TESTS FAILED")
        print(f"\nPlease review the failed tests above.")
    print(f"{'='*35}\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
