#!/usr/bin/env python3
"""
诊断脚本：检查运行时的实际horizon值

运行方式：
python tests/diagnose_horizon.py --horizon 5
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
from c2o_drive.config import get_global_config, update_horizon
from c2o_drive.algorithms.c2osr.config import C2OSRPlannerConfig


def main():
    parser = argparse.ArgumentParser(description="诊断horizon配置")
    parser.add_argument("--horizon", type=int, default=5, help="目标horizon值")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"HORIZON配置诊断")
    print(f"{'='*70}\n")

    # 1. 更新全局配置
    print(f"步骤1: 设置全局horizon = {args.horizon}")
    update_horizon(args.horizon)
    gc = get_global_config()
    print(f"  ✓ GlobalConfig.time.default_horizon = {gc.time.default_horizon}")
    print()

    # 2. 创建planner配置
    print(f"步骤2: 创建C2OSRPlannerConfig")
    planner_config = C2OSRPlannerConfig()
    print(f"  ├─ lattice.horizon = {planner_config.lattice.horizon}")
    print(f"  └─ q_value.horizon = {planner_config.q_value.horizon}")
    print()

    # 3. 模拟创建Dirichlet Bank
    print(f"步骤3: 模拟创建DirichletBank")
    from c2o_drive.algorithms.c2osr.spatial_dirichlet import (
        DirichletParams,
        OptimizedMultiTimestepSpatialDirichletBank
    )

    params = DirichletParams()
    bank = OptimizedMultiTimestepSpatialDirichletBank(
        K=1000,
        params=params,
        horizon=planner_config.lattice.horizon  # 像planner中一样传递
    )
    print(f"  └─ DirichletBank.horizon = {bank.horizon}")
    print()

    # 4. 初始化一个agent并检查timestep范围
    print(f"步骤4: 初始化agent并检查timestep范围")
    reachable_sets = {t: [100, 101, 102] for t in range(1, args.horizon + 1)}
    bank.init_agent(agent_id=1, reachable_sets=reachable_sets)

    actual_timesteps = list(bank.agent_alphas[1].keys())
    print(f"  └─ 实际初始化的timesteps: {actual_timesteps}")
    print()

    # 5. 验证结果
    print(f"{'='*70}")
    print(f"验证结果")
    print(f"{'='*70}")

    success = True

    if gc.time.default_horizon != args.horizon:
        print(f"  ✗ 全局horizon不正确: {gc.time.default_horizon} != {args.horizon}")
        success = False
    else:
        print(f"  ✓ 全局horizon正确")

    if planner_config.lattice.horizon != args.horizon:
        print(f"  ✗ Lattice horizon不正确: {planner_config.lattice.horizon} != {args.horizon}")
        success = False
    else:
        print(f"  ✓ Lattice horizon正确")

    if planner_config.q_value.horizon != args.horizon:
        print(f"  ✗ Q-value horizon不正确: {planner_config.q_value.horizon} != {args.horizon}")
        success = False
    else:
        print(f"  ✓ Q-value horizon正确")

    if bank.horizon != args.horizon:
        print(f"  ✗ DirichletBank horizon不正确: {bank.horizon} != {args.horizon}")
        success = False
    else:
        print(f"  ✓ DirichletBank horizon正确")

    expected_timesteps = list(range(1, args.horizon + 1))
    if actual_timesteps != expected_timesteps:
        print(f"  ✗ Bank初始化的timestep不正确:")
        print(f"    预期: {expected_timesteps}")
        print(f"    实际: {actual_timesteps}")
        success = False
    else:
        print(f"  ✓ Bank初始化的timestep正确: 1-{args.horizon}")

    print()
    if success:
        print(f"{'='*70}")
        print(f"✅ 所有检查通过！")
        print(f"   可视化应该只生成timestep 1-{args.horizon}的图片")
        print(f"{'='*70}\n")
        return 0
    else:
        print(f"{'='*70}")
        print(f"✗ 检查失败！请检查上述错误")
        print(f"{'='*70}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
