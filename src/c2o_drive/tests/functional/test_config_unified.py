#!/usr/bin/env python3
"""测试 run_c2osr_scenario.py 是否使用了 config.py 的默认值"""

import sys
import argparse

# 模拟命令行参数
class MockArgs:
    def __init__(self):
        self.horizon = 5
        self.dt = 1.0
        self.grid_size = 100.0
        self.config_preset = "default"

# 导入配置类
from c2o_drive.algorithms.c2osr.config import (
    C2OSRPlannerConfig,
    GridConfig,
    LatticePlannerConfig,
    QValueConfig,
)

# 导入 run_c2osr_scenario 的 create_planner_config 函数
sys.path.insert(0, '/home/zwt/code/C2O-Drive/examples')
from run_c2osr_scenario import create_planner_config


def test_default_config_uses_config_py_defaults():
    """测试默认配置是否使用 config.py 的默认值"""

    print("=" * 60)
    print("测试配置参数统一性")
    print("=" * 60)

    # 1. 获取 config.py 的默认值
    print("\n1. config.py 的默认值:")
    default_config = C2OSRPlannerConfig()

    print(f"\n  GridConfig:")
    print(f"    grid_size_m: {default_config.grid.grid_size_m}")
    print(f"    bounds_x: {default_config.grid.bounds_x}")
    print(f"    bounds_y: {default_config.grid.bounds_y}")

    print(f"\n  LatticePlannerConfig:")
    print(f"    lateral_offsets: {default_config.lattice.lateral_offsets}")
    print(f"    speed_variations: {default_config.lattice.speed_variations}")
    print(f"    num_trajectories: {default_config.lattice.num_trajectories}")
    print(f"    horizon: {default_config.lattice.horizon}")
    print(f"    dt: {default_config.lattice.dt}")

    print(f"\n  QValueConfig:")
    print(f"    horizon: {default_config.q_value.horizon}")
    print(f"    n_samples: {default_config.q_value.n_samples}")
    print(f"    selection_percentile: {default_config.q_value.selection_percentile}")
    print(f"    gamma: {default_config.q_value.gamma}")

    # 2. 通过 run_c2osr_scenario 创建配置
    print("\n2. run_c2osr_scenario.py 创建的配置:")
    args = MockArgs()
    scenario_config = create_planner_config(args)

    print(f"\n  GridConfig:")
    print(f"    grid_size_m: {scenario_config.grid.grid_size_m}")
    print(f"    bounds_x: {scenario_config.grid.bounds_x}")
    print(f"    bounds_y: {scenario_config.grid.bounds_y}")

    print(f"\n  LatticePlannerConfig:")
    print(f"    lateral_offsets: {scenario_config.lattice.lateral_offsets}")
    print(f"    speed_variations: {scenario_config.lattice.speed_variations}")
    print(f"    num_trajectories: {scenario_config.lattice.num_trajectories}")
    print(f"    horizon: {scenario_config.lattice.horizon}")
    print(f"    dt: {scenario_config.lattice.dt}")

    print(f"\n  QValueConfig:")
    print(f"    horizon: {scenario_config.q_value.horizon}")
    print(f"    n_samples: {scenario_config.q_value.n_samples}")
    print(f"    selection_percentile: {scenario_config.q_value.selection_percentile}")
    print(f"    gamma: {scenario_config.q_value.gamma}")

    # 3. 验证
    print("\n" + "=" * 60)
    print("验证结果")
    print("=" * 60)

    errors = []

    # GridConfig
    if scenario_config.grid.grid_size_m != default_config.grid.grid_size_m:
        errors.append(f"✗ grid_size_m 不一致: {scenario_config.grid.grid_size_m} != {default_config.grid.grid_size_m}")
    else:
        print(f"✓ grid_size_m 一致: {scenario_config.grid.grid_size_m}")

    # LatticePlannerConfig - 算法参数
    if scenario_config.lattice.lateral_offsets != default_config.lattice.lateral_offsets:
        errors.append(f"✗ lateral_offsets 不一致")
        print(f"  scenario: {scenario_config.lattice.lateral_offsets}")
        print(f"  default:  {default_config.lattice.lateral_offsets}")
    else:
        print(f"✓ lateral_offsets 一致: {scenario_config.lattice.lateral_offsets}")

    if scenario_config.lattice.speed_variations != default_config.lattice.speed_variations:
        errors.append(f"✗ speed_variations 不一致")
        print(f"  scenario: {scenario_config.lattice.speed_variations}")
        print(f"  default:  {default_config.lattice.speed_variations}")
    else:
        print(f"✓ speed_variations 一致: {scenario_config.lattice.speed_variations}")

    if scenario_config.lattice.num_trajectories != default_config.lattice.num_trajectories:
        errors.append(f"✗ num_trajectories 不一致: {scenario_config.lattice.num_trajectories} != {default_config.lattice.num_trajectories}")
    else:
        print(f"✓ num_trajectories 一致: {scenario_config.lattice.num_trajectories}")

    # LatticePlannerConfig - 运行时参数（允许不同）
    print(f"\n运行时参数（允许覆盖）:")
    print(f"  horizon: {scenario_config.lattice.horizon} (来自 args.horizon={args.horizon})")
    print(f"  dt: {scenario_config.lattice.dt} (来自 args.dt={args.dt})")

    # QValueConfig
    if scenario_config.q_value.n_samples != default_config.q_value.n_samples:
        errors.append(f"✗ n_samples 不一致: {scenario_config.q_value.n_samples} != {default_config.q_value.n_samples}")
    else:
        print(f"✓ n_samples 一致: {scenario_config.q_value.n_samples}")

    if scenario_config.q_value.selection_percentile != default_config.q_value.selection_percentile:
        errors.append(f"✗ selection_percentile 不一致: {scenario_config.q_value.selection_percentile} != {default_config.q_value.selection_percentile}")
    else:
        print(f"✓ selection_percentile 一致: {scenario_config.q_value.selection_percentile}")

    if scenario_config.q_value.gamma != default_config.q_value.gamma:
        errors.append(f"✗ gamma 不一致: {scenario_config.q_value.gamma} != {default_config.q_value.gamma}")
    else:
        print(f"✓ gamma 一致: {scenario_config.q_value.gamma}")

    # 总结
    print("\n" + "=" * 60)
    if errors:
        print("❌ 测试失败，发现以下不一致:")
        for error in errors:
            print(f"  {error}")
        return False
    else:
        print("✅ 所有算法参数都使用 config.py 的默认值!")
        print("\n现在你可以:")
        print("  1. 修改 carla_c2osr/algorithms/c2osr/config.py 中的默认值")
        print("  2. 运行 examples/run_c2osr_scenario.py 会自动使用新值")
        print("  3. 无需修改 run_c2osr_scenario.py")
        return True


if __name__ == '__main__':
    success = test_default_config_uses_config_py_defaults()
    sys.exit(0 if success else 1)
