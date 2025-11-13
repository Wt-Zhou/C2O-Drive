#!/usr/bin/env python3
"""
测试统一的Horizon配置

验证所有模块都从C2OSRPlannerConfig.horizon统一读取
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from carla_c2osr.algorithms.c2osr.config import C2OSRPlannerConfig


def test_unified_horizon():
    print(f"\n{'='*70}")
    print("测试统一的Horizon配置")
    print(f"{'='*70}\n")

    # 测试1: 使用默认值
    print("测试1: 使用config.py中的默认值")
    config1 = C2OSRPlannerConfig()
    print(f"  C2OSRPlannerConfig.horizon = {config1.horizon}")
    print(f"  lattice.horizon = {config1.lattice.horizon}")
    print(f"  q_value.horizon = {config1.q_value.horizon}")

    assert config1.horizon == 10, f"Expected 10, got {config1.horizon}"
    assert config1.lattice.horizon == 10, f"Expected 10, got {config1.lattice.horizon}"
    assert config1.q_value.horizon == 10, f"Expected 10, got {config1.q_value.horizon}"
    print(f"  ✓ 所有horizon统一为10\n")

    # 测试2: 显式设置horizon=5
    print("测试2: 显式设置horizon=5")
    config2 = C2OSRPlannerConfig(horizon=5)
    print(f"  C2OSRPlannerConfig.horizon = {config2.horizon}")
    print(f"  lattice.horizon = {config2.lattice.horizon}")
    print(f"  q_value.horizon = {config2.q_value.horizon}")

    assert config2.horizon == 5, f"Expected 5, got {config2.horizon}"
    assert config2.lattice.horizon == 5, f"Expected 5, got {config2.lattice.horizon}"
    assert config2.q_value.horizon == 5, f"Expected 5, got {config2.q_value.horizon}"
    print(f"  ✓ 所有horizon统一为5\n")

    # 测试3: 显式设置horizon=15
    print("测试3: 显式设置horizon=15")
    config3 = C2OSRPlannerConfig(horizon=15)
    print(f"  C2OSRPlannerConfig.horizon = {config3.horizon}")
    print(f"  lattice.horizon = {config3.lattice.horizon}")
    print(f"  q_value.horizon = {config3.q_value.horizon}")

    assert config3.horizon == 15, f"Expected 15, got {config3.horizon}"
    assert config3.lattice.horizon == 15, f"Expected 15, got {config3.lattice.horizon}"
    assert config3.q_value.horizon == 15, f"Expected 15, got {config3.q_value.horizon}"
    print(f"  ✓ 所有horizon统一为15\n")

    # 测试4: 验证from_global_config
    print("测试4: 从全局配置创建")
    from carla_c2osr.config import update_horizon
    update_horizon(8)
    config4 = C2OSRPlannerConfig.from_global_config()
    print(f"  GlobalConfig.time.default_horizon = 8")
    print(f"  C2OSRPlannerConfig.horizon = {config4.horizon}")
    print(f"  lattice.horizon = {config4.lattice.horizon}")
    print(f"  q_value.horizon = {config4.q_value.horizon}")

    assert config4.horizon == 8, f"Expected 8, got {config4.horizon}"
    assert config4.lattice.horizon == 8, f"Expected 8, got {config4.lattice.horizon}"
    assert config4.q_value.horizon == 8, f"Expected 8, got {config4.q_value.horizon}"
    print(f"  ✓ 所有horizon统一为8\n")

    print(f"{'='*70}")
    print("✅ 所有测试通过！")
    print()
    print("配置方式：")
    print("  1. 在 config.py 中修改: C2OSRPlannerConfig.horizon = X")
    print("  2. 代码中创建: config = C2OSRPlannerConfig(horizon=X)")
    print("  3. 命令行参数: --horizon X")
    print()
    print("所有子配置（lattice, q_value）将自动同步！")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    test_unified_horizon()
