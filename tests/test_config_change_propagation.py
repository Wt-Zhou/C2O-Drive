#!/usr/bin/env python3
"""演示修改 config.py 后会自动影响 run_c2osr_scenario.py"""

import sys

# 模拟命令行参数
class MockArgs:
    def __init__(self):
        self.horizon = 5
        self.dt = 1.0
        self.grid_size = 100.0
        self.config_preset = "default"

# 导入配置类
from carla_c2osr.algorithms.c2osr.config import (
    C2OSRPlannerConfig,
    LatticePlannerConfig,
)

# 导入 run_c2osr_scenario 的 create_planner_config 函数
sys.path.insert(0, '/home/zwt/code/C2O-Drive/examples')
from run_c2osr_scenario import create_planner_config


print("=" * 60)
print("演示：修改 config.py 的默认值会自动生效")
print("=" * 60)

# 场景1：使用当前的默认值
print("\n场景1：使用 config.py 的当前默认值")
print("-" * 60)

args = MockArgs()
config1 = create_planner_config(args)

print(f"lateral_offsets: {config1.lattice.lateral_offsets}")
print(f"speed_variations: {config1.lattice.speed_variations}")
print(f"num_trajectories: {config1.lattice.num_trajectories}")

# 场景2：创建一个自定义的 config.py 默认值
print("\n场景2：假设你修改了 config.py 的默认值")
print("-" * 60)
print("假设你在 config.py 中修改了:")
print("  lateral_offsets: [-10.0, -5.0, 0.0, 5.0, 10.0]")
print("  speed_variations: [3.0, 5.0, 7.0, 9.0]")
print("  num_trajectories: 100")

# 创建一个自定义配置来模拟修改后的 config.py
custom_lattice = LatticePlannerConfig(
    lateral_offsets=[-10.0, -5.0, 0.0, 5.0, 10.0],
    speed_variations=[3.0, 5.0, 7.0, 9.0],
    num_trajectories=100,
    horizon=5,
    dt=1.0,
)

custom_planner_config = C2OSRPlannerConfig(lattice=custom_lattice)

print("\n使用自定义配置创建 planner:")
from carla_c2osr.algorithms.c2osr.planner import C2OSRPlanner

planner = C2OSRPlanner(custom_planner_config)

print(f"lateral_offsets: {planner.lattice_planner.lateral_offsets}")
print(f"speed_variations: {planner.lattice_planner.speed_variations}")
print(f"num_trajectories: {planner.lattice_planner.num_trajectories}")

print("\n" + "=" * 60)
print("✅ 总结")
print("=" * 60)
print("""
现在配置已经统一，你只需要：

1. 修改 carla_c2osr/algorithms/c2osr/config.py 中的默认值
   例如：

   @dataclass
   class LatticePlannerConfig:
       lateral_offsets: List[float] = field(default_factory=lambda: [-10.0, -5.0, 0.0, 5.0, 10.0])
       speed_variations: List[float] = field(default_factory=lambda: [3.0, 5.0, 7.0])
       num_trajectories: int = 100
       ...

2. 运行 examples/run_c2osr_scenario.py
   它会自动使用新的默认值！

3. 不需要修改 run_c2osr_scenario.py

特殊情况：
- 如果使用 --config-preset fast 或 high-precision，
  它们会使用硬编码的预设值（这是故意的）
- 只有默认模式（不指定 preset）才会使用 config.py 的默认值
""")
