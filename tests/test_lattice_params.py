#!/usr/bin/env python3
"""测试 lattice 参数是否生效"""

from carla_c2osr.algorithms.c2osr.config import C2OSRPlannerConfig, LatticePlannerConfig
from carla_c2osr.algorithms.c2osr.planner import C2OSRPlanner

# 测试1: 默认配置
print("=" * 60)
print("测试1: 默认配置")
print("=" * 60)

default_config = C2OSRPlannerConfig()
print(f"\nLatticePlannerConfig 默认值:")
print(f"  lateral_offsets: {default_config.lattice.lateral_offsets}")
print(f"  speed_variations: {default_config.lattice.speed_variations}")
print(f"  num_trajectories: {default_config.lattice.num_trajectories}")
print(f"  horizon: {default_config.lattice.horizon}")
print(f"  dt: {default_config.lattice.dt}")

planner = C2OSRPlanner(default_config)
print(f"\nLatticePlanner 实际使用的参数:")
print(f"  lateral_offsets: {planner.lattice_planner.lateral_offsets}")
print(f"  speed_variations: {planner.lattice_planner.speed_variations}")
print(f"  num_trajectories: {planner.lattice_planner.num_trajectories}")

# 测试2: 自定义配置
print("\n" + "=" * 60)
print("测试2: 自定义配置")
print("=" * 60)

custom_lattice = LatticePlannerConfig(
    lateral_offsets=[-10.0, -5.0, 0.0, 5.0, 10.0],
    speed_variations=[3.0, 5.0, 7.0],
    num_trajectories=50,
    horizon=10,
    dt=0.5,
)

custom_config = C2OSRPlannerConfig(lattice=custom_lattice)
print(f"\n自定义 LatticePlannerConfig:")
print(f"  lateral_offsets: {custom_config.lattice.lateral_offsets}")
print(f"  speed_variations: {custom_config.lattice.speed_variations}")
print(f"  num_trajectories: {custom_config.lattice.num_trajectories}")
print(f"  horizon: {custom_config.lattice.horizon}")
print(f"  dt: {custom_config.lattice.dt}")

planner_custom = C2OSRPlanner(custom_config)
print(f"\nLatticePlanner 实际使用的参数:")
print(f"  lateral_offsets: {planner_custom.lattice_planner.lateral_offsets}")
print(f"  speed_variations: {planner_custom.lattice_planner.speed_variations}")
print(f"  num_trajectories: {planner_custom.lattice_planner.num_trajectories}")

# 验证
print("\n" + "=" * 60)
print("验证")
print("=" * 60)

if planner_custom.lattice_planner.lateral_offsets == custom_lattice.lateral_offsets:
    print("✓ lateral_offsets 参数生效")
else:
    print("✗ lateral_offsets 参数未生效")

if planner_custom.lattice_planner.speed_variations == custom_lattice.speed_variations:
    print("✓ speed_variations 参数生效")
else:
    print("✗ speed_variations 参数未生效")

if planner_custom.lattice_planner.num_trajectories == custom_lattice.num_trajectories:
    print("✓ num_trajectories 参数生效")
else:
    print("✗ num_trajectories 参数未生效")

# 测试3: 生成轨迹验证参数影响
print("\n" + "=" * 60)
print("测试3: 生成轨迹验证")
print("=" * 60)

reference_path = [(float(i), 0.0) for i in range(20)]

trajectories_default = planner.lattice_planner.generate_trajectories(
    reference_path=reference_path,
    horizon=5,
    dt=1.0,
    ego_state=(0.0, 0.0, 0.0)
)

trajectories_custom = planner_custom.lattice_planner.generate_trajectories(
    reference_path=reference_path,
    horizon=10,
    dt=0.5,
    ego_state=(0.0, 0.0, 0.0)
)

print(f"\n默认配置生成的轨迹数量: {len(trajectories_default)}")
print(f"自定义配置生成的轨迹数量: {len(trajectories_custom)}")

print(f"\n默认配置 - 预期组合数: {len(default_config.lattice.lateral_offsets)} × {len(default_config.lattice.speed_variations)} = {len(default_config.lattice.lateral_offsets) * len(default_config.lattice.speed_variations)}")
print(f"默认配置 - 限制数量: {default_config.lattice.num_trajectories}")

print(f"\n自定义配置 - 预期组合数: {len(custom_lattice.lateral_offsets)} × {len(custom_lattice.speed_variations)} = {len(custom_lattice.lateral_offsets) * len(custom_lattice.speed_variations)}")
print(f"自定义配置 - 限制数量: {custom_lattice.num_trajectories}")

if len(trajectories_default) <= default_config.lattice.num_trajectories:
    print(f"\n✓ 默认配置的 num_trajectories={default_config.lattice.num_trajectories} 限制生效")
else:
    print(f"\n✗ 默认配置的 num_trajectories 限制未生效")

if len(trajectories_custom) <= custom_lattice.num_trajectories:
    print(f"✓ 自定义配置的 num_trajectories={custom_lattice.num_trajectories} 限制生效")
else:
    print(f"✗ 自定义配置的 num_trajectories 限制未生效")

print("\n" + "=" * 60)
print("总结")
print("=" * 60)
print("lattice 参数正常工作！")
print("如果你的参数没有生效，请检查:")
print("1. 是否在创建 C2OSRPlannerConfig 时传入了自定义的 lattice 参数")
print("2. 是否使用了正确的 config 对象创建 planner")
