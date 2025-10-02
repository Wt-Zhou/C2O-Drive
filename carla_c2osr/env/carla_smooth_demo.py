#!/usr/bin/env python3
"""
CARLA平滑运动演示

展示修复后的效果：
1. 环境车辆朝向正确
2. 自车平滑移动（不跳跃）
3. 相机自动跟随
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from carla_c2osr.env.carla_scenario_1 import CarlaSimulator, carla_transform_from_position


def main():
    print("\n" + "="*70)
    print("CARLA平滑运动演示 - 修复后效果展示")
    print("="*70)

    # 创建仿真器
    print("\n📡 连接CARLA仿真器...")
    sim = CarlaSimulator(town="Town03", dt=0.1, no_rendering=False)

    # 创建场景
    print("\n🏗️  创建场景...")
    ego_spawn = carla_transform_from_position(x=5.5, y=-90, yaw=-90)  # 朝向南（-90度）

    # 环境车辆：不同朝向测试
    agent_spawns = [
        carla_transform_from_position(x=10, y=-100, yaw=-90),   # 朝向南（与自车同向）
        carla_transform_from_position(x=3, y=-95, yaw=0),       # 朝向东
        carla_transform_from_position(x=15, y=-110, yaw=180),   # 朝向西
    ]

    world_state = sim.create_scenario(ego_spawn, agent_spawns)

    print(f"\n✅ 场景创建成功")
    print(f"  - 自车位置: {world_state.ego.position_m}")
    print(f"  - 自车朝向: {world_state.ego.yaw_rad:.2f} rad (南)")
    print(f"  - 环境车辆: {len(world_state.agents)}辆")
    print(f"  - 📷 相机已聚焦俯视图 (60m)")

    # 暂停观察初始状态
    print("\n⏸️  请观察CARLA窗口：")
    print("  1. 自车（蓝色）应朝向南（下方）")
    print("  2. 环境车（红色）应朝向各自指定方向，且匀速移动")
    print("  3. 相机俯视图应清晰显示所有车辆")
    print("\n按回车开始自车轨迹执行...")
    input()

    # 执行自车轨迹（平滑模式）
    print("\n🚗 执行自车轨迹（平滑控制模式）...")
    trajectory = [
        (5.5, -90),   # 起点
        (5.5, -95),
        (5.5, -100),
        (5.5, -105),
        (5.5, -110),
        (5.5, -115),
        (5.5, -120),
        (5.5, -125),
        (5.5, -130),
        (5.5, -135),
    ]

    print("  📷 相机将自动跟随自车...")
    print("  ⚡ 观察要点：")
    print("    - 自车应该平滑移动（不跳跃）")
    print("    - 环境车应该继续匀速移动")
    print("    - 相机应该始终跟随自车")

    states = sim.execute_trajectory(trajectory, horizon=10, velocity=5.0, smooth=True)

    print(f"\n✅ 轨迹执行完成")
    print(f"  - 总步数: {len(states)}")
    print(f"  - 最终位置: {states[-1].ego.position_m}")

    # 显示环境车辆状态
    print(f"\n🚙 环境车辆状态：")
    for i, agent in enumerate(states[-1].agents):
        print(f"  - 车辆{i+1}: 位置={agent.position_m}, "
              f"速度=({agent.velocity_mps[0]:.2f}, {agent.velocity_mps[1]:.2f})")

    # 对比测试：传送模式
    print("\n\n📊 对比测试：传送模式（会跳跃）...")
    print("按回车查看传送模式效果（仅供对比）...")
    input()

    # 重置场景
    sim.cleanup()
    world_state = sim.create_scenario(ego_spawn, agent_spawns)

    trajectory_short = [
        (5.5, -90),
        (5.5, -100),
        (5.5, -110),
        (5.5, -120),
    ]

    print("  ⚠️  传送模式：车辆会跳跃式移动...")
    states = sim.execute_trajectory(trajectory_short, horizon=4, velocity=5.0, smooth=False)

    print("\n📝 对比结论：")
    print("  ✅ 平滑模式 (smooth=True)：车辆平滑移动，真实感强")
    print("  ❌ 传送模式 (smooth=False)：车辆跳跃移动，不真实")

    # 清理
    print("\n🧹 清理场景...")
    sim.cleanup()

    print("\n" + "="*70)
    print("演示完成！")
    print("="*70)
    print("\n🎉 修复总结：")
    print("  1. ✅ 环境车辆朝向正确（根据spawn的yaw设置）")
    print("  2. ✅ 环境车辆有初始速度（2m/s朝向前方）")
    print("  3. ✅ 自车平滑移动（速度控制模式，默认smooth=True）")
    print("  4. ✅ 相机自动跟随（俯视图，60m高度）")
    print("  5. ✅ 无actor销毁错误（增加is_alive检查）")


if __name__ == "__main__":
    print("🚀 CARLA平滑运动演示")
    print("\n📋 准备工作：")
    print("  1. 确保CARLA服务器正在运行: ./CarlaUE4.sh")
    print("  2. CARLA窗口应该可见（观察效果）")
    print("\n按回车开始...")
    input()

    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  演示已中断")
    except Exception as e:
        print(f"\n\n❌ 演示失败: {e}")
        import traceback
        traceback.print_exc()
