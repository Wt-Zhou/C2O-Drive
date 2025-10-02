#!/usr/bin/env python3
"""
CARLA接口演示 - 相机自动跟随俯视图

演示CarlaSimulator的相机自动跟随功能。
相机会自动聚焦在自车正上方，提供清晰的俯视图视角。
"""

import sys
from pathlib import Path
import time

# 添加项目根目录到路径
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from carla_c2osr.env.carla_scenario_1 import CarlaSimulator, carla_transform_from_position


def demo_camera_follow():
    """演示相机自动跟随功能"""
    print("\n" + "="*60)
    print("CARLA接口演示 - 相机自动跟随俯视图")
    print("="*60)

    # 创建仿真器
    sim = CarlaSimulator(town="Town03", dt=0.1, no_rendering=False)

    # 创建场景
    print("\n1️⃣ 创建场景...")
    ego_spawn = carla_transform_from_position(x=5.5, y=-90, yaw=-90)
    agent_spawns = [
        carla_transform_from_position(x=10, y=-100, yaw=0),
        carla_transform_from_position(x=3, y=-95, yaw=90),
    ]

    world_state = sim.create_scenario(ego_spawn, agent_spawns)
    print(f"✅ 场景已创建")
    print(f"  - 自车位置: {world_state.ego.position_m}")
    print(f"  - 环境车辆: {len(world_state.agents)}辆")
    print(f"  - 📷 相机已自动聚焦到自车俯视图")

    # 等待用户观察初始场景
    print("\n⏸️  请在CARLA窗口查看俯视图，按回车继续...")
    input()

    # 演示相机高度调整
    print("\n2️⃣ 演示相机高度调整...")
    heights = [30, 60, 100, 150]
    for height in heights:
        sim.set_camera_view(height=height, pitch=-90)
        print(f"  📷 相机高度: {height}m")
        time.sleep(2)

    # 恢复默认高度
    sim.set_camera_view(height=60, pitch=-90)
    print("  📷 恢复默认高度: 60m")

    # 演示轨迹执行和相机跟随
    print("\n3️⃣ 演示平滑轨迹执行和相机自动跟随...")
    trajectory = [
        (5.5, -90),
        (5.5, -95),
        (5.5, -100),
        (5.5, -105),
        (5.5, -110),
        (5.5, -115),
        (5.5, -120),
        (5.5, -125),
    ]

    print("  🚗 自车开始沿轨迹平滑移动（速度控制模式）...")
    print("  📷 相机自动跟随...")
    states = sim.execute_trajectory(trajectory, horizon=8, velocity=5.0, smooth=True)

    for i, state in enumerate(states):
        print(f"    t={i}: 自车位置={state.ego.position_m}")

    print("  ✅ 轨迹执行完成，相机全程跟随自车")

    # 演示倾斜视角
    print("\n4️⃣ 演示倾斜视角（45度斜视）...")
    sim.set_camera_view(height=80, pitch=-45)
    print("  📷 相机视角: 45度斜视（高度80m）")
    time.sleep(3)

    # 恢复俯视
    sim.set_camera_view(height=60, pitch=-90)
    print("  📷 恢复俯视角度")

    # 清理
    print("\n5️⃣ 清理场景...")
    sim.cleanup()
    print("✅ 清理完成")

    print("\n" + "="*60)
    print("演示结束")
    print("="*60)
    print("\n📝 总结:")
    print("  ✅ 相机自动聚焦自车俯视图")
    print("  ✅ 相机自动跟随自车移动")
    print("  ✅ 支持高度和俯仰角动态调整")
    print("  ✅ 提供清晰的场景观察视角")


if __name__ == "__main__":
    print("🚀 CARLA接口演示")
    print("请确保CARLA服务器正在运行: ./CarlaUE4.sh")
    print("按回车开始演示...")
    input()

    try:
        demo_camera_follow()
    except KeyboardInterrupt:
        print("\n\n⏹️  演示已中断")
    except Exception as e:
        print(f"\n\n❌ 演示失败: {e}")
        import traceback
        traceback.print_exc()
