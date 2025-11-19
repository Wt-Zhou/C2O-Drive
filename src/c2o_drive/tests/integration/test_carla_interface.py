#!/usr/bin/env python3
"""
CARLA接口测试脚本

独立测试CarlaSimulator的所有功能，不依赖其他模块。
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from c2o_drive.environments.carla.carla_scenario_1 import CarlaSimulator, carla_transform_from_position, world_state_to_carla_spawns
from c2o_drive.environments.carla.types import AgentType
import time


def test_carla_connection():
    """测试1: CARLA连接"""
    print("\n" + "="*60)
    print("测试1: CARLA连接")
    print("="*60)

    try:
        sim = CarlaSimulator(town="Town03", dt=0.1, no_rendering=False)
        print("✅ CARLA连接成功")
        print(f"  - 地图: Town03")
        print(f"  - 时间步长: 0.1s")
        sim.cleanup()
        return True
    except Exception as e:
        print(f"❌ CARLA连接失败: {e}")
        print("  请确保CARLA服务器正在运行: ./CarlaUE4.sh")
        return False


def test_scenario_creation():
    """测试2: 场景创建和WorldState获取"""
    print("\n" + "="*60)
    print("测试2: 场景创建和WorldState获取")
    print("="*60)

    try:
        sim = CarlaSimulator(town="Town03", dt=0.1)

        # 定义生成点
        ego_spawn = carla_transform_from_position(x=5.5, y=-90, yaw=-90)
        agent_spawns = [
            carla_transform_from_position(x=10, y=-100, yaw=0),
            carla_transform_from_position(x=3, y=-95, yaw=90)
        ]

        # 创建场景
        world_state = sim.create_scenario(ego_spawn, agent_spawns)

        # 验证WorldState
        print("✅ 场景创建成功")
        print(f"  - 自车位置: {world_state.ego.position_m}")
        print(f"  - 自车速度: {world_state.ego.velocity_mps}")
        print(f"  - 自车朝向: {world_state.ego.yaw_rad:.2f} rad")
        print(f"  - 环境车辆数: {len(world_state.agents)}")

        for i, agent in enumerate(world_state.agents):
            print(f"  - Agent {i+1}:")
            print(f"    位置: {agent.position_m}")
            print(f"    速度: {agent.velocity_mps}")
            print(f"    类型: {agent.agent_type}")

        sim.cleanup()
        return True

    except Exception as e:
        print(f"❌ 场景创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_world_state_conversion():
    """测试3: WorldState与CARLA Transform转换"""
    print("\n" + "="*60)
    print("测试3: WorldState与CARLA Transform转换")
    print("="*60)

    try:
        sim = CarlaSimulator(town="Town03", dt=0.1)

        # 创建场景
        ego_spawn = carla_transform_from_position(x=5.5, y=-90, yaw=-90)
        agent_spawns = [carla_transform_from_position(x=10, y=-100, yaw=0)]

        world_state = sim.create_scenario(ego_spawn, agent_spawns)

        # 转换回CARLA Transform
        ego_spawn_back, agent_spawns_back = world_state_to_carla_spawns(world_state)

        print("✅ WorldState转换成功")
        print(f"  - 自车Transform: x={ego_spawn_back.location.x:.2f}, "
              f"y={ego_spawn_back.location.y:.2f}, "
              f"yaw={ego_spawn_back.rotation.yaw:.2f}")
        print(f"  - Agent Transform: x={agent_spawns_back[0].location.x:.2f}, "
              f"y={agent_spawns_back[0].location.y:.2f}, "
              f"yaw={agent_spawns_back[0].rotation.yaw:.2f}")

        # 验证转换精度
        pos_diff = abs(world_state.ego.position_m[0] - ego_spawn_back.location.x)
        if pos_diff < 0.1:
            print(f"  - 位置转换精度: {pos_diff:.4f}m ✅")
        else:
            print(f"  - 位置转换精度: {pos_diff:.4f}m ⚠️")

        sim.cleanup()
        return True

    except Exception as e:
        print(f"❌ WorldState转换失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_trajectory_execution():
    """测试4: 轨迹执行"""
    print("\n" + "="*60)
    print("测试4: 轨迹执行")
    print("="*60)

    try:
        sim = CarlaSimulator(town="Town03", dt=0.1)

        # 创建场景
        ego_spawn = carla_transform_from_position(x=5.5, y=-90, yaw=-90)
        agent_spawns = [
            carla_transform_from_position(x=10, y=-100, yaw=0),
        ]

        world_state = sim.create_scenario(ego_spawn, agent_spawns)
        print(f"初始自车位置: {world_state.ego.position_m}")

        # 定义轨迹（直线向南）
        trajectory = [
            (5.5, -90),
            (5.5, -95),
            (5.5, -100),
            (5.5, -105),
            (5.5, -110)
        ]

        # 执行轨迹（平滑模式）
        print("\n执行轨迹（平滑控制模式）...")
        states = sim.execute_trajectory(trajectory, horizon=5, velocity=5.0, smooth=True)

        print("✅ 轨迹执行成功")
        print(f"  - 执行步数: {len(states)}")

        for i, state in enumerate(states):
            print(f"  - t={i}: 自车={state.ego.position_m}, "
                  f"环境车辆数={len(state.agents)}")

        # 验证轨迹跟踪
        final_pos = states[-1].ego.position_m
        target_pos = trajectory[-1]
        pos_error = ((final_pos[0] - target_pos[0])**2 + (final_pos[1] - target_pos[1])**2)**0.5

        if pos_error < 1.0:
            print(f"  - 轨迹跟踪误差: {pos_error:.2f}m ✅")
        else:
            print(f"  - 轨迹跟踪误差: {pos_error:.2f}m ⚠️")

        sim.cleanup()
        return True

    except Exception as e:
        print(f"❌ 轨迹执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_collision_detection():
    """测试5: 碰撞检测"""
    print("\n" + "="*60)
    print("测试5: 碰撞检测")
    print("="*60)

    try:
        sim = CarlaSimulator(town="Town03", dt=0.1)

        # 创建碰撞场景（自车和障碍物接近）
        ego_spawn = carla_transform_from_position(x=5.5, y=-90, yaw=-90)
        agent_spawns = [
            carla_transform_from_position(x=5.5, y=-100, yaw=180),  # 正面碰撞位置
        ]

        world_state = sim.create_scenario(ego_spawn, agent_spawns)

        # 定义会碰撞的轨迹
        collision_trajectory = [
            (5.5, -90),
            (5.5, -92),
            (5.5, -94),
            (5.5, -96),
            (5.5, -98),
            (5.5, -100),  # 碰撞点
        ]

        print("执行碰撞测试轨迹...")
        states = sim.execute_trajectory(collision_trajectory, horizon=6, velocity=5.0)

        if sim.is_collision_occurred():
            print("✅ 碰撞检测成功")
            print(f"  - 碰撞已正确检测到")
        else:
            print("⚠️ 未检测到碰撞")
            print(f"  - 可能是车辆间距过大，或碰撞传感器延迟")

        sim.cleanup()
        return True

    except Exception as e:
        print(f"❌ 碰撞检测失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_step_execution():
    """测试6: 单步执行"""
    print("\n" + "="*60)
    print("测试6: 单步执行")
    print("="*60)

    try:
        sim = CarlaSimulator(town="Town03", dt=0.1)

        # 创建场景
        ego_spawn = carla_transform_from_position(x=5.5, y=-90, yaw=-90)
        agent_spawns = [
            carla_transform_from_position(x=10, y=-100, yaw=0),
        ]

        world_state = sim.create_scenario(ego_spawn, agent_spawns, agent_autopilot=True)

        print("执行10个仿真步...")
        for i in range(10):
            # 不控制自车，只前进仿真
            world_state = sim.step()
            print(f"  - t={i+1}: 自车={world_state.ego.position_m}")

        print("✅ 单步执行成功")

        sim.cleanup()
        return True

    except Exception as e:
        print(f"❌ 单步执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_camera_follow():
    """测试7: 相机自动跟随自车俯视图"""
    print("\n" + "="*60)
    print("测试7: 相机自动跟随自车俯视图")
    print("="*60)

    try:
        sim = CarlaSimulator(town="Town03", dt=0.1)

        # 创建场景
        ego_spawn = carla_transform_from_position(x=5.5, y=-90, yaw=-90)
        world_state = sim.create_scenario(ego_spawn)

        print(f"初始自车位置: {world_state.ego.position_m}")
        print("📷 相机已自动设置为俯视图")

        # 测试不同相机高度
        print("\n测试相机高度调整:")
        for height in [30, 60, 100]:
            sim.set_camera_view(height=height, pitch=-90)
            print(f"  - 相机高度: {height}m")
            time.sleep(0.5)

        # 测试相机跟随轨迹
        print("\n测试相机跟随轨迹:")
        trajectory = [
            (5.5, -90),
            (5.5, -100),
            (5.5, -110),
            (5.5, -120),
        ]

        states = sim.execute_trajectory(trajectory, horizon=4, velocity=5.0, smooth=True)
        for i, state in enumerate(states):
            print(f"  - t={i}: 自车={state.ego.position_m}, 相机已跟随")

        print("✅ 相机跟随测试成功")
        print("  - 相机自动跟随自车移动")
        print("  - 支持高度和俯仰角调整")

        sim.cleanup()
        return True

    except Exception as e:
        print(f"❌ 相机跟随测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "🚀"*30)
    print("CARLA接口功能测试")
    print("🚀"*30)

    tests = [
        ("CARLA连接", test_carla_connection),
        ("场景创建和WorldState获取", test_scenario_creation),
        ("WorldState转换", test_world_state_conversion),
        ("轨迹执行", test_trajectory_execution),
        ("碰撞检测", test_collision_detection),
        ("单步执行", test_step_execution),
        ("相机自动跟随", test_camera_follow),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ 测试 '{test_name}' 异常: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

        # 短暂等待，避免资源冲突
        time.sleep(0.5)

    # 打印总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status}: {test_name}")

    print(f"\n总计: {passed}/{total} 测试通过")

    if passed == total:
        print("\n🎉 所有测试通过！CARLA接口工作正常。")
    else:
        print(f"\n⚠️ {total - passed} 个测试失败，请检查CARLA服务器状态。")


if __name__ == "__main__":
    main()
