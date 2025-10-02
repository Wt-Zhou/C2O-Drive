#!/usr/bin/env python3
"""
对向碰撞风险场景演示

场景设置：
1. 自车从起点向前行驶（轨迹由模型给出）
2. 逆行车从前方朝向自车驶来，带有随机横向偏移
3. 测试场景的碰撞风险

此场景可用于测试您的规划模型。
"""

import sys
from pathlib import Path
import numpy as np

# 添加项目根目录到路径
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from carla_c2osr.env.carla_scenario_1 import (
    CarlaSimulator,
    carla_transform_from_position,
    generate_oncoming_trajectory,
    generate_straight_trajectory
)


def create_oncoming_collision_scenario(sim: CarlaSimulator, scenario_difficulty: str = "medium"):
    """创建对向碰撞风险场景

    Args:
        sim: CARLA仿真器实例
        scenario_difficulty: 场景难度 ("easy", "medium", "hard")

    Returns:
        (ego_spawn, agent_spawns, initial_world_state)
    """
    # 场景参数
    if scenario_difficulty == "easy":
        ego_start_y = -90
        oncoming_start_y = -150
        lateral_offset_range = (-1.0, 1.0)  # 小偏移
        oncoming_speed = 4.0
    elif scenario_difficulty == "hard":
        ego_start_y = -90
        oncoming_start_y = -140
        lateral_offset_range = (-3.0, 3.0)  # 大偏移
        oncoming_speed = 8.0
    else:  # medium
        ego_start_y = -90
        oncoming_start_y = -145
        lateral_offset_range = (-2.0, 2.0)  # 中等偏移
        oncoming_speed = 6.0

    # 自车生成位置（朝向南，-90度）
    ego_spawn = carla_transform_from_position(
        x=5.5,
        y=ego_start_y,
        yaw=-90  # 朝向南
    )

    # 逆行车生成位置（朝向北，90度）
    oncoming_spawn = carla_transform_from_position(
        x=5.5,
        y=oncoming_start_y,
        yaw=90  # 朝向北（逆行）
    )

    agent_spawns = [oncoming_spawn]

    # 创建场景
    world_state = sim.create_scenario(ego_spawn, agent_spawns)

    print(f"\n📋 场景配置:")
    print(f"  - 难度: {scenario_difficulty}")
    print(f"  - 自车起点: y={ego_start_y}, 朝向: 南(-90°)")
    print(f"  - 逆行车起点: y={oncoming_start_y}, 朝向: 北(90°)")
    print(f"  - 横向偏移范围: {lateral_offset_range}")
    print(f"  - 逆行车速度: {oncoming_speed} m/s")

    return ego_spawn, agent_spawns, world_state, {
        'ego_start_y': ego_start_y,
        'oncoming_start_y': oncoming_start_y,
        'lateral_offset_range': lateral_offset_range,
        'oncoming_speed': oncoming_speed
    }


def demo_with_dummy_planner():
    """使用简单规划器的演示（模拟模型输出）"""
    print("\n" + "="*70)
    print("对向碰撞风险场景演示 - 使用简单规划器")
    print("="*70)

    # 创建仿真器
    print("\n📡 连接CARLA仿真器...")
    sim = CarlaSimulator(town="Town03", dt=0.1, no_rendering=False)

    # 设置相机高度
    sim.set_camera_view(height=80, pitch=-90)

    # 创建场景
    print("\n🏗️  创建对向碰撞风险场景...")
    ego_spawn, agent_spawns, world_state, scenario_params = create_oncoming_collision_scenario(
        sim, scenario_difficulty="medium"
    )

    print("\n⏸️  请在CARLA窗口观察场景:")
    print("  - 蓝色车（自车）朝向下方")
    print("  - 红色车（逆行车）朝向上方")
    print("  - 两车将对向行驶")
    print("\n按回车开始场景执行...")
    input()

    # 生成轨迹
    horizon = 50

    # 1. 自车轨迹（简单直行，实际应由您的模型生成）
    print("\n🚗 生成自车轨迹（简单直行）...")
    ego_trajectory = generate_straight_trajectory(
        start_x=5.5,
        start_y=scenario_params['ego_start_y'],
        direction_yaw=-90,  # 朝向南
        distance=50,
        horizon=horizon
    )
    print(f"  ✅ 自车轨迹: {len(ego_trajectory)} 点")

    # 2. 逆行车轨迹（带随机偏移）
    print("\n🚙 生成逆行车轨迹（随机偏移）...")
    oncoming_trajectory = generate_oncoming_trajectory(
        start_x=5.5,
        start_y=scenario_params['oncoming_start_y'],
        end_y=scenario_params['ego_start_y'] - 10,  # 接近自车
        horizon=horizon,
        lateral_offset_range=scenario_params['lateral_offset_range'],
        seed=42  # 固定种子可复现
    )
    print(f"  ✅ 逆行车轨迹: {len(oncoming_trajectory)} 点")

    # 3. 同时执行自车和逆行车轨迹
    print("\n▶️  执行场景...")
    print("  📷 相机将跟随自车...")

    agent_trajectories = {0: oncoming_trajectory}  # 车辆索引0 = 逆行车
    agent_velocities = {0: scenario_params['oncoming_speed']}

    states = sim.execute_multi_vehicle_trajectories(
        ego_trajectory=ego_trajectory,
        agent_trajectories=agent_trajectories,
        horizon=horizon,
        ego_velocity=5.0,
        agent_velocities=agent_velocities,
        smooth=True
    )

    # 分析结果
    print(f"\n📊 场景执行结果:")
    print(f"  - 总步数: {len(states)}")
    print(f"  - 自车最终位置: {states[-1].ego.position_m}")

    if sim.is_collision_occurred():
        print(f"  - ⚠️  碰撞发生！")
    else:
        print(f"  - ✅ 无碰撞")

    # 计算最小距离
    min_distance = float('inf')
    min_distance_time = 0

    for t, state in enumerate(states):
        if len(state.agents) > 0:
            ego_pos = np.array(state.ego.position_m)
            agent_pos = np.array(state.agents[0].position_m)
            distance = np.linalg.norm(ego_pos - agent_pos)

            if distance < min_distance:
                min_distance = distance
                min_distance_time = t

    print(f"  - 最小距离: {min_distance:.2f}m (t={min_distance_time})")

    # 清理
    print("\n🧹 清理场景...")
    sim.cleanup()

    print("\n" + "="*70)
    print("演示完成！")
    print("="*70)


def demo_with_model_interface():
    """展示如何与您的模型集成"""
    print("\n" + "="*70)
    print("对向碰撞场景 - 模型接口示例")
    print("="*70)

    # 创建仿真器
    sim = CarlaSimulator(town="Town03", dt=0.1, no_rendering=False)
    sim.set_camera_view(height=80, pitch=-90)

    # 创建场景
    ego_spawn, agent_spawns, world_state, scenario_params = create_oncoming_collision_scenario(
        sim, scenario_difficulty="medium"
    )

    # 生成逆行车轨迹
    horizon = 50
    oncoming_trajectory = generate_oncoming_trajectory(
        start_x=5.5,
        start_y=scenario_params['oncoming_start_y'],
        end_y=scenario_params['ego_start_y'] - 10,
        horizon=horizon,
        lateral_offset_range=scenario_params['lateral_offset_range'],
        seed=42
    )

    print("\n🤖 模型接口使用示例:")
    print("="*70)
    print("""
    # 1. 获取当前WorldState
    world_state = sim.get_world_state()

    # 2. 调用您的规划模型
    # ego_trajectory = your_planner.plan(
    #     current_state=world_state,
    #     horizon=50,
    #     dt=0.1
    # )

    # 3. 对于演示，使用简单直行轨迹
    ego_trajectory = generate_straight_trajectory(
        start_x=5.5,
        start_y=-90,
        direction_yaw=-90,
        distance=50,
        horizon=50
    )

    # 4. 执行轨迹
    agent_trajectories = {0: oncoming_trajectory}
    states = sim.execute_multi_vehicle_trajectories(
        ego_trajectory=ego_trajectory,
        agent_trajectories=agent_trajectories,
        horizon=50,
        ego_velocity=5.0,
        agent_velocities={0: 6.0}
    )

    # 5. 评估结果
    collision_occurred = sim.is_collision_occurred()
    final_state = states[-1]
    """)
    print("="*70)

    # 实际执行演示
    ego_trajectory = generate_straight_trajectory(
        start_x=5.5,
        start_y=scenario_params['ego_start_y'],
        direction_yaw=-90,
        distance=50,
        horizon=horizon
    )

    print("\n▶️  执行模型轨迹...")
    agent_trajectories = {0: oncoming_trajectory}
    states = sim.execute_multi_vehicle_trajectories(
        ego_trajectory=ego_trajectory,
        agent_trajectories=agent_trajectories,
        horizon=horizon,
        ego_velocity=5.0,
        agent_velocities={0: 6.0}
    )

    print(f"\n✅ 执行完成")
    print(f"  - 碰撞: {'是' if sim.is_collision_occurred() else '否'}")

    sim.cleanup()


def main():
    """主函数"""
    print("🚀 对向碰撞风险场景演示")
    print("\n选择演示模式:")
    print("  1. 简单规划器演示")
    print("  2. 模型接口示例")

    choice = input("\n请选择 (1/2，默认1): ").strip() or "1"

    if choice == "2":
        demo_with_model_interface()
    else:
        demo_with_dummy_planner()


if __name__ == "__main__":
    print("📋 准备工作:")
    print("  1. 确保CARLA服务器正在运行: ./CarlaUE4.sh")
    print("  2. CARLA窗口应该可见")
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
