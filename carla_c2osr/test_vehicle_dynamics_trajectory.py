#!/usr/bin/env python3
"""
测试车辆动力学轨迹生成

验证新的轨迹生成器是否正确复用了车辆动力学模型。
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# 添加项目根目录到路径
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from carla_c2osr.env.types import AgentState, AgentType
from carla_c2osr.utils.simple_trajectory_generator import SimpleTrajectoryGenerator


def test_vehicle_dynamics_trajectory():
    """测试车辆动力学轨迹生成"""
    
    # 创建车辆智能体
    vehicle = AgentState(
        agent_id="test-vehicle",
        position_m=(2.0, 2.0),
        velocity_mps=(4.0, 0.0),
        heading_rad=0.0,
        agent_type=AgentType.VEHICLE
    )
    
    print(f"测试车辆初始状态:")
    print(f"  位置: {vehicle.position_m}")
    print(f"  速度: {vehicle.velocity_mps}")
    print(f"  朝向: {np.degrees(vehicle.heading_rad):.1f}度")
    
    # 创建轨迹生成器
    generator = SimpleTrajectoryGenerator(grid_bounds=(-50.0, 50.0))
    
    # 生成轨迹
    horizon = 20
    trajectory = generator.generate_agent_trajectory(vehicle, horizon, dt=1.0)
    
    print(f"\n轨迹分析:")
    print(f"轨迹长度: {len(trajectory)}")
    
    # 分析轨迹
    positions = np.array(trajectory)
    headings = []
    speeds = []
    heading_changes = []
    speed_changes = []
    
    for i in range(len(positions)):
        # 计算朝向
        if i < len(positions) - 1:
            dx = positions[i+1][0] - positions[i][0]
            dy = positions[i+1][1] - positions[i][1]
            heading = np.arctan2(dy, dx)
            headings.append(heading)
            
            # 计算速度
            speed = np.sqrt(dx**2 + dy**2)
            speeds.append(speed)
            
            # 计算朝向变化
            if i > 0:
                heading_change = abs(heading - headings[i-1])
                # 处理角度跳变
                if heading_change > np.pi:
                    heading_change = 2 * np.pi - heading_change
                heading_changes.append(np.degrees(heading_change))
            
            # 计算速度变化
            if i > 0:
                speed_change = abs(speed - speeds[i-1])
                speed_changes.append(speed_change)
    
    # 打印分析结果
    print(f"平均速度: {np.mean(speeds):.2f} m/s")
    print(f"速度变化范围: {np.min(speed_changes):.3f} - {np.max(speed_changes):.3f} m/s")
    print(f"朝向变化范围: {np.min(heading_changes):.2f} - {np.max(heading_changes):.2f} 度")
    print(f"最大朝向变化: {np.max(heading_changes):.2f} 度/步")
    
    # 检查是否符合车辆动力学约束
    from carla_c2osr.env.types import AgentDynamicsParams
    dynamics = AgentDynamicsParams.for_agent_type(AgentType.VEHICLE)
    
    print(f"\n动力学约束检查:")
    print(f"最大速度: {dynamics.max_speed_mps} m/s")
    print(f"最大加速度: {dynamics.max_accel_mps2} m/s²")
    print(f"最大偏航角速度: {np.degrees(dynamics.max_yaw_rate_rps):.1f} 度/s")
    print(f"轴距: {dynamics.wheelbase_m} m")
    
    # 验证约束
    max_speed_observed = np.max(speeds)
    max_accel_observed = np.max(speed_changes)  # 简化计算
    max_yaw_rate_observed = np.max(heading_changes)  # 度/步
    
    print(f"\n约束验证:")
    print(f"最大观测速度: {max_speed_observed:.2f} m/s (限制: {dynamics.max_speed_mps} m/s)")
    print(f"最大观测加速度: {max_accel_observed:.2f} m/s² (限制: {dynamics.max_accel_mps2} m/s²)")
    print(f"最大观测偏航角速度: {max_yaw_rate_observed:.2f} 度/步 (限制: {np.degrees(dynamics.max_yaw_rate_rps):.1f} 度/s)")
    
    # 可视化轨迹
    plot_trajectory(positions, headings, speeds, heading_changes, speed_changes)
    
    return trajectory


def plot_trajectory(positions, headings, speeds, heading_changes, speed_changes):
    """可视化轨迹"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 轨迹图
    axes[0, 0].plot(positions[:, 0], positions[:, 1], 'b-o', linewidth=2, markersize=4)
    axes[0, 0].plot(positions[0, 0], positions[0, 1], 'go', markersize=8, label='起点')
    axes[0, 0].plot(positions[-1, 0], positions[-1, 1], 'ro', markersize=8, label='终点')
    
    # 添加速度向量
    for i in range(0, len(positions)-1, 3):  # 每3步画一个向量
        dx = positions[i+1, 0] - positions[i, 0]
        dy = positions[i+1, 1] - positions[i, 1]
        axes[0, 0].arrow(positions[i, 0], positions[i, 1], dx*0.3, dy*0.3, 
                        head_width=0.2, head_length=0.3, fc='red', ec='red', alpha=0.7)
    
    axes[0, 0].set_xlabel('X (m)')
    axes[0, 0].set_ylabel('Y (m)')
    axes[0, 0].set_title('车辆轨迹')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axis('equal')
    
    # 朝向变化
    if heading_changes:
        axes[0, 1].plot(range(1, len(heading_changes)+1), heading_changes, 'g-o', linewidth=2)
        axes[0, 1].set_xlabel('时间步')
        axes[0, 1].set_ylabel('朝向变化 (度)')
        axes[0, 1].set_title('朝向变化率')
        axes[0, 1].grid(True, alpha=0.3)
    
    # 速度变化
    if speed_changes:
        axes[1, 0].plot(range(1, len(speed_changes)+1), speed_changes, 'r-o', linewidth=2)
        axes[1, 0].set_xlabel('时间步')
        axes[1, 0].set_ylabel('速度变化 (m/s)')
        axes[1, 0].set_title('速度变化率')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 速度曲线
    if speeds:
        axes[1, 1].plot(range(len(speeds)), speeds, 'b-o', linewidth=2)
        axes[1, 1].set_xlabel('时间步')
        axes[1, 1].set_ylabel('速度 (m/s)')
        axes[1, 1].set_title('速度曲线')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('vehicle_dynamics_trajectory.png', dpi=150, bbox_inches='tight')
    print(f"\n轨迹图已保存为: vehicle_dynamics_trajectory.png")
    plt.show()


def test_different_vehicle_types():
    """测试不同车辆类型的轨迹"""
    print(f"\n=== 测试不同车辆类型 ===")
    
    vehicle_types = [
        (AgentType.VEHICLE, "轿车"),
        (AgentType.BICYCLE, "自行车"),
        (AgentType.MOTORCYCLE, "摩托车")
    ]
    
    generator = SimpleTrajectoryGenerator(grid_bounds=(-50.0, 50.0))
    
    for agent_type, name in vehicle_types:
        print(f"\n--- {name} ---")
        
        vehicle = AgentState(
            agent_id=f"test-{name}",
            position_m=(2.0, 2.0),
            velocity_mps=(4.0, 0.0),
            heading_rad=0.0,
            agent_type=agent_type
        )
        
        trajectory = generator.generate_agent_trajectory(vehicle, horizon=10, dt=1.0)
        positions = np.array(trajectory)
        
        # 计算总移动距离
        start_pos = positions[0]
        end_pos = positions[-1]
        total_distance = np.linalg.norm(end_pos - start_pos)
        
        print(f"总移动距离: {total_distance:.2f} 米")
        print(f"最终位置: ({end_pos[0]:.2f}, {end_pos[1]:.2f})")


if __name__ == "__main__":
    test_vehicle_dynamics_trajectory()
    test_different_vehicle_types()

