#!/usr/bin/env python3
"""
测试轨迹平滑性

可视化修复后的车辆轨迹，检查是否还有突然转向的问题。
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# 添加项目根目录到路径
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from carla_c2osr.env.types import AgentState, AgentType
from carla_c2osr.utils.simple_trajectory_generator import SimpleTrajectoryGenerator


def test_vehicle_trajectory_smoothness():
    """测试车辆轨迹的平滑性"""
    
    # 创建车辆智能体
    vehicle = AgentState(
        agent_id="test-vehicle",
        position_m=(2.0, 2.0),
        velocity_mps=(4.0, 0.0),
        heading_rad=0.0,
        agent_type=AgentType.VEHICLE
    )
    
    # 创建轨迹生成器
    generator = SimpleTrajectoryGenerator(grid_bounds=(-9.0, 9.0))
    
    # 生成轨迹
    horizon = 20
    trajectory = generator.generate_agent_trajectory(vehicle, horizon, dt=1.0)
    
    # 转换为numpy数组
    trajectory_array = np.array(trajectory)
    
    # 计算朝向变化
    headings = []
    for i in range(len(trajectory)):
        if i < len(trajectory) - 1:
            # 计算相邻点之间的方向
            dx = trajectory[i+1][0] - trajectory[i][0]
            dy = trajectory[i+1][1] - trajectory[i][1]
            heading = np.arctan2(dy, dx)
            headings.append(heading)
        else:
            # 最后一点使用前一点的方向
            headings.append(headings[-1] if headings else 0.0)
    
    headings = np.array(headings)
    
    # 计算朝向变化率
    heading_changes = np.diff(headings)
    # 处理角度跳跃
    heading_changes = np.where(heading_changes > np.pi, heading_changes - 2*np.pi, heading_changes)
    heading_changes = np.where(heading_changes < -np.pi, heading_changes + 2*np.pi, heading_changes)
    
    # 创建可视化
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 轨迹图
    axes[0, 0].plot(trajectory_array[:, 0], trajectory_array[:, 1], 'b-o', linewidth=2, markersize=6)
    axes[0, 0].plot(trajectory_array[0, 0], trajectory_array[0, 1], 'go', markersize=10, label='Start')
    axes[0, 0].plot(trajectory_array[-1, 0], trajectory_array[-1, 1], 'ro', markersize=10, label='End')
    axes[0, 0].set_xlabel('X (m)')
    axes[0, 0].set_ylabel('Y (m)')
    axes[0, 0].set_title('Vehicle Trajectory')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    axes[0, 0].set_aspect('equal')
    
    # 2. 朝向变化
    time_steps = np.arange(len(headings))
    axes[0, 1].plot(time_steps, np.degrees(headings), 'b-o', linewidth=2)
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Heading (degrees)')
    axes[0, 1].set_title('Heading Over Time')
    axes[0, 1].grid(True)
    
    # 3. 朝向变化率
    time_steps_changes = np.arange(len(heading_changes))
    axes[1, 0].plot(time_steps_changes, np.degrees(heading_changes), 'r-o', linewidth=2)
    axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Heading Change (degrees/step)')
    axes[1, 0].set_title('Heading Change Rate')
    axes[1, 0].grid(True)
    
    # 4. 速度变化
    speeds = []
    for i in range(len(trajectory)):
        if i < len(trajectory) - 1:
            dx = trajectory[i+1][0] - trajectory[i][0]
            dy = trajectory[i+1][1] - trajectory[i][1]
            speed = np.sqrt(dx**2 + dy**2)
            speeds.append(speed)
        else:
            speeds.append(speeds[-1] if speeds else 0.0)
    
    speeds = np.array(speeds)
    axes[1, 1].plot(time_steps, speeds, 'g-o', linewidth=2)
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Speed (m/s)')
    axes[1, 1].set_title('Speed Over Time')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    # 保存图片
    output_path = Path("outputs/trajectory_smoothness_test.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    # 分析结果
    max_heading_change = np.max(np.abs(heading_changes))
    mean_heading_change = np.mean(np.abs(heading_changes))
    max_speed_change = np.max(np.abs(np.diff(speeds)))
    
    print(f"轨迹平滑性分析结果:")
    print(f"  最大朝向变化: {np.degrees(max_heading_change):.2f} 度/步")
    print(f"  平均朝向变化: {np.degrees(mean_heading_change):.2f} 度/步")
    print(f"  最大速度变化: {max_speed_change:.2f} m/s/步")
    
    # 判断是否平滑
    is_smooth = max_heading_change < 0.3  # 小于约17度/步
    print(f"  轨迹是否平滑: {'是' if is_smooth else '否'}")
    
    return is_smooth


if __name__ == "__main__":
    test_vehicle_trajectory_smoothness()
