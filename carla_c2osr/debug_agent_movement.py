#!/usr/bin/env python3
"""
调试环境车agent移动问题

检查为什么环境车有初始速度但位置不变。
"""

import numpy as np
import sys
from pathlib import Path

# 添加项目根目录到路径
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from carla_c2osr.env.types import AgentState, AgentType
from carla_c2osr.utils.simple_trajectory_generator import SimpleTrajectoryGenerator


def debug_agent_movement():
    """调试环境车agent的移动问题"""
    
    # 创建车辆智能体（与场景管理器中的设置相同）
    vehicle = AgentState(
        agent_id="vehicle-1",
        position_m=(2.0, 2.0),
        velocity_mps=(4.0, 0.0),
        heading_rad=0.0,
        agent_type=AgentType.VEHICLE
    )
    
    print(f"初始车辆状态:")
    print(f"  位置: {vehicle.position_m}")
    print(f"  速度: {vehicle.velocity_mps}")
    print(f"  朝向: {np.degrees(vehicle.heading_rad):.1f}度")
    
    # 测试不同的边界设置
    test_cases = [
        ("默认边界", (-9.0, 9.0)),
        ("网格边界", (-50.0, 50.0)),
        ("大边界", (-100.0, 100.0))
    ]
    
    for case_name, bounds in test_cases:
        print(f"\n=== 测试 {case_name}: {bounds} ===")
        
        # 创建轨迹生成器
        generator = SimpleTrajectoryGenerator(grid_bounds=bounds)
        
        # 生成轨迹
        horizon = 10
        trajectory = generator.generate_agent_trajectory(vehicle, horizon, dt=1.0)
        
        print(f"轨迹长度: {len(trajectory)}")
        print(f"轨迹位置:")
        for i, pos in enumerate(trajectory):
            print(f"  t={i}: {pos}")
        
        # 检查是否有移动
        start_pos = np.array(trajectory[0])
        end_pos = np.array(trajectory[-1])
        movement = np.linalg.norm(end_pos - start_pos)
        print(f"总移动距离: {movement:.2f} 米")
        
        # 检查速度变化
        speeds = []
        for i in range(len(trajectory) - 1):
            pos1 = np.array(trajectory[i])
            pos2 = np.array(trajectory[i + 1])
            speed = np.linalg.norm(pos2 - pos1)
            speeds.append(speed)
        
        print(f"速度变化: {speeds}")
        print(f"平均速度: {np.mean(speeds):.2f} m/s")


def test_boundary_conditions():
    """测试边界条件"""
    print(f"\n=== 边界条件测试 ===")
    
    # 模拟轨迹生成中的边界检查逻辑
    current_pos = np.array([2.0, 2.0])
    current_speed = 4.0
    fixed_heading = 0.0
    dt = 1.0
    
    test_bounds = [(-9.0, 9.0), (-50.0, 50.0), (-100.0, 100.0)]
    
    for bounds in test_bounds:
        min_bound, max_bound = bounds
        print(f"\n边界: {bounds}")
        
        for t in range(5):
            # 计算下一位置
            next_x = current_pos[0] + current_speed * np.cos(fixed_heading) * dt
            next_y = current_pos[1] + current_speed * np.sin(fixed_heading) * dt
            
            print(f"  t={t}: 计算位置=({next_x:.1f}, {next_y:.1f})")
            
            # 边界检查
            if next_x < min_bound or next_x > max_bound or next_y < min_bound or next_y > max_bound:
                print(f"    超出边界！重置为当前位置")
                next_x = current_pos[0]
                next_y = current_pos[1]
            else:
                print(f"    在边界内，正常移动")
            
            current_pos = np.array([next_x, next_y])
            print(f"    最终位置=({current_pos[0]:.1f}, {current_pos[1]:.1f})")


if __name__ == "__main__":
    debug_agent_movement()
    test_boundary_conditions()



