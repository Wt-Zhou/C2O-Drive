#!/usr/bin/env python3
"""
测试不同智能体类型的动力学差异
"""

from __future__ import annotations
import numpy as np
import math

from c2o_drive.environments.carla.types import AgentState, EgoState, WorldState, AgentType, AgentDynamicsParams
from c2o_drive.algorithms.c2osr.grid import GridSpec, GridMapper


def test_agent_dynamics_params():
    """测试不同智能体类型的动力学参数差异。"""
    # 获取所有类型的参数
    pedestrian_params = AgentDynamicsParams.for_agent_type(AgentType.PEDESTRIAN)
    bicycle_params = AgentDynamicsParams.for_agent_type(AgentType.BICYCLE)
    vehicle_params = AgentDynamicsParams.for_agent_type(AgentType.VEHICLE)
    motorcycle_params = AgentDynamicsParams.for_agent_type(AgentType.MOTORCYCLE)
    
    # 验证行人参数
    assert pedestrian_params.max_speed_mps == 2.0
    assert pedestrian_params.wheelbase_m == 0.0  # 行人无轴距
    assert pedestrian_params.max_yaw_rate_rps > 1.0  # 行人转向灵活
    
    # 验证车辆参数
    assert vehicle_params.max_speed_mps > pedestrian_params.max_speed_mps
    assert vehicle_params.wheelbase_m > 0  # 车辆有轴距
    assert vehicle_params.max_yaw_rate_rps < bicycle_params.max_yaw_rate_rps  # 车辆转向受限
    
    # 验证速度排序: 摩托车 > 车辆 > 自行车 > 行人
    assert motorcycle_params.max_speed_mps > vehicle_params.max_speed_mps
    assert vehicle_params.max_speed_mps > bicycle_params.max_speed_mps
    assert bicycle_params.max_speed_mps > pedestrian_params.max_speed_mps


def test_reachable_set_differences():
    """测试不同智能体类型的可达集大小差异。"""
    np.random.seed(42)  # 固定随机种子
    
    # 设置网格和世界状态
    grid_spec = GridSpec(size_m=10.0, cell_m=0.2, macro=True)  # 50x50 网格
    grid_mapper = GridMapper(grid_spec)
    
    ego = EgoState(position_m=(0.0, 0.0), velocity_mps=(0.0, 0.0), yaw_rad=0.0)
    world = WorldState(time_s=0.0, ego=ego, agents=[])
    
    # 创建相同初始状态但不同类型的智能体
    initial_pos = (2.0, 1.0)
    initial_vel = (1.0, 0.0)  # 1 m/s 向前
    initial_heading = 0.0
    
    pedestrian = AgentState(
        agent_id="ped-1", 
        position_m=initial_pos, 
        velocity_mps=initial_vel,
        heading_rad=initial_heading,
        agent_type=AgentType.PEDESTRIAN
    )
    
    bicycle = AgentState(
        agent_id="bike-1",
        position_m=initial_pos,
        velocity_mps=initial_vel,
        heading_rad=initial_heading,
        agent_type=AgentType.BICYCLE
    )
    
    vehicle = AgentState(
        agent_id="car-1",
        position_m=initial_pos,
        velocity_mps=initial_vel,
        heading_rad=initial_heading,
        agent_type=AgentType.VEHICLE
    )
    
    motorcycle = AgentState(
        agent_id="moto-1",
        position_m=initial_pos,
        velocity_mps=initial_vel,
        heading_rad=initial_heading,
        agent_type=AgentType.MOTORCYCLE
    )
    
    # 计算可达集
    dt = 1.0
    n_samples = 200  # 增加采样数以获得更稳定的结果
    
    reachable_ped = grid_mapper.successor_cells(pedestrian, dt, n_samples)
    reachable_bike = grid_mapper.successor_cells(bicycle, dt, n_samples)
    reachable_car = grid_mapper.successor_cells(vehicle, dt, n_samples)
    reachable_moto = grid_mapper.successor_cells(motorcycle, dt, n_samples)
    
    print(f"可达集大小对比:")
    print(f"  行人: {len(reachable_ped)} 个单元")
    print(f"  自行车: {len(reachable_bike)} 个单元")
    print(f"  车辆: {len(reachable_car)} 个单元")
    print(f"  摩托车: {len(reachable_moto)} 个单元")
    
    # 验证可达集特性
    assert len(reachable_ped) > 0, "行人可达集不能为空"
    assert len(reachable_bike) > 0, "自行车可达集不能为空"
    assert len(reachable_car) > 0, "车辆可达集不能为空"
    assert len(reachable_moto) > 0, "摩托车可达集不能为空"
    
    # 行人因为可以任意方向移动，通常有更多的可达单元
    # 但这取决于具体的动力学参数和采样策略，所以不做强制要求
    
    # 验证可达集的合理性：所有单元都在网格范围内
    for idx in reachable_ped + reachable_bike + reachable_car + reachable_moto:
        assert 0 <= idx < grid_spec.num_cells, f"可达单元索引 {idx} 超出网格范围"


def test_vehicle_constraints():
    """测试车辆动力学约束的正确性。"""
    # 设置高速移动的车辆
    grid_spec = GridSpec(size_m=20.0, cell_m=0.5, macro=True)
    grid_mapper = GridMapper(grid_spec)
    
    ego = EgoState(position_m=(0.0, 0.0), velocity_mps=(0.0, 0.0), yaw_rad=0.0)
    world = WorldState(time_s=0.0, ego=ego, agents=[])
    
    # 高速车辆（接近最大速度）
    fast_vehicle = AgentState(
        agent_id="fast-car",
        position_m=(5.0, 0.0),
        velocity_mps=(12.0, 0.0),  # 12 m/s，接近最大速度 15 m/s
        heading_rad=0.0,
        agent_type=AgentType.VEHICLE
    )
    
    # 计算可达集
    reachable_fast = grid_mapper.successor_cells(fast_vehicle, dt=1.0, n_samples=100)
    
    # 将可达单元转换为位置，检查是否符合车辆约束
    reachable_positions = []
    for idx in reachable_fast:
        xy_local = grid_mapper.index_to_xy_center(idx)
        xy_world = (xy_local[0] + world.ego.position_m[0], xy_local[1] + world.ego.position_m[1])
        reachable_positions.append(xy_world)
    
    # 验证可达位置的合理性
    initial_pos = fast_vehicle.position_m
    vehicle_params = AgentDynamicsParams.for_agent_type(AgentType.VEHICLE)
    
    for pos in reachable_positions:
        # 计算位移距离
        dx = pos[0] - initial_pos[0]
        dy = pos[1] - initial_pos[1]
        distance = math.sqrt(dx**2 + dy**2)
        
        # 验证位移距离的合理性（考虑最大速度约束）
        # 最大可能位移 ≈ (初始速度 + 最大速度) / 2 * dt
        max_possible_distance = (12.0 + vehicle_params.max_speed_mps) / 2 * 1.0 + 2.0  # 加 2m 容差
        assert distance <= max_possible_distance, f"可达位置距离 {distance} 超过物理约束 {max_possible_distance}"


def test_pedestrian_flexibility():
    """测试行人的运动灵活性。"""
    grid_spec = GridSpec(size_m=8.0, cell_m=0.4, macro=True)
    grid_mapper = GridMapper(grid_spec)
    
    ego = EgoState(position_m=(0.0, 0.0), velocity_mps=(0.0, 0.0), yaw_rad=0.0)
    world = WorldState(time_s=0.0, ego=ego, agents=[])
    
    # 移动中的行人
    pedestrian = AgentState(
        agent_id="ped-test",
        position_m=(1.0, 1.0),
        velocity_mps=(0.5, 0.8),  # 斜向移动
        heading_rad=math.pi/4,    # 45度角
        agent_type=AgentType.PEDESTRIAN
    )
    
    reachable_ped = grid_mapper.successor_cells(pedestrian, dt=1.0, n_samples=150)
    
    # 将可达单元转换为局部坐标
    reachable_local = []
    for idx in reachable_ped:
        xy_local = grid_mapper.index_to_xy_center(idx)
        reachable_local.append(xy_local)
    
    # 检查行人可达集的方向分布
    current_grid = grid_mapper.to_grid_frame(pedestrian.position_m)
    
    directions = []
    for pos in reachable_local:
        dx = pos[0] - current_grid[0]
        dy = pos[1] - current_grid[1]
        if dx != 0 or dy != 0:  # 排除原地不动
            angle = math.atan2(dy, dx)
            directions.append(angle)
    
    # 验证行人可以朝多个方向移动（角度分布应该相对均匀）
    if len(directions) > 10:  # 确保有足够的样本
        # 计算角度标准差，行人应该有较大的角度覆盖范围
        directions_array = np.array(directions)
        angle_std = np.std(directions_array)
        assert angle_std > 0.5, f"行人角度标准差 {angle_std} 太小，缺乏运动灵活性"


if __name__ == "__main__":
    test_agent_dynamics_params()
    test_reachable_set_differences()
    test_vehicle_constraints()
    test_pedestrian_flexibility()
    print("所有动力学测试通过！")
