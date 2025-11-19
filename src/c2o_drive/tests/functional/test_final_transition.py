#!/usr/bin/env python3
"""测试成功完成的episodes是否正确存储数据"""

from c2o_drive.environments.carla.types import EgoState, AgentState, WorldState, AgentType
from c2o_drive.algorithms.c2osr.planner import C2OSRPlanner
from c2o_drive.algorithms.c2osr.config import C2OSRPlannerConfig
from c2o_drive.core.planner import Transition

print("="*70)
print("测试: 成功完成的episodes（无terminated/truncated）")
print("="*70)

config = C2OSRPlannerConfig()
config.trajectory_storage_multiplier = 5
config.min_buffer_size = 0
planner = C2OSRPlanner(config)

print(f"\n初始 buffer size: {len(planner.trajectory_buffer)}")

# Episode 1: 模拟成功完成（不设置terminated/truncated）
print("\n" + "="*70)
print("Episode 1: 5 steps (成功完成，无碰撞无超时)")
print("="*70)

planner.reset()

for step in range(5):
    ego = EgoState(position_m=(step * 2.0, 0.0), velocity_mps=(5.0, 0.0), yaw_rad=0.0)
    agent = AgentState(
        agent_id="agent_1",
        position_m=(step * 2.0 + 10.0, 2.0),
        velocity_mps=(4.0, 0.0),
        heading_rad=0.0,
        agent_type=AgentType.VEHICLE
    )
    world_state = WorldState(time_s=step * 1.0, ego=ego, agents=[agent])

    # 正常步骤：不设置terminated/truncated
    transition = Transition(
        state=world_state,
        action=0,
        reward=1.0,
        next_state=world_state,
        terminated=False,
        truncated=False,
    )
    planner.update(transition)
    print(f"  Step {step}: Buffer size = {len(planner.trajectory_buffer)}")

print(f"\n循环结束后 buffer size: {len(planner.trajectory_buffer)}")
print("⚠️  此时buffer应该还是0，因为没有发送final transition")

# 模拟run_c2osr_scenario.py的final transition
print("\n发送final transition (truncated=True)...")
final_transition = Transition(
    state=world_state,
    action=0,
    reward=0.0,
    next_state=world_state,
    terminated=False,
    truncated=True,  # 标记episode完成
    info={},
)
planner.update(final_transition)

final_size = len(planner.trajectory_buffer)
expected = 6 * config.trajectory_storage_multiplier  # 5 steps + 1 final transition
print(f"\n发送final transition后 buffer size: {final_size}")
print(f"Expected: {expected} (6 transitions: 5 steps + 1 final × {config.trajectory_storage_multiplier} multiplier)")

print("\n" + "="*70)
if final_size == expected:
    print("✅ 成功! Final transition触发了buffer存储")
    print(f"   Buffer size从0增长到{final_size}")
else:
    print(f"❌ 失败! Expected {expected}, got {final_size}")
print("="*70)

# Episode 2: 测试碰撞情况（terminated=True）
print("\n" + "="*70)
print("Episode 2: 碰撞场景（terminated=True）")
print("="*70)

planner.reset()

for step in range(3):  # 只运行3步就碰撞
    ego = EgoState(position_m=(step * 2.0, 0.0), velocity_mps=(5.0, 0.0), yaw_rad=0.0)
    agent = AgentState(
        agent_id="agent_1",
        position_m=(step * 2.0 + 10.0, 2.0),
        velocity_mps=(4.0, 0.0),
        heading_rad=0.0,
        agent_type=AgentType.VEHICLE
    )
    world_state = WorldState(time_s=step * 1.0, ego=ego, agents=[agent])

    is_collision = (step == 2)
    transition = Transition(
        state=world_state,
        action=0,
        reward=-10.0 if is_collision else 1.0,
        next_state=world_state,
        terminated=is_collision,  # 最后一步碰撞
        truncated=False,
    )
    planner.update(transition)

collision_size = len(planner.trajectory_buffer)
expected_collision = expected + 3 * config.trajectory_storage_multiplier  # Previous 30 + 3 collision steps
print(f"\n碰撞episode后 buffer size: {collision_size}")
print(f"Expected: {expected_collision} ({expected} + 3 steps × {config.trajectory_storage_multiplier})")

print("\n" + "="*70)
if collision_size == expected_collision:
    print("✅ 成功! 碰撞episode也正确存储了")
else:
    print(f"❌ 失败! Expected {expected_collision}, got {collision_size}")
print("="*70)
