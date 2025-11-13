#!/usr/bin/env python3
"""测试通过update()收集数据（不调用select_action）"""

from carla_c2osr.env.types import EgoState, AgentState, WorldState, AgentType
from carla_c2osr.algorithms.c2osr.planner import C2OSRPlanner
from carla_c2osr.algorithms.c2osr.config import C2OSRPlannerConfig
from carla_c2osr.core.planner import Transition

print("="*70)
print("测试: 通过update()收集数据（模拟run_c2osr_scenario.py的模式）")
print("="*70)

# 创建planner
config = C2OSRPlannerConfig()
config.trajectory_storage_multiplier = 5
config.min_buffer_size = 0
planner = C2OSRPlanner(config)

print(f"\n初始 buffer size: {len(planner.trajectory_buffer)}")

# 模拟episode (不调用select_action，只调用update)
print("\n" + "="*70)
print("Episode 1: 5 steps")
print("="*70)

planner.reset()

for step in range(5):
    # 创建world state
    ego = EgoState(
        position_m=(step * 2.0, 0.0),
        velocity_mps=(5.0, 0.0),
        yaw_rad=0.0
    )
    agent = AgentState(
        agent_id="agent_1",
        position_m=(step * 2.0 + 10.0, 2.0),
        velocity_mps=(4.0, 0.0),
        heading_rad=0.0,
        agent_type=AgentType.VEHICLE
    )
    world_state = WorldState(time_s=step * 1.0, ego=ego, agents=[agent])

    # 只调用update (不调用select_action)
    is_last_step = (step == 4)
    transition = Transition(
        state=world_state,
        action=0,  # Dummy action
        reward=1.0,
        next_state=world_state,
        terminated=is_last_step,
        truncated=False
    )

    planner.update(transition)
    print(f"  Step {step}: Buffer size = {len(planner.trajectory_buffer)}")

print(f"\nEpisode 1 结束后 buffer size: {len(planner.trajectory_buffer)}")
stats = planner.trajectory_buffer.get_stats()
print(f"Expected: ~{5 * config.trajectory_storage_multiplier} (5 steps × {config.trajectory_storage_multiplier} multiplier)")
print(f"Per-agent counts: {stats['per_agent_counts']}")

# Episode 2
print("\n" + "="*70)
print("Episode 2: 5 steps (相同的轨迹)")
print("="*70)

planner.reset()

for step in range(5):
    ego = EgoState(
        position_m=(step * 2.0, 0.0),
        velocity_mps=(5.0, 0.0),
        yaw_rad=0.0
    )
    agent = AgentState(
        agent_id="agent_1",
        position_m=(step * 2.0 + 10.0, 2.0),
        velocity_mps=(4.0, 0.0),
        heading_rad=0.0,
        agent_type=AgentType.VEHICLE
    )
    world_state = WorldState(time_s=step * 1.0, ego=ego, agents=[agent])

    is_last_step = (step == 4)
    transition = Transition(
        state=world_state,
        action=0,
        reward=1.0,
        next_state=world_state,
        terminated=is_last_step,
        truncated=False
    )

    planner.update(transition)
    print(f"  Step {step}: Buffer size = {len(planner.trajectory_buffer)}")

final_size = len(planner.trajectory_buffer)
expected = 5 * 2 * config.trajectory_storage_multiplier  # 2 episodes
print(f"\nEpisode 2 结束后 buffer size: {final_size}")
print(f"Expected: ~{expected} (2 episodes × 5 steps × {config.trajectory_storage_multiplier} multiplier)")

print("\n" + "="*70)
if final_size > 0:
    print("✅ 成功! 通过update()收集数据正常工作")
    print("   Buffer size从0增长到了", final_size)
else:
    print("❌ 失败! Buffer size仍然是0")
print("="*70)
