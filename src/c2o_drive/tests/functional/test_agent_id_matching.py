#!/usr/bin/env python3
"""测试Agent ID匹配修复后的效果"""

import sys

# 设置verbose level为2来启用详细调试
from c2o_drive.config import get_global_config
config = get_global_config()
config.visualization.verbose_level = 2  # Enable detailed debug logging

from c2o_drive.environments.carla.types import EgoState, AgentState, WorldState, AgentType
from c2o_drive.algorithms.c2osr.config import C2OSRPlannerConfig
from c2o_drive.algorithms.c2osr.planner import C2OSRPlanner
from c2o_drive.core.planner import Transition


def create_world_state(timestep: int, ego_x: float = 0.0) -> WorldState:
    """创建简单的world state"""
    ego = EgoState(
        position_m=(ego_x, 0.0),
        velocity_mps=(5.0, 0.0),
        yaw_rad=0.0,
    )

    agent = AgentState(
        agent_id="agent_1",
        position_m=(ego_x + 10.0, 2.0),
        velocity_mps=(4.0, 0.0),
        heading_rad=0.0,
        agent_type=AgentType.VEHICLE,
    )

    return WorldState(
        time_s=timestep * 1.0,
        ego=ego,
        agents=[agent]
    )


def main():
    print("=" * 70)
    print("测试Agent ID匹配修复")
    print("=" * 70)

    # 创建planner
    planner_config = C2OSRPlannerConfig()
    planner_config.trajectory_storage_multiplier = 10  # 使用较小的倍数方便观察
    planner_config.min_buffer_size = 0

    planner = C2OSRPlanner(planner_config)

    print(f"\n配置:")
    print(f"  storage_multiplier: {planner.config.trajectory_storage_multiplier}")
    print(f"  verbose_level: {config.visualization.verbose_level}")

    # Episode 1: 存储数据
    print(f"\n{'='*70}")
    print("Episode 1: 存储初始数据")
    print("=" * 70)

    for t in range(3):
        world_state = create_world_state(t, ego_x=t * 2.0)
        action = planner.select_action(world_state)
        print(f"  Timestep {t}: ego_x={t*2.0}, buffer_size={len(planner.trajectory_buffer)}")

    # 结束episode 1
    final_state = create_world_state(3, ego_x=6.0)
    transition = Transition(
        state=final_state,
        action=0,
        reward=0.0,
        next_state=final_state,
        terminated=True,
    )
    planner.update(transition)

    buffer_stats = planner.trajectory_buffer.get_stats()
    print(f"\nEpisode 1结束:")
    print(f"  Buffer总episodes: {buffer_stats['total_episodes']}")
    print(f"  可用agent IDs: {sorted(buffer_stats['per_agent_counts'].keys())}")
    for agent_id, count in buffer_stats['per_agent_counts'].items():
        print(f"    Agent {agent_id}: {count} episodes")

    # Episode 2: 尝试匹配
    print(f"\n{'='*70}")
    print("Episode 2: 尝试匹配历史数据")
    print("=" * 70)
    print("(查看详细的Debug日志，了解匹配过程)\n")

    for t in range(3):
        world_state = create_world_state(t, ego_x=t * 2.0)  # 相同的初始位置
        action = planner.select_action(world_state)

    print(f"\n{'='*70}")
    print("总结")
    print("=" * 70)
    print("如果看到'可用的 agent IDs: [1]'，说明Agent ID修复成功")
    print("如果仍然'Matched 0 cells'，说明是阈值或其他匹配条件的问题")


if __name__ == '__main__':
    main()
