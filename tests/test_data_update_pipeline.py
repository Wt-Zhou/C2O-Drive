#!/usr/bin/env python3
"""测试数据更新链路是否正常工作"""

import numpy as np
from carla_c2osr.algorithms.c2osr.config import C2OSRPlannerConfig
from carla_c2osr.algorithms.c2osr.planner import C2OSRPlanner
from carla_c2osr.env.types import WorldState, EgoState, AgentState, EgoControl, AgentType
from carla_c2osr.core.planner import Transition


def create_dummy_world_state(t: int) -> WorldState:
    """创建一个假的 WorldState 用于测试"""
    ego = EgoState(
        position_m=(float(t), 0.0),
        velocity_mps=(1.0, 0.0),
        yaw_rad=0.0,
    )

    # 创建一个 agent
    agent = AgentState(
        agent_id=1,
        position_m=(float(t) + 5.0, 0.0),
        velocity_mps=(0.5, 0.0),
        heading_rad=0.0,
        agent_type=AgentType.VEHICLE,
    )

    return WorldState(ego=ego, agents=[agent], time_s=float(t))


def test_data_update_pipeline():
    """测试数据更新链路"""
    print("=" * 60)
    print("测试数据更新链路")
    print("=" * 60)

    # 创建 planner
    config = C2OSRPlannerConfig(
        buffer_capacity=1000,
        min_buffer_size=1,
        trajectory_storage_multiplier=10,
    )
    planner = C2OSRPlanner(config)

    print(f"\n初始 buffer 大小: {len(planner.trajectory_buffer)}")
    assert len(planner.trajectory_buffer) == 0, "Buffer 应该初始为空"

    # 模拟 Episode 1
    print("\n" + "=" * 60)
    print("Episode 1")
    print("=" * 60)

    planner.reset()
    episode_steps = 5

    for t in range(episode_steps):
        # 获取观测
        obs = create_dummy_world_state(t)

        # 选择动作
        action = planner.select_action(obs)
        print(f"  步骤 {t}: 选择动作 {action}")

        # 创建 transition
        next_obs = create_dummy_world_state(t + 1)
        is_done = (t == episode_steps - 1)
        transition = Transition(
            state=obs,
            action=action,
            reward=1.0,
            next_state=next_obs,
            terminated=is_done,
            truncated=False,
        )

        # 更新 planner
        metrics = planner.update(transition)
        print(f"    Buffer 大小: {metrics.custom.get('buffer_size', 0)}")

    print(f"\nEpisode 1 结束后 buffer 大小: {len(planner.trajectory_buffer)}")
    buffer_size_after_ep1 = len(planner.trajectory_buffer)
    assert buffer_size_after_ep1 > 0, "Episode 1 结束后 buffer 应该有数据！"

    # 模拟 Episode 2
    print("\n" + "=" * 60)
    print("Episode 2")
    print("=" * 60)

    planner.reset()

    for t in range(episode_steps):
        obs = create_dummy_world_state(t)
        action = planner.select_action(obs)
        print(f"  步骤 {t}: 选择动作 {action}")

        next_obs = create_dummy_world_state(t + 1)
        is_done = (t == episode_steps - 1)
        transition = Transition(
            state=obs,
            action=action,
            reward=1.0,
            next_state=next_obs,
            terminated=is_done,
            truncated=False,
        )

        metrics = planner.update(transition)
        print(f"    Buffer 大小: {metrics.custom.get('buffer_size', 0)}")

    print(f"\nEpisode 2 结束后 buffer 大小: {len(planner.trajectory_buffer)}")
    buffer_size_after_ep2 = len(planner.trajectory_buffer)
    assert buffer_size_after_ep2 > buffer_size_after_ep1, "Episode 2 结束后 buffer 应该增长！"

    # 模拟 Episode 3
    print("\n" + "=" * 60)
    print("Episode 3 (测试持续增长)")
    print("=" * 60)

    planner.reset()

    for t in range(episode_steps):
        obs = create_dummy_world_state(t)
        action = planner.select_action(obs)

        next_obs = create_dummy_world_state(t + 1)
        is_done = (t == episode_steps - 1)
        transition = Transition(
            state=obs,
            action=action,
            reward=1.0,
            next_state=next_obs,
            terminated=is_done,
            truncated=False,
        )

        metrics = planner.update(transition)

    print(f"\nEpisode 3 结束后 buffer 大小: {len(planner.trajectory_buffer)}")
    buffer_size_after_ep3 = len(planner.trajectory_buffer)
    assert buffer_size_after_ep3 > buffer_size_after_ep2, "Episode 3 结束后 buffer 应该继续增长！"

    # 验证 storage_multiplier 效果
    print("\n" + "=" * 60)
    print("验证 storage_multiplier")
    print("=" * 60)

    # 每个 episode 有 5 个 timesteps
    # 每个 timestep 有 1 个 agent
    # storage_multiplier = 10
    # 预期: 5 * 1 * 10 = 50 个 episodes per episode
    expected_per_episode = episode_steps * 1 * config.trajectory_storage_multiplier
    print(f"每个 episode 预期存储: {expected_per_episode} 条数据")
    print(f"3 个 episodes 预期总共: {expected_per_episode * 3} 条数据")
    print(f"实际 buffer 大小: {buffer_size_after_ep3}")

    # 允许一些误差（因为可能有一些数据被过滤）
    assert buffer_size_after_ep3 >= expected_per_episode * 2, \
        f"Buffer 大小应该接近 {expected_per_episode * 3}，但只有 {buffer_size_after_ep3}"

    # 测试 buffer capacity 限制
    print("\n" + "=" * 60)
    print("测试 buffer capacity 限制")
    print("=" * 60)

    config_small = C2OSRPlannerConfig(
        buffer_capacity=100,
        min_buffer_size=1,
        trajectory_storage_multiplier=10,
    )
    planner_small = C2OSRPlanner(config_small)

    # 运行足够多的 episodes 以填满 buffer
    for ep in range(5):
        planner_small.reset()
        for t in range(episode_steps):
            obs = create_dummy_world_state(t)
            action = planner_small.select_action(obs)
            next_obs = create_dummy_world_state(t + 1)
            is_done = (t == episode_steps - 1)
            transition = Transition(
                state=obs,
                action=action,
                reward=1.0,
                next_state=next_obs,
                terminated=is_done,
                truncated=False,
            )
            planner_small.update(transition)

        print(f"  Episode {ep + 1}: Buffer 大小 = {len(planner_small.trajectory_buffer)}")

    print(f"\n最终 buffer 大小: {len(planner_small.trajectory_buffer)}")
    print(f"Buffer capacity: {planner_small.trajectory_buffer.capacity}")
    assert len(planner_small.trajectory_buffer) <= planner_small.trajectory_buffer.capacity, \
        "Buffer 大小不应超过 capacity"

    print("\n" + "=" * 60)
    print("✓ 所有测试通过！")
    print("=" * 60)
    print("\n总结:")
    print(f"  ✓ Episode 数据收集正常")
    print(f"  ✓ update() 方法正确触发存储")
    print(f"  ✓ Buffer 持续增长")
    print(f"  ✓ storage_multiplier 参数生效")
    print(f"  ✓ Buffer capacity 限制正常工作")
    print(f"\n数据更新链路修复成功！")


if __name__ == '__main__':
    try:
        test_data_update_pipeline()
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"✗ 测试失败: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        exit(1)
