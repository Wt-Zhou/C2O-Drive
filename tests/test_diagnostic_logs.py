"""
测试诊断日志功能
验证Alpha增长、Q值分布统计、匹配密度等日志是否正常工作
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from carla_c2osr.config import GlobalConfig, set_global_config
from carla_c2osr.algorithms.c2osr.planner import C2OSRPlanner
from carla_c2osr.algorithms.c2osr.config import C2OSRPlannerConfig
from carla_c2osr.env.types import WorldState, EgoState, AgentState, AgentType
from carla_c2osr.core.planner import Transition

def create_test_state(step: int = 0, num_agents: int = 2):
    """创建测试用的WorldState"""
    ego = EgoState(
        position_m=(float(step) * 2.0, 0.0),
        velocity_mps=(5.0, 0.0),
        yaw_rad=0.0
    )

    agents = []
    for i in range(num_agents):
        agent = AgentState(
            agent_id=f"agent_{i+1}",
            position_m=(float(step) * 2.0 + 10.0 + i * 5, float(i) * 3),
            velocity_mps=(3.0, 0.0),
            heading_rad=0.0,
            agent_type=AgentType.VEHICLE
        )
        agents.append(agent)

    return WorldState(time_s=float(step), ego=ego, agents=agents)

def main():
    print("="*70)
    print("测试诊断日志功能")
    print("="*70)

    # 配置全局config
    global_config = GlobalConfig()
    global_config.visualization.verbose_level = 2  # 开启详细日志
    set_global_config(global_config)

    # 配置planner
    planner_config = C2OSRPlannerConfig()
    planner_config.trajectory_storage_multiplier = 5
    planner_config.learning_rate = 1.0
    planner_config.alpha_in = 50.0  # 使用默认的强先验
    planner_config.min_buffer_size = 0

    print(f"\n配置:")
    print(f"  verbose_level = {global_config.visualization.verbose_level}")
    print(f"  storage_multiplier = {planner_config.trajectory_storage_multiplier}")
    print(f"  learning_rate = {planner_config.learning_rate}")
    print(f"  alpha_in = {planner_config.alpha_in}")

    # 创建planner
    planner = C2OSRPlanner(config=planner_config)

    print(f"\n初始 buffer size: {len(planner.trajectory_buffer)}")

    # 模拟3个episodes
    for episode in range(3):
        print(f"\n{'='*70}")
        print(f"Episode {episode}: 模拟5步 + 成功完成")
        print(f"{'='*70}")

        # 5个正常步骤
        for step in range(5):
            state = create_test_state(step)
            action = 0

            transition = Transition(
                state=state,
                action=action,
                reward=0.0,
                next_state=create_test_state(step + 1),
                terminated=False,
                truncated=False,
                info={},
            )
            planner.update(transition)

        # 成功完成的final transition
        final_state = create_test_state(5)
        final_transition = Transition(
            state=final_state,
            action=0,
            reward=0.0,
            next_state=final_state,
            terminated=False,
            truncated=True,  # 标记episode完成
            info={},
        )

        print(f"\n发送final transition (truncated=True)...")
        planner.update(final_transition)

        buffer_size = len(planner.trajectory_buffer)
        print(f"\nEpisode {episode} 后 buffer size: {buffer_size}")

        # 在第1个episode之后尝试select_action（会触发Q值计算）
        if episode >= 1:
            print(f"\n{'─'*70}")
            print(f"Episode {episode}: 调用select_action（应该看到诊断日志）")
            print(f"{'─'*70}")

            try:
                # select_action会触发plan，进而触发Q值计算
                test_state = create_test_state(0)
                action = planner.select_action(test_state)
                print(f"\n✓ select_action成功返回: {action}")
            except Exception as e:
                print(f"\n⚠️  select_action出错（可能需要更多设置）: {type(e).__name__}")
                # 这是预期的，因为我们没有设置完整的lattice等，但数据存储已经验证

    print(f"\n{'='*70}")
    print(f"测试完成")
    print(f"{'='*70}")

    # 获取buffer统计
    stats = planner.trajectory_buffer.get_stats()
    print(f"\nBuffer统计:")
    print(f"  Total episodes: {stats['total_episodes']}")
    print(f"  Per agent: {stats['per_agent_counts']}")
    print(f"  Capacity: {stats['capacity']}")

    print(f"\n{'='*70}")
    print("预期日志输出:")
    print("="*70)
    print("""
应该看到以下诊断日志：

1. ✅ Alpha增长日志 (verbose_level >= 2):
   [Dirichlet Update] Agent X, t=Y:
     Alpha: 50.00 → 51.00 (+1.00)
     Updated cells: Z/W (max_α=..., mean_α=...)

2. ✅ 匹配密度日志 (verbose_level >= 1):
   [Agent X] Matched Y/Z cells (W.X%) across M timesteps,
   Updated N timesteps, Buffer size: B

3. ✅ Q值分布统计 (verbose_level >= 2):
   Q值分布: P5=..., P50=..., P95=..., Mean=..., Std=...

如果Alpha一直停在50.00，说明更新没生效！
如果匹配密度<0.1%，说明阈值太严格！
如果Q值分布的Std很小(<0.1)，说明采样可能有问题！
""")

if __name__ == "__main__":
    main()
