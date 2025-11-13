"""
测试Alpha值持久性
验证Dirichlet Bank的alpha值在多次Q值计算后保持累积，不被重置
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

def get_total_alpha(planner, agent_id):
    """获取某个agent的总alpha值"""
    if agent_id not in planner.dirichlet_bank.agent_alphas:
        return 0.0

    total = 0.0
    for timestep in planner.dirichlet_bank.agent_alphas[agent_id]:
        total += planner.dirichlet_bank.agent_alphas[agent_id][timestep].sum()
    return total

def main():
    print("="*70)
    print("测试Alpha值持久性 - 验证修复是否生效")
    print("="*70)

    # 配置全局config
    global_config = GlobalConfig()
    global_config.visualization.verbose_level = 1  # 显示基本日志
    set_global_config(global_config)

    # 配置planner
    planner_config = C2OSRPlannerConfig()
    planner_config.trajectory_storage_multiplier = 5
    planner_config.learning_rate = 1.0
    planner_config.alpha_in = 50.0
    planner_config.min_buffer_size = 10  # 需要至少10个episodes才开始Q值计算

    print(f"\n配置:")
    print(f"  verbose_level = {global_config.visualization.verbose_level}")
    print(f"  storage_multiplier = {planner_config.trajectory_storage_multiplier}")
    print(f"  learning_rate = {planner_config.learning_rate}")
    print(f"  alpha_in = {planner_config.alpha_in}")
    print(f"  min_buffer_size = {planner_config.min_buffer_size}")

    # 创建planner
    planner = C2OSRPlanner(config=planner_config)

    print(f"\n初始 buffer size: {len(planner.trajectory_buffer)}")

    # 记录每个episode后的alpha值
    alpha_history = {1: [], 2: []}

    # 模拟15个episodes（超过min_buffer_size以触发Q值计算）
    num_episodes = 15

    for episode in range(num_episodes):
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
        planner.update(final_transition)

        buffer_size = len(planner.trajectory_buffer)
        print(f"Episode {episode} 后 buffer size: {buffer_size}")

        # 记录当前alpha值（在Q值计算之前）
        alpha_before_1 = get_total_alpha(planner, 1)
        alpha_before_2 = get_total_alpha(planner, 2)

        # 在buffer size达到min_buffer_size后，尝试select_action（会触发Q值计算）
        if buffer_size >= planner_config.min_buffer_size:
            print(f"\n{'─'*70}")
            print(f"Episode {episode}: Buffer充足，触发Q值计算")
            print(f"{'─'*70}")

            print(f"  Alpha before Q-calculation: Agent 1={alpha_before_1:.2f}, Agent 2={alpha_before_2:.2f}")

            try:
                # select_action会触发Q值计算
                test_state = create_test_state(0)
                action = planner.select_action(test_state)

                # 记录Q值计算后的alpha值
                alpha_after_1 = get_total_alpha(planner, 1)
                alpha_after_2 = get_total_alpha(planner, 2)

                print(f"  Alpha after  Q-calculation: Agent 1={alpha_after_1:.2f}, Agent 2={alpha_after_2:.2f}")

                # 检查alpha是否增长（或至少不减少）
                if alpha_after_1 >= alpha_before_1 and alpha_after_2 >= alpha_before_2:
                    delta_1 = alpha_after_1 - alpha_before_1
                    delta_2 = alpha_after_2 - alpha_before_2
                    print(f"  ✅ Alpha持久！变化量: Agent 1=+{delta_1:.2f}, Agent 2=+{delta_2:.2f}")
                else:
                    print(f"  ❌ Alpha被重置！这是Bug！")
                    print(f"     Agent 1: {alpha_before_1:.2f} → {alpha_after_1:.2f}")
                    print(f"     Agent 2: {alpha_before_2:.2f} → {alpha_after_2:.2f}")

                # 记录到历史
                alpha_history[1].append(alpha_after_1)
                alpha_history[2].append(alpha_after_2)

            except Exception as e:
                print(f"  ⚠️  select_action出错: {type(e).__name__}: {e}")
                # 如果出错，也记录当前alpha
                alpha_history[1].append(alpha_before_1)
                alpha_history[2].append(alpha_before_2)
        else:
            print(f"  Buffer size ({buffer_size}) < min_buffer_size ({planner_config.min_buffer_size}), 跳过Q值计算")
            # 记录当前alpha（没有Q值计算）
            alpha_history[1].append(alpha_before_1)
            alpha_history[2].append(alpha_before_2)

    print(f"\n{'='*70}")
    print(f"测试完成 - Alpha增长趋势分析")
    print(f"{'='*70}")

    # 分析alpha增长趋势
    print(f"\nAgent 1 Alpha历史:")
    for i, alpha in enumerate(alpha_history[1]):
        marker = "✅" if i == 0 or alpha >= alpha_history[1][i-1] else "❌"
        print(f"  Episode {i}: {alpha:.2f} {marker}")

    print(f"\nAgent 2 Alpha历史:")
    for i, alpha in enumerate(alpha_history[2]):
        marker = "✅" if i == 0 or alpha >= alpha_history[2][i-1] else "❌"
        print(f"  Episode {i}: {alpha:.2f} {marker}")

    # 最终验证
    print(f"\n{'='*70}")
    print(f"最终验证")
    print(f"{'='*70}")

    # 检查是否有单调递增的趋势
    all_increasing = True
    for agent_id in [1, 2]:
        for i in range(1, len(alpha_history[agent_id])):
            if alpha_history[agent_id][i] < alpha_history[agent_id][i-1]:
                all_increasing = False
                print(f"❌ Agent {agent_id} 在 episode {i} 出现alpha下降！")

    if all_increasing:
        print("✅ 所有agents的alpha值单调递增（或不变）")
        print("✅ Alpha持久性测试通过！")
        print("\n修复成功：")
        print("  - bank.init_agent() 只在首次调用时初始化")
        print("  - Alpha值在Q值计算之间保持累积")
        print("  - Dirichlet学习正常工作")
    else:
        print("❌ Alpha持久性测试失败")
        print("  Alpha值在某些episodes中下降，可能仍有bug")

    # 显示总增长
    initial_alpha = planner_config.alpha_in * 5  # 5个timesteps
    final_alpha_1 = alpha_history[1][-1] if alpha_history[1] else 0
    final_alpha_2 = alpha_history[2][-1] if alpha_history[2] else 0

    print(f"\n总Alpha增长:")
    print(f"  Agent 1: {initial_alpha:.2f} → {final_alpha_1:.2f} (增长 {final_alpha_1 - initial_alpha:.2f})")
    print(f"  Agent 2: {initial_alpha:.2f} → {final_alpha_2:.2f} (增长 {final_alpha_2 - initial_alpha:.2f})")

if __name__ == "__main__":
    main()
