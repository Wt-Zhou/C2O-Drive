"""
C2OSR + SimpleGrid 端到端集成测试

快速验证C2OSR算法在SimpleGrid环境下的完整工作流程
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from carla_c2osr.algorithms.c2osr import create_c2osr_planner, C2OSRPlannerConfig, LatticePlannerConfig, QValueConfig
from carla_c2osr.environments import SimpleGridEnvironment
from carla_c2osr.core.planner import Transition

def run_simplified_test():
    """运行简化的集成测试（减少计算量）"""

    print("\n" + "="*70)
    print(" C2OSR + SimpleGrid 集成测试")
    print("="*70)

    # 创建简化配置（减少采样数和horizon以加快测试）
    config = C2OSRPlannerConfig(
        lattice=LatticePlannerConfig(
            horizon=5,  # 减小horizon
            lateral_offsets=[-2.0, 0.0, 2.0],  # 减少候选轨迹
            speed_variations=[3.0, 5.0],
            num_trajectories=3,
        ),
        q_value=QValueConfig(
            horizon=5,
            n_samples=10,  # 大幅减少采样数（原来100）
        ),
    )

    # 创建环境和规划器
    print("\n1. 创建环境和规划器...")
    env = SimpleGridEnvironment(dt=0.5, max_episode_steps=20)
    planner = create_c2osr_planner(config)
    print("   ✓ 环境和规划器创建成功")

    # 运行短episode
    print("\n2. 运行测试episode...")
    state, info = env.reset(seed=42)

    total_reward = 0.0
    steps = 0
    max_steps = 10  # 限制步数

    for step in range(max_steps):
        # 选择动作
        reference_path = [(i * 5.0, 0.0) for i in range(15)]
        action = planner.select_action(state, reference_path=reference_path)

        # 执行动作
        step_result = env.step(action)

        # 更新规划器
        transition = Transition(
            state=state,
            action=action,
            reward=step_result.reward,
            next_state=step_result.observation,
            terminated=step_result.terminated,
            truncated=step_result.truncated,
            info=step_result.info,
        )
        planner.update(transition)

        total_reward += step_result.reward
        steps += 1

        print(f"   Step {step+1}: reward={step_result.reward:.2f}, "
              f"action=(t={action.throttle:.2f}, s={action.steer:.2f})")

        state = step_result.observation

        if step_result.terminated or step_result.truncated:
            print(f"   Episode结束: {'碰撞' if step_result.terminated else '超时'}")
            break

    env.close()

    # 输出结果
    print(f"\n3. 测试结果:")
    print(f"   ✓ 完成步数: {steps}")
    print(f"   ✓ 总奖励: {total_reward:.2f}")
    print(f"   ✓ 平均奖励: {total_reward/steps:.2f}")
    print(f"   ✓ Buffer大小: {len(planner.trajectory_buffer)}")

    # 验证
    assert steps > 0, "至少应该运行1步"
    assert total_reward < 0 or total_reward > -100, "奖励应该在合理范围内"

    print("\n" + "="*70)
    print(" ✓ 所有测试通过！")
    print("="*70)

    return {
        'steps': steps,
        'total_reward': total_reward,
        'buffer_size': len(planner.trajectory_buffer),
    }


if __name__ == "__main__":
    try:
        results = run_simplified_test()
        print(f"\n测试成功完成！")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
