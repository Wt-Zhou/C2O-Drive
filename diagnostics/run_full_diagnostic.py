"""完整诊断流程 - 运行所有分析并生成报告"""

from __future__ import annotations
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from c2o_drive.algorithms.c2osr.trajectory_buffer import HighPerformanceTrajectoryBuffer
from c2o_drive.config.global_config import GlobalConfig

from buffer_inspector import BufferInspector
from analyze_matching_issue import MatchingDiagnostic
from matching_visualizer import MatchingVisualizer


def main():
    """运行完整诊断流程"""
    print("="*80)
    print(" C2O-Drive 数据匹配问题完整诊断")
    print("="*80)

    # 1. 加载Buffer
    print("\n[步骤 1/4] 正在加载Buffer...")
    config = GlobalConfig()

    # 创建buffer实例
    buffer = HighPerformanceTrajectoryBuffer(
        capacity=1000,  # 默认容量
        horizon=config.time.default_horizon,
    )

    # 查找buffer文件
    possible_paths = [
        Path("data/trajectory_buffer.pkl"),
        Path("checkpoints/trajectory_buffer.pkl"),
        Path("results/trajectory_buffer.pkl"),
    ]

    buffer_path = None
    for path in possible_paths:
        if path.exists():
            buffer_path = path
            break

    if buffer_path is None:
        print(f"\n❌ 错误: Buffer文件不存在")
        print("\n尝试查找的路径:")
        for path in possible_paths:
            print(f"  - {path}")
        print("\n建议:")
        print("  1. 先运行 'python examples/run_c2osr_carla.py --scenario s4 --episodes 5' 生成历史数据")
        print("  2. 或者指定buffer文件路径: python run_full_diagnostic.py --buffer <path>")
        return

    buffer.load(str(buffer_path))
    print(f"✓ 成功加载Buffer: {buffer_path}")

    # 2. Buffer基本统计
    print("\n[步骤 2/4] 分析Buffer内容...")
    inspector = BufferInspector(buffer)
    inspector.print_summary()

    # 获取统计数据
    basic_stats = inspector.get_basic_stats()
    max_timestep = min(int(basic_stats['max_episode_length']) + 2, 15)
    timestep_availability = inspector.get_timestep_data_availability(max_timestep)

    # 3. 匹配问题诊断
    print("\n[步骤 3/4] 诊断匹配问题...")
    diagnostic = MatchingDiagnostic(buffer, config)

    num_episodes = min(10, len(buffer.agent_data))
    timestep_stats = diagnostic.analyze_multiple_episodes(
        num_episodes=num_episodes,
        sample_strategy="recent"
    )

    # 4. 生成可视化报告
    print("\n[步骤 4/4] 生成可视化报告...")
    visualizer = MatchingVisualizer(save_dir="./diagnostics/results")

    if timestep_stats:
        visualizer.generate_full_report(
            buffer=buffer,
            timestep_stats=timestep_stats,
            timestep_availability=timestep_availability
        )
    else:
        print("⚠️  没有足够的数据生成可视化报告")

    # 5. 总结和建议
    print("\n" + "="*80)
    print(" 诊断总结与建议")
    print("="*80)

    if timestep_stats:
        # 找出问题严重的timestep
        problem_timesteps = []
        for t, stats in timestep_stats.items():
            success_rate = stats["success_count"] / stats["total_count"] * 100
            if success_rate < 20:
                problem_timesteps.append((t, success_rate))

        if problem_timesteps:
            print("\n⚠️  发现问题:")
            print(f"  以下timestep的匹配成功率低于20%:")
            for t, rate in sorted(problem_timesteps):
                print(f"    - Timestep {t}: {rate:.1f}%")

            # 分析主要原因
            print("\n📊 主要失败原因:")
            reason_counts = {}
            for t, _ in problem_timesteps:
                for reason, count in timestep_stats[t]["failure_reasons"].items():
                    reason_counts[reason] = reason_counts.get(reason, 0) + count

            for reason, count in sorted(reason_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"    - {reason}: {count}次")

            # 根据原因给出建议
            print("\n💡 建议的修复方案:")

            if any("action_dist_too_large" in reason for reason in reason_counts):
                print("\n  [修复1] Action距离阈值问题")
                print("    问题: Ego action trajectory的距离超过阈值")
                print("    原因: 后期timestep的padding导致轨迹失真")
                print("    方案: 改进padding策略，使用速度外推而非重复最后位置")
                print("    文件: src/c2o_drive/algorithms/c2osr/trajectory_buffer.py")
                print("          store_episode_trajectories_by_timestep() 方法")

            if any("no_data" in reason or "no_spatial" in reason for reason in reason_counts):
                print("\n  [修复2] 历史数据不足")
                print("    问题: 某些timestep缺乏足够的历史数据")
                print("    原因: 大多数episode较短，后期timestep数据稀少")
                print("    方案: ")
                print("      - 增加episode运行步数（调整max_episode_steps）")
                print("      - 或者使用自适应阈值，后期timestep放宽匹配条件")

            if any("padding" in str(timestep_availability[t]) for t in problem_timesteps
                   if t in timestep_availability):
                print("\n  [修复3] Padding比例过高")
                print("    问题: 轨迹被过度填充，失去真实性")
                print("    方案: 存储实际轨迹长度，匹配时只比较有效部分")

        else:
            print("\n✓ 匹配性能良好!")
            print("  所有timestep的匹配成功率都在可接受范围内。")

    # 数据量建议
    if basic_stats['total_episodes'] < 50:
        print("\n⚠️  历史数据量较少:")
        print(f"    当前: {basic_stats['total_episodes']} episodes")
        print(f"    建议: 至少收集100+ episodes以获得稳定的匹配性能")

    print("\n" + "="*80)
    print("诊断完成! 请查看 diagnostics/results/ 目录中的可视化图表")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
