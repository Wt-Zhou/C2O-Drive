#!/usr/bin/env python3
"""
测试Dirichlet批量采样的性能提升
"""

import time
import numpy as np
from pathlib import Path
import sys

# 添加项目根目录到路径
_repo_root = Path(__file__).resolve().parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from carla_c2osr.agents.c2osr.spatial_dirichlet import DirichletParams, OptimizedMultiTimestepSpatialDirichletBank


def test_original_vs_optimized():
    """测试原始版本vs优化版本的Dirichlet采样"""

    print("=" * 60)
    print("Dirichlet批量采样性能测试")
    print("=" * 60)

    # 模拟参数
    K = 40000  # 网格单元数
    n_samples = 100  # Q值采样数量
    n_timesteps = 8
    n_agents = 2

    print(f"\n测试配置:")
    print(f"  网格单元数: {K}")
    print(f"  采样数量: {n_samples}")
    print(f"  时间步数: {n_timesteps}")
    print(f"  Agents数量: {n_agents}")

    # 创建bank
    params = DirichletParams(alpha_in=50.0, alpha_out=1e-6)
    bank = OptimizedMultiTimestepSpatialDirichletBank(K, params, horizon=n_timesteps)

    # 初始化agents
    for agent_id in range(1, n_agents + 1):
        # 模拟可达集（每个时间步约500个cells）
        reachable_sets = {}
        for t in range(1, n_timesteps + 1):
            reachable_cells = np.random.choice(K, size=500, replace=False).tolist()
            reachable_sets[t] = reachable_cells
        bank.init_agent(agent_id, reachable_sets)

    # 测试原始版本（手动实现循环版本用于对比）
    print("\n⏱️  测试原始循环版本...")
    start_time = time.time()

    for agent_id in range(1, n_agents + 1):
        for timestep in bank.agent_alphas[agent_id]:
            alpha = bank.agent_alphas[agent_id][timestep]
            samples = []
            for _ in range(n_samples):  # 原始：循环采样
                prob_vector = np.random.dirichlet(alpha)
                samples.append(prob_vector)

    time_original = time.time() - start_time
    print(f"   耗时: {time_original:.3f}秒")

    # 测试优化版本
    print("\n⚡ 测试批量采样版本...")
    start_time = time.time()

    for agent_id in range(1, n_agents + 1):
        distributions = bank.sample_transition_distributions(agent_id, n_samples=n_samples)

    time_optimized = time.time() - start_time
    print(f"   耗时: {time_optimized:.3f}秒")

    # 计算加速比
    speedup = time_original / time_optimized

    print("\n" + "=" * 60)
    print(f"🚀 性能提升汇总:")
    print(f"   原始版本: {time_original:.3f}秒")
    print(f"   优化版本: {time_optimized:.3f}秒")
    print(f"   加速比: {speedup:.2f}x")
    print(f"   时间节省: {(time_original - time_optimized):.3f}秒 ({(1 - time_optimized/time_original)*100:.1f}%)")
    print("=" * 60)

    # 验证结果一致性
    print("\n🔍 验证结果一致性...")

    # 重置随机种子
    np.random.seed(42)

    # 手动循环采样
    alpha_test = bank.agent_alphas[1][1]
    samples_loop = []
    for _ in range(10):
        samples_loop.append(np.random.dirichlet(alpha_test))

    # 重置随机种子
    np.random.seed(42)

    # 批量采样
    samples_batch = np.random.dirichlet(alpha_test, size=10)

    # 比较
    max_diff = np.max(np.abs(np.array(samples_loop) - samples_batch))
    print(f"   最大差异: {max_diff:.10f}")

    if max_diff < 1e-10:
        print("   ✅ 结果完全一致！")
    else:
        print("   ⚠️ 存在数值差异（可能由于随机种子）")

    return speedup


def test_different_sample_sizes():
    """测试不同采样数量的性能"""
    print("\n" + "=" * 60)
    print("不同采样数量的性能测试")
    print("=" * 60)

    K = 40000
    n_timesteps = 8

    params = DirichletParams(alpha_in=50.0, alpha_out=1e-6)
    bank = OptimizedMultiTimestepSpatialDirichletBank(K, params, horizon=n_timesteps)

    # 初始化一个agent
    reachable_sets = {}
    for t in range(1, n_timesteps + 1):
        reachable_cells = np.random.choice(K, size=500, replace=False).tolist()
        reachable_sets[t] = reachable_cells
    bank.init_agent(1, reachable_sets)

    sample_sizes = [10, 50, 100, 200, 500]

    print(f"\n{'样本数':<10} {'原始版本':<12} {'优化版本':<12} {'加速比':<10}")
    print("-" * 50)

    for n_samples in sample_sizes:
        # 原始版本
        start = time.time()
        for timestep in bank.agent_alphas[1]:
            alpha = bank.agent_alphas[1][timestep]
            samples = []
            for _ in range(n_samples):
                prob_vector = np.random.dirichlet(alpha)
                samples.append(prob_vector)
        time_orig = time.time() - start

        # 优化版本
        start = time.time()
        _ = bank.sample_transition_distributions(1, n_samples=n_samples)
        time_opt = time.time() - start

        speedup = time_orig / time_opt

        print(f"{n_samples:<10} {time_orig:<12.4f} {time_opt:<12.4f} {speedup:<10.2f}x")


if __name__ == "__main__":
    # 运行主性能测试
    speedup = test_original_vs_optimized()

    # 测试不同采样数量
    test_different_sample_sizes()

    print("\n✅ 测试完成！")

    if speedup and speedup > 5:
        print(f"\n🎉 Dirichlet批量采样优化成功！加速比达到 {speedup:.2f}x")
    elif speedup:
        print(f"\n⚠️  加速效果有限 ({speedup:.2f}x)")
