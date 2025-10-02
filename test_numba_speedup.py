#!/usr/bin/env python3
"""
测试numba优化版本的可达集计算性能提升
"""

import time
import numpy as np
from pathlib import Path
import sys

# 添加项目根目录到路径
_repo_root = Path(__file__).resolve().parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from carla_c2osr.agents.c2osr.grid import GridSpec, GridMapper, NUMBA_AVAILABLE
from carla_c2osr.env.types import AgentState, AgentType
from carla_c2osr.config import get_global_config

def create_test_agent(agent_type=AgentType.VEHICLE):
    """创建测试用的agent"""
    return AgentState(
        agent_id="test_agent_1",
        position_m=(10.0, 5.0),
        velocity_mps=(5.0, 0.0),
        heading_rad=0.0,
        agent_type=agent_type
    )

def benchmark_reachable_set_computation():
    """性能测试：对比原始版本和numba优化版本"""

    print("=" * 60)
    print("可达集计算性能测试")
    print("=" * 60)

    if not NUMBA_AVAILABLE:
        print("❌ Numba未安装，无法进行性能对比测试")
        return

    print(f"✅ Numba已安装，版本：{NUMBA_AVAILABLE}")

    # 创建网格
    config = get_global_config()
    grid_spec = GridSpec(
        size_m=config.grid.grid_size_m,
        cell_m=config.grid.cell_size_m,
        macro=True
    )
    grid = GridMapper(grid_spec, world_center=(0.0, 0.0))

    # 创建测试agent
    agent = create_test_agent(AgentType.VEHICLE)

    # 测试参数
    horizon = 8
    dt = 1.0
    n_samples = 2000  # 使用默认的采样数量

    print(f"\n测试配置:")
    print(f"  网格尺寸: {grid_spec.size_m}m × {grid_spec.size_m}m")
    print(f"  单元尺寸: {grid_spec.cell_m}m")
    print(f"  网格单元数: {grid.spec.num_cells}")
    print(f"  Horizon: {horizon} 时间步")
    print(f"  采样数量: {n_samples}")
    print(f"  Agent类型: {agent.agent_type.name}")

    # 预热numba JIT编译（第一次调用会触发编译）
    print("\n🔥 预热numba JIT编译...")
    _ = grid._multi_timestep_successor_cells_numba(agent, horizon=horizon, dt=dt, n_samples=100)
    print("   JIT编译完成！")

    # 测试原始版本
    print("\n⏱️  测试原始版本...")
    start_time = time.time()
    result_original = grid._multi_timestep_successor_cells_original(
        agent, horizon=horizon, dt=dt, n_samples=n_samples
    )
    time_original = time.time() - start_time

    print(f"   耗时: {time_original:.3f}秒")
    print(f"   可达集大小: {sum(len(cells) for cells in result_original.values())} cells")

    # 测试numba优化版本
    print("\n⚡ 测试numba优化版本...")
    start_time = time.time()
    result_numba = grid._multi_timestep_successor_cells_numba(
        agent, horizon=horizon, dt=dt, n_samples=n_samples
    )
    time_numba = time.time() - start_time

    print(f"   耗时: {time_numba:.3f}秒")
    print(f"   可达集大小: {sum(len(cells) for cells in result_numba.values())} cells")

    # 计算加速比
    speedup = time_original / time_numba

    print("\n" + "=" * 60)
    print(f"🚀 性能提升汇总:")
    print(f"   原始版本: {time_original:.3f}秒")
    print(f"   Numba版本: {time_numba:.3f}秒")
    print(f"   加速比: {speedup:.2f}x")
    print(f"   时间节省: {(time_original - time_numba):.3f}秒 ({(1 - time_numba/time_original)*100:.1f}%)")
    print("=" * 60)

    # 验证结果一致性（允许略微不同，因为随机性）
    print("\n🔍 验证结果一致性...")
    for timestep in result_original.keys():
        size_original = len(result_original[timestep])
        size_numba = len(result_numba[timestep])
        diff_percent = abs(size_original - size_numba) / size_original * 100

        if diff_percent < 10:  # 允许10%的差异（由于随机采样）
            status = "✅"
        else:
            status = "⚠️"

        print(f"   {status} Timestep {timestep}: 原始={size_original}, Numba={size_numba}, "
              f"差异={diff_percent:.1f}%")

    return speedup

def test_different_sampling_rates():
    """测试不同采样数量下的性能"""
    print("\n" + "=" * 60)
    print("不同采样数量的性能测试")
    print("=" * 60)

    if not NUMBA_AVAILABLE:
        print("❌ Numba未安装，跳过测试")
        return

    config = get_global_config()
    grid_spec = GridSpec(size_m=100.0, cell_m=0.5, macro=True)
    grid = GridMapper(grid_spec, world_center=(0.0, 0.0))
    agent = create_test_agent(AgentType.VEHICLE)

    horizon = 8
    dt = 1.0
    sample_sizes = [500, 1000, 2000, 5000]

    # 预热
    _ = grid._multi_timestep_successor_cells_numba(agent, horizon=horizon, dt=dt, n_samples=100)

    print(f"\n{'样本数':<10} {'原始版本':<12} {'Numba版本':<12} {'加速比':<10}")
    print("-" * 50)

    for n_samples in sample_sizes:
        # 原始版本
        start = time.time()
        _ = grid._multi_timestep_successor_cells_original(agent, horizon=horizon, dt=dt, n_samples=n_samples)
        time_orig = time.time() - start

        # Numba版本
        start = time.time()
        _ = grid._multi_timestep_successor_cells_numba(agent, horizon=horizon, dt=dt, n_samples=n_samples)
        time_nb = time.time() - start

        speedup = time_orig / time_nb

        print(f"{n_samples:<10} {time_orig:<12.3f} {time_nb:<12.3f} {speedup:<10.2f}x")

if __name__ == "__main__":
    # 运行主性能测试
    speedup = benchmark_reachable_set_computation()

    # 测试不同采样数量
    test_different_sampling_rates()

    print("\n✅ 测试完成！")

    if speedup and speedup > 2:
        print(f"\n🎉 Numba优化成功！加速比达到 {speedup:.2f}x")
    elif speedup:
        print(f"\n⚠️  加速效果有限 ({speedup:.2f}x)，可能需要进一步优化")
