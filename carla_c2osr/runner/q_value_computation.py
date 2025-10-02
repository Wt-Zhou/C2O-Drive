"""
Q值计算辅助模块

从 run_episode 中提取的Q值计算逻辑,职责单一,易于测试。
"""

from typing import Dict, List, Tuple, Optional
import numpy as np

from carla_c2osr.env.types import WorldState
from carla_c2osr.evaluation.q_value_calculator import QValueCalculator, QValueConfig
from carla_c2osr.config import get_global_config
from carla_c2osr.runner.episode_context import EpisodeContext


def compute_q_value_for_episode(
    ctx: EpisodeContext,
    world_state: WorldState,
    timestep: int
) -> Optional[Dict]:
    """计算episode的Q值(仅在timestep=0时执行)

    Args:
        ctx: Episode运行上下文
        world_state: 当前世界状态
        timestep: 当前时间步

    Returns:
        Q值计算结果字典,如果不需要计算则返回None
        {
            'q_values': List[float],
            'avg_q_value': float,
            'detailed_info': Dict
        }
    """
    # 只在第一个时间步计算Q值
    if timestep != 0:
        return None

    # 构造自车未来动作轨迹
    ego_action_trajectory = _extract_ego_action_trajectory(ctx, timestep)

    # 获取verbose级别
    from carla_c2osr.config import get_global_config
    verbose = get_global_config().visualization.verbose_level

    if verbose >= 2:
        print(f"    开始Q值计算: 自车动作序列长度={len(ego_action_trajectory)}")

    try:
        # 创建Q值配置和计算器
        q_config = QValueConfig.from_global_config()
        global_config = get_global_config()
        reward_config = global_config.reward

        q_calculator = QValueCalculator(q_config, reward_config)

        # 计算Q值
        q_values, detailed_info = q_calculator.compute_q_value(
            current_world_state=world_state,
            ego_action_trajectory=ego_action_trajectory,
            trajectory_buffer=ctx.trajectory_buffer,
            grid=ctx.grid,
            bank=ctx.bank,
            rng=ctx.rng
        )

        # 计算统计信息
        avg_q_value = np.mean(q_values)

        # 打印结果
        _print_q_value_results(q_values, avg_q_value, detailed_info)

        # 记录到tracker
        if ctx.q_tracker is not None:
            _record_q_value_to_tracker(ctx, avg_q_value, q_values, detailed_info)

        # 生成可视化(每5个episode)
        if ctx.should_visualize():
            _generate_q_value_visualizations(
                ctx, world_state, ego_action_trajectory,
                q_calculator, timestep
            )

        return {
            'q_values': q_values,
            'avg_q_value': avg_q_value,
            'detailed_info': detailed_info
        }

    except Exception as e:
        print(f"    Q值计算失败: {e}")
        _record_q_value_failure(ctx, e)
        return None


def _extract_ego_action_trajectory(ctx: EpisodeContext, timestep: int) -> List[Tuple[float, float]]:
    """提取自车动作轨迹"""
    ego_action_trajectory = []
    for action_t in range(timestep, min(timestep + ctx.horizon, len(ctx.ego_trajectory))):
        ego_action_trajectory.append(tuple(ctx.ego_trajectory[action_t]))
    return ego_action_trajectory


def _print_q_value_results(q_values: List[float], avg_q_value: float, detailed_info: Dict):
    """打印Q值计算结果"""
    from carla_c2osr.config import get_global_config
    verbose = get_global_config().visualization.verbose_level

    # 始终打印摘要(单行)
    min_q = min(q_values) if q_values else 0.0
    collision_rate = detailed_info['reward_breakdown']['collision_rate']
    print(f"    Q: Avg={avg_q_value:.2f}, Min={min_q:.2f}, Collision={collision_rate:.1%}")

    # 详细信息(仅verbose >= 2)
    if verbose >= 2:
        print(f"    所有Q值: {[f'{q:.2f}' for q in q_values]}")
        print(f"    Q值标准差: {detailed_info['reward_breakdown']['q_value_std']:.2f}")

        # 打印每个智能体的信息
        for agent_id, agent_info in detailed_info.get('agent_info', {}).items():
            reachable_total = agent_info.get('reachable_cells_total', 0)
            historical_total = agent_info.get('historical_data_count', 0)
            print(f"    Agent {agent_id}: 可达集(总计)={reachable_total}, 历史数据={historical_total}")


def _record_q_value_to_tracker(
    ctx: EpisodeContext,
    avg_q_value: float,
    q_values: List[float],
    detailed_info: Dict
):
    """记录Q值到tracker"""
    q_distribution = detailed_info['reward_breakdown']['all_q_values']
    collision_rate = detailed_info['reward_breakdown']['collision_rate']

    ctx.q_tracker.add_episode_data(
        episode_id=ctx.episode_id,
        q_value=avg_q_value,
        q_distribution=q_distribution,
        collision_rate=collision_rate,
        detailed_info=detailed_info
    )


def _record_q_value_failure(ctx: EpisodeContext, error: Exception):
    """记录Q值计算失败"""
    if ctx.q_tracker is None:
        return

    config = get_global_config()
    n_samples = config.sampling.q_value_samples

    ctx.q_tracker.add_episode_data(
        episode_id=ctx.episode_id,
        q_value=0.0,
        q_distribution=[0.0] * n_samples,
        collision_rate=0.0,
        detailed_info={'error': str(error)}
    )


def _generate_q_value_visualizations(
    ctx: EpisodeContext,
    world_state: WorldState,
    ego_action_trajectory: List[Tuple[float, float]],
    q_calculator: QValueCalculator,
    timestep: int
):
    """生成transition和Dirichlet分布可视化"""
    try:
        from carla_c2osr.visualization.transition_visualizer import (
            visualize_transition_distributions,
            visualize_dirichlet_distributions
        )

        from carla_c2osr.config import get_global_config
        verbose = get_global_config().visualization.verbose_level
        if verbose >= 1:
            print(f"  🎨 生成transition分布和Dirichlet分布可视化...")

        # 获取transition分布数据
        agent_transition_samples = q_calculator._build_agent_transition_distributions(
            world_state, ego_action_trajectory, ctx.trajectory_buffer,
            ctx.grid, ctx.bank, ctx.horizon
        )

        # 可视化transition分布
        visualize_transition_distributions(
            agent_transition_samples=agent_transition_samples,
            current_world_state=world_state,
            grid=ctx.grid,
            episode_idx=ctx.episode_id,
            output_dir=ctx.output_dir
        )

        # 可视化Dirichlet分布
        visualize_dirichlet_distributions(
            bank=ctx.bank,
            current_world_state=world_state,
            grid=ctx.grid,
            episode_idx=ctx.episode_id,
            output_dir=ctx.output_dir
        )

    except Exception as e:
        print(f"  ⚠️ 可视化生成失败: {e}")
        import traceback
        traceback.print_exc()
