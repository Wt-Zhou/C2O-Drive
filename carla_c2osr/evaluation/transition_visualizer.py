from __future__ import annotations
from typing import List, Dict, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
import os
from carla_c2osr.env.types import WorldState
from carla_c2osr.agents.c2osr.grid import GridMapper


def visualize_transition_distributions(
    agent_transition_samples: Dict[int, Dict[str, Any]],
    current_world_state: WorldState,
    grid: GridMapper,
    episode_idx: int,
    output_dir: str,
    grid_size_m: float = 20.0
) -> None:
    """可视化每个agent的transition分布
    
    Args:
        agent_transition_samples: 从_build_agent_transition_distributions返回的数据
        current_world_state: 当前世界状态
        grid: 网格映射器
        episode_idx: episode索引
        output_dir: 输出目录
        grid_size_m: 网格物理尺寸
    """
    # 创建输出目录
    episode_dir = os.path.join(output_dir, f"ep_{episode_idx:02d}")
    transition_dir = os.path.join(episode_dir, "transition_distributions")
    os.makedirs(transition_dir, exist_ok=True)
    
    N = grid.N
    
    for agent_id, transition_info in agent_transition_samples.items():
        distributions = transition_info['distributions']
        reachable_sets = transition_info['reachable_sets']
        
        # 获取agent位置
        if agent_id <= len(current_world_state.agents):
            agent = current_world_state.agents[agent_id - 1]
            agent_world_pos = agent.position_m
            agent_cell_idx = grid.world_to_cell(agent_world_pos)
            # 将cell索引转换为网格坐标
            agent_grid_pos = np.array([agent_cell_idx % grid.N, agent_cell_idx // grid.N])
        else:
            continue
        
        # 为每个时间步创建可视化
        for timestep in sorted(distributions.keys()):
            if timestep not in reachable_sets:
                continue
                
            reachable_cells = reachable_sets[timestep]
            timestep_distributions = distributions[timestep]  # List[np.ndarray]
            
            # 使用网格布局显示多个样本
            _create_grid_transition_visualization(
                timestep_distributions=timestep_distributions,
                reachable_cells=reachable_cells,
                agent_grid_pos=agent_grid_pos,
                ego_grid_pos=np.array([grid.world_to_cell(current_world_state.ego.position_m) % grid.N, 
                                      grid.world_to_cell(current_world_state.ego.position_m) // grid.N]),
                N=N,
                timestep=timestep,
                agent_id=agent_id,
                output_path=os.path.join(transition_dir, f"agent_{agent_id}_timestep_{timestep}_grid.png"),
                grid_size_m=grid_size_m,
                max_samples=25  # 最多显示25个样本
            )


def _create_single_transition_visualization(
    sample_distribution: np.ndarray,
    reachable_cells: List[int],
    agent_grid_pos: np.ndarray,
    ego_grid_pos: np.ndarray,
    N: int,
    timestep: int,
    agent_id: int,
    sample_idx: int,
    output_path: str,
    grid_size_m: float = 20.0
) -> None:
    """为单个transition样本创建可视化"""
    
    if len(sample_distribution) == 0:
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # 坐标转换
    half_size = grid_size_m / 2
    extent = [-half_size, half_size, -half_size, half_size]
    cell_m = grid_size_m / float(N)
    x0 = -half_size + cell_m * 0.5
    y0 = -half_size + cell_m * 0.5
    
    # 创建概率网格
    prob_grid = np.zeros(N * N)
    for i, cell_idx in enumerate(reachable_cells):
        if i < len(sample_distribution):
            prob_grid[cell_idx] = sample_distribution[i]
    
    prob_grid_2d = prob_grid.reshape(N, N)
    
    # 使用hot颜色映射
    im = ax.imshow(prob_grid_2d, origin='lower', cmap='hot', 
                   vmin=0, vmax=np.max(sample_distribution), extent=extent)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect('equal', adjustable='box')
    
    # 标记agent和ego位置
    agent_x = x0 + agent_grid_pos[0] * cell_m
    agent_y = y0 + agent_grid_pos[1] * cell_m
    ax.plot(agent_x, agent_y, 'r*', markersize=15, markeredgewidth=2, 
           markeredgecolor='white', label='Agent', zorder=10)
    
    ego_x = x0 + ego_grid_pos[0] * cell_m
    ego_y = y0 + ego_grid_pos[1] * cell_m
    ax.plot(ego_x, ego_y, 'w^', markersize=12, markeredgewidth=2, 
           markeredgecolor='black', label='Ego', zorder=10)
    
    ax.set_title(f'Transition Probability - Sample {sample_idx}\nMax: {np.max(sample_distribution):.3f}, Mean: {np.mean(sample_distribution):.3f}')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend()
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Probability')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def _create_timestep_transition_visualization(
    timestep_distributions: List[np.ndarray],
    reachable_cells: List[int],
    agent_grid_pos: np.ndarray,
    ego_grid_pos: np.ndarray,
    N: int,
    timestep: int,
    agent_id: int,
    output_path: str,
    grid_size_m: float = 20.0
) -> None:
    """为单个时间步创建transition分布可视化 - 显示平均概率分布热力图"""
    
    # 计算所有样本的平均概率分布
    n_samples = len(timestep_distributions)
    if n_samples == 0:
        return
    
    # 计算平均概率分布
    avg_prob_vector = np.zeros(len(reachable_cells))
    for prob_vector in timestep_distributions:
        for i in range(min(len(prob_vector), len(avg_prob_vector))):
            avg_prob_vector[i] += prob_vector[i]
    avg_prob_vector /= n_samples
    
    # 计算概率分布的统计信息
    all_probs = np.concatenate(timestep_distributions)
    prob_stats = {
        'mean': np.mean(all_probs),
        'std': np.std(all_probs),
        'max': np.max(all_probs),
        'min': np.min(all_probs)
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 坐标转换
    half_size = grid_size_m / 2
    extent = [-half_size, half_size, -half_size, half_size]
    cell_m = grid_size_m / float(N)
    x0 = -half_size + cell_m * 0.5
    y0 = -half_size + cell_m * 0.5
    
    # 左图：平均概率分布热力图
    avg_prob_grid = np.zeros(N * N)
    for i, cell_idx in enumerate(reachable_cells):
        if i < len(avg_prob_vector):
            avg_prob_grid[cell_idx] = avg_prob_vector[i]
    
    avg_prob_grid_2d = avg_prob_grid.reshape(N, N)
    
    # 使用更好的颜色映射，让概率分布更清晰
    im1 = ax1.imshow(avg_prob_grid_2d, origin='lower', cmap='hot', 
                    vmin=0, vmax=np.max(avg_prob_vector), extent=extent)
    ax1.set_xlim(extent[0], extent[1])
    ax1.set_ylim(extent[2], extent[3])
    ax1.set_aspect('equal', adjustable='box')
    
    # 不显示可达集边界，只显示概率分布
    
    # 标记agent和ego位置（使用更明显的标记）
    agent_x = x0 + agent_grid_pos[0] * cell_m
    agent_y = y0 + agent_grid_pos[1] * cell_m
    ax1.plot(agent_x, agent_y, 'r*', markersize=15, markeredgewidth=2, 
            markeredgecolor='white', label='Agent', zorder=10)
    
    ego_x = x0 + ego_grid_pos[0] * cell_m
    ego_y = y0 + ego_grid_pos[1] * cell_m
    ax1.plot(ego_x, ego_y, 'w^', markersize=12, markeredgewidth=2, 
            markeredgecolor='black', label='Ego', zorder=10)
    
    ax1.set_title(f'Average Transition Probability\nMax: {np.max(avg_prob_vector):.3f}, Mean: {np.mean(avg_prob_vector):.3f}')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.legend()
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label='Probability')
    
    # 右图：概率分布统计直方图
    ax2.hist(all_probs, bins=min(30, len(all_probs)), alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(prob_stats['mean'], color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {prob_stats["mean"]:.3f}')
    ax2.axvline(np.median(all_probs), color='green', linestyle='--', linewidth=2, 
               label=f'Median: {np.median(all_probs):.3f}')
    ax2.set_xlabel('Transition Probability')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Transition Probability Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'Agent {agent_id} - Timestep {timestep} Transition Distributions\n'
                f'({n_samples} samples, {len(reachable_cells)} reachable cells)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  📊 Transition分布可视化已保存: {output_path}")


def visualize_dirichlet_distributions(
    bank,
    current_world_state: WorldState,
    grid: GridMapper,
    episode_idx: int,
    output_dir: str,
    grid_size_m: float = 20.0
) -> None:
    """可视化Dirichlet分布的alpha值
    
    Args:
        bank: OptimizedMultiTimestepSpatialDirichletBank实例
        current_world_state: 当前世界状态
        grid: 网格映射器
        episode_idx: episode索引
        output_dir: 输出目录
        grid_size_m: 网格物理尺寸
    """
    # 创建输出目录
    episode_dir = os.path.join(output_dir, f"ep_{episode_idx:02d}")
    dirichlet_dir = os.path.join(episode_dir, "dirichlet_distributions")
    os.makedirs(dirichlet_dir, exist_ok=True)
    
    N = grid.N
    
    # 获取所有agent的alpha值
    for agent_id in range(1, len(current_world_state.agents) + 1):
        try:
            reachable_sets = bank.get_reachable_sets(agent_id)
            if not reachable_sets:
                continue
                
            # 获取agent位置
            agent = current_world_state.agents[agent_id - 1]
            agent_world_pos = agent.position_m
            agent_cell_idx = grid.world_to_cell(agent_world_pos)
            # 将cell索引转换为网格坐标
            agent_grid_pos = np.array([agent_cell_idx % grid.N, agent_cell_idx // grid.N])
            
            # 为每个时间步创建可视化
            for timestep in sorted(reachable_sets.keys()):
                reachable_cells = reachable_sets[timestep]
                alpha_vector = bank.get_agent_alpha(agent_id, timestep)
                
                if alpha_vector is None or len(alpha_vector) == 0:
                    continue
                
                _create_timestep_dirichlet_visualization(
                    alpha_vector=alpha_vector,
                    reachable_cells=reachable_cells,
                    agent_grid_pos=agent_grid_pos,
                    ego_grid_pos=np.array([grid.world_to_cell(current_world_state.ego.position_m) % grid.N, 
                                      grid.world_to_cell(current_world_state.ego.position_m) // grid.N]),
                    N=N,
                    timestep=timestep,
                    agent_id=agent_id,
                    output_path=os.path.join(dirichlet_dir, f"agent_{agent_id}_timestep_{timestep}_alpha.png"),
                    grid_size_m=grid_size_m
                )
                
        except Exception as e:
            print(f"  ⚠️  Agent {agent_id} Dirichlet可视化失败: {e}")
            continue


def _create_timestep_dirichlet_visualization(
    alpha_vector: np.ndarray,
    reachable_cells: List[int],
    agent_grid_pos: np.ndarray,
    ego_grid_pos: np.ndarray,
    N: int,
    timestep: int,
    agent_id: int,
    output_path: str,
    grid_size_m: float = 20.0
) -> None:
    """为单个时间步创建Dirichlet alpha值可视化"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 坐标转换
    half_size = grid_size_m / 2
    extent = [-half_size, half_size, -half_size, half_size]
    cell_m = grid_size_m / float(N)
    x0 = -half_size + cell_m * 0.5
    y0 = -half_size + cell_m * 0.5
    
    # 左图：Alpha值热力图
    alpha_grid = np.zeros(N * N)
    for i, cell_idx in enumerate(reachable_cells):
        if i < len(alpha_vector):
            alpha_grid[cell_idx] = alpha_vector[i]
    
    alpha_grid_2d = alpha_grid.reshape(N, N)
    
    im1 = ax1.imshow(alpha_grid_2d, origin='lower', cmap='plasma', 
                    vmin=0, vmax=np.max(alpha_vector), extent=extent)
    ax1.set_xlim(extent[0], extent[1])
    ax1.set_ylim(extent[2], extent[3])
    ax1.set_aspect('equal', adjustable='box')
    
    # 标记agent和ego位置
    agent_x = x0 + agent_grid_pos[0] * cell_m
    agent_y = y0 + agent_grid_pos[1] * cell_m
    ax1.plot(agent_x, agent_y, 'ro', markersize=10, markeredgewidth=2, 
            markeredgecolor='white', label='Agent')
    
    ego_x = x0 + ego_grid_pos[0] * cell_m
    ego_y = y0 + ego_grid_pos[1] * cell_m
    ax1.plot(ego_x, ego_y, 'ko', markersize=10, markeredgewidth=2, 
            markeredgecolor='white', label='Ego')
    
    # 不显示可达集边界，只显示alpha值分布
    
    ax1.set_title(f'Alpha Values\nMax: {np.max(alpha_vector):.2f}, Sum: {np.sum(alpha_vector):.2f}')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.legend()
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label='Alpha Value')
    
    # 右图：Alpha值分布直方图
    ax2.hist(alpha_vector, bins=min(20, len(alpha_vector)), alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(np.mean(alpha_vector), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(alpha_vector):.2f}')
    ax2.axvline(np.median(alpha_vector), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(alpha_vector):.2f}')
    ax2.set_xlabel('Alpha Value')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Alpha Value Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'Agent {agent_id} - Timestep {timestep} Dirichlet Alpha Distribution\n'
                f'({len(reachable_cells)} reachable cells)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  📊 Dirichlet分布可视化已保存: {output_path}")


def _create_grid_transition_visualization(
    timestep_distributions: List[np.ndarray],
    reachable_cells: List[int],
    agent_grid_pos: np.ndarray,
    ego_grid_pos: np.ndarray,
    N: int,
    timestep: int,
    agent_id: int,
    output_path: str,
    grid_size_m: float = 20.0,
    max_samples: int = 25  # 最多显示25个样本
) -> None:
    """创建网格布局显示多个transition样本"""
    
    n_samples = min(len(timestep_distributions), max_samples)
    if n_samples == 0:
        return
    
    # 计算网格布局
    n_cols = min(5, n_samples)  # 最多5列
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    if n_samples == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    else:
        axes = axes.flatten()
    
    # 坐标转换
    half_size = grid_size_m / 2
    extent = [-half_size, half_size, -half_size, half_size]
    cell_m = grid_size_m / float(N)
    x0 = -half_size + cell_m * 0.5
    y0 = -half_size + cell_m * 0.5
    
    for sample_idx in range(n_samples):
        sample_distribution = timestep_distributions[sample_idx]
        ax = axes[sample_idx]
        
        # 创建概率网格
        prob_grid = np.zeros(N * N)
        for i, cell_idx in enumerate(reachable_cells):
            if i < len(sample_distribution):
                prob_grid[cell_idx] = sample_distribution[i]
        
        prob_grid_2d = prob_grid.reshape(N, N)
        
        # 绘制热力图
        im = ax.imshow(prob_grid_2d, origin='lower', cmap='hot', 
                      vmin=0, vmax=np.max(sample_distribution), extent=extent)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect('equal', adjustable='box')
        
        # 标记agent和ego位置
        agent_x = x0 + agent_grid_pos[0] * cell_m
        agent_y = y0 + agent_grid_pos[1] * cell_m
        ax.plot(agent_x, agent_y, 'r*', markersize=8, markeredgewidth=1, 
               markeredgecolor='white', zorder=10)
        
        ego_x = x0 + ego_grid_pos[0] * cell_m
        ego_y = y0 + ego_grid_pos[1] * cell_m
        ax.plot(ego_x, ego_y, 'w^', markersize=6, markeredgewidth=1, 
               markeredgecolor='black', zorder=10)
        
        ax.set_title(f'Sample {sample_idx}\nMax: {np.max(sample_distribution):.2f}')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
    
    # 隐藏多余的子图
    for sample_idx in range(n_samples, n_rows * n_cols):
        axes[sample_idx].set_visible(False)
    
    plt.suptitle(f'Agent {agent_id} - Timestep {timestep} Transition Samples\n'
                f'({n_samples} samples, {len(reachable_cells)} reachable cells)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  📊 网格Transition分布可视化已保存: {output_path}")

