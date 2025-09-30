"""
增强的可视化函数，支持显示agent历史轨迹数据
"""

from __future__ import annotations
from typing import List, Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
import imageio


def grid_heatmap_with_history(
    prob: np.ndarray,
    N: int,
    ego_xy: np.ndarray,
    agents_xy: List[np.ndarray],
    title: str,
    out_path: str,
    grid_size_m: float = 20.0,
    reachable_sets: List[List[int]] | None = None,
    reachable_colors: List[str] | None = None,
    multi_timestep_reachable_sets: Dict[int, Dict[int, List[int]]] | None = None,
    agent_historical_data: Dict[int, List[int]] | None = None,  # 新增：agent历史轨迹数据
    grid_mapper = None,  # 新增：网格映射器用于坐标转换
) -> None:
    """渲染概率热力图并叠加轨迹点，包含agent历史轨迹数据。
    
    Args:
        prob: K维后验均值，将reshape为(N,N)
        N: 网格边长
        ego_xy: 自车位置 (x,y) 网格坐标
        agents_xy: 环境智能体位置列表，每个为(x,y) 网格坐标
        title: 图表标题
        out_path: 输出PNG路径
        grid_size_m: 网格物理尺寸
        reachable_sets: 当前时刻可达集 (向后兼容)
        reachable_colors: 可达集颜色 (向后兼容)
        multi_timestep_reachable_sets: 多时间步可达集 {agent_id: {timestep: [cell_indices]}}
        agent_historical_data: agent历史轨迹数据 {agent_id: [cell_indices]}
        grid_mapper: 网格映射器用于坐标转换
    """
    assert prob.shape[0] == N * N, f"prob shape {prob.shape} != N*N={N*N}"
    
    # Reshape为2D网格
    prob_grid = prob.reshape(N, N)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 坐标轴转换为米制（固定网格尺寸）
    half_size = grid_size_m / 2
    extent = [-half_size, half_size, -half_size, half_size]
    # 单次绘制热力图：使用 'gray_r'（0=白，1=黑），高概率更深
    im = ax.imshow(prob_grid, origin='lower', cmap='gray_r', vmin=0, vmax=1, extent=extent)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect('equal', adjustable='box')
    
    # 叠加自车位置（大黑点）
    ax.plot(ego_xy[0], ego_xy[1], 'ko', markersize=12, markeredgewidth=2, 
            markeredgecolor='white', label='自车')
    
    # 叠加环境智能体位置（小彩色点）
    colors = ['blue', 'red', 'green', 'orange']
    for i, agent_xy in enumerate(agents_xy):
        color = colors[i % len(colors)]
        ax.plot(agent_xy[0], agent_xy[1], 'o', color=color, markersize=8, 
                markeredgewidth=1, markeredgecolor='white', label=f'Agent{i+1}')

    # 显示agent历史轨迹数据（黑色小点）
    if agent_historical_data is not None and grid_mapper is not None:
        for agent_id, historical_cells in agent_historical_data.items():
            if historical_cells:
                # 将网格单元索引转换为世界坐标
                historical_positions = []
                for cell_idx in historical_cells:
                    # 将单元索引转换为网格坐标
                    grid_xy = grid_mapper.index_to_xy_center(cell_idx)
                    # 转换为世界坐标
                    world_xy = grid_mapper.grid_to_world(np.array(grid_xy))
                    historical_positions.append(world_xy)
                
                if historical_positions:
                    # 提取x和y坐标
                    hist_x = [pos[0] for pos in historical_positions]
                    hist_y = [pos[1] for pos in historical_positions]
                    
                    # 用黑色小点显示历史轨迹
                    ax.plot(hist_x, hist_y, 'ko', markersize=3, alpha=0.7, 
                           label=f'Agent{agent_id}历史轨迹')

    # 优先使用多时间步可达集，否则回退到单时间步可达集
    if multi_timestep_reachable_sets is not None:
        # 绘制多时间步可达集
        cell_m = grid_size_m / float(N)
        x0 = -half_size + cell_m * 0.5
        y0 = -half_size + cell_m * 0.5
        
        # 为不同时间步定义不同的透明度和线型
        timestep_styles = [
            {'alpha': 1.0, 'linestyle': '-', 'linewidth': 2.0, 'suffix': 't1'},
            {'alpha': 0.8, 'linestyle': '--', 'linewidth': 1.8, 'suffix': 't2'},
            {'alpha': 0.6, 'linestyle': '-.', 'linewidth': 1.5, 'suffix': 't3'},
            {'alpha': 0.4, 'linestyle': ':', 'linewidth': 1.2, 'suffix': 't4'},
            {'alpha': 0.3, 'linestyle': '-', 'linewidth': 1.0, 'suffix': 't5+'},
        ]
        
        # 基础颜色为每个agent
        base_colors = ['cyan', 'magenta', 'lime', 'orange', 'red', 'blue']
        
        for agent_id, timestep_data in multi_timestep_reachable_sets.items():
            base_color = base_colors[(agent_id - 1) % len(base_colors)]
            
            for timestep, reach_indices in timestep_data.items():
                if not reach_indices:
                    continue
                
                # 获取对应时间步的样式
                style_idx = min(timestep - 1, len(timestep_styles) - 1)
                style = timestep_styles[style_idx]
                
                xs = []
                ys = []
                for idx in reach_indices:
                    iy = idx // N
                    ix = idx % N
                    xs.append(x0 + ix * cell_m)
                    ys.append(y0 + iy * cell_m)
                
                # marker大小随时间步递减
                base_size = max(5.0, (300.0 / N) ** 2)
                marker_size = base_size * (1.0 - 0.1 * (timestep - 1))
                
                ax.scatter(
                    xs,
                    ys,
                    marker='s',
                    facecolors='none',
                    edgecolors=base_color,
                    linewidths=style['linewidth'],
                    linestyles=style['linestyle'],
                    alpha=style['alpha'],
                    s=marker_size,
                    label=f'A{agent_id}-{style["suffix"]}',
                )
    
    elif reachable_sets is not None and len(reachable_sets) > 0:
        # 回退到原有的单时间步可达集绘制逻辑
        cell_m = grid_size_m / float(N)
        x0 = -half_size + cell_m * 0.5
        y0 = -half_size + cell_m * 0.5
        
        if reachable_colors is None:
            reachable_colors = ['cyan', 'magenta', 'lime', 'orange']
        
        for i, reach_indices in enumerate(reachable_sets):
            if not reach_indices:
                continue
            
            color = reachable_colors[i % len(reachable_colors)]
            xs = []
            ys = []
            for idx in reach_indices:
                iy = idx // N
                ix = idx % N
                xs.append(x0 + ix * cell_m)
                ys.append(y0 + iy * cell_m)
            
            ax.scatter(xs, ys, marker='s', facecolors='none', edgecolors=color,
                      linewidths=2, s=max(5.0, (300.0 / N) ** 2), label=f'可达集{i+1}')
    
    # 设置图例和标题
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8)
    ax.set_title(title, fontsize=12, pad=20)
    ax.set_xlabel('X (m)', fontsize=10)
    ax.set_ylabel('Y (m)', fontsize=10)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('概率', rotation=270, labelpad=15, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def make_gif(frame_paths: List[str], out_path: str, fps: int = 2) -> None:
    """从帧图片创建GIF动画。
    
    Args:
        frame_paths: 帧图片路径列表
        out_path: 输出GIF路径
        fps: 帧率
    """
    assert len(frame_paths) > 0, "Frame paths cannot be empty"
    
    images = []
    for path in frame_paths:
        images.append(imageio.imread(path))
    
    # 计算duration（毫秒）
    duration = int(1000 / fps)
    imageio.mimsave(out_path, images, duration=duration, loop=0)
