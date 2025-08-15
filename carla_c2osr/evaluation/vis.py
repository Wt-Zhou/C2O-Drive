from __future__ import annotations
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import imageio


def grid_heatmap(
    prob: np.ndarray,
    N: int,
    ego_xy: np.ndarray,
    agents_xy: List[np.ndarray],
    title: str,
    out_path: str,
    grid_size_m: float = 20.0,
    reachable_sets: List[List[int]] | None = None,
    reachable_colors: List[str] | None = None,
) -> None:
    """渲染概率热力图并叠加轨迹点。
    
    Args:
        prob: K维后验均值，将reshape为(N,N)
        N: 网格边长
        ego_xy: 自车位置 (x,y) 网格坐标
        agents_xy: 环境智能体位置列表，每个为(x,y) 网格坐标
        title: 图表标题
        out_path: 输出PNG路径
        grid_size_m: 网格物理尺寸
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

    # 可选：叠加当前时刻可达集（方框轮廓，不遮挡热力图）
    if reachable_sets is not None and len(reachable_sets) > 0:
        cell_m = grid_size_m / float(N)
        x0 = -half_size + cell_m * 0.5
        y0 = -half_size + cell_m * 0.5
        # 颜色为每个agent提供不同轮廓颜色
        if reachable_colors is None:
            reachable_colors = ['cyan', 'magenta', 'lime', 'orange']
        for ridx, reach_indices in enumerate(reachable_sets):
            if not reach_indices:
                continue
            xs = []
            ys = []
            for idx in reach_indices:
                iy = idx // N
                ix = idx % N
                xs.append(x0 + ix * cell_m)
                ys.append(y0 + iy * cell_m)
            # 使用方形marker，只有边框不填充，避免与热力图颜色冲突
            edge_color = reachable_colors[ridx % len(reachable_colors)]
            # marker大小（以点为单位），与格子尺寸近似匹配
            marker_size_pts2 = max(10.0, (300.0 / N) ** 2)
            ax.scatter(
                xs,
                ys,
                marker='s',
                facecolors='none',
                edgecolors=edge_color,
                linewidths=1.0,
                s=marker_size_pts2,
                label=f'Reachable A{ridx+1}',
            )
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('占据概率')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def make_gif(frame_paths: List[str], out_path: str, fps: int = 2) -> None:
    """用imageio合成GIF动画。
    
    Args:
        frame_paths: PNG帧文件路径列表
        out_path: 输出GIF路径
        fps: 帧率
    """
    assert len(frame_paths) > 0, "Frame paths cannot be empty"
    
    images = []
    for path in frame_paths:
        images.append(imageio.imread(path))
    
    imageio.mimsave(out_path, images, fps=fps, loop=0)
