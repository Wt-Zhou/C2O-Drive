from __future__ import annotations
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt
import imageio
import warnings

# 配置matplotlib - 禁用中文字体警告
warnings.filterwarnings('ignore', message='Glyph.*missing from current font')
plt.rcParams['axes.unicode_minus'] = False


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
    multi_timestep_reachable_sets: Dict[int, Dict[int, List[int]]] | None = None,
    historical_data_sets: Dict[int, Dict[int, List[int]]] | None = None,
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
        reachable_sets: 当前时刻可达集 (向后兼容)
        reachable_colors: 可达集颜色 (向后兼容)
        multi_timestep_reachable_sets: 多时间步可达集 {agent_id: {timestep: [cell_indices]}}
        historical_data_sets: 历史轨迹数据 {agent_id: {timestep: [cell_indices]}}
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

    # 优先使用多时间步可达集，否则回退到单时间步可达集
    if multi_timestep_reachable_sets is not None:
        # 绘制多时间步可达集
        cell_m = grid_size_m / float(N)
        x0 = -half_size + cell_m * 0.5
        y0 = -half_size + cell_m * 0.5
        
        # 为不同时间步定义不同的颜色和样式
        timestep_colors = [
            '#FF0000',  # 红色 - t1 (最近)
            '#FF8000',  # 橙色 - t2
            '#FFFF00',  # 黄色 - t3
            '#00FF00',  # 绿色 - t4
            '#00FFFF',  # 青色 - t5
            '#0080FF',  # 蓝色 - t6
            '#8000FF',  # 紫色 - t7
            '#FF00FF',  # 洋红 - t8+
        ]
        
        timestep_styles = [
            {'alpha': 0.9, 'linewidth': 2.5, 'suffix': 't1'},
            {'alpha': 0.8, 'linewidth': 2.2, 'suffix': 't2'},
            {'alpha': 0.7, 'linewidth': 2.0, 'suffix': 't3'},
            {'alpha': 0.6, 'linewidth': 1.8, 'suffix': 't4'},
            {'alpha': 0.5, 'linewidth': 1.6, 'suffix': 't5'},
            {'alpha': 0.4, 'linewidth': 1.4, 'suffix': 't6'},
            {'alpha': 0.3, 'linewidth': 1.2, 'suffix': 't7'},
            {'alpha': 0.2, 'linewidth': 1.0, 'suffix': 't8+'},
        ]
        
        # 基础颜色为每个agent（用于区分不同agent）
        base_colors = ['cyan', 'magenta', 'lime', 'orange', 'red', 'blue']
        
        for agent_id, timestep_data in multi_timestep_reachable_sets.items():
            base_color = base_colors[(agent_id - 1) % len(base_colors)]
            
            for timestep, reach_indices in timestep_data.items():
                if not reach_indices:
                    continue
                
                # 获取对应时间步的颜色和样式
                color_idx = min(timestep - 1, len(timestep_colors) - 1)
                style_idx = min(timestep - 1, len(timestep_styles) - 1)
                
                timestep_color = timestep_colors[color_idx]
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
                    edgecolors=timestep_color,  # 使用时间步颜色而不是agent基础颜色
                    linewidths=style['linewidth'],
                    alpha=style['alpha'],
                    s=marker_size,
                    label=f'A{agent_id}-{style["suffix"]}',
                )
    
    # 绘制历史轨迹数据（黑色圆点）
    if historical_data_sets is not None:
        cell_m = grid_size_m / float(N)
        x0 = -half_size + cell_m * 0.5
        y0 = -half_size + cell_m * 0.5
        
        # 为不同时间步定义不同的透明度
        historical_timestep_styles = [
            {'alpha': 1.0, 'size': 40, 'suffix': 't1'},
            {'alpha': 0.8, 'size': 35, 'suffix': 't2'},
            {'alpha': 0.6, 'size': 30, 'suffix': 't3'},
            {'alpha': 0.4, 'size': 25, 'suffix': 't4'},
            {'alpha': 0.3, 'size': 20, 'suffix': 't5+'},
        ]
        
        for agent_id, timestep_data in historical_data_sets.items():
            for timestep, historical_cells in timestep_data.items():
                if not historical_cells:
                    continue
                
                # 获取对应时间步的样式
                style_idx = min(timestep - 1, len(historical_timestep_styles) - 1)
                style = historical_timestep_styles[style_idx]
                
                xs = []
                ys = []
                for idx in historical_cells:
                    iy = idx // N
                    ix = idx % N
                    xs.append(x0 + ix * cell_m)
                    ys.append(y0 + iy * cell_m)
                
                # 绘制黑色圆点表示历史数据
                ax.scatter(
                    xs,
                    ys,
                    marker='o',
                    facecolors='black',
                    edgecolors='white',
                    linewidths=1.0,
                    alpha=style['alpha'],
                    s=style['size'],
                    label=f'历史A{agent_id}-{style["suffix"]}',
                )
    
    elif reachable_sets is not None and len(reachable_sets) > 0:
        # 回退到原有的单时间步可达集绘制逻辑
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
            marker_size_pts2 = max(5.0, (300.0 / N) ** 2)
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
    cbar.set_label('Probability')
    
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
        try:
            img = imageio.imread(path)
            images.append(img)
        except Exception as e:
            print(f"警告: 跳过损坏的图像 {path}: {e}")
            continue
    
    if not images:
        print("错误: 没有有效的图像可以生成GIF")
        return
    
    # 统一所有图像尺寸
    from PIL import Image
    first_shape = images[0].shape
    valid_images = []

    for img in images:
        if img.shape != first_shape:
            # resize到统一尺寸
            pil_img = Image.fromarray(img)
            pil_img = pil_img.resize((first_shape[1], first_shape[0]), Image.LANCZOS)
            img = np.array(pil_img)
        valid_images.append(img)

    if not valid_images:
        print("错误: 没有有效的图像")
        return
    
    # 计算duration（毫秒）
    duration = int(1000 / fps)
    try:
        imageio.mimsave(out_path, valid_images, duration=duration, loop=0)
        print(f"GIF生成成功: {out_path} ({len(valid_images)}帧)")
    except Exception as e:
        print(f"GIF生成失败: {e}")
