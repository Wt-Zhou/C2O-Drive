"""
可视化管理模块
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
from pathlib import Path

from c2o_drive.visualization.vis import make_gif
from c2o_drive.visualization.lattice_visualizer import (
    visualize_lattice_selection,
    visualize_lattice_trajectories_detailed
)
from .episode_context import EpisodeContext


class VisualizationManager:
    """可视化管理"""

    def __init__(self, ctx: EpisodeContext):
        self.ctx = ctx

    def visualize_trajectory_selection(self,
                                       trajectory_q_values: List[Dict[str, Any]],
                                       selected_trajectory_info: Optional[Dict[str, Any]]):
        """
        可视化lattice轨迹选择

        Args:
            trajectory_q_values: 所有候选轨迹的Q值信息
            selected_trajectory_info: 选中的轨迹信息
        """
        if not trajectory_q_values or selected_trajectory_info is None:
            return

        ep_dir = self.ctx.get_episode_dir()

        try:
            # 主要可视化：轨迹 + Q值柱状图
            visualize_lattice_selection(
                trajectory_q_values=trajectory_q_values,
                selected_trajectory_info=selected_trajectory_info,
                current_world_state=self.ctx.world_init,
                grid=self.ctx.grid,
                episode_idx=self.ctx.episode_id,
                output_dir=ep_dir
            )

            # 详细可视化：Q值分布箱线图
            visualize_lattice_trajectories_detailed(
                trajectory_q_values=trajectory_q_values,
                selected_trajectory_info=selected_trajectory_info,
                current_world_state=self.ctx.world_init,
                grid=self.ctx.grid,
                episode_idx=self.ctx.episode_id,
                output_dir=ep_dir
            )

        except Exception as e:
            print(f"  警告: Lattice可视化失败: {e}")

    def visualize_q_evolution(self):
        """
        可视化所有轨迹的Q值演化（从第1个episode到当前episode）
        """
        if self.ctx.q_tracker is None:
            return

        try:
            # 生成所有轨迹Q值演化图
            output_path = self.ctx.output_dir / f"all_trajectories_q_evolution_ep{self.ctx.episode_id:02d}.png"
            self.ctx.q_tracker.plot_all_trajectories_q_evolution(str(output_path))

        except Exception as e:
            print(f"  警告: Q值演化可视化失败: {e}")

    def generate_episode_gif(self, frame_paths: List[str]) -> str:
        """
        生成episode GIF

        Args:
            frame_paths: 帧图片路径列表

        Returns:
            GIF文件路径
        """
        gif_path = self.ctx.output_dir / f"episode_{self.ctx.episode_id:02d}.gif"
        make_gif(frame_paths, str(gif_path), fps=2)
        return str(gif_path)

    @staticmethod
    def generate_summary_gif(summary_frames: List[str], output_dir: Path) -> str:
        """
        生成汇总GIF

        Args:
            summary_frames: 每个episode最后一帧的路径列表
            output_dir: 输出目录

        Returns:
            GIF文件路径
        """
        summary_gif_path = output_dir / "summary.gif"
        make_gif(summary_frames, str(summary_gif_path), fps=1)
        return str(summary_gif_path)
