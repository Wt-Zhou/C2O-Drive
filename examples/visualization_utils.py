"""
可视化工具模块（新架构版本）

封装原有的可视化功能，使其能在新架构下工作。
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np

# 导入原有的可视化函数
from carla_c2osr.visualization.vis import grid_heatmap, make_gif
from carla_c2osr.visualization.lattice_visualizer import (
    visualize_lattice_selection,
    visualize_lattice_trajectories_detailed
)
from carla_c2osr.visualization.transition_visualizer import (
    visualize_transition_distributions,
    visualize_dirichlet_distributions
)
from carla_c2osr.evaluation.q_distribution_tracker import QDistributionTracker


class EpisodeVisualizer:
    """Episode 级别的可视化管理器"""

    def __init__(
        self,
        episode_id: int,
        output_dir: Path,
        grid_mapper,
        world_state,
        horizon: int,
        verbose: bool = True,
    ):
        """初始化 Episode 可视化器

        Args:
            episode_id: Episode ID
            output_dir: 输出目录
            grid_mapper: 网格映射器
            world_state: 初始世界状态
            horizon: 规划时域
            verbose: 是否输出详细信息
        """
        self.episode_id = episode_id
        self.output_dir = Path(output_dir)
        self.grid_mapper = grid_mapper
        self.world_state = world_state
        self.horizon = horizon
        self.verbose = verbose

        # 创建 episode 子目录
        self.episode_dir = self.output_dir / f"ep_{episode_id:02d}"
        self.episode_dir.mkdir(parents=True, exist_ok=True)

        # 存储帧路径
        self.frame_paths: List[str] = []

    def visualize_trajectory_selection(
        self,
        trajectory_q_values: List[Dict[str, Any]],
        selected_trajectory_info: Dict[str, Any],
    ):
        """可视化 lattice 轨迹选择

        Args:
            trajectory_q_values: 所有候选轨迹的 Q 值信息
            selected_trajectory_info: 选中的轨迹信息
        """
        if not trajectory_q_values or selected_trajectory_info is None:
            return

        try:
            # 修正数据格式：将 LatticeTrajectory 对象替换为 waypoints 列表
            # lattice visualizer 期望 'trajectory' 字段是一个位置列表，而不是对象
            corrected_trajectory_q_values = []
            for traj_info in trajectory_q_values:
                corrected_info = traj_info.copy()
                # 使用 waypoints 列表代替 LatticeTrajectory 对象
                corrected_info['trajectory'] = traj_info['waypoints']
                corrected_trajectory_q_values.append(corrected_info)

            corrected_selected_info = selected_trajectory_info.copy()
            corrected_selected_info['trajectory'] = selected_trajectory_info['waypoints']

            # 主要可视化：轨迹 + Q 值柱状图
            visualize_lattice_selection(
                trajectory_q_values=corrected_trajectory_q_values,
                selected_trajectory_info=corrected_selected_info,
                current_world_state=self.world_state,
                grid=self.grid_mapper,
                episode_idx=self.episode_id,
                output_dir=self.episode_dir
            )

            # 详细可视化：Q 值分布箱线图
            visualize_lattice_trajectories_detailed(
                trajectory_q_values=corrected_trajectory_q_values,
                selected_trajectory_info=corrected_selected_info,
                current_world_state=self.world_state,
                grid=self.grid_mapper,
                episode_idx=self.episode_id,
                output_dir=self.episode_dir
            )

        except Exception as e:
            print(f"  警告: Lattice 可视化失败: {e}")
            import traceback
            traceback.print_exc()

    def render_timestep_heatmap(
        self,
        timestep: int,
        current_world_state,
        prob_grid: np.ndarray,
        multi_timestep_reachable_sets: Optional[Dict] = None,
        buffer_size: Optional[int] = None,
        matched_transitions: Optional[int] = None,
        total_alpha: Optional[float] = None,
    ) -> str:
        """渲染时间步热力图

        Args:
            timestep: 时间步索引
            current_world_state: 当前世界状态
            prob_grid: 概率网格
            multi_timestep_reachable_sets: 多时间步可达集
            buffer_size: 缓冲区大小
            matched_transitions: 匹配的transition数量
            total_alpha: 总alpha值

        Returns:
            帧图片路径
        """
        # 转换自车和 agents 的位置到网格坐标
        ego_pos_m = current_world_state.ego.position_m
        ego_xy = np.array(ego_pos_m)

        agents_xy = []
        for agent in current_world_state.agents:
            agents_xy.append(np.array(agent.position_m))

        # 输出路径
        frame_path = str(self.episode_dir / f"t_{timestep:02d}.png")

        # 构建标题（包含统计信息）
        title = f"Episode {self.episode_id} - Timestep {timestep}"

        # 添加统计信息到标题
        stats_parts = []
        if buffer_size is not None:
            stats_parts.append(f"Buffer: {buffer_size}")
        if matched_transitions is not None:
            stats_parts.append(f"Transitions: {matched_transitions}")
        if total_alpha is not None:
            stats_parts.append(f"Alpha: {total_alpha:.0f}")

        if stats_parts:
            title += " | " + " | ".join(stats_parts)

        # 渲染热力图
        try:
            grid_heatmap(
                prob=prob_grid,
                N=self.grid_mapper.N,
                ego_xy=ego_xy,
                agents_xy=agents_xy,
                title=title,
                out_path=frame_path,
                grid_size_m=self.grid_mapper.size_m,
                multi_timestep_reachable_sets=multi_timestep_reachable_sets,
            )

            self.frame_paths.append(frame_path)
            return frame_path

        except Exception as e:
            print(f"  警告: 时间步 {timestep} 热力图渲染失败: {e}")
            return ""

    def generate_episode_gif(self) -> str:
        """生成 episode GIF

        Returns:
            GIF 文件路径
        """
        if not self.frame_paths:
            print(f"  警告: Episode {self.episode_id} 没有帧可生成 GIF")
            return ""

        gif_path = str(self.output_dir / f"episode_{self.episode_id:02d}.gif")

        try:
            make_gif(self.frame_paths, gif_path, fps=2)
            return gif_path
        except Exception as e:
            print(f"  警告: Episode {self.episode_id} GIF 生成失败: {e}")
            return ""

    def visualize_distributions(
        self,
        q_calculator,
        world_state,
        ego_action_trajectory,
        trajectory_buffer,
        bank,
    ):
        """可视化 Transition 和 Dirichlet 分布（可选，每 N 个 episode）

        Args:
            q_calculator: Q 值计算器
            world_state: 世界状态
            ego_action_trajectory: 自车动作轨迹
            trajectory_buffer: 轨迹缓冲区
            bank: Dirichlet bank
        """
        try:
            if self.verbose:
                print(f"  生成 Transition/Dirichlet 分布可视化...")

            # 获取 transition 分布数据
            agent_transition_samples = q_calculator._build_agent_transition_distributions(
                world_state,
                ego_action_trajectory,
                trajectory_buffer,
                self.grid_mapper,
                bank,
                self.horizon
            )

            # 可视化 transition 分布
            visualize_transition_distributions(
                agent_transition_samples=agent_transition_samples,
                current_world_state=world_state,
                grid=self.grid_mapper,
                episode_idx=self.episode_id,
                output_dir=self.episode_dir
            )

            # 可视化 Dirichlet 分布
            visualize_dirichlet_distributions(
                bank=bank,
                current_world_state=world_state,
                grid=self.grid_mapper,
                episode_idx=self.episode_id,
                output_dir=self.episode_dir
            )

            if self.verbose:
                print(f"  ✓ Transition/Dirichlet 分布可视化完成")

        except Exception as e:
            print(f"  警告: 分布可视化失败: {e}")
            import traceback
            traceback.print_exc()


class GlobalVisualizer:
    """全局可视化管理器（跨 episodes）"""

    def __init__(self, output_dir: Path, q_tracker: QDistributionTracker):
        """初始化全局可视化器

        Args:
            output_dir: 输出目录
            q_tracker: Q 值分布跟踪器
        """
        self.output_dir = Path(output_dir)
        self.q_tracker = q_tracker
        self.summary_frames: List[str] = []

    def add_summary_frame(self, frame_path: str):
        """添加汇总帧（每个 episode 的最后一帧）

        Args:
            frame_path: 帧图片路径
        """
        if frame_path and Path(frame_path).exists():
            self.summary_frames.append(frame_path)

    def generate_summary_gif(self) -> str:
        """生成汇总 GIF

        Returns:
            GIF 文件路径
        """
        if not self.summary_frames:
            print(f"  警告: 没有汇总帧可生成 GIF")
            return ""

        summary_gif_path = str(self.output_dir / "summary.gif")

        try:
            make_gif(self.summary_frames, summary_gif_path, fps=1)
            return summary_gif_path
        except Exception as e:
            print(f"  警告: 汇总 GIF 生成失败: {e}")
            return ""

    def visualize_q_evolution(self, episode_id: int):
        """可视化所有轨迹的 Q 值演化（从第 1 个 episode 到当前 episode）

        Args:
            episode_id: 当前 episode ID
        """
        try:
            output_path = str(self.output_dir / f"all_trajectories_q_evolution_ep{episode_id:02d}.png")
            self.q_tracker.plot_all_trajectories_q_evolution(output_path)
        except Exception as e:
            print(f"  警告: Q 值演化可视化失败: {e}")

    def generate_final_plots(self):
        """生成最终的全局统计图

        生成：
        - Q 分布演化图
        - 碰撞率演化图
        """
        try:
            # Q 分布演化图
            q_dist_path = str(self.output_dir / "q_distribution_evolution.png")
            self.q_tracker.plot_q_distribution_evolution(q_dist_path)

            # 碰撞率演化图
            collision_rate_path = str(self.output_dir / "collision_rate_evolution.png")
            self.q_tracker.plot_collision_rate_evolution(collision_rate_path)

            print(f"\n  ✓ 全局统计图生成完成:")
            print(f"    - {q_dist_path}")
            print(f"    - {collision_rate_path}")

        except Exception as e:
            print(f"  警告: 全局统计图生成失败: {e}")
            import traceback
            traceback.print_exc()


def create_visualization_pipeline(
    output_dir: Path,
    enable_visualization: bool = True,
) -> Tuple[QDistributionTracker, GlobalVisualizer]:
    """创建可视化管道

    Args:
        output_dir: 输出目录
        enable_visualization: 是否启用可视化

    Returns:
        (q_tracker, global_visualizer)
    """
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建 Q 值跟踪器
    q_tracker = QDistributionTracker()

    # 创建全局可视化器
    global_visualizer = None
    if enable_visualization:
        global_visualizer = GlobalVisualizer(output_dir, q_tracker)

    return q_tracker, global_visualizer
