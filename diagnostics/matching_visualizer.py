"""可视化匹配统计的工具"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from c2o_drive.algorithms.c2osr.trajectory_buffer import HighPerformanceTrajectoryBuffer
from c2o_drive.config.global_config import GlobalConfig

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("警告: matplotlib未安装，无法生成图表")


class MatchingVisualizer:
    """匹配统计可视化工具"""

    def __init__(self, save_dir: str = "./diagnostics/results"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def plot_timestep_success_rate(
        self,
        timestep_stats: Dict[int, Dict[str, Any]],
        save_name: str = "matching_success_rate.png"
    ):
        """绘制每个timestep的匹配成功率曲线

        Args:
            timestep_stats: {timestep: {"success_count": int, "total_count": int, ...}}
            save_name: 保存文件名
        """
        if not MATPLOTLIB_AVAILABLE:
            print("跳过可视化: matplotlib未安装")
            return

        timesteps = sorted(timestep_stats.keys())
        success_rates = []
        avg_cells = []

        for t in timesteps:
            stats = timestep_stats[t]
            rate = stats["success_count"] / stats["total_count"] * 100 if stats["total_count"] > 0 else 0
            cells = (stats["total_matched_cells"] / stats["success_count"]
                    if stats["success_count"] > 0 else 0)

            success_rates.append(rate)
            avg_cells.append(cells)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # 子图1: 成功率
        ax1.plot(timesteps, success_rates, 'b-o', linewidth=2, markersize=6, label='匹配成功率')
        ax1.axhline(y=50, color='orange', linestyle='--', linewidth=1.5, label='50%基准线')
        ax1.axhline(y=20, color='red', linestyle='--', linewidth=1.5, label='20%警戒线')

        ax1.set_xlabel('Timestep', fontsize=12)
        ax1.set_ylabel('匹配成功率 (%)', fontsize=12)
        ax1.set_title('每个Timestep的历史数据匹配成功率', fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 105)

        # 标记问题区域
        for i, (t, rate) in enumerate(zip(timesteps, success_rates)):
            if rate < 20:
                ax1.text(t, rate + 3, '❌', ha='center', fontsize=12, color='red')
            elif rate < 50:
                ax1.text(t, rate + 3, '⚠️', ha='center', fontsize=10, color='orange')

        # 子图2: 平均匹配cells数
        ax2.plot(timesteps, avg_cells, 'g-s', linewidth=2, markersize=6, label='平均匹配Cells数')

        ax2.set_xlabel('Timestep', fontsize=12)
        ax2.set_ylabel('平均匹配Cells数', fontsize=12)
        ax2.set_title('每个Timestep匹配到的平均历史数据量', fontsize=14, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)

        # 标记数据稀少区域
        for i, (t, cells) in enumerate(zip(timesteps, avg_cells)):
            if cells < 5:
                ax2.text(t, cells + 1, '数据稀少', ha='center', fontsize=8, color='red')

        plt.tight_layout()

        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"可视化已保存: {save_path}")
        plt.close(fig)

    def plot_data_availability(
        self,
        timestep_availability: Dict[int, Dict[str, Any]],
        save_name: str = "data_availability.png"
    ):
        """绘制每个timestep的数据可用性

        Args:
            timestep_availability: {timestep: {"num_episodes": int, "padding_ratio": float, ...}}
            save_name: 保存文件名
        """
        if not MATPLOTLIB_AVAILABLE:
            print("跳过可视化: matplotlib未安装")
            return

        timesteps = sorted(timestep_availability.keys())
        num_episodes = []
        padding_ratios = []

        for t in timesteps:
            stats = timestep_availability[t]
            num_episodes.append(stats["num_episodes"])
            padding_ratios.append(stats["padding_ratio"] * 100)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # 子图1: 数据量
        bars1 = ax1.bar(timesteps, num_episodes, color='skyblue', edgecolor='black', linewidth=0.5)

        # 给数据量少的bar标红
        for i, (t, count) in enumerate(zip(timesteps, num_episodes)):
            if count < 10:
                bars1[i].set_color('red')
                bars1[i].set_alpha(0.7)

        ax1.set_xlabel('Timestep', fontsize=12)
        ax1.set_ylabel('历史Episode数量', fontsize=12)
        ax1.set_title('每个Timestep可用的历史数据量', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')

        # 添加数值标签
        for i, (t, count) in enumerate(zip(timesteps, num_episodes)):
            ax1.text(t, count + max(num_episodes) * 0.02, str(count),
                    ha='center', fontsize=9)

        # 子图2: Padding比例
        bars2 = ax2.bar(timesteps, padding_ratios, color='lightcoral', edgecolor='black', linewidth=0.5)

        # 给严重padding的bar标深红
        for i, (t, ratio) in enumerate(zip(timesteps, padding_ratios)):
            if ratio > 60:
                bars2[i].set_color('darkred')
            elif ratio > 30:
                bars2[i].set_color('orange')

        ax2.axhline(y=30, color='orange', linestyle='--', linewidth=1.5, label='30% 警戒线')
        ax2.axhline(y=60, color='red', linestyle='--', linewidth=1.5, label='60% 危险线')

        ax2.set_xlabel('Timestep', fontsize=12)
        ax2.set_ylabel('Padding比例 (%)', fontsize=12)
        ax2.set_title('每个Timestep的轨迹Padding比例', fontsize=14, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim(0, 105)

        # 添加数值标签
        for i, (t, ratio) in enumerate(zip(timesteps, padding_ratios)):
            ax2.text(t, ratio + 3, f'{ratio:.0f}%',
                    ha='center', fontsize=9)

        plt.tight_layout()

        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"可视化已保存: {save_path}")
        plt.close(fig)

    def plot_failure_reasons(
        self,
        timestep_stats: Dict[int, Dict[str, Any]],
        save_name: str = "failure_reasons.png"
    ):
        """绘制失败原因分布

        Args:
            timestep_stats: {timestep: {"failure_reasons": {"reason": count, ...}, ...}}
            save_name: 保存文件名
        """
        if not MATPLOTLIB_AVAILABLE:
            print("跳过可视化: matplotlib未安装")
            return

        # 收集所有失败原因
        all_reasons = set()
        for stats in timestep_stats.values():
            all_reasons.update(stats.get("failure_reasons", {}).keys())

        if not all_reasons:
            print("没有失败数据可供可视化")
            return

        all_reasons = sorted(all_reasons)
        timesteps = sorted(timestep_stats.keys())

        # 构建数据矩阵
        data_matrix = np.zeros((len(all_reasons), len(timesteps)))

        for j, t in enumerate(timesteps):
            stats = timestep_stats[t]
            failure_reasons = stats.get("failure_reasons", {})
            total = stats.get("total_count", 1)

            for i, reason in enumerate(all_reasons):
                count = failure_reasons.get(reason, 0)
                data_matrix[i, j] = count / total * 100  # 转换为百分比

        # 绘制堆叠条形图
        fig, ax = plt.subplots(figsize=(14, 8))

        colors = plt.cm.Set3(np.linspace(0, 1, len(all_reasons)))
        bottom = np.zeros(len(timesteps))

        for i, reason in enumerate(all_reasons):
            ax.bar(timesteps, data_matrix[i, :], bottom=bottom,
                  label=reason, color=colors[i], edgecolor='black', linewidth=0.5)
            bottom += data_matrix[i, :]

        ax.set_xlabel('Timestep', fontsize=12)
        ax.set_ylabel('失败原因占比 (%)', fontsize=12)
        ax.set_title('每个Timestep的匹配失败原因分布', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 100)

        plt.tight_layout()

        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"可视化已保存: {save_path}")
        plt.close(fig)

    def generate_full_report(
        self,
        buffer: HighPerformanceTrajectoryBuffer,
        timestep_stats: Dict[int, Dict[str, Any]],
        timestep_availability: Dict[int, Dict[str, Any]]
    ):
        """生成完整的可视化报告

        Args:
            buffer: TrajectoryBuffer实例
            timestep_stats: 匹配统计数据
            timestep_availability: 数据可用性统计
        """
        print("\n正在生成可视化报告...")

        # 绘制成功率曲线
        self.plot_timestep_success_rate(timestep_stats)

        # 绘制数据可用性
        self.plot_data_availability(timestep_availability)

        # 绘制失败原因
        self.plot_failure_reasons(timestep_stats)

        print(f"\n所有图表已保存到: {self.save_dir}")
        print("生成的文件:")
        print("  - matching_success_rate.png: 匹配成功率曲线")
        print("  - data_availability.png: 数据可用性分析")
        print("  - failure_reasons.png: 失败原因分布")


def main():
    """独立运行可视化（需要先运行analyze_matching_issue.py生成数据）"""
    print("这是一个工具模块，请通过其他脚本调用")
    print("可以运行 analyze_matching_issue.py 来生成完整报告")


if __name__ == "__main__":
    main()
