"""
简化的Q值分布追踪器
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import json
import os


class QDistributionTracker:
    """Q值分布追踪器 - 简化版本"""

    def __init__(self):
        """初始化追踪器"""
        self.episode_data = []
        self.q_value_history = []
        self.percentile_q_history = []  # 新增：存储percentile Q值
        self.q_distribution_history = []
        self.collision_rate_history = []
        self.detailed_info_history = []

    def _get_percentile_q_value(self, q_values: np.ndarray, percentile: float) -> float:
        """
        根据百分位数获取Q值

        Args:
            q_values: Q值数组
            percentile: 百分位数 [0.0, 1.0]，0.0表示最小值，1.0表示最大值

        Returns:
            百分位对应的Q值（使用线性插值）
        """
        if len(q_values) == 0:
            return 0.0

        if len(q_values) == 1:
            return float(q_values[0])

        # 排序Q值
        sorted_q = np.sort(q_values)
        n = len(sorted_q)

        # 计算精确位置（0-based index）
        position = percentile * (n - 1)

        # 下界和上界索引
        lower_idx = int(np.floor(position))
        upper_idx = int(np.ceil(position))

        # 如果正好是整数位置
        if lower_idx == upper_idx:
            return float(sorted_q[lower_idx])

        # 线性插值
        weight = position - lower_idx
        return float(sorted_q[lower_idx] * (1 - weight) + sorted_q[upper_idx] * weight)

    def add_episode_data(self, episode_id: int, q_value: float, q_distribution: List[float],
                        collision_rate: float, detailed_info: Dict):
        """添加episode的Q值数据"""
        # 计算percentile Q值
        from carla_c2osr.evaluation.q_value_calculator import QValueConfig
        q_config = QValueConfig.from_global_config()
        percentile_q = self._get_percentile_q_value(np.array(q_distribution), q_config.q_selection_percentile)

        self.episode_data.append({
            'episode_id': episode_id,
            'q_value': q_value,
            'percentile_q': percentile_q,  # 新增：保存percentile Q值
            'q_distribution': q_distribution.copy(),
            'collision_rate': collision_rate,
            'detailed_info': detailed_info
        })

        self.q_value_history.append(q_value)
        self.percentile_q_history.append(percentile_q)  # 新增：记录percentile Q历史
        self.q_distribution_history.append(q_distribution.copy())
        self.collision_rate_history.append(collision_rate)
        self.detailed_info_history.append(detailed_info)
    
    def plot_q_distribution_evolution(self, output_path: str,
                                    figsize: Tuple[int, int] = (15, 8)) -> None:
        """绘制所有Q值随episode变化的曲线图（使用Percentile Q）"""
        if len(self.q_distribution_history) == 0:
            print("没有Q值分布数据可绘制")
            return

        # 获取percentile配置
        from carla_c2osr.evaluation.q_value_calculator import QValueConfig
        q_config = QValueConfig.from_global_config()
        percentile = q_config.q_selection_percentile

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Plot all Q-value samples for each episode
        episodes = list(range(len(self.q_distribution_history)))

        # Create color map
        colors = plt.cm.viridis(np.linspace(0, 1, len(episodes)))

        for episode_idx, q_distribution in enumerate(self.q_distribution_history):
            # Plot all samples for each episode
            episode_points = [episode_idx] * len(q_distribution)
            ax.scatter(episode_points, q_distribution, alpha=0.6, s=15,
                      color=colors[episode_idx], label=f'Episode {episode_idx}' if episode_idx < 10 else "")

        # Plot percentile Q-value trend line (instead of mean)
        ax.plot(episodes, self.percentile_q_history, 'r-o', linewidth=3, markersize=8,
               label=f'P{int(percentile*100)} Q-value', zorder=10)

        # Calculate and plot standard deviation band (based on percentile Q)
        if len(self.q_distribution_history) > 0:
            try:
                percentile_qs = self.percentile_q_history
                stds = []
                for dist in self.q_distribution_history:
                    if len(dist) > 0:
                        stds.append(np.std(dist))
                    else:
                        stds.append(0.0)

                # Plot std band
                upper_bound = [p + s for p, s in zip(percentile_qs, stds)]
                lower_bound = [p - s for p, s in zip(percentile_qs, stds)]
                ax.fill_between(episodes, lower_bound, upper_bound, alpha=0.2, color='red', label='±1 Std')

            except Exception as e:
                print(f"Std calculation failed: {e}")

        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Q Value', fontsize=12)
        ax.set_title('Q-Value Distribution Evolution across Episodes', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Show only key legend items
        handles, labels = ax.get_legend_handles_labels()
        key_handles = [h for h, l in zip(handles, labels) if f'P{int(percentile*100)}' in l or 'Std' in l]
        key_labels = [l for l in labels if f'P{int(percentile*100)}' in l or 'Std' in l]
        ax.legend(key_handles, key_labels, loc='upper right')

        # Add statistics text (based on percentile Q)
        if len(self.percentile_q_history) > 0:
            overall_percentile_mean = np.mean(self.percentile_q_history)
            overall_percentile_std = np.std(self.percentile_q_history)
            max_collision_rate = max(self.collision_rate_history) if self.collision_rate_history else 0

            stats_text = f'Statistics:\nP{int(percentile*100)} Q (mean): {overall_percentile_mean:.2f}\nP{int(percentile*100)} Q (std): {overall_percentile_std:.2f}\nMax Collision: {max_collision_rate:.3f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Q值分布演化图已保存到: {output_path}")
    
    def plot_collision_rate_evolution(self, output_path: str, 
                                     figsize: Tuple[int, int] = (12, 6)) -> None:
        """绘制碰撞率变化图"""
        if len(self.collision_rate_history) == 0:
            print("没有碰撞率数据可绘制")
            return
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        episodes = list(range(len(self.collision_rate_history)))

        # Plot collision rate line
        ax.plot(episodes, self.collision_rate_history, 'r-o', linewidth=3, markersize=8)

        # Add zero baseline
        ax.axhline(y=0, color='green', linestyle='--', alpha=0.7, label='Zero Collision')

        # Mark collisions if any
        for i, rate in enumerate(self.collision_rate_history):
            if rate > 0:
                ax.annotate(f'{rate:.3f}', (i, rate), textcoords="offset points",
                           xytext=(0,10), ha='center', fontsize=9, color='red')

        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Collision Rate', fontsize=12)
        ax.set_title('Collision Rate Evolution', fontsize=14, fontweight='bold')
        ax.set_ylim([-0.05, max(1.05, max(self.collision_rate_history) + 0.1)])
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Add statistics
        total_collisions = sum(self.collision_rate_history)
        avg_collision_rate = np.mean(self.collision_rate_history)

        stats_text = f'Collision Stats:\nTotal: {total_collisions:.3f}\nAverage: {avg_collision_rate:.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"碰撞率变化图已保存到: {output_path}")
    
    def print_summary(self):
        """打印统计摘要"""
        if len(self.episode_data) == 0:
            print("没有episode数据")
            return

        # 获取percentile配置
        from carla_c2osr.evaluation.q_value_calculator import QValueConfig
        q_config = QValueConfig.from_global_config()
        percentile = q_config.q_selection_percentile

        print("\n" + "="*50)
        print("Q值分布统计摘要")
        print("="*50)

        # Q值统计（Mean Q）
        print(f"总Episodes: {len(self.q_value_history)}")
        print(f"平均Q值 (Mean): {np.mean(self.q_value_history):.4f}")
        print(f"Q值标准差 (Mean): {np.std(self.q_value_history):.4f}")
        print(f"Q值范围 (Mean): [{np.min(self.q_value_history):.4f}, {np.max(self.q_value_history):.4f}]")

        # Percentile Q值统计
        print(f"\nP{int(percentile*100)} Q值统计:")
        print(f"平均P{int(percentile*100)} Q值: {np.mean(self.percentile_q_history):.4f}")
        print(f"P{int(percentile*100)} Q值标准差: {np.std(self.percentile_q_history):.4f}")
        print(f"P{int(percentile*100)} Q值范围: [{np.min(self.percentile_q_history):.4f}, {np.max(self.percentile_q_history):.4f}]")

        # 碰撞统计
        print(f"\n碰撞率统计:")
        print(f"平均碰撞率: {np.mean(self.collision_rate_history):.4f}")
        print(f"最大碰撞率: {np.max(self.collision_rate_history):.4f}")
        print(f"零碰撞Episodes: {sum(1 for rate in self.collision_rate_history if rate == 0.0)}/{len(self.collision_rate_history)}")

        # 分布统计
        if self.q_distribution_history:
            all_q_values = []
            for dist in self.q_distribution_history:
                all_q_values.extend(dist)

            print(f"\n所有样本统计:")
            print(f"总样本数: {len(all_q_values)}")
            print(f"样本平均值: {np.mean(all_q_values):.4f}")
            print(f"样本标准差: {np.std(all_q_values):.4f}")

        print("="*50)
    
    def save_data(self, output_path: str):
        """保存数据到JSON文件"""
        # 获取percentile配置
        from carla_c2osr.evaluation.q_value_calculator import QValueConfig
        q_config = QValueConfig.from_global_config()
        percentile = q_config.q_selection_percentile

        data = {
            'episode_data': self.episode_data,
            'q_value_history': self.q_value_history,
            'percentile_q_history': self.percentile_q_history,  # 新增：保存percentile Q历史
            'collision_rate_history': self.collision_rate_history,
            'summary_stats': {
                'total_episodes': len(self.q_value_history),
                'percentile_used': float(percentile),  # 新增：记录使用的百分位数
                'mean_q_value': float(np.mean(self.q_value_history)) if self.q_value_history else 0,
                'std_q_value': float(np.std(self.q_value_history)) if self.q_value_history else 0,
                'mean_percentile_q_value': float(np.mean(self.percentile_q_history)) if self.percentile_q_history else 0,  # 新增
                'std_percentile_q_value': float(np.std(self.percentile_q_history)) if self.percentile_q_history else 0,  # 新增
                'mean_collision_rate': float(np.mean(self.collision_rate_history)) if self.collision_rate_history else 0
            }
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"数据已保存到: {output_path}")

    @staticmethod
    def load_data(input_path: str) -> 'QDistributionTracker':
        """从JSON文件加载数据并恢复tracker状态

        Args:
            input_path: JSON文件路径

        Returns:
            恢复状态的QDistributionTracker实例
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 创建新的tracker实例
        tracker = QDistributionTracker()

        # 恢复所有历史数据
        tracker.episode_data = data.get('episode_data', [])
        tracker.q_value_history = data.get('q_value_history', [])
        tracker.percentile_q_history = data.get('percentile_q_history', [])
        tracker.collision_rate_history = data.get('collision_rate_history', [])

        # 恢复q_distribution_history（从episode_data中提取）
        tracker.q_distribution_history = [
            ep['q_distribution'] for ep in tracker.episode_data
        ]

        # 恢复detailed_info_history（从episode_data中提取）
        tracker.detailed_info_history = [
            ep.get('detailed_info', {}) for ep in tracker.episode_data
        ]

        print(f"数据已从 {input_path} 加载")
        print(f"恢复了 {len(tracker.q_value_history)} 个episodes的数据")

        return tracker
