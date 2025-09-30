"""
Q值分布追踪和可视化模块

用于跟踪和可视化Q值分布随episode的变化。
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json


class QDistributionTracker:
    """Q值分布追踪器"""
    
    def __init__(self):
        self.episode_data: List[Dict] = []
        self.q_value_history: List[float] = []
        self.q_distribution_history: List[List[float]] = []
        self.collision_rate_history: List[float] = []
        self.detailed_info_history: List[Dict] = []
    
    def add_episode_data(self, episode_id: int, q_value: float, q_distribution: List[float], 
                        collision_rate: float, detailed_info: Dict):
        """添加episode的Q值数据"""
        self.episode_data.append({
            'episode_id': episode_id,
            'q_value': q_value,
            'q_distribution': q_distribution.copy(),
            'collision_rate': collision_rate,
            'detailed_info': detailed_info
        })
        
        self.q_value_history.append(q_value)
        self.q_distribution_history.append(q_distribution.copy())
        self.collision_rate_history.append(collision_rate)
        self.detailed_info_history.append(detailed_info)
    
    def plot_q_distribution_evolution(self, output_path: str, 
                                    figsize: Tuple[int, int] = (15, 10)) -> None:
        """绘制Q值分布随episode的演化图"""
        if len(self.q_distribution_history) == 0:
            print("没有Q值分布数据可绘制")
            return
        
        # 设置中文字体
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
        except:
            pass
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Q值分布演化分析', fontsize=16, fontweight='bold')
        
        # 1. Q值分布热力图
        ax1 = axes[0, 0]
        if len(self.q_distribution_history) > 1:
            try:
                # 检查并统一数组长度
                max_len = max(len(dist) for dist in self.q_distribution_history)
                
                # 填充较短的数组到相同长度
                padded_distributions = []
                for dist in self.q_distribution_history:
                    if len(dist) < max_len:
                        padded = list(dist) + [0.0] * (max_len - len(dist))
                    else:
                        padded = list(dist)
                    padded_distributions.append(padded)
                
                # 转换为矩阵形式
                q_matrix = np.array(padded_distributions).T
                im1 = ax1.imshow(q_matrix, aspect='auto', cmap='viridis', origin='lower')
                ax1.set_xlabel('Episode')
                ax1.set_ylabel('Sample Index')
                ax1.set_title('Q值分布热力图')
                plt.colorbar(im1, ax=ax1, label='Q Value')
            except Exception as e:
                ax1.text(0.5, 0.5, f'Q值分布矩阵创建失败:\n{str(e)}', ha='center', va='center', 
                        transform=ax1.transAxes, fontsize=10)
                ax1.set_title('Q值分布热力图 (创建失败)')
        else:
            ax1.text(0.5, 0.5, '需要至少2个episode的数据', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Q值分布热力图 (数据不足)')
        
        # 2. 平均Q值变化
        ax2 = axes[0, 1]
        episodes = list(range(len(self.q_value_history)))
        ax2.plot(episodes, self.q_value_history, 'b-o', linewidth=2, markersize=6)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Average Q Value')
        ax2.set_title('平均Q值变化趋势')
        ax2.grid(True, alpha=0.3)
        
        # 3. Q值分布统计信息
        ax3 = axes[1, 0]
        if len(self.q_distribution_history) > 0:
            q_means = [np.mean(dist) for dist in self.q_distribution_history]
            q_stds = [np.std(dist) for dist in self.q_distribution_history]
            q_mins = [np.min(dist) for dist in self.q_distribution_history]
            q_maxs = [np.max(dist) for dist in self.q_distribution_history]
            
            ax3.plot(episodes, q_means, 'g-o', label='Mean', linewidth=2)
            ax3.fill_between(episodes, 
                           [m - s for m, s in zip(q_means, q_stds)], 
                           [m + s for m, s in zip(q_means, q_stds)], 
                           alpha=0.3, color='green', label='±1 Std')
            ax3.plot(episodes, q_mins, 'r--', label='Min', alpha=0.7)
            ax3.plot(episodes, q_maxs, 'b--', label='Max', alpha=0.7)
            
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Q Value')
            ax3.set_title('Q值分布统计')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, '无Q值分布数据', ha='center', va='center', transform=ax3.transAxes)
        
        # 4. 碰撞率变化
        ax4 = axes[1, 1]
        if len(self.collision_rate_history) > 0:
            ax4.plot(episodes, self.collision_rate_history, 'r-o', linewidth=2, markersize=6)
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Collision Rate')
            ax4.set_title('碰撞率变化趋势')
            ax4.set_ylim([0, 1])
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, '无碰撞率数据', ha='center', va='center', transform=ax4.transAxes)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Q值分布演化图已保存到: {output_path}")
    
    def plot_q_value_boxplot(self, output_path: str, 
                           figsize: Tuple[int, int] = (12, 8)) -> None:
        """绘制每个episode的Q值分布箱线图"""
        if len(self.q_distribution_history) == 0:
            print("没有Q值分布数据可绘制")
            return
        
        # 设置中文字体
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
        except:
            pass
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 创建箱线图
        box_data = self.q_distribution_history
        episode_labels = [f'Episode {i}' for i in range(len(box_data))]
        
        bp = ax.boxplot(box_data, labels=episode_labels, patch_artist=True)
        
        # 美化箱线图
        colors = plt.cm.viridis(np.linspace(0, 1, len(bp['boxes'])))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # 添加平均值点
        means = [np.mean(dist) for dist in self.q_distribution_history]
        ax.plot(range(1, len(means) + 1), means, 'ro-', linewidth=2, markersize=8, label='Mean')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Q Value')
        ax.set_title('Q值分布箱线图 - 各Episode对比')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 旋转x轴标签
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Q值分布箱线图已保存到: {output_path}")
    
    def save_data(self, output_path: str) -> None:
        """保存数据到JSON文件"""
        data = {
            'episode_data': self.episode_data,
            'summary': {
                'total_episodes': len(self.episode_data),
                'q_value_stats': {
                    'mean': float(np.mean(self.q_value_history)) if self.q_value_history else 0,
                    'std': float(np.std(self.q_value_history)) if self.q_value_history else 0,
                    'min': float(np.min(self.q_value_history)) if self.q_value_history else 0,
                    'max': float(np.max(self.q_value_history)) if self.q_value_history else 0
                },
                'collision_rate_stats': {
                    'mean': float(np.mean(self.collision_rate_history)) if self.collision_rate_history else 0,
                    'std': float(np.std(self.collision_rate_history)) if self.collision_rate_history else 0
                }
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Q值分布数据已保存到: {output_path}")
    
    def print_summary(self) -> None:
        """打印统计摘要"""
        if len(self.q_value_history) == 0:
            print("没有Q值数据")
            return
        
        print("\n=== Q值分布统计摘要 ===")
        print(f"Episode数量: {len(self.q_value_history)}")
        print(f"平均Q值: {np.mean(self.q_value_history):.3f} ± {np.std(self.q_value_history):.3f}")
        print(f"Q值范围: [{np.min(self.q_value_history):.3f}, {np.max(self.q_value_history):.3f}]")
        print(f"平均碰撞率: {np.mean(self.collision_rate_history):.3f} ± {np.std(self.collision_rate_history):.3f}")
        
        # 最后几个episode的趋势
        if len(self.q_value_history) >= 3:
            recent_trend = np.polyfit(range(len(self.q_value_history)), self.q_value_history, 1)[0]
            trend_desc = "上升" if recent_trend > 0 else "下降" if recent_trend < 0 else "稳定"
            print(f"Q值趋势: {trend_desc} (斜率: {recent_trend:.4f})")
