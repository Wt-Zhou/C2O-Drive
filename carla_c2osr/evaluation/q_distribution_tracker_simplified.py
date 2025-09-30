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
        self.q_distribution_history = []
        self.collision_rate_history = []
        self.detailed_info_history = []
    
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
                                    figsize: Tuple[int, int] = (15, 8)) -> None:
        """绘制所有Q值随episode变化的曲线图"""
        if len(self.q_distribution_history) == 0:
            print("没有Q值分布数据可绘制")
            return
        
        # 设置中文字体
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
        except:
            pass
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # 绘制每个episode的所有Q值样本
        episodes = list(range(len(self.q_distribution_history)))
        
        # 创建颜色映射
        colors = plt.cm.viridis(np.linspace(0, 1, len(episodes)))
        
        for episode_idx, q_distribution in enumerate(self.q_distribution_history):
            # 为每个episode的所有样本绘制点
            episode_points = [episode_idx] * len(q_distribution)
            ax.scatter(episode_points, q_distribution, alpha=0.6, s=15, 
                      color=colors[episode_idx], label=f'Episode {episode_idx}' if episode_idx < 10 else "")
        
        # 绘制平均Q值趋势线
        ax.plot(episodes, self.q_value_history, 'r-o', linewidth=3, markersize=8, 
               label='平均Q值', zorder=10)
        
        # 计算并绘制标准差区间
        if len(self.q_distribution_history) > 0:
            try:
                means = self.q_value_history
                stds = []
                for dist in self.q_distribution_history:
                    if len(dist) > 0:
                        stds.append(np.std(dist))
                    else:
                        stds.append(0.0)
                
                # 绘制标准差区间
                upper_bound = [m + s for m, s in zip(means, stds)]
                lower_bound = [m - s for m, s in zip(means, stds)]
                ax.fill_between(episodes, lower_bound, upper_bound, alpha=0.2, color='red', label='±1 标准差')
                
            except Exception as e:
                print(f"标准差区间计算失败: {e}")
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Q Value', fontsize=12)
        ax.set_title('所有Q值随Episode变化的分布', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 只显示关键的图例项
        handles, labels = ax.get_legend_handles_labels()
        key_handles = [h for h, l in zip(handles, labels) if '平均' in l or '标准差' in l]
        key_labels = [l for l in labels if '平均' in l or '标准差' in l]
        ax.legend(key_handles, key_labels, loc='upper right')
        
        # 添加统计信息文本
        if len(self.q_value_history) > 0:
            overall_mean = np.mean(self.q_value_history)
            overall_std = np.std(self.q_value_history)
            max_collision_rate = max(self.collision_rate_history) if self.collision_rate_history else 0
            
            stats_text = f'总体统计:\n平均Q值: {overall_mean:.2f}\n标准差: {overall_std:.2f}\n最大碰撞率: {max_collision_rate:.3f}'
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
        
        # 设置中文字体
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
        except:
            pass
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        episodes = list(range(len(self.collision_rate_history)))
        
        # 绘制碰撞率折线图
        ax.plot(episodes, self.collision_rate_history, 'r-o', linewidth=3, markersize=8)
        
        # 添加零碰撞率基线
        ax.axhline(y=0, color='green', linestyle='--', alpha=0.7, label='零碰撞率')
        
        # 如果有碰撞，标记出来
        for i, rate in enumerate(self.collision_rate_history):
            if rate > 0:
                ax.annotate(f'{rate:.3f}', (i, rate), textcoords="offset points", 
                           xytext=(0,10), ha='center', fontsize=9, color='red')
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Collision Rate', fontsize=12)
        ax.set_title('碰撞率随Episode变化', fontsize=14, fontweight='bold')
        ax.set_ylim([-0.05, max(1.05, max(self.collision_rate_history) + 0.1)])
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 添加统计信息
        total_collisions = sum(self.collision_rate_history)
        avg_collision_rate = np.mean(self.collision_rate_history)
        
        stats_text = f'碰撞统计:\n总碰撞率: {total_collisions:.3f}\n平均碰撞率: {avg_collision_rate:.3f}'
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
        
        print("\n" + "="*50)
        print("Q值分布统计摘要")
        print("="*50)
        
        # Q值统计
        print(f"总Episodes: {len(self.q_value_history)}")
        print(f"平均Q值: {np.mean(self.q_value_history):.4f}")
        print(f"Q值标准差: {np.std(self.q_value_history):.4f}")
        print(f"Q值范围: [{np.min(self.q_value_history):.4f}, {np.max(self.q_value_history):.4f}]")
        
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
        data = {
            'episode_data': self.episode_data,
            'q_value_history': self.q_value_history,
            'collision_rate_history': self.collision_rate_history,
            'summary_stats': {
                'total_episodes': len(self.q_value_history),
                'mean_q_value': float(np.mean(self.q_value_history)) if self.q_value_history else 0,
                'std_q_value': float(np.std(self.q_value_history)) if self.q_value_history else 0,
                'mean_collision_rate': float(np.mean(self.collision_rate_history)) if self.collision_rate_history else 0
            }
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"数据已保存到: {output_path}")
