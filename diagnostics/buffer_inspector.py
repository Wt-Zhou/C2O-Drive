"""Buffer内容检查工具 - 只读分析，不修改任何数据"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from c2o_drive.algorithms.c2osr.trajectory_buffer import HighPerformanceTrajectoryBuffer
from c2o_drive.config.global_config import GlobalConfig


class BufferInspector:
    """Buffer数据检查器（只读）"""

    def __init__(self, buffer: HighPerformanceTrajectoryBuffer):
        self.buffer = buffer
        self.config = GlobalConfig()

    def get_basic_stats(self) -> Dict[str, Any]:
        """获取基本统计信息"""
        total_episodes = len(self.buffer.agent_data)

        if total_episodes == 0:
            return {
                "total_episodes": 0,
                "avg_episode_length": 0,
                "total_timesteps": 0,
            }

        # 统计每个episode的长度
        episode_lengths = []
        for episode_id, data in self.buffer.agent_data.items():
            # 通过检查有多少个唯一的timestep来确定episode长度
            timesteps = set()
            for agent_id, agent_data_list in data.items():
                for agent_data in agent_data_list:
                    timesteps.add(agent_data.timestep)
            episode_lengths.append(max(timesteps) if timesteps else 0)

        return {
            "total_episodes": total_episodes,
            "avg_episode_length": np.mean(episode_lengths) if episode_lengths else 0,
            "max_episode_length": max(episode_lengths) if episode_lengths else 0,
            "min_episode_length": min(episode_lengths) if episode_lengths else 0,
            "total_timesteps": sum(episode_lengths),
        }

    def get_timestep_data_availability(self, max_timestep: int = 15) -> Dict[int, Dict[str, Any]]:
        """分析每个timestep的数据可用性

        Returns:
            {timestep: {
                "num_episodes": int,  # 有多少episode包含这个timestep
                "num_padded": int,    # 其中有多少是padding的
                "padding_ratio": float,
                "avg_trajectory_length": float,
            }}
        """
        timestep_stats = {}

        for timestep in range(1, max_timestep + 1):
            episodes_with_timestep = []
            padded_count = 0

            for episode_id, data in self.buffer.agent_data.items():
                # 检查这个episode是否有这个timestep的数据
                has_timestep = False
                for agent_id, agent_data_list in data.items():
                    for agent_data in agent_data_list:
                        if agent_data.timestep == timestep:
                            has_timestep = True
                            # 检查是否是padding（通过检查trajectory是否有重复点）
                            ego_traj = agent_data.initial_mdp.ego_action_trajectory
                            if self._is_padded_trajectory(ego_traj):
                                padded_count += 1
                            break
                    if has_timestep:
                        break

                if has_timestep:
                    episodes_with_timestep.append(episode_id)

            num_episodes = len(episodes_with_timestep)
            timestep_stats[timestep] = {
                "num_episodes": num_episodes,
                "num_padded": padded_count,
                "padding_ratio": padded_count / num_episodes if num_episodes > 0 else 0,
                "episode_ids": episodes_with_timestep[:5],  # 只保存前5个用于调试
            }

        return timestep_stats

    def _is_padded_trajectory(self, trajectory: List[Tuple[float, float]], threshold: float = 0.1) -> bool:
        """判断轨迹是否被padding（检查是否有连续重复点）

        Args:
            trajectory: [(x1, y1), (x2, y2), ...]
            threshold: 位置差异阈值（米）

        Returns:
            如果有超过3个连续点位置几乎相同，认为是padding
        """
        if len(trajectory) < 4:
            return False

        consecutive_same = 0
        for i in range(1, len(trajectory)):
            dist = np.linalg.norm(np.array(trajectory[i]) - np.array(trajectory[i-1]))
            if dist < threshold:
                consecutive_same += 1
                if consecutive_same >= 3:
                    return True
            else:
                consecutive_same = 0

        return False

    def get_agent_count_distribution(self) -> Dict[int, int]:
        """获取agent数量分布

        Returns:
            {num_agents: num_episodes}
        """
        agent_count_dist = {}

        for episode_id, data in self.buffer.agent_data.items():
            # 获取第一个timestep的agent数量作为代表
            num_agents = len(data) - 1  # 减去ego
            agent_count_dist[num_agents] = agent_count_dist.get(num_agents, 0) + 1

        return agent_count_dist

    def analyze_action_trajectory_patterns(self, num_samples: int = 20) -> Dict[str, Any]:
        """分析ego action trajectory的模式

        Returns:
            {
                "avg_trajectory_length": float,
                "avg_displacement_per_step": float,
                "samples": List[Dict],  # 一些样本用于检查
            }
        """
        samples = []
        all_displacements = []

        episode_ids = list(self.buffer.agent_data.keys())[:num_samples]

        for episode_id in episode_ids:
            data = self.buffer.agent_data[episode_id]

            # 获取第一个timestep的数据
            for agent_id, agent_data_list in data.items():
                if agent_data_list:
                    agent_data = agent_data_list[0]
                    ego_traj = agent_data.initial_mdp.ego_action_trajectory

                    # 计算轨迹总位移和平均位移
                    displacements = []
                    for i in range(1, len(ego_traj)):
                        dist = np.linalg.norm(np.array(ego_traj[i]) - np.array(ego_traj[i-1]))
                        displacements.append(dist)

                    if displacements:
                        all_displacements.extend(displacements)
                        samples.append({
                            "episode_id": episode_id,
                            "timestep": agent_data.timestep,
                            "trajectory_length": len(ego_traj),
                            "avg_displacement": np.mean(displacements),
                            "is_padded": self._is_padded_trajectory(ego_traj),
                        })
                    break
                break

        return {
            "num_samples": len(samples),
            "avg_displacement_per_step": np.mean(all_displacements) if all_displacements else 0,
            "std_displacement": np.std(all_displacements) if all_displacements else 0,
            "samples": samples,
        }

    def print_summary(self):
        """打印完整的统计摘要"""
        print("=" * 70)
        print("Buffer 数据统计摘要")
        print("=" * 70)

        # 基本统计
        basic_stats = self.get_basic_stats()
        print("\n[基本信息]")
        print(f"总Episode数: {basic_stats['total_episodes']}")
        print(f"平均Episode长度: {basic_stats['avg_episode_length']:.2f} 步")
        print(f"最长Episode: {basic_stats['max_episode_length']} 步")
        print(f"最短Episode: {basic_stats['min_episode_length']} 步")
        print(f"总Timestep数: {basic_stats['total_timesteps']}")

        # Agent数量分布
        agent_dist = self.get_agent_count_distribution()
        print("\n[Agent数量分布]")
        for num_agents, count in sorted(agent_dist.items()):
            print(f"  {num_agents} agents: {count} episodes")

        # Timestep数据可用性
        max_timestep = min(int(basic_stats['max_episode_length']) + 2, 15)
        timestep_stats = self.get_timestep_data_availability(max_timestep)
        print("\n[每个Timestep的数据可用性]")
        print(f"{'Timestep':<10} {'Episodes':<12} {'Padded':<10} {'Padding%':<12} {'状态'}")
        print("-" * 70)

        for t in range(1, max_timestep + 1):
            if t in timestep_stats:
                stats = timestep_stats[t]
                padding_pct = stats['padding_ratio'] * 100

                # 状态标记
                if padding_pct > 60:
                    status = "❌ 严重padding"
                elif padding_pct > 30:
                    status = "⚠️  轻度padding"
                elif stats['num_episodes'] < 10:
                    status = "⚠️  数据稀少"
                else:
                    status = "✓ 正常"

                print(f"{t:<10} {stats['num_episodes']:<12} {stats['num_padded']:<10} "
                      f"{padding_pct:<12.1f} {status}")

        # Trajectory模式分析
        pattern_stats = self.analyze_action_trajectory_patterns()
        print("\n[Ego Action Trajectory 模式]")
        print(f"样本数: {pattern_stats['num_samples']}")
        print(f"平均步间位移: {pattern_stats['avg_displacement_per_step']:.2f} ± "
              f"{pattern_stats['std_displacement']:.2f} 米")

        print("\n[前5个样本详情]")
        for sample in pattern_stats['samples'][:5]:
            padding_flag = "[PADDED]" if sample['is_padded'] else ""
            print(f"  Episode {sample['episode_id']}, t={sample['timestep']}: "
                  f"traj_len={sample['trajectory_length']}, "
                  f"avg_disp={sample['avg_displacement']:.2f}m {padding_flag}")

        print("\n" + "=" * 70)


def main():
    """主函数：加载buffer并打印统计信息"""
    print("正在加载Buffer...")

    config = GlobalConfig()
    buffer = HighPerformanceTrajectoryBuffer(
        capacity=1000,
        horizon=config.time.default_horizon,
    )

    # 查找buffer文件
    possible_paths = [
        Path("data/trajectory_buffer.pkl"),
        Path("checkpoints/trajectory_buffer.pkl"),
        Path("results/trajectory_buffer.pkl"),
    ]

    buffer_path = None
    for path in possible_paths:
        if path.exists():
            buffer_path = path
            break

    if buffer_path:
        buffer.load(str(buffer_path))
        print(f"成功加载Buffer: {buffer_path}")
    else:
        print(f"警告: Buffer文件不存在")
        print("将分析空Buffer（用于测试）")

    # 创建检查器并打印摘要
    inspector = BufferInspector(buffer)
    inspector.print_summary()


if __name__ == "__main__":
    main()
