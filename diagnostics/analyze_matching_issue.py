"""诊断匹配失败问题的主脚本 - 模拟匹配过程并分析失败原因"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from collections import defaultdict

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from c2o_drive.algorithms.c2osr.trajectory_buffer import HighPerformanceTrajectoryBuffer
from c2o_drive.algorithms.c2osr.types import MDPState, AgentMDP
from c2o_drive.config.global_config import GlobalConfig
from c2o_drive.core.types import AgentType


class MatchingDiagnostic:
    """匹配失败诊断工具"""

    def __init__(self, buffer: HighPerformanceTrajectoryBuffer, config: GlobalConfig):
        self.buffer = buffer
        self.config = config

    def analyze_matching_for_episode(
        self,
        episode_id: int,
        verbose: bool = True
    ) -> Dict[int, Dict[str, Any]]:
        """分析某个episode中每个timestep的匹配情况

        Args:
            episode_id: 要分析的episode ID
            verbose: 是否打印详细信息

        Returns:
            {timestep: {
                "query_state": MDPState,
                "num_candidates": int,
                "num_matched": int,
                "matched_cells": List[cells],
                "failure_reason": str,
                "distance_stats": Dict,
            }}
        """
        if episode_id not in self.buffer.agent_data:
            print(f"错误: Episode {episode_id} 不在buffer中")
            return {}

        episode_data = self.buffer.agent_data[episode_id]
        results = {}

        if verbose:
            print(f"\n{'='*70}")
            print(f"分析 Episode {episode_id} 的匹配情况")
            print(f"{'='*70}")

        # 遍历这个episode的每个timestep
        for agent_id, agent_data_list in episode_data.items():
            for agent_data in agent_data_list:
                timestep = agent_data.timestep
                query_mdp = agent_data.initial_mdp

                # 模拟匹配过程
                match_result = self._simulate_matching(
                    query_mdp=query_mdp,
                    agent_id=agent_id,
                    timestep=timestep,
                    exclude_episode_id=episode_id,  # 排除自己
                )

                results[timestep] = match_result

                if verbose:
                    self._print_timestep_result(timestep, match_result)

        return results

    def _simulate_matching(
        self,
        query_mdp: MDPState,
        agent_id: int,
        timestep: int,
        exclude_episode_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """模拟buffer的匹配过程

        Returns:
            {
                "query_mdp": MDPState,
                "num_candidates_index": int,  # 索引过滤后的候选数
                "num_candidates_strict": int,  # 严格匹配后的候选数
                "num_matched_timesteps": int,  # 匹配的timestep数
                "matched_cells": List,
                "failure_reason": str,
                "distance_stats": Dict,
            }
        """
        result = {
            "query_mdp": query_mdp,
            "num_candidates_index": 0,
            "num_candidates_strict": 0,
            "num_matched_timesteps": 0,
            "matched_cells": [],
            "failure_reason": "unknown",
            "distance_stats": {},
        }

        # Step 1: 多级索引过滤
        agent_count = len(query_mdp.agents)
        ego_pos = query_mdp.ego_position

        # 获取spatial key
        spatial_key = self.buffer._get_spatial_key(ego_pos)

        # Level 1: Agent count
        if agent_count not in self.buffer.agent_count_index:
            result["failure_reason"] = f"no_data_for_{agent_count}_agents"
            return result

        candidate_ids = set(self.buffer.agent_count_index[agent_count])

        # Level 2: Spatial
        neighbor_keys = self.buffer._get_neighbor_spatial_keys(spatial_key)
        spatial_candidates = set()
        for key in neighbor_keys:
            if key in self.buffer.spatial_index:
                spatial_candidates.update(self.buffer.spatial_index[key])

        candidate_ids &= spatial_candidates

        if not candidate_ids:
            result["failure_reason"] = "no_spatial_match"
            return result

        # Level 3: Action signature
        action_sig = self.buffer._get_action_signature(query_mdp.ego_action_trajectory)
        if action_sig in self.buffer.action_index:
            action_candidates = set(self.buffer.action_index[action_sig])
            candidate_ids &= action_candidates

        if exclude_episode_id is not None:
            candidate_ids.discard(exclude_episode_id)

        result["num_candidates_index"] = len(candidate_ids)

        if not candidate_ids:
            result["failure_reason"] = "no_action_signature_match"
            return result

        # Step 2: 严格MDP匹配
        all_matched_cells = []
        all_distances = {"ego": [], "agents": [], "action": []}
        strict_match_count = 0

        for candidate_id in candidate_ids:
            if candidate_id not in self.buffer.agent_data:
                continue

            candidate_data = self.buffer.agent_data[candidate_id]

            # 假设我们要匹配同一个agent_id的数据
            if agent_id not in candidate_data:
                continue

            for candidate_agent_data in candidate_data[agent_id]:
                candidate_mdp = candidate_agent_data.initial_mdp

                # 检查ego position距离
                ego_dist = np.linalg.norm(
                    np.array(query_mdp.ego_position) - np.array(candidate_mdp.ego_position)
                )
                all_distances["ego"].append(ego_dist)

                if ego_dist > self.config.matching.ego_state_threshold:
                    continue

                # 检查agents匹配（简化版：只检查数量和平均距离）
                if len(query_mdp.agents) != len(candidate_mdp.agents):
                    continue

                agent_matched = True
                for q_agent, c_agent in zip(query_mdp.agents, candidate_mdp.agents):
                    agent_dist = np.linalg.norm(
                        np.array(q_agent.position) - np.array(c_agent.position)
                    )
                    all_distances["agents"].append(agent_dist)

                    if agent_dist > self.config.matching.agents_state_threshold:
                        agent_matched = False
                        break

                if not agent_matched:
                    continue

                # 检查ego action trajectory（逐timestep）
                matched_timesteps = []
                min_len = min(
                    len(query_mdp.ego_action_trajectory),
                    len(candidate_mdp.ego_action_trajectory)
                )

                for t in range(min_len):
                    action_dist = np.linalg.norm(
                        np.array(query_mdp.ego_action_trajectory[t]) -
                        np.array(candidate_mdp.ego_action_trajectory[t])
                    )
                    all_distances["action"].append(action_dist)

                    if action_dist <= self.config.matching.ego_action_threshold:
                        matched_timesteps.append(t + 1)

                if matched_timesteps:
                    strict_match_count += 1
                    # 收集这个候选的agent trajectory cells
                    agent_traj_cells = candidate_agent_data.agent_trajectory_cells
                    for t in matched_timesteps:
                        if t <= len(agent_traj_cells):
                            all_matched_cells.append(agent_traj_cells[t - 1])

        result["num_candidates_strict"] = strict_match_count
        result["num_matched_timesteps"] = len(all_matched_cells)
        result["matched_cells"] = all_matched_cells

        # 分析失败原因
        if strict_match_count == 0:
            if all_distances["ego"]:
                avg_ego_dist = np.mean(all_distances["ego"])
                if avg_ego_dist > self.config.matching.ego_state_threshold:
                    result["failure_reason"] = f"ego_dist_too_large (avg={avg_ego_dist:.2f}m)"
            if all_distances["agents"]:
                avg_agent_dist = np.mean(all_distances["agents"])
                if avg_agent_dist > self.config.matching.agents_state_threshold:
                    result["failure_reason"] = f"agent_dist_too_large (avg={avg_agent_dist:.2f}m)"
            if all_distances["action"]:
                avg_action_dist = np.mean(all_distances["action"])
                if avg_action_dist > self.config.matching.ego_action_threshold:
                    result["failure_reason"] = f"action_dist_too_large (avg={avg_action_dist:.2f}m)"
        elif len(all_matched_cells) == 0:
            result["failure_reason"] = "no_timestep_matched"
        else:
            result["failure_reason"] = "success"

        result["distance_stats"] = {
            "ego_dist_mean": np.mean(all_distances["ego"]) if all_distances["ego"] else None,
            "ego_dist_std": np.std(all_distances["ego"]) if all_distances["ego"] else None,
            "agent_dist_mean": np.mean(all_distances["agents"]) if all_distances["agents"] else None,
            "action_dist_mean": np.mean(all_distances["action"]) if all_distances["action"] else None,
            "action_dist_max": np.max(all_distances["action"]) if all_distances["action"] else None,
        }

        return result

    def _print_timestep_result(self, timestep: int, result: Dict[str, Any]):
        """打印单个timestep的诊断结果"""
        print(f"\n--- Timestep {timestep} ---")
        print(f"  索引过滤后候选数: {result['num_candidates_index']}")
        print(f"  严格匹配候选数: {result['num_candidates_strict']}")
        print(f"  匹配到的cells数: {result['num_matched_timesteps']}")
        print(f"  状态: {result['failure_reason']}")

        if result['distance_stats']:
            stats = result['distance_stats']
            print(f"  距离统计:")
            if stats['ego_dist_mean'] is not None:
                print(f"    Ego距离: {stats['ego_dist_mean']:.2f} ± {stats['ego_dist_std']:.2f} m "
                      f"(阈值: {self.config.matching.ego_state_threshold}m)")
            if stats['agent_dist_mean'] is not None:
                print(f"    Agent距离: {stats['agent_dist_mean']:.2f} m "
                      f"(阈值: {self.config.matching.agents_state_threshold}m)")
            if stats['action_dist_mean'] is not None:
                print(f"    Action距离: {stats['action_dist_mean']:.2f} m (max: {stats['action_dist_max']:.2f} m) "
                      f"(阈值: {self.config.matching.ego_action_threshold}m)")

    def analyze_multiple_episodes(
        self,
        num_episodes: int = 10,
        sample_strategy: str = "recent"
    ) -> Dict[str, Any]:
        """分析多个episode的匹配情况

        Args:
            num_episodes: 分析多少个episode
            sample_strategy: "recent" (最近) 或 "random" (随机)

        Returns:
            聚合的统计信息
        """
        episode_ids = list(self.buffer.agent_data.keys())

        if not episode_ids:
            print("错误: Buffer中没有数据")
            return {}

        if sample_strategy == "recent":
            selected_ids = episode_ids[-num_episodes:]
        else:  # random
            np.random.shuffle(episode_ids)
            selected_ids = episode_ids[:num_episodes]

        print(f"\n{'='*70}")
        print(f"分析 {len(selected_ids)} 个Episode的匹配情况")
        print(f"{'='*70}")

        # 按timestep聚合统计
        timestep_aggregated = defaultdict(lambda: {
            "success_count": 0,
            "total_count": 0,
            "total_matched_cells": 0,
            "failure_reasons": defaultdict(int),
        })

        for episode_id in selected_ids:
            results = self.analyze_matching_for_episode(episode_id, verbose=False)

            for timestep, result in results.items():
                stats = timestep_aggregated[timestep]
                stats["total_count"] += 1

                if result["failure_reason"] == "success":
                    stats["success_count"] += 1
                    stats["total_matched_cells"] += result["num_matched_timesteps"]
                else:
                    stats["failure_reasons"][result["failure_reason"]] += 1

        # 打印聚合结果
        print(f"\n{'='*70}")
        print("聚合统计结果")
        print(f"{'='*70}\n")
        print(f"{'Timestep':<10} {'总查询':<10} {'成功':<8} {'成功率':<10} "
              f"{'平均Cells':<12} {'主要失败原因'}")
        print("-" * 80)

        for timestep in sorted(timestep_aggregated.keys()):
            stats = timestep_aggregated[timestep]
            success_rate = stats["success_count"] / stats["total_count"] * 100
            avg_cells = (stats["total_matched_cells"] / stats["success_count"]
                        if stats["success_count"] > 0 else 0)

            # 找出最主要的失败原因
            if stats["failure_reasons"]:
                main_reason = max(stats["failure_reasons"].items(), key=lambda x: x[1])
                main_reason_str = f"{main_reason[0]} ({main_reason[1]}次)"
            else:
                main_reason_str = "-"

            # 状态标记
            if success_rate < 20:
                status = "❌"
            elif success_rate < 50:
                status = "⚠️ "
            else:
                status = "✓"

            print(f"{timestep:<10} {stats['total_count']:<10} {stats['success_count']:<8} "
                  f"{status} {success_rate:>5.1f}%   {avg_cells:<12.1f} {main_reason_str}")

        return dict(timestep_aggregated)


def main():
    """主函数"""
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

    if buffer_path is None:
        print(f"错误: Buffer文件不存在")
        print("请先运行 'python examples/run_c2osr_carla.py' 生成数据")
        return

    buffer.load(str(buffer_path))
    print(f"成功加载Buffer: {buffer_path}\n")

    # 创建诊断工具
    diagnostic = MatchingDiagnostic(buffer, config)

    # 分析多个episode
    num_episodes = min(10, len(buffer.agent_data))
    diagnostic.analyze_multiple_episodes(num_episodes=num_episodes, sample_strategy="recent")


if __name__ == "__main__":
    main()
