"""统一测试框架

提供算法-环境组合的统一测试接口
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from c2o_drive.core.planner import Transition


@dataclass
class TestResults:
    """测试结果数据类"""
    algorithm_name: str
    env_name: str
    num_episodes: int
    total_steps: int
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    success_count: int = 0
    collision_count: int = 0
    timeout_count: int = 0
    total_time: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.success_count / self.num_episodes if self.num_episodes > 0 else 0.0

    @property
    def avg_reward(self) -> float:
        return np.mean(self.episode_rewards) if self.episode_rewards else 0.0

    @property
    def avg_length(self) -> float:
        return np.mean(self.episode_lengths) if self.episode_lengths else 0.0


class TestRunner:
    """统一测试运行器"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def run_algorithm_env_combination(
        self,
        algorithm,  # Planner实例
        environment,  # Environment实例
        algorithm_name: str,
        env_name: str,
        num_episodes: int = 10,
        max_steps_per_episode: int = 100,
    ) -> TestResults:
        """运行算法-环境组合测试

        Args:
            algorithm: 算法规划器
            environment: 环境实例
            algorithm_name: 算法名称
            env_name: 环境名称
            num_episodes: episode数量
            max_steps_per_episode: 每个episode最大步数

        Returns:
            TestResults对象
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print(f" 测试: {algorithm_name} × {env_name}")
            print(f" Episodes: {num_episodes}, Max steps: {max_steps_per_episode}")
            print(f"{'='*70}\n")

        results = TestResults(
            algorithm_name=algorithm_name,
            env_name=env_name,
            num_episodes=num_episodes,
            total_steps=0,
        )

        start_time = time.time()

        for episode in range(num_episodes):
            episode_reward, episode_length, outcome = self._run_episode(
                algorithm,
                environment,
                episode,
                max_steps_per_episode,
            )

            results.episode_rewards.append(episode_reward)
            results.episode_lengths.append(episode_length)
            results.total_steps += episode_length

            if outcome == 'success':
                results.success_count += 1
            elif outcome == 'collision':
                results.collision_count += 1
            elif outcome == 'timeout':
                results.timeout_count += 1

            if self.verbose:
                print(f"  Episode {episode+1}/{num_episodes}: "
                      f"reward={episode_reward:.2f}, steps={episode_length}, "
                      f"outcome={outcome}")

        results.total_time = time.time() - start_time

        if self.verbose:
            print(f"\n{'='*70}")
            print(f" 测试完成!")
            print(f" 成功率: {results.success_rate*100:.1f}%")
            print(f" 平均奖励: {results.avg_reward:.2f}")
            print(f" 平均步数: {results.avg_length:.1f}")
            print(f" 总耗时: {results.total_time:.2f}s")
            print(f"{'='*70}\n")

        return results

    def _run_episode(
        self,
        algorithm,
        environment,
        episode_idx: int,
        max_steps: int,
    ) -> Tuple[float, int, str]:
        """运行单个episode

        Returns:
            (总奖励, 步数, 结果类型)
        """
        state, _ = environment.reset(seed=42 + episode_idx)
        algorithm.reset()

        total_reward = 0.0
        steps = 0

        for step in range(max_steps):
            # 生成参考路径（简单直线）
            reference_path = [(i * 5.0, 0.0) for i in range(20)]

            # 选择动作
            action = algorithm.select_action(state, reference_path=reference_path)

            # 执行动作
            step_result = environment.step(action)

            # 更新算法
            transition = Transition(
                state=state,
                action=action,
                reward=step_result.reward,
                next_state=step_result.observation,
                terminated=step_result.terminated,
                truncated=step_result.truncated,
                info=step_result.info,
            )
            algorithm.update(transition)

            total_reward += step_result.reward
            steps += 1
            state = step_result.observation

            if step_result.terminated:
                return total_reward, steps, 'collision'
            if step_result.truncated:
                return total_reward, steps, 'timeout'

        return total_reward, steps, 'success'

    def compare_results(self, results_list: List[TestResults]) -> Dict[str, Any]:
        """对比多个测试结果

        Args:
            results_list: TestResults列表

        Returns:
            对比数据字典
        """
        comparison = {
            'results': results_list,
            'summary': {}
        }

        for results in results_list:
            key = f"{results.algorithm_name}_{results.env_name}"
            comparison['summary'][key] = {
                'success_rate': results.success_rate,
                'avg_reward': results.avg_reward,
                'avg_length': results.avg_length,
                'total_time': results.total_time,
            }

        return comparison


__all__ = ['TestRunner', 'TestResults']
