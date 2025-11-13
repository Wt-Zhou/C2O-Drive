#!/usr/bin/env python3
"""
C2OSR + CARLA 集成运行脚本

将C2OSR算法与CARLA仿真环境集成，支持真实的3D仿真场景。

特性：
- 使用C2OSRPlanner（标准化算法接口）
- 使用CarlaEnvironment（Gym标准接口）
- 支持预定义场景库
- 支持自定义CARLA配置
- 实时性能监控
- 可视化和数据导出

使用前提：
- CARLA服务器必须已经启动（默认localhost:2000）
- 确保carla python包已安装或路径已配置
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import time

# 添加项目根目录到路径
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# C2OSR算法组件
from carla_c2osr.algorithms.c2osr import (
    create_c2osr_planner,
    C2OSRPlannerConfig,
    LatticePlannerConfig,
    QValueConfig,
    GridConfig,
    DirichletConfig,
)

# CARLA环境组件
from carla_c2osr.environments import CarlaEnvironment
from carla_c2osr.env.carla_scenarios import CarlaScenarioLibrary, get_scenario, list_scenarios
from carla_c2osr.core.planner import Transition
from carla_c2osr.config.global_config import GlobalConfig, CarlaConfig


class CarlaEpisodeRunner:
    """CARLA环境下的Episode运行器"""

    def __init__(
        self,
        planner,
        env: CarlaEnvironment,
        output_dir: Optional[Path] = None,
        verbose: bool = True,
        save_trajectory: bool = True,
    ):
        self.planner = planner
        self.env = env
        self.output_dir = output_dir or Path("outputs/c2osr_carla")
        self.verbose = verbose
        self.save_trajectory = save_trajectory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_episode(
        self,
        episode_id: int,
        max_steps: int,
        scenario_name: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """运行单个episode

        Args:
            episode_id: Episode ID
            max_steps: 最大步数
            scenario_name: 场景名称（可选）
            seed: 随机种子

        Returns:
            Episode统计数据字典
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Episode {episode_id}")
            if scenario_name:
                print(f"场景: {scenario_name}")
            print(f"{'='*60}")

        episode_start_time = time.time()

        # 重置环境和规划器
        options = {}
        if scenario_name:
            scenario_def = get_scenario(scenario_name)
            options['scenario_config'] = {
                'scenario': scenario_def,
            }
            if self.verbose:
                print(f"场景: {scenario_def.description}")
                print(f"难度: {scenario_def.difficulty}")

        state, info = self.env.reset(seed=seed, options=options)
        self.planner.reset()

        # Episode统计
        total_reward = 0.0
        steps = 0
        outcome = 'success'
        collision_occurred = False

        # 主循环
        for step in range(max_steps):
            # 选择动作
            try:
                action = self.planner.select_action(
                    observation=state,
                    deterministic=False,
                    reference_path=info.get('reference_path'),
                )
            except Exception as e:
                if self.verbose:
                    print(f"  ✗ 规划失败: {e}")
                outcome = 'planning_failed'
                break

            # 执行动作
            step_result = self.env.step(action)

            # 更新规划器
            transition = Transition(
                state=state,
                action=action,
                reward=step_result.reward,
                next_state=step_result.observation,
                terminated=step_result.terminated,
                truncated=step_result.truncated,
                info=step_result.info,
            )
            self.planner.update(transition)

            # 更新统计
            total_reward += step_result.reward
            steps += 1
            state = step_result.observation

            # 输出进度
            if self.verbose and (step + 1) % 10 == 0:
                collision_info = ""
                if step_result.info.get('collision_sensor'):
                    collision_info = " [碰撞传感器触发]"
                print(f"  Step {step+1}/{max_steps}: "
                      f"reward={step_result.reward:.2f}, "
                      f"total={total_reward:.2f}, "
                      f"accel={step_result.info.get('acceleration', 0):.2f}"
                      f"{collision_info}")

            # 检查终止条件
            if step_result.terminated:
                outcome = 'collision'
                collision_occurred = True
                if self.verbose:
                    print(f"  ✗ 碰撞！Episode在第{steps}步结束")
                break

            if step_result.truncated:
                outcome = 'timeout'
                if self.verbose:
                    print(f"  ⏱ 达到最大步数")
                break

        episode_time = time.time() - episode_start_time

        if outcome == 'success' and self.verbose:
            print(f"  ✓ 成功完成{steps}步！")

        # 保存轨迹
        trajectory_file = ""
        if self.save_trajectory:
            trajectory = self.env.get_episode_trajectory()
            if len(trajectory) > 0:
                trajectory_file = self.output_dir / f"trajectory_ep{episode_id}.npy"
                np.save(trajectory_file, trajectory)
                if self.verbose:
                    print(f"  轨迹已保存: {trajectory_file}")

        # 统计信息
        stats = {
            'episode_id': episode_id,
            'steps': steps,
            'total_reward': total_reward,
            'outcome': outcome,
            'collision': collision_occurred,
            'episode_time': episode_time,
            'trajectory_file': str(trajectory_file),
            'scenario_name': scenario_name or 'default',
        }

        if self.verbose:
            print(f"  完成时间: {episode_time:.2f}s")
            print(f"  平均步时: {episode_time/max(steps, 1):.3f}s/step")

        return stats


def create_planner_config(args) -> C2OSRPlannerConfig:
    """根据命令行参数创建规划器配置"""

    # 应用配置预设
    if args.config_preset == "fast":
        # 快速测试配置
        lattice_config = LatticePlannerConfig(
            lateral_offsets=[-2.0, 0.0, 2.0],
            speed_variations=[4.0],
            dt=args.dt,
            horizon=args.horizon,
        )
        q_value_config = QValueConfig(
            n_samples=20,
            horizon=args.horizon,
        )
    elif args.config_preset == "high-precision":
        # 高精度配置
        lattice_config = LatticePlannerConfig(
            lateral_offsets=[-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0],
            speed_variations=[3.0, 4.0, 5.0],
            dt=args.dt,
            horizon=args.horizon,
        )
        q_value_config = QValueConfig(
            n_samples=100,
            horizon=args.horizon,
        )
    else:
        # 默认配置
        lattice_config = LatticePlannerConfig(
            lateral_offsets=[-3.0, -2.0, 0.0, 2.0, 3.0],
            speed_variations=[4.0],
            dt=args.dt,
            horizon=args.horizon,
        )
        q_value_config = QValueConfig(
            n_samples=50,
            horizon=args.horizon,
        )

    # 创建配置
    config = C2OSRPlannerConfig(
        horizon=args.horizon,
        grid=GridConfig(
            grid_size_m=args.grid_size,
            cell_size_m=0.5,
            x_min=-args.grid_size,
            x_max=args.grid_size,
            y_min=-args.grid_size,
            y_max=args.grid_size,
        ),
        dirichlet=DirichletConfig(
            alpha_in=50.0,
            alpha_out=1e-6,
        ),
        lattice=lattice_config,
        q_value=q_value_config,
    )

    return config


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="C2OSR + CARLA集成实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本运行
  python run_c2osr_carla.py

  # 指定场景
  python run_c2osr_carla.py --scenario oncoming_medium

  # 自定义配置
  python run_c2osr_carla.py --town Town04 --num-vehicles 20 --episodes 5

  # 查看所有可用场景
  python run_c2osr_carla.py --list-scenarios

可用场景: oncoming_easy, oncoming_medium, oncoming_hard, lane_change_left,
          lane_change_right, overtake, intersection, multi_agent, highway
        """
    )

    # 基本运行参数
    parser.add_argument("--episodes", type=int, default=5,
                       help="执行episode数")
    parser.add_argument("--max-steps", type=int, default=500,
                       help="每个episode的最大步数")
    parser.add_argument("--seed", type=int, default=2025,
                       help="随机种子")

    # CARLA连接参数
    parser.add_argument("--host", type=str, default="localhost",
                       help="CARLA服务器地址")
    parser.add_argument("--port", type=int, default=2000,
                       help="CARLA服务器端口")

    # CARLA场景参数
    parser.add_argument("--town", type=str, default="Town03",
                       help="CARLA地图名称 (Town01-Town10)")
    parser.add_argument("--scenario", type=str, default=None,
                       help="预定义场景名称")
    parser.add_argument("--list-scenarios", action="store_true",
                       help="列出所有可用场景并退出")
    parser.add_argument("--num-vehicles", type=int, default=10,
                       help="环境车辆数量")
    parser.add_argument("--num-pedestrians", type=int, default=5,
                       help="行人数量")
    parser.add_argument("--no-rendering", action="store_true",
                       help="禁用CARLA渲染（提升性能）")

    # C2OSR配置参数
    parser.add_argument("--config-preset",
                       choices=["default", "fast", "high-precision"],
                       default="default",
                       help="配置预设")
    parser.add_argument("--horizon", type=int, default=10,
                       help="规划时域（步数）")
    parser.add_argument("--dt", type=float, default=0.5,
                       help="时间步长（秒）")
    parser.add_argument("--grid-size", type=float, default=50.0,
                       help="网格大小（米）")

    # 输出参数
    parser.add_argument("--output-dir", type=str,
                       default="outputs/c2osr_carla",
                       help="输出目录")
    parser.add_argument("--save-trajectory", action="store_true",
                       default=True,
                       help="保存轨迹数据")
    parser.add_argument("--quiet", action="store_true",
                       help="静默模式（减少输出）")

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_arguments()

    # 处理--list-scenarios
    if args.list_scenarios:
        print("\n可用场景:")
        print("="*60)
        for scenario_name in list_scenarios():
            scenario = get_scenario(scenario_name)
            print(f"\n{scenario_name}:")
            print(f"  描述: {scenario.description}")
            print(f"  难度: {scenario.difficulty}")
            print(f"  路径模式: {scenario.reference_path_mode}")
        print("\n")
        return

    # 打印配置
    print(f"\n{'='*70}")
    print(f" C2OSR + CARLA 集成实验")
    print(f"{'='*70}")
    print(f"\nCARLA配置:")
    print(f"  服务器: {args.host}:{args.port}")
    print(f"  地图: {args.town}")
    print(f"  场景: {args.scenario or '默认'}")
    print(f"  车辆数: {args.num_vehicles}")
    print(f"  行人数: {args.num_pedestrians}")
    print(f"  渲染: {'禁用' if args.no_rendering else '启用'}")

    print(f"\nC2OSR配置:")
    print(f"  Episodes: {args.episodes}")
    print(f"  最大步数: {args.max_steps}")
    print(f"  Horizon: {args.horizon}, dt: {args.dt}s")
    print(f"  配置预设: {args.config_preset}")
    print(f"  网格大小: {args.grid_size}m")
    print(f"  种子: {args.seed}")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n输出目录: {output_dir}")

    # 创建环境
    print(f"\n正在连接CARLA服务器...")
    try:
        env = CarlaEnvironment(
            host=args.host,
            port=args.port,
            town=args.town,
            dt=args.dt,
            max_episode_steps=args.max_steps,
            num_vehicles=args.num_vehicles,
            num_pedestrians=args.num_pedestrians,
        )
        print(f"✓ 成功连接到CARLA服务器")
    except Exception as e:
        print(f"✗ 连接CARLA失败: {e}")
        print(f"\n请确保CARLA服务器已启动：")
        print(f"  cd /path/to/CARLA")
        print(f"  ./CarlaUE4.sh")
        return

    # 创建规划器
    print(f"\n创建C2OSR规划器...")
    planner_config = create_planner_config(args)
    planner = create_c2osr_planner(planner_config)
    print(f"✓ 规划器创建完成")

    # 创建运行器
    runner = CarlaEpisodeRunner(
        planner=planner,
        env=env,
        output_dir=output_dir,
        verbose=not args.quiet,
        save_trajectory=args.save_trajectory,
    )

    # 运行episodes
    print(f"\n{'='*70}")
    print(f"开始运行 {args.episodes} 个episodes")
    print(f"{'='*70}")

    all_stats = []
    total_start_time = time.time()

    for episode_id in range(args.episodes):
        episode_seed = args.seed + episode_id if args.seed else None

        try:
            stats = runner.run_episode(
                episode_id=episode_id,
                max_steps=args.max_steps,
                scenario_name=args.scenario,
                seed=episode_seed,
            )
            all_stats.append(stats)

        except KeyboardInterrupt:
            print(f"\n\n用户中断，提前结束")
            break
        except Exception as e:
            print(f"\n✗ Episode {episode_id} 失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    total_time = time.time() - total_start_time

    # 打印总结
    if len(all_stats) > 0:
        print(f"\n{'='*70}")
        print(f" 实验总结")
        print(f"{'='*70}")

        total_episodes = len(all_stats)
        successful = sum(1 for s in all_stats if s['outcome'] == 'success')
        collisions = sum(1 for s in all_stats if s['collision'])
        avg_reward = np.mean([s['total_reward'] for s in all_stats])
        avg_steps = np.mean([s['steps'] for s in all_stats])

        print(f"\n完成Episodes: {total_episodes}/{args.episodes}")
        print(f"成功: {successful} ({successful/total_episodes*100:.1f}%)")
        print(f"碰撞: {collisions} ({collisions/total_episodes*100:.1f}%)")
        print(f"平均奖励: {avg_reward:.2f}")
        print(f"平均步数: {avg_steps:.1f}")
        print(f"总用时: {total_time:.1f}s")
        print(f"平均episode用时: {total_time/total_episodes:.1f}s")

        # 保存统计数据
        stats_file = output_dir / "experiment_stats.npy"
        np.save(stats_file, all_stats)
        print(f"\n统计数据已保存: {stats_file}")

    # 清理
    print(f"\n清理资源...")
    env.close()
    print(f"✓ 完成")


if __name__ == "__main__":
    main()
