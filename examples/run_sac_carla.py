#!/usr/bin/env python3
"""
SAC + CARLA 集成训练脚本

将SAC算法与CARLA仿真环境集成,支持真实的3D仿真场景训练。

特性:
- 使用SACAgent（标准化强化学习算法）
- 使用CarlaEnvironment（Gym标准接口）
- 支持预定义场景库
- TensorBoard日志记录
- 模型检查点保存与加载
- 评估模式

使用前提:
- CARLA服务器必须已经启动（默认localhost:2000）
- 确保carla python包已安装或路径已配置
- 建议使用GPU加速训练（--device cuda）
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import time
import torch
from torch.utils.tensorboard import SummaryWriter

# 添加项目根目录到路径
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

_src_path = _repo_root / "src"
if _src_path.exists() and str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

# SAC算法组件
from c2o_drive.algorithms.sac.agent import SACAgent
from c2o_drive.algorithms.sac.config import SACConfig

# CARLA环境组件
from c2o_drive.environments.carla_env import CarlaEnvironment
from c2o_drive.environments.carla.scenarios import get_scenario, list_scenarios, CarlaScenarioLibrary
from c2o_drive.config import get_global_config
from c2o_drive.core.types import EgoControl

# Lattice Planner组件
from c2o_drive.utils.lattice_planner import LatticePlanner


class SACTrainer:
    """SAC训练器"""

    def __init__(
        self,
        agent: SACAgent,
        env: CarlaEnvironment,
        config: SACConfig,
        output_dir: Path,
        log_dir: Path,
        checkpoint_dir: Path,
        lattice_planner: LatticePlanner,
        horizon: int = 8,
        dt: float = 1.0,
        verbose: bool = True,
    ):
        self.agent = agent
        self.env = env
        self.config = config
        self.output_dir = output_dir
        self.verbose = verbose
        self.lattice_planner = lattice_planner
        self.horizon = horizon
        self.dt = dt

        # 创建目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(log_dir))

        # 训练统计
        self.total_steps = 0
        self.episode_count = 0

        # 动作范围配置（从config读取）
        gc = get_global_config()
        self.lateral_offset_range = gc.baseline.sac_lateral_offset_range
        self.target_speed_range = gc.baseline.sac_target_speed_range

    def train_episode(
        self,
        episode_id: int,
        max_steps: int,
        scenario_name: Optional[str] = None,
        seed: Optional[int] = None,
        training: bool = True,
    ) -> Dict[str, Any]:
        """运行单个训练episode

        Args:
            episode_id: Episode ID
            max_steps: 最大步数
            scenario_name: 场景名称（可选）
            seed: 随机种子
            training: 是否为训练模式（False则为评估模式）

        Returns:
            Episode统计数据字典
        """
        if self.verbose:
            mode_str = "训练" if training else "评估"
            print(f"\n{'='*60}")
            print(f"Episode {episode_id} ({mode_str})")
            if scenario_name:
                print(f"场景: {scenario_name}")
            print(f"{'='*60}")

        episode_start_time = time.time()

        # 重置环境（与C2OSR对齐）
        options = {}
        reference_path = None
        if scenario_name:
            scenario_def = get_scenario(scenario_name)
            options['scenario_config'] = {
                'scenario': scenario_def,
                'scenario_name': scenario_name,
            }
            # 生成 reference_path（关键！用于环境车生成）
            reference_path = CarlaScenarioLibrary.get_reference_path(
                scenario_def,
                horizon=self.horizon,
                dt=self.dt,
            )
            options['reference_path'] = reference_path
            if self.verbose:
                print(f"场景: {scenario_def.description}")
                print(f"难度: {scenario_def.difficulty}")

        state, info = self.env.reset(seed=seed, options=options)

        # 获取参考路径
        reference_path = info.get('reference_path', [])
        if reference_path:
            reference_path = [(float(p[0]), float(p[1])) for p in reference_path]

        # 提取初始状态特征
        initial_state_features = self.agent._extract_state_features(state)

        # SAC 输出归一化动作 [-1, 1]（只在episode开始时生成一次，与C2OSR对齐）
        action = self.agent.select_action(initial_state_features, training=training)

        # Rescale 到实际的 lattice 参数范围
        lateral_offset, target_speed = self.rescale_action(action)

        if self.verbose:
            print(f"  选择动作: lateral_offset={lateral_offset:.2f}m, target_speed={target_speed:.2f}m/s")

        # 使用 lattice planner 生成轨迹（只生成一次）
        trajectory = self.generate_trajectory(
            state,
            lateral_offset,
            target_speed,
            reference_path,
        )

        if trajectory is None:
            if self.verbose:
                print(f"  ✗ 轨迹生成失败")
            return {
                'episode_id': episode_id,
                'steps': 0,
                'total_reward': 0.0,
                'outcome': 'planning_failed',
                'collision': False,
                'episode_time': 0.0,
                'scenario_name': scenario_name or 'default',
            }

        # 调整 max_steps 为轨迹长度（与C2OSR对齐）
        num_waypoints = len(trajectory.waypoints)
        if num_waypoints < 2:
            if self.verbose:
                print("  ✗ 选中轨迹缺少waypoints")
            return {
                'episode_id': episode_id,
                'steps': 0,
                'total_reward': 0.0,
                'outcome': 'planning_failed',
                'collision': False,
                'episode_time': 0.0,
                'scenario_name': scenario_name or 'default',
            }

        if max_steps > num_waypoints - 1:
            if self.verbose:
                print(f"  调整: max_steps从{max_steps}调整为{num_waypoints - 1}（轨迹长度限制）")
            max_steps = num_waypoints - 1

        # Episode统计
        episode_reward = 0.0
        episode_steps = 0
        episode_losses = []
        collision_occurred = False
        outcome = 'success'

        # 执行轨迹的每一步（与C2OSR对齐）
        for step in range(max_steps):
            # 转换轨迹当前步为控制命令
            control = self.trajectory_to_control(state, trajectory, step)

            # 执行动作
            step_result = self.env.step(control)

            # 提取当前状态特征（用于经验存储）
            current_state_features = self.agent._extract_state_features(state)

            # 提取下一状态特征
            next_state_features = self.agent._extract_state_features(step_result.observation)

            # 存储经验（仅在训练模式）
            # 注意：同一个动作(lateral_offset, target_speed)用于整条轨迹的所有步骤
            # 这样agent可以学习到选择某个动作后会经历的所有状态转换
            if training:
                done = step_result.terminated or step_result.truncated
                self.agent.store_transition(
                    current_state_features,
                    action,
                    step_result.reward,
                    next_state_features,
                    done,
                )
                self.total_steps += 1

            # 更新统计
            episode_reward += step_result.reward
            episode_steps += 1
            state = step_result.observation

            if self.verbose and (step + 1) % 10 == 0:
                print(f"  Step {step+1}/{max_steps}: reward={step_result.reward:.2f}, "
                      f"total={episode_reward:.2f}")

            # 检查终止条件
            if step_result.terminated:
                outcome = 'collision'
                collision_occurred = True
                if self.verbose:
                    print(f"  ✗ 碰撞！Episode在第{episode_steps}步结束")
                break

            if step_result.truncated:
                outcome = 'timeout'
                if self.verbose:
                    print(f"  ⏱ Episode在第{episode_steps}步因truncated结束")
                break

        # Episode结束后训练（仅训练模式）
        if training and self.agent.replay_buffer.is_ready(self.config.batch_size):
            if self.verbose:
                print(f"\n  开始训练网络...")

            # 从BaselineConfig读取updates_per_episode
            gc = get_global_config()
            num_updates = gc.baseline.sac_updates_per_episode

            for update_idx in range(num_updates):
                metrics = self.agent.train_step()
                if metrics:
                    episode_losses.append(metrics)

                    # 每10次更新记录一次
                    if (update_idx + 1) % 10 == 0 and self.verbose:
                        print(f"    Update {update_idx+1}/{num_updates}: "
                              f"critic_loss={metrics['critic_loss']:.3f}, "
                              f"actor_loss={metrics['actor_loss']:.3f}, "
                              f"alpha={metrics['alpha']:.3f}")

        episode_time = time.time() - episode_start_time

        if outcome == 'success' and self.verbose:
            print(f"  ✓ 成功完成{episode_steps}步！")

        # 统计信息
        stats = {
            'episode_id': episode_id,
            'steps': episode_steps,
            'total_reward': episode_reward,
            'avg_reward': episode_reward / max(episode_steps, 1),
            'outcome': outcome,
            'collision': collision_occurred,
            'episode_time': episode_time,
            'scenario_name': scenario_name or 'default',
        }

        # 添加训练指标
        if episode_losses:
            stats['avg_critic_loss'] = np.mean([m['critic_loss'] for m in episode_losses])
            stats['avg_actor_loss'] = np.mean([m['actor_loss'] for m in episode_losses])
            stats['avg_alpha'] = np.mean([m['alpha'] for m in episode_losses])
            stats['avg_q_value'] = np.mean([m['q_mean'] for m in episode_losses])

        if self.verbose:
            print(f"  Episode时间: {episode_time:.2f}s")
            print(f"  平均步时: {episode_time/max(episode_steps, 1):.3f}s/step")

        return stats

    def log_episode(self, stats: Dict[str, Any], mode: str = "train"):
        """记录episode统计到TensorBoard

        Args:
            stats: Episode统计数据
            mode: 模式（train/eval）
        """
        episode_id = stats['episode_id']

        # 基础指标
        self.writer.add_scalar(f'{mode}/episode_reward', stats['total_reward'], episode_id)
        self.writer.add_scalar(f'{mode}/episode_steps', stats['steps'], episode_id)
        self.writer.add_scalar(f'{mode}/avg_reward', stats['avg_reward'], episode_id)
        self.writer.add_scalar(f'{mode}/collision', 1.0 if stats['collision'] else 0.0, episode_id)

        # 训练指标
        if 'avg_critic_loss' in stats:
            self.writer.add_scalar(f'{mode}/critic_loss', stats['avg_critic_loss'], episode_id)
            self.writer.add_scalar(f'{mode}/actor_loss', stats['avg_actor_loss'], episode_id)
            self.writer.add_scalar(f'{mode}/alpha', stats['avg_alpha'], episode_id)
            self.writer.add_scalar(f'{mode}/q_value', stats['avg_q_value'], episode_id)

        # Replay buffer状态
        buffer_size = len(self.agent.replay_buffer)
        self.writer.add_scalar('train/buffer_size', buffer_size, episode_id)

    def save_checkpoint(self, episode_id: int):
        """保存模型检查点

        Args:
            episode_id: Episode ID
        """
        checkpoint_path = self.checkpoint_dir / f"sac_episode_{episode_id}.pth"
        self.agent.save(str(checkpoint_path))
        if self.verbose:
            print(f"  ✓ 保存检查点: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """加载模型检查点

        Args:
            checkpoint_path: 检查点文件路径
        """
        self.agent.load(checkpoint_path)
        if self.verbose:
            print(f"  ✓ 加载检查点: {checkpoint_path}")

    def rescale_action(self, action: np.ndarray) -> Tuple[float, float]:
        """将SAC输出的[-1,1]动作rescale到实际范围

        Args:
            action: SAC输出的归一化动作 [lateral_offset_norm, target_speed_norm]

        Returns:
            (lateral_offset, target_speed) in meters and m/s
        """
        # Rescale lateral_offset from [-1, 1] to range
        lateral_min, lateral_max = self.lateral_offset_range
        lateral_offset = (action[0] + 1.0) / 2.0 * (lateral_max - lateral_min) + lateral_min

        # Rescale target_speed from [-1, 1] to range
        speed_min, speed_max = self.target_speed_range
        target_speed = (action[1] + 1.0) / 2.0 * (speed_max - speed_min) + speed_min

        return float(lateral_offset), float(target_speed)

    def generate_trajectory(
        self,
        state,
        lateral_offset: float,
        target_speed: float,
        reference_path: Optional[List] = None,
    ):
        """使用lattice planner生成轨迹

        Args:
            state: 当前WorldState
            lateral_offset: 横向偏移（米）
            target_speed: 目标速度（m/s）
            reference_path: 参考路径

        Returns:
            LatticeTrajectory对象
        """
        # 准备参考路径
        if reference_path is None:
            ego_x, ego_y = state.ego.position_m
            reference_path = [
                (ego_x + i * 5.0, ego_y)
                for i in range(self.horizon + 1)
            ]

        # 生成单条轨迹（使用指定的参数）
        ego_state_tuple = (
            state.ego.position_m[0],
            state.ego.position_m[1],
            state.ego.yaw_rad,
        )

        # 临时创建单轨迹planner
        temp_planner = LatticePlanner(
            lateral_offsets=[lateral_offset],
            speed_variations=[target_speed],
            num_trajectories=1
        )

        trajectories = temp_planner.generate_trajectories(
            reference_path=reference_path,
            horizon=self.horizon,
            dt=self.dt,
            ego_state=ego_state_tuple,
        )

        return trajectories[0] if trajectories else None

    def trajectory_to_control(
        self,
        current_state,
        trajectory,
        step_idx: int,
    ) -> EgoControl:
        """将轨迹waypoint转换为控制指令（复用C2OSR的逻辑）

        Args:
            current_state: 当前状态
            trajectory: LatticeTrajectory对象
            step_idx: 当前步骤索引

        Returns:
            EgoControl对象
        """
        if step_idx + 1 >= len(trajectory.waypoints):
            return EgoControl(throttle=0.0, steer=0.0, brake=1.0)

        target_x, target_y = trajectory.waypoints[step_idx + 1]
        current_x, current_y = current_state.ego.position_m

        dx = target_x - current_x
        dy = target_y - current_y
        target_heading = np.arctan2(dy, dx)
        heading_error = target_heading - current_state.ego.yaw_rad
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

        steer = np.clip(heading_error * 0.5, -1.0, 1.0)

        current_speed = np.linalg.norm(np.array(current_state.ego.velocity_mps))
        speed_error = trajectory.target_speed - current_speed

        if speed_error > 0.5:
            throttle = 0.6
            brake = 0.0
        elif speed_error < -0.5:
            throttle = 0.0
            brake = 0.5
        else:
            throttle = 0.3
            brake = 0.0

        return EgoControl(throttle=throttle, steer=steer, brake=brake)

    def close(self):
        """关闭训练器"""
        self.writer.close()


def create_sac_agent(args) -> SACAgent:
    """根据命令行参数和配置创建SAC agent"""
    gc = get_global_config()
    baseline_config = gc.baseline

    # 创建SAC配置
    sac_config = SACConfig(
        state_dim=baseline_config.sac_state_dim,
        action_dim=baseline_config.sac_action_dim,
        max_action=baseline_config.sac_max_action,
        hidden_dims=baseline_config.sac_hidden_dims,
        learning_rate=baseline_config.sac_learning_rate,
        gamma=baseline_config.sac_gamma,
        tau=baseline_config.sac_tau,
        initial_alpha=baseline_config.sac_alpha,
        batch_size=baseline_config.sac_batch_size,
        buffer_size=baseline_config.sac_buffer_size,
        device=args.device,
    )

    # 创建agent
    agent = SACAgent(sac_config)

    return agent


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="SAC + CARLA集成训练",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本训练
  python run_sac_carla.py --episodes 1000 --device cuda

  # 指定场景训练
  python run_sac_carla.py --scenario s4_wrong_way --episodes 500

  # 评估模式
  python run_sac_carla.py --eval --load-checkpoint checkpoints/sac_carla/sac_episode_500.pth

  # 查看所有可用场景
  python run_sac_carla.py --list-scenarios
        """
    )

    # 基本训练参数
    parser.add_argument("--episodes", type=int, default=None,
                       help="训练episodes数（默认读取config）")
    parser.add_argument("--max-steps", type=int, default=None,
                       help="每个episode的最大步数（默认读取config）")
    parser.add_argument("--seed", type=int, default=2025,
                       help="随机种子")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"],
                       help="训练设备")

    # CARLA连接参数
    parser.add_argument("--host", type=str, default="localhost",
                       help="CARLA服务器地址")
    parser.add_argument("--port", type=int, default=2000,
                       help="CARLA服务器端口")

    # CARLA场景参数
    parser.add_argument("--town", type=str, default="Town03",
                       help="CARLA地图名称 (Town01-Town10)")
    parser.add_argument("--scenario", type=str, default="s4_wrong_way",
                       help="预定义场景名称（默认: s4_wrong_way）")
    parser.add_argument("--list-scenarios", action="store_true",
                       help="列出所有可用场景并退出")
    parser.add_argument("--num-vehicles", type=int, default=10,
                       help="环境车辆数量")
    parser.add_argument("--num-pedestrians", type=int, default=5,
                       help="行人数量")
    parser.add_argument("--no-rendering", action="store_true",
                       help="禁用CARLA渲染（提升性能）")

    # 检查点和评估参数
    parser.add_argument("--save-interval", type=int, default=None,
                       help="模型保存间隔（episodes）")
    parser.add_argument("--eval-interval", type=int, default=None,
                       help="评估间隔（episodes）")
    parser.add_argument("--eval-episodes", type=int, default=None,
                       help="每次评估的episodes数")
    parser.add_argument("--eval", action="store_true",
                       help="评估模式（不训练）")
    parser.add_argument("--load-checkpoint", type=str, default=None,
                       help="加载检查点文件路径")

    # 输出参数
    parser.add_argument("--output-dir", type=str, default=None,
                       help="输出目录（默认读取config）")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                       help="检查点目录（默认读取config）")
    parser.add_argument("--log-dir", type=str, default=None,
                       help="TensorBoard日志目录（默认读取config）")
    parser.add_argument("--quiet", action="store_true",
                       help="静默模式（减少输出）")

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_arguments()

    # 加载全局配置
    gc = get_global_config()
    baseline_config = gc.baseline

    # 设置默认值（从config读取）
    if args.episodes is None:
        args.episodes = baseline_config.sac_num_episodes
    if args.max_steps is None:
        args.max_steps = baseline_config.sac_max_steps_per_episode
    if args.save_interval is None:
        args.save_interval = baseline_config.sac_save_interval
    if args.eval_interval is None:
        args.eval_interval = baseline_config.sac_eval_interval
    if args.eval_episodes is None:
        args.eval_episodes = baseline_config.sac_eval_episodes
    if args.output_dir is None:
        args.output_dir = "outputs/sac_carla"
    if args.checkpoint_dir is None:
        args.checkpoint_dir = baseline_config.sac_checkpoint_dir
    if args.log_dir is None:
        args.log_dir = baseline_config.sac_log_dir

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
    print(f" SAC + CARLA 集成训练")
    print(f"{'='*70}")
    print(f"\nCARLA配置:")
    print(f"  服务器: {args.host}:{args.port}")
    print(f"  地图: {args.town}")
    print(f"  场景: {args.scenario or '默认'}")
    print(f"  车辆数: {args.num_vehicles}")
    print(f"  行人数: {args.num_pedestrians}")
    print(f"  渲染: {'禁用' if args.no_rendering else '启用'}")

    print(f"\nSAC配置:")
    print(f"  模式: {'评估' if args.eval else '训练'}")
    print(f"  Episodes: {args.episodes}")
    print(f"  最大步数/episode: {args.max_steps}")
    print(f"  设备: {args.device}")
    print(f"  批次大小: {baseline_config.sac_batch_size}")
    print(f"  学习率: {baseline_config.sac_learning_rate}")
    print(f"  Gamma: {baseline_config.sac_gamma}")
    print(f"  种子: {args.seed}")

    if not args.eval:
        print(f"\n训练配置:")
        print(f"  保存间隔: {args.save_interval} episodes")
        print(f"  评估间隔: {args.eval_interval} episodes")
        print(f"  每次评估: {args.eval_episodes} episodes")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n输出目录:")
    print(f"  输出: {output_dir}")
    print(f"  检查点: {checkpoint_dir}")
    print(f"  日志: {log_dir}")

    # 检查CUDA可用性
    if args.device == "cuda" and not torch.cuda.is_available():
        print(f"\n⚠ CUDA不可用，切换到CPU")
        args.device = "cpu"

    # 创建环境
    print(f"\n正在连接CARLA服务器...")
    try:
        env = CarlaEnvironment(
            host=args.host,
            port=args.port,
            town=args.town,
            dt=gc.time.dt,
            max_episode_steps=args.max_steps,
            num_vehicles=args.num_vehicles,
            num_pedestrians=args.num_pedestrians,
            no_rendering=args.no_rendering,
        )
        print(f"✓ 成功连接到CARLA服务器")
    except Exception as e:
        print(f"✗ 连接CARLA失败: {e}")
        print(f"\n请确保CARLA服务器已启动：")
        print(f"  cd /path/to/CARLA")
        print(f"  ./CarlaUE4.sh")
        return

    # 创建SAC agent
    print(f"\n创建SAC agent...")
    agent = create_sac_agent(args)
    print(f"✓ SAC agent创建完成")
    print(f"  状态维度: {baseline_config.sac_state_dim}")
    print(f"  动作维度: {baseline_config.sac_action_dim}")
    print(f"  网络结构: {baseline_config.sac_hidden_dims}")
    print(f"  动作空间: [lateral_offset: {baseline_config.sac_lateral_offset_range}, "
          f"target_speed: {baseline_config.sac_target_speed_range}]")

    # 创建Lattice Planner（用于将SAC的动作转换为轨迹）
    print(f"\n创建Lattice Planner...")
    lattice_planner = LatticePlanner(
        lateral_offsets=gc.lattice.lateral_offsets,
        speed_variations=gc.lattice.speed_variations,
        num_trajectories=len(gc.lattice.lateral_offsets) * len(gc.lattice.speed_variations),
    )
    print(f"✓ Lattice Planner创建完成")
    print(f"  Lateral offsets: {gc.lattice.lateral_offsets}")
    print(f"  Speed variations: {gc.lattice.speed_variations}")

    # 创建训练器
    trainer = SACTrainer(
        agent=agent,
        env=env,
        config=agent.config,
        output_dir=output_dir,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        lattice_planner=lattice_planner,
        horizon=gc.time.default_horizon,
        dt=gc.time.dt,
        verbose=not args.quiet,
    )

    # 加载检查点（如果指定）
    if args.load_checkpoint:
        print(f"\n加载检查点: {args.load_checkpoint}")
        trainer.load_checkpoint(args.load_checkpoint)

    # 评估模式
    if args.eval:
        print(f"\n{'='*70}")
        print(f"开始评估 {args.eval_episodes} 个episodes")
        print(f"{'='*70}")

        eval_stats = []
        for episode_id in range(args.eval_episodes):
            episode_seed = args.seed + episode_id if args.seed else None

            try:
                stats = trainer.train_episode(
                    episode_id=episode_id,
                    max_steps=args.max_steps,
                    scenario_name=args.scenario,
                    seed=episode_seed,
                    training=False,
                )
                eval_stats.append(stats)
                trainer.log_episode(stats, mode="eval")

            except KeyboardInterrupt:
                print(f"\n\n用户中断，提前结束")
                break
            except Exception as e:
                print(f"\n✗ Episode {episode_id} 失败: {e}")
                import traceback
                traceback.print_exc()
                continue

        # 打印评估总结
        if eval_stats:
            print(f"\n{'='*70}")
            print(f" 评估总结")
            print(f"{'='*70}")
            avg_reward = np.mean([s['total_reward'] for s in eval_stats])
            avg_steps = np.mean([s['steps'] for s in eval_stats])
            success_rate = sum(1 for s in eval_stats if s['outcome'] == 'success') / len(eval_stats)
            collision_rate = sum(1 for s in eval_stats if s['collision']) / len(eval_stats)

            print(f"\nEpisodes: {len(eval_stats)}")
            print(f"平均奖励: {avg_reward:.2f}")
            print(f"平均步数: {avg_steps:.1f}")
            print(f"成功率: {success_rate*100:.1f}%")
            print(f"碰撞率: {collision_rate*100:.1f}%")

    # 训练模式
    else:
        print(f"\n{'='*70}")
        print(f"开始训练 {args.episodes} 个episodes")
        print(f"{'='*70}")

        all_stats = []
        total_start_time = time.time()

        for episode_id in range(args.episodes):
            episode_seed = args.seed + episode_id if args.seed else None

            try:
                # 训练episode
                stats = trainer.train_episode(
                    episode_id=episode_id,
                    max_steps=args.max_steps,
                    scenario_name=args.scenario,
                    seed=episode_seed,
                    training=True,
                )
                all_stats.append(stats)
                trainer.log_episode(stats, mode="train")

                # 保存检查点
                if (episode_id + 1) % args.save_interval == 0:
                    trainer.save_checkpoint(episode_id + 1)

                # 定期评估
                if (episode_id + 1) % args.eval_interval == 0:
                    print(f"\n{'='*60}")
                    print(f"运行评估 (Episode {episode_id + 1})")
                    print(f"{'='*60}")

                    eval_stats = []
                    for eval_id in range(args.eval_episodes):
                        eval_seed = args.seed + 10000 + eval_id if args.seed else None
                        eval_stat = trainer.train_episode(
                            episode_id=episode_id,
                            max_steps=args.max_steps,
                            scenario_name=args.scenario,
                            seed=eval_seed,
                            training=False,
                        )
                        eval_stats.append(eval_stat)

                    # 记录评估结果
                    avg_eval_reward = np.mean([s['total_reward'] for s in eval_stats])
                    trainer.writer.add_scalar('eval/avg_reward', avg_eval_reward, episode_id)

                    if not args.quiet:
                        print(f"\n  评估平均奖励: {avg_eval_reward:.2f}")

            except KeyboardInterrupt:
                print(f"\n\n用户中断，提前结束")
                break
            except Exception as e:
                print(f"\n✗ Episode {episode_id} 失败: {e}")
                import traceback
                traceback.print_exc()
                continue

        total_time = time.time() - total_start_time

        # 打印训练总结
        if len(all_stats) > 0:
            print(f"\n{'='*70}")
            print(f" 训练总结")
            print(f"{'='*70}")

            total_episodes = len(all_stats)
            avg_reward = np.mean([s['total_reward'] for s in all_stats])
            avg_steps = np.mean([s['steps'] for s in all_stats])
            success_rate = sum(1 for s in all_stats if s['outcome'] == 'success') / total_episodes
            collision_rate = sum(1 for s in all_stats if s['collision']) / total_episodes

            print(f"\n完成Episodes: {total_episodes}/{args.episodes}")
            print(f"平均奖励: {avg_reward:.2f}")
            print(f"平均步数: {avg_steps:.1f}")
            print(f"成功率: {success_rate*100:.1f}%")
            print(f"碰撞率: {collision_rate*100:.1f}%")
            print(f"总用时: {total_time:.1f}s")
            print(f"平均episode用时: {total_time/total_episodes:.1f}s")
            print(f"Buffer大小: {len(agent.replay_buffer)}")

        # 保存最终模型
        trainer.save_checkpoint('final')
        print(f"\n✓ 保存最终模型")

    # 清理
    print(f"\n清理资源...")
    trainer.close()
    env.close()
    print(f"✓ 完成")


if __name__ == "__main__":
    main()
