"""
训练状态的Checkpoint管理器

支持保存和加载完整的训练状态，包括：
- TrajectoryBuffer（历史轨迹数据）
- DirichletBank（学习到的先验分布）
- QDistributionTracker（Q值分布历史）
- 训练进度和随机数生成器状态
"""

from __future__ import annotations
import json
import hashlib
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime


class CheckpointManager:
    """训练checkpoint管理器"""

    CHECKPOINT_VERSION = "1.0.0"

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        """初始化checkpoint管理器

        Args:
            checkpoint_dir: checkpoint保存的根目录
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self,
                       episode_id: int,
                       trajectory_buffer,
                       dirichlet_bank,
                       q_tracker,
                       config: Dict[str, Any],
                       rng_state: Optional[Dict] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> Path:
        """保存完整的训练checkpoint

        Args:
            episode_id: 当前episode编号
            trajectory_buffer: TrajectoryBuffer实例
            dirichlet_bank: DirichletBank实例
            q_tracker: QDistributionTracker实例
            config: 全局配置字典
            rng_state: 随机数生成器状态（可选）
            metadata: 额外的元数据（可选）

        Returns:
            checkpoint目录路径
        """
        # 创建checkpoint目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"checkpoint_ep{episode_id:04d}_{timestamp}"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        print(f"\n💾 保存checkpoint: {checkpoint_name}")

        # 1. 保存元数据
        config_hash = self._compute_config_hash(config)
        meta = {
            "version": self.CHECKPOINT_VERSION,
            "episode_id": episode_id,
            "timestamp": timestamp,
            "config_hash": config_hash,
            **(metadata or {})
        }
        with open(checkpoint_path / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        # 2. 保存配置
        with open(checkpoint_path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        # 3. 保存TrajectoryBuffer
        self._save_trajectory_buffer(trajectory_buffer, checkpoint_path)

        # 4. 保存DirichletBank
        self._save_dirichlet_bank(dirichlet_bank, checkpoint_path)

        # 5. 保存QDistributionTracker
        q_tracker.save_data(str(checkpoint_path / "q_tracker.json"))

        # 6. 保存训练状态
        training_state = {
            "episode_id": episode_id,
            "rng_state": rng_state
        }
        with open(checkpoint_path / "training_state.json", "w") as f:
            json.dump(training_state, f, indent=2)

        print(f"✅ Checkpoint已保存到: {checkpoint_path}")
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """加载训练checkpoint

        Args:
            checkpoint_path: checkpoint目录路径

        Returns:
            包含所有状态的字典
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint不存在: {checkpoint_path}")

        print(f"\n📂 加载checkpoint: {checkpoint_path.name}")

        # 1. 加载元数据
        with open(checkpoint_path / "metadata.json", "r") as f:
            metadata = json.load(f)

        version = metadata.get("version", "unknown")
        if version != self.CHECKPOINT_VERSION:
            print(f"⚠️  警告: Checkpoint版本不匹配 (期望 {self.CHECKPOINT_VERSION}, 实际 {version})")

        # 2. 加载配置
        with open(checkpoint_path / "config.json", "r") as f:
            config = json.load(f)

        # 3. 加载TrajectoryBuffer状态
        trajectory_buffer_data = self._load_trajectory_buffer(checkpoint_path)

        # 4. 加载DirichletBank状态
        dirichlet_bank_data = self._load_dirichlet_bank(checkpoint_path)

        # 5. 加载QDistributionTracker状态
        with open(checkpoint_path / "q_tracker.json", "r") as f:
            q_tracker_data = json.load(f)

        # 6. 加载训练状态
        with open(checkpoint_path / "training_state.json", "r") as f:
            training_state = json.load(f)

        print(f"✅ Checkpoint已加载 (Episode {metadata['episode_id']})")

        return {
            "metadata": metadata,
            "config": config,
            "trajectory_buffer_data": trajectory_buffer_data,
            "dirichlet_bank_data": dirichlet_bank_data,
            "q_tracker_data": q_tracker_data,
            "training_state": training_state
        }

    def list_checkpoints(self) -> List[Path]:
        """列出所有可用的checkpoint"""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_ep*"))
        return checkpoints

    def _save_trajectory_buffer(self, buffer, checkpoint_path: Path):
        """保存TrajectoryBuffer状态"""
        buffer_dict = buffer.to_dict()

        # 分离numpy数组和元数据
        numpy_arrays = {}
        metadata = {}

        for key, value in buffer_dict.items():
            if isinstance(value, np.ndarray):
                numpy_arrays[key] = value
            else:
                metadata[key] = value

        # 保存numpy数组
        if numpy_arrays:
            np.savez_compressed(checkpoint_path / "trajectory_buffer.npz", **numpy_arrays)

        # 保存元数据
        with open(checkpoint_path / "trajectory_buffer_meta.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def _load_trajectory_buffer(self, checkpoint_path: Path) -> Dict[str, Any]:
        """加载TrajectoryBuffer状态"""
        # 加载元数据
        with open(checkpoint_path / "trajectory_buffer_meta.json", "r") as f:
            metadata = json.load(f)

        # 加载numpy数组
        npz_path = checkpoint_path / "trajectory_buffer.npz"
        numpy_data = {}
        if npz_path.exists():
            with np.load(npz_path, allow_pickle=True) as data:
                numpy_data = {key: data[key] for key in data.keys()}

        return {**metadata, **numpy_data}

    def _save_dirichlet_bank(self, bank, checkpoint_path: Path):
        """保存DirichletBank状态"""
        bank_dict = bank.to_dict()

        # 保存alpha数组（使用npz压缩格式）
        alpha_arrays = {}
        for agent_id, timesteps in bank_dict["agent_alphas"].items():
            for timestep, alpha in timesteps.items():
                key = f"agent_{agent_id}_t_{timestep}"
                alpha_arrays[key] = alpha

        np.savez_compressed(checkpoint_path / "dirichlet_bank.npz", **alpha_arrays)

        # 保存其他元数据（不包括numpy数组）
        metadata = {
            "K": bank_dict["K"],
            "horizon": bank_dict["horizon"],
            "params": bank_dict["params"],
            "agent_reachable_sets": bank_dict["agent_reachable_sets"]
        }

        with open(checkpoint_path / "dirichlet_bank_meta.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def _load_dirichlet_bank(self, checkpoint_path: Path) -> Dict[str, Any]:
        """加载DirichletBank状态"""
        # 加载元数据
        with open(checkpoint_path / "dirichlet_bank_meta.json", "r") as f:
            metadata = json.load(f)

        # 加载alpha数组
        agent_alphas = {}
        with np.load(checkpoint_path / "dirichlet_bank.npz") as data:
            for key in data.keys():
                # 解析key: "agent_{agent_id}_t_{timestep}"
                parts = key.split("_")
                agent_id = int(parts[1])
                timestep = int(parts[3])

                if agent_id not in agent_alphas:
                    agent_alphas[agent_id] = {}
                agent_alphas[agent_id][timestep] = data[key]

        return {
            **metadata,
            "agent_alphas": agent_alphas
        }

    def _compute_config_hash(self, config: Dict[str, Any]) -> str:
        """计算配置的hash值用于版本控制"""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:16]
