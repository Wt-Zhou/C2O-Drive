# carla_c2osr

C2OSR-Drive: 基于狄利克雷过程 (DP) 的部署期自我优化驾驶 (Deployment-time learning) 验证仓库骨架。

快速开始：

```bash
conda env create -f environment.yml
conda activate c2o-drive
python -m pip install -U pip
pytest -q               # 先跑占位测试
python runner/run_sim.py --mock-env  # 跑一圈假环境
```

### 目录结构

- `docs/architecture.md`: 总体架构、数据模型与接口签名、规划与推断流程。
- `configs/`: 实验与仿真配置（`sim.yaml`、`lattice.yaml`、`baselines.yaml`、`scenarios.yaml`）。
- `env/`: 与 CARLA 或 mock 环境对接的封装与类型定义。
- `agents/`: 智能体实现。`agents/c2osr/` 为 DP + 保守规划主方法；`agents/baselines/` 为基线占位。
- `evaluation/`: 指标、日志与可视化占位实现。
- `utils/`: 通用工具（几何、随机种子、计时）。
- `runner/`: 可执行脚本（仿真、基线训练、开放环回放）。
- `tests/`: 轻量占位测试，确保骨架可运行。

### 状态与假设

- 本仓库当前不安装 CARLA，仅提供接口与 mock 环境，便于先行调试与 CI。
- DP 混合模块与规划器为最小可运行占位实现，重点是清晰的接口、类型与文档；复杂数学细节标注 TODO。

### 许可证

学术研究优先，后续可按项目需要添加具体 LICENSE。
