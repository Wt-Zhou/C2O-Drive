#!/usr/bin/env python3
"""
简单CARLA测试 - 逐步诊断
"""

import sys
import os

print("Python版本:", sys.version)
print("工作目录:", os.getcwd())

# 测试1: 基本导入
print("\n[测试1] 基本库导入...")
try:
    import numpy as np
    import glob
    print("  ✓ numpy, glob导入成功")
except Exception as e:
    print(f"  ✗ 失败: {e}")
    sys.exit(1)

# 测试2: 查找CARLA
print("\n[测试2] 查找CARLA路径...")
possible_paths = [
    '../carla/dist/carla-*-py3.*.egg',
    '~/carla/dist/carla-*-py3.*.egg',
    '/opt/carla/PythonAPI/carla/dist/carla-*-py3.*.egg',
]

found_carla = False
for pattern in possible_paths:
    expanded = os.path.expanduser(pattern)
    matches = glob.glob(expanded)
    if matches:
        print(f"  找到CARLA: {matches[0]}")
        found_carla = True
        break

if not found_carla:
    print("  ✗ 未找到CARLA .egg文件")
    print("\n请检查:")
    print("  1. CARLA是否已安装")
    print("  2. 设置环境变量: export CARLA_ROOT=/path/to/CARLA")
    sys.exit(1)

# 测试3: 不导入carla，直接测试项目模块
print("\n[测试3] 测试项目模块（不使用CARLA）...")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from carla_c2osr.env.types import WorldState, EgoState, AgentState, EgoControl
    print("  ✓ 项目类型定义导入成功")
except Exception as e:
    print(f"  ✗ 失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    from carla_c2osr.env.carla_scenarios import list_scenarios, get_scenario
    print("  ✓ 场景库导入成功")
    scenarios = list_scenarios()
    print(f"    可用场景: {len(scenarios)}个")
    for s in scenarios[:3]:
        print(f"      - {s}")
except Exception as e:
    print(f"  ✗ 失败: {e}")
    import traceback
    traceback.print_exc()

try:
    from carla_c2osr.config.global_config import CarlaConfig
    print("  ✓ CarlaConfig导入成功")
    config = CarlaConfig()
    print(f"    默认town: {config.town}")
except Exception as e:
    print(f"  ✗ 失败: {e}")
    import traceback
    traceback.print_exc()

# 测试4: 检查CARLA服务器
print("\n[测试4] 检查CARLA服务器状态...")
import socket

def check_port(host, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    try:
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except:
        return False

if check_port('localhost', 2000):
    print("  ✓ CARLA服务器端口2000正在监听")
else:
    print("  ✗ CARLA服务器端口2000未开启")
    print("\n请启动CARLA服务器:")
    print("  cd /path/to/CARLA")
    print("  ./CarlaUE4.sh")

print("\n" + "="*70)
print(" 诊断完成")
print("="*70)
