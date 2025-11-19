#!/usr/bin/env python3
"""
CARLA连接诊断脚本

用于测试和诊断CARLA连接问题
"""

import sys
from pathlib import Path

# 添加项目根目录
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

print("="*70)
print(" CARLA连接诊断")
print("="*70)

# 步骤1: 测试CARLA库导入
print("\n[1/5] 测试CARLA库导入...")
try:
    import glob
    import os

    # 查找CARLA .egg文件
    carla_egg_path = None
    search_paths = [
        '../carla/dist/',
        '~/carla/dist/',
        '/opt/carla/PythonAPI/carla/dist/',
        os.environ.get('CARLA_ROOT', '') + '/PythonAPI/carla/dist/',
    ]

    for search_path in search_paths:
        if not search_path:
            continue
        expanded_path = os.path.expanduser(search_path)
        pattern = os.path.join(expanded_path, 'carla-*%d.%d-%s.egg' % (
            sys.version_info.major,
            sys.version_info.minor,
            'win-amd64' if os.name == 'nt' else 'linux-x86_64'
        ))
        matches = glob.glob(pattern)
        if matches:
            carla_egg_path = matches[0]
            print(f"  找到CARLA .egg文件: {carla_egg_path}")
            sys.path.append(carla_egg_path)
            break

    if not carla_egg_path:
        print("  ⚠️ 警告: 未找到CARLA .egg文件，尝试直接导入...")

    import carla
    print(f"  ✓ CARLA库导入成功")
    print(f"  CARLA版本: {carla.version if hasattr(carla, 'version') else '未知'}")

except ImportError as e:
    print(f"  ✗ CARLA库导入失败: {e}")
    print("\n解决方案:")
    print("  1. 确保CARLA已安装")
    print("  2. 设置CARLA_ROOT环境变量:")
    print("     export CARLA_ROOT=/path/to/CARLA")
    print("  3. 或将CARLA .egg文件路径添加到PYTHONPATH")
    sys.exit(1)

# 步骤2: 测试CARLA服务器连接
print("\n[2/5] 测试CARLA服务器连接...")
try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    # 获取服务器版本
    server_version = client.get_server_version()
    client_version = client.get_client_version()

    print(f"  ✓ 成功连接到CARLA服务器")
    print(f"  服务器版本: {server_version}")
    print(f"  客户端版本: {client_version}")

    if server_version != client_version:
        print(f"  ⚠️ 警告: 服务器和客户端版本不匹配")

except RuntimeError as e:
    print(f"  ✗ 无法连接到CARLA服务器: {e}")
    print("\n解决方案:")
    print("  1. 启动CARLA服务器:")
    print("     cd /path/to/CARLA")
    print("     ./CarlaUE4.sh")
    print("  2. 或使用无渲染模式:")
    print("     ./CarlaUE4.sh -RenderOffScreen")
    print("  3. 检查端口2000是否被占用:")
    print("     netstat -an | grep 2000")
    sys.exit(1)

# 步骤3: 测试获取世界
print("\n[3/5] 测试获取世界...")
try:
    world = client.get_world()
    map_name = world.get_map().name
    print(f"  ✓ 成功获取世界")
    print(f"  当前地图: {map_name}")

except Exception as e:
    print(f"  ✗ 获取世界失败: {e}")
    sys.exit(1)

# 步骤4: 测试CarlaSimulator
print("\n[4/5] 测试CarlaSimulator...")
try:
    from carla_c2osr.env.carla_scenario_1 import CarlaSimulator

    print("  创建CarlaSimulator实例...")
    sim = CarlaSimulator(
        host='localhost',
        port=2000,
        town='Town03',
        dt=0.1,
        no_rendering=False
    )
    print(f"  ✓ CarlaSimulator创建成功")

    # 清理
    print("  清理资源...")
    sim.cleanup()
    print(f"  ✓ 清理完成")

except Exception as e:
    print(f"  ✗ CarlaSimulator测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 步骤5: 测试CarlaEnvironment
print("\n[5/5] 测试CarlaEnvironment...")
try:
    from carla_c2osr.environments import CarlaEnvironment

    print("  创建CarlaEnvironment实例...")
    env = CarlaEnvironment(
        host='localhost',
        port=2000,
        town='Town03',
        dt=0.1,
        max_episode_steps=10,
        num_vehicles=2,
        num_pedestrians=0,
    )
    print(f"  ✓ CarlaEnvironment创建成功")

    print("  测试reset...")
    state, info = env.reset(seed=42)
    print(f"  ✓ Reset成功")
    print(f"    Ego位置: {state.ego.position_m}")
    print(f"    Agent数量: {len(state.agents)}")

    print("  测试step...")
    from carla_c2osr.env.types import EgoControl
    action = EgoControl(throttle=0.3, steer=0.0, brake=0.0)
    result = env.step(action)
    print(f"  ✓ Step成功")
    print(f"    奖励: {result.reward:.2f}")
    print(f"    Info: {result.info}")

    print("  清理环境...")
    env.close()
    print(f"  ✓ 清理完成")

except Exception as e:
    print(f"  ✗ CarlaEnvironment测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print(" ✓ 所有测试通过！CARLA集成工作正常")
print("="*70)
print("\n现在可以运行:")
print("  python examples/run_c2osr_carla.py --scenario s4_wrong_way --episodes 2")
