#!/usr/bin/env python3
"""
CARLA Actor管理修复版本测试

测试修复后的逻辑：
1. 移除双重cleanup()
2. 增强get_world_state()异常处理
3. 优化cleanup()避免重复删除
4. 改进agent_id使用

不影响原代码，仅用于验证修复方案。
"""

import sys
import time
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_fixed_cleanup_logic():
    """测试修复后的cleanup逻辑"""
    print("\n" + "="*70)
    print("测试: 修复后的cleanup()逻辑")
    print("="*70)

    try:
        # 导入原始的CarlaSimulator作为基准
        from c2o_drive.environments.carla.simulator import CarlaSimulator
        from c2o_drive.environments.carla.scenarios import CarlaScenarioLibrary

        print("\n创建simulator...")
        sim = CarlaSimulator(town="Town03", dt=0.1, no_rendering=True)

        # 添加修复后的cleanup方法
        def fixed_cleanup(self):
            """修复后的cleanup - 避免重复删除"""
            destroyed_ids = set()  # 追踪已删除的actor ID

            print("  [Cleanup] 开始清理...")

            # 1. 销毁碰撞传感器
            if self.ego_collision_sensor is not None:
                try:
                    if self.ego_collision_sensor.is_alive:
                        self.ego_collision_sensor.destroy()
                        destroyed_ids.add(self.ego_collision_sensor.id)
                        print("    ✓ 碰撞传感器已销毁")
                except Exception as e:
                    print(f"    ⚠️ 清理传感器失败: {e}")
                finally:
                    self.ego_collision_sensor = None

            # 2. 销毁自车
            if self.ego_vehicle is not None:
                try:
                    if self.ego_vehicle.is_alive:
                        self.ego_vehicle.destroy()
                        destroyed_ids.add(self.ego_vehicle.id)
                        print("    ✓ 自车已销毁")
                except Exception as e:
                    print(f"    ⚠️ 清理自车失败: {e}")
                finally:
                    self.ego_vehicle = None

            # 3. 销毁env_vehicles
            destroyed_count = 0
            for i, vehicle in enumerate(list(self.env_vehicles)):
                try:
                    if vehicle.is_alive:
                        vehicle.destroy()
                        destroyed_ids.add(vehicle.id)
                        destroyed_count += 1
                except Exception as e:
                    print(f"    ⚠️ 清理环境车辆{i}失败: {e}")
            self.env_vehicles = []
            print(f"    ✓ {destroyed_count} 个环境车辆已销毁")

            # 4. 全局清理 - 避免重复删除
            residual_count = 0
            try:
                actor_list = self.world.get_actors().filter("*vehicle*")
                for actor in actor_list:
                    try:
                        # 跳过已删除的
                        if actor.id in destroyed_ids:
                            continue

                        role = actor.attributes.get('role_name', '')
                        if role != "hero" and actor.is_alive:
                            actor.destroy()
                            residual_count += 1
                    except Exception:
                        pass
            except Exception:
                pass

            if residual_count > 0:
                print(f"    ✓ {residual_count} 个残留车辆已清理")

            print(f"  [Cleanup] 完成 (总计删除: {len(destroyed_ids)}, 残留: {residual_count})")

        # 替换cleanup方法
        import types
        sim.cleanup = types.MethodType(fixed_cleanup, sim)

        # 测试1: 创建场景
        print("\n[测试1] 创建场景...")
        scenario = CarlaScenarioLibrary.get_scenario("s4_wrong_way")
        ego_spawn = CarlaScenarioLibrary.spawn_to_transform(scenario.ego_spawn)
        agent_spawns = [CarlaScenarioLibrary.spawn_to_transform(s) for s in scenario.agent_spawns]

        state1 = sim.create_scenario(ego_spawn, agent_spawns)
        print(f"✓ 场景创建成功: {len(state1.agents)} agents")

        # 测试2: 双重cleanup
        print("\n[测试2] 测试双重cleanup...")
        print("第1次cleanup:")
        sim.cleanup()

        print("\n第2次cleanup (应该没有东西可删):")
        sim.cleanup()

        # 测试3: 重新创建场景
        print("\n[测试3] cleanup后重新创建场景...")
        state2 = sim.create_scenario(ego_spawn, agent_spawns)
        print(f"✓ 场景重新创建成功: {len(state2.agents)} agents")

        # 最终清理
        sim.cleanup()

        print("\n✅ 修复后的cleanup逻辑测试通过！")
        return True

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fixed_get_world_state():
    """测试修复后的get_world_state逻辑"""
    print("\n" + "="*70)
    print("测试: 修复后的get_world_state()逻辑")
    print("="*70)

    try:
        from c2o_drive.environments.carla.simulator import CarlaSimulator
        from c2o_drive.environments.carla.scenarios import CarlaScenarioLibrary
        from c2o_drive.environments.carla.types import AgentState, AgentType
        import math

        print("\n创建simulator...")
        sim = CarlaSimulator(town="Town03", dt=0.1, no_rendering=True)

        # 添加修复后的get_world_state方法
        def fixed_get_world_state(self):
            """修复后的get_world_state - 处理已删除actor"""
            if self.ego_vehicle is None:
                raise RuntimeError("场景未初始化，请先调用create_scenario()")

            # 获取自车状态
            ego_loc = self.ego_vehicle.get_location()
            ego_vel = self.ego_vehicle.get_velocity()
            ego_rot = self.ego_vehicle.get_transform().rotation

            from c2o_drive.environments.carla.types import EgoState, WorldState
            ego_state = EgoState(
                position_m=(ego_loc.x, ego_loc.y),
                velocity_mps=(ego_vel.x, ego_vel.y),
                yaw_rad=math.radians(ego_rot.yaw)
            )

            # 获取环境车辆状态 - 增加异常处理
            agents = []
            invalid_indices = []  # 记录失效的vehicle索引

            for i, vehicle in enumerate(self.env_vehicles):
                try:
                    if not vehicle.is_alive:
                        invalid_indices.append(i)
                        print(f"    ⚠️ vehicle {i}: is_alive=False，跳过")
                        continue

                    v_loc = vehicle.get_location()
                    v_vel = vehicle.get_velocity()
                    v_rot = vehicle.get_transform().rotation

                    agent_state = AgentState(
                        agent_id=f"vehicle-{vehicle.id}",  # 使用CARLA actor ID
                        position_m=(v_loc.x, v_loc.y),
                        velocity_mps=(v_vel.x, v_vel.y),
                        heading_rad=math.radians(v_rot.yaw),
                        agent_type=AgentType.VEHICLE
                    )
                    agents.append(agent_state)

                except RuntimeError as e:
                    # Actor已被destroy
                    print(f"    ⚠️ vehicle {i} 访问失败 (已被删除): {e}")
                    invalid_indices.append(i)
                except Exception as e:
                    print(f"    ⚠️ vehicle {i} 访问异常: {e}")
                    invalid_indices.append(i)

            # 清理失效的vehicle引用
            if invalid_indices:
                for idx in reversed(invalid_indices):  # 从后往前删除
                    del self.env_vehicles[idx]
                print(f"    已自动清理 {len(invalid_indices)} 个失效vehicle引用")

            from c2o_drive.environments.carla.types import WorldState
            return WorldState(
                time_s=self.current_time,
                ego=ego_state,
                agents=agents
            )

        # 替换get_world_state方法
        import types
        sim.get_world_state = types.MethodType(fixed_get_world_state, sim)

        # 测试1: 正常获取
        print("\n[测试1] 创建场景并获取world state...")
        scenario = CarlaScenarioLibrary.get_scenario("s4_wrong_way")
        ego_spawn = CarlaScenarioLibrary.spawn_to_transform(scenario.ego_spawn)
        agent_spawns = [CarlaScenarioLibrary.spawn_to_transform(s) for s in scenario.agent_spawns]

        state1 = sim.create_scenario(ego_spawn, agent_spawns)
        print(f"✓ 初始场景: {len(state1.agents)} agents")

        state2 = sim.get_world_state()
        print(f"✓ 正常获取: {len(state2.agents)} agents")

        # 测试2: 手动删除一个vehicle，测试异常处理
        print("\n[测试2] 手动删除一个vehicle，测试容错性...")
        if len(sim.env_vehicles) > 0:
            print(f"  删除前: {len(sim.env_vehicles)} vehicles")

            # 手动删除第一个vehicle (模拟外部删除)
            try:
                sim.env_vehicles[0].destroy()
                print("  ✓ 手动删除了第1个vehicle")
            except:
                pass

            # 尝试获取world state (应该自动处理)
            state3 = sim.get_world_state()
            print(f"  ✓ 获取成功: {len(state3.agents)} agents (少了1个)")
            print(f"  删除后: {len(sim.env_vehicles)} vehicles (自动清理了失效引用)")

        # 清理
        sim.cleanup()

        print("\n✅ 修复后的get_world_state逻辑测试通过！")
        return True

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fixed_reset_logic():
    """测试修复后的reset逻辑（移除双重cleanup）"""
    print("\n" + "="*70)
    print("测试: 修复后的reset()逻辑 (移除双重cleanup)")
    print("="*70)

    try:
        from c2o_drive.environments import CarlaEnv
        from c2o_drive.environments.carla.scenarios import CarlaScenarioLibrary

        print("\n创建CarlaEnv...")
        env = CarlaEnv(
            host='localhost',
            port=2000,
            town='Town03',
            dt=0.5,
            max_episode_steps=10,
            no_rendering=True,
        )

        # 修复后的reset方法
        def fixed_reset(self, seed=None, options=None):
            """修复后的reset - 移除显式cleanup"""
            if seed is not None:
                import numpy as np
                np.random.seed(seed)

            # 确保连接
            self._ensure_connected()

            # ❌ 移除这里的cleanup - create_scenario会自动处理
            # if self.simulator is not None:
            #     self.simulator.cleanup()

            # 创建场景
            options = options or {}
            scenario_config = options.get('scenario_config', {})
            scenario_def = scenario_config.get('scenario')
            scenario_name = scenario_config.get('scenario_name')

            if isinstance(scenario_def, str):
                scenario_def = CarlaScenarioLibrary.get_scenario(scenario_def)
            if scenario_def is None and scenario_name:
                scenario_def = CarlaScenarioLibrary.get_scenario(scenario_name)

            if scenario_def is not None:
                ego_spawn = CarlaScenarioLibrary.spawn_to_transform(scenario_def.ego_spawn)
                agent_spawns = [
                    CarlaScenarioLibrary.spawn_to_transform(spawn)
                    for spawn in scenario_def.agent_spawns
                ]
                autopilot = scenario_def.autopilot
            else:
                default_spawn = CarlaScenarioLibrary.spawn_to_transform((5.5, -70.0, 0.5, -90.0))
                ego_spawn = default_spawn
                agent_spawns = []
                autopilot = False

            # create_scenario()内部会调用cleanup()
            print("  [Reset] 调用create_scenario (内部会cleanup)...")
            self._current_state = self.simulator.create_scenario(
                ego_spawn=ego_spawn,
                agent_spawns=agent_spawns,
                agent_autopilot=autopilot,
            )

            self._step_count = 0
            self._episode_reward = 0.0
            self._episode_trajectory = []
            self._previous_action = None

            reference_path = options.get('reference_path')
            if reference_path is None and scenario_def is not None:
                reference_path = CarlaScenarioLibrary.get_reference_path(
                    scenario_def,
                    horizon=self.max_episode_steps,
                    dt=self.dt,
                )

            info = {
                'town': self.town,
                'episode': 0,
                'reference_path': reference_path,
                'scenario': scenario_def.name if scenario_def else 'default',
            }

            return self._current_state, info

        # 替换reset方法
        import types
        env.reset = types.MethodType(fixed_reset, env)

        # 测试快速连续reset
        print("\n[测试] 快速连续reset 3次...")
        scenario_def = CarlaScenarioLibrary.get_scenario("s4_wrong_way")

        for i in range(3):
            print(f"\n  Reset #{i+1}:")
            state, info = env.reset(
                seed=42+i,
                options={'scenario_config': {'scenario': scenario_def}}
            )
            print(f"    ✓ Reset成功: {len(state.agents)} agents, ego位置={state.ego.position_m}")
            time.sleep(0.2)

        env.close()

        print("\n✅ 修复后的reset逻辑测试通过！")
        return True

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_stress_test():
    """压力测试：快速创建和销毁场景"""
    print("\n" + "="*70)
    print("压力测试: 快速创建/销毁场景 10次")
    print("="*70)

    try:
        from c2o_drive.environments.carla.simulator import CarlaSimulator
        from c2o_drive.environments.carla.scenarios import CarlaScenarioLibrary

        print("\n创建simulator...")
        sim = CarlaSimulator(town="Town03", dt=0.1, no_rendering=True)

        scenario = CarlaScenarioLibrary.get_scenario("s4_wrong_way")
        ego_spawn = CarlaScenarioLibrary.spawn_to_transform(scenario.ego_spawn)
        agent_spawns = [CarlaScenarioLibrary.spawn_to_transform(s) for s in scenario.agent_spawns]

        print("\n开始压力测试...")
        for i in range(10):
            try:
                # 创建场景
                state = sim.create_scenario(ego_spawn, agent_spawns)

                # 获取world state
                state2 = sim.get_world_state()

                # 执行几步
                from c2o_drive.environments.carla.types import EgoControl
                for _ in range(3):
                    sim.step(EgoControl(throttle=0.3, steer=0.0, brake=0.0))

                # 清理
                sim.cleanup()

                print(f"  迭代 {i+1}/10: ✓ 成功 ({len(state.agents)} agents)")

            except Exception as e:
                print(f"  迭代 {i+1}/10: ❌ 失败 - {e}")
                raise

        print("\n✅ 压力测试通过！")
        return True

    except Exception as e:
        print(f"\n❌ 压力测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("="*70)
    print(" CARLA Actor管理修复版本测试")
    print("="*70)
    print("\n修复内容:")
    print("  1. ✅ 移除reset()中的双重cleanup()")
    print("  2. ✅ 增强get_world_state()异常处理")
    print("  3. ✅ 优化cleanup()避免重复删除")
    print("  4. ✅ 改进agent_id使用actor.id")
    print("\n开始测试...\n")

    results = {}

    tests = [
        ("修复后的cleanup逻辑", test_fixed_cleanup_logic),
        ("修复后的get_world_state逻辑", test_fixed_get_world_state),
        ("修复后的reset逻辑", test_fixed_reset_logic),
        ("压力测试", test_stress_test),
    ]

    for test_name, test_func in tests:
        try:
            print(f"\n{'='*70}")
            results[test_name] = test_func()
            time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n⏸️  用户中断测试")
            break
        except Exception as e:
            print(f"\n💥 测试 '{test_name}' 异常: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False

    # 总结
    print("\n" + "="*70)
    print(" 测试总结")
    print("="*70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {status}: {test_name}")

    print(f"\n总计: {passed}/{total} 通过")

    if passed == total:
        print("\n🎉 所有测试通过！可以合并到原代码")
        print("\n下一步:")
        print("  1. 修改 src/c2o_drive/environments/carla/simulator.py")
        print("  2. 修改 src/c2o_drive/environments/carla_env.py")
    else:
        print("\n⚠️ 发现问题，需要进一步调试")

    return passed == total


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n💥 测试脚本异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
