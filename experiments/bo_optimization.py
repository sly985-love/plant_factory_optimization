# -*- coding: utf-8 -*-
"""
BO排程优化主脚本

使用方法:
    python experiments/bo_optimization.py

【重要】rho2 和 A1_A2 现在是连续搜索变量，不再使用离散值映射。

来源: 论文方法部分 2.4
"""

import os
import sys
import argparse
import numpy as np
import yaml
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.envs import MultiBatchPlantFactoryEnv, load_all_configs
from src.bo import BayesianOptimizer
from src.utils.common import set_random_seed


def parse_args():
    parser = argparse.ArgumentParser(description='贝叶斯优化排程')
    parser.add_argument('--results_dir', type=str, default='results/bo',
                        help='结果保存目录')
    parser.add_argument('--n_iter', type=int, default=40,
                        help='BO迭代次数')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--n_eval_repeats', type=int, default=3,
                        help='每个排程的评估重复次数')
    return parser.parse_args()


def create_objective_func(env_config, n_repeats=3):
    """
    创建BO目标函数

    目标函数: 给定排程，评估全年利润
    利润 = 收益 - 能耗成本
    """

    def objective(schedule):
        rewards = []

        for _ in range(n_repeats):
            # 创建环境
            env = MultiBatchPlantFactoryEnv(env_config)
            options = {'schedule': schedule}
            obs, _ = env.reset(options=options)

            done = False
            ep_reward = 0
            ep_steps = 0

            # 简单随机控制器（评估排程本身的质量）
            # 动作从配置读取
            ep = env_config.get('equipment_params', {})
            I_max = ep.get('I_max', 600.0)
            default_action = np.array([
                200.0, 200.0, 0.0,
                ep.get('co2_supply_max', 0.5) * 0.2,
                ep.get('c_vent_fan_cap', 0.5) * 0.4,
                ep.get('c_dehum_cap', 0.002) * 0.5,
            ], dtype=np.float32)

            while not done and ep_steps < 1000:
                action = default_action
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                ep_reward += reward
                ep_steps += 1

            rewards.append(ep_reward)

        return np.mean(rewards)

    return objective


def main():
    args = parse_args()
    set_random_seed(args.seed)

    # 配置目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    config_dir = os.path.join(project_dir, 'configs')

    # 创建结果目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(args.results_dir, f'bo_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)

    # 加载配置
    print("加载配置文件...")
    configs = load_all_configs(config_dir)

    env_config = {
        'schedule': {'t1': 14, 't2': 21, 'rho2': 35.0, 'A1_A2': 0.5},
        'seed': args.seed,
        'dt': 3600.0,
    }

    for key in ['container_params', 'crop_params', 'equipment_params', 'reward_params', 'schedule_params']:
        if key in configs:
            env_config[key] = configs[key]

    # BO配置
    bo_config = {
        'n_iter': args.n_iter,
        'n_initial_points': 10,
        'acquisition': 'EI',
        'seed': args.seed,
        'infeasible_penalty': -1e6,
    }

    # 排程参数（从schedule_params.yaml加载）
    schedule_params = configs.get('schedule_params', {})
    # 确保密度约束参数存在
    schedule_params.setdefault('rho1_min', 30)
    schedule_params.setdefault('rho1_max', 60)
    schedule_params.setdefault('A_total', 40.0)
    schedule_params.setdefault('rho2_min', 20.0)
    schedule_params.setdefault('rho2_max', 80.0)
    schedule_params.setdefault('A1_A2_min', 0.1)
    schedule_params.setdefault('A1_A2_max', 5.0)
    schedule_params.setdefault('t1_min', 10)
    schedule_params.setdefault('t1_max', 18)
    schedule_params.setdefault('t2_min', 18)
    schedule_params.setdefault('t2_max', 26)

    print(f"\nBO优化配置:")
    print(f"  迭代次数: {bo_config['n_iter']}")
    print(f"  初始点数: {bo_config['n_initial_points']}")
    print(f"  采集函数: {bo_config['acquisition']}")
    print(f"  评估重复: {args.n_eval_repeats}")
    print(f"  t1范围: [{schedule_params['t1_min']}, {schedule_params['t1_max']}]")
    print(f"  t2范围: [{schedule_params['t2_min']}, {schedule_params['t2_max']}]")
    print(f"  rho2范围: [{schedule_params['rho2_min']}, {schedule_params['rho2_max']}]")
    print(f"  A1_A2范围: [{schedule_params['A1_A2_min']}, {schedule_params['A1_A2_max']}]")

    # 创建目标函数
    objective_func = create_objective_func(env_config, args.n_eval_repeats)

    # 创建BO优化器
    print("\n开始BO优化...")
    optimizer = BayesianOptimizer(
        config=bo_config,
        objective_func=objective_func,
        schedule_params=schedule_params
    )

    # 执行优化
    result = optimizer.optimize()

    # 输出结果
    print("\n" + "=" * 60)
    print("BO优化结果")
    print("=" * 60)
    print(f"最优排程:")
    print(f"  t1 (育苗期): {result['best_schedule']['t1']} 天")
    print(f"  t2 (定植期): {result['best_schedule']['t2']} 天")
    print(f"  rho2 (定植密度): {result['best_schedule']['rho2']:.2f} 株/m²")
    print(f"  A1/A2 (面积比): {result['best_schedule']['A1_A2']:.4f}")
    print(f"  A1: {result['best_schedule']['A1']:.2f} m²")
    print(f"  A2: {result['best_schedule']['A2']:.2f} m²")
    print(f"最优利润: {result['best_profit']:.2f}")

    # 保存结果
    result_path = os.path.join(results_dir, 'bo_results.yaml')
    with open(result_path, 'w', encoding='utf-8') as f:
        yaml.dump(result, f, default_flow_style=False, allow_unicode=True)
    print(f"\n结果已保存: {result_path}")


if __name__ == '__main__':
    main()
