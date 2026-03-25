# -*- coding: utf-8 -*-
"""
训练主脚本

使用方法:
    python experiments/train.py --config configs/rl_params.yaml

来源: 论文方法部分 2.3.4
"""

import os
import sys
import argparse
import yaml
import numpy as np
from datetime import datetime

# 添加src到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.envs import MultiBatchPlantFactoryEnv, load_all_configs
from src.rl import ContextualPPOTrainer
from src.utils import set_random_seed


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练上下文PPO策略')
    parser.add_argument('--config', type=str, default='configs/rl_params.yaml',
                        help='RL配置文件路径')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='结果保存目录')
    parser.add_argument('--no_wandb', action='store_true',
                        help='禁用WandB日志')
    parser.add_argument('--eval_only', action='store_true',
                        help='仅运行评估')
    parser.add_argument('--model_path', type=str, default=None,
                        help='模型加载路径（用于评估）')
    parser.add_argument('--total_timesteps', type=int, default=None,
                        help='训练总步数（覆盖配置文件）')
    parser.add_argument('--viz', action='store_true', default=False,
                        help='训练结束后生成可视化')
    return parser.parse_args()


def load_configs(config_dir: str, args) -> dict:
    """加载所有配置"""
    configs = load_all_configs(config_dir)

    # 合并配置
    env_config = {
        'schedule': {
            't1': 14, 't2': 21, 'rho2': 35, 'A1_A2': 0.5
        },
        'seed': args.seed,
        'dt': 3600.0,
    }

    # 添加各模块参数
    if 'container_params' in configs:
        env_config['container_params'] = configs['container_params']
    if 'crop_params' in configs:
        env_config['crop_params'] = configs['crop_params']
    if 'equipment_params' in configs:
        env_config['equipment_params'] = configs['equipment_params']
    if 'reward_params' in configs:
        env_config['reward_params'] = configs['reward_params']
    if 'schedule_params' in configs:
        env_config['schedule_params'] = configs['schedule_params']

    # RL参数
    rl_params = {}
    if 'rl_params' in configs:
        rl_params = configs['rl_params']

    # 命令行覆盖
    if args.no_wandb:
        rl_params['use_wandb'] = False
    if args.total_timesteps is not None:
        rl_params['total_timesteps'] = args.total_timesteps

    return env_config, rl_params


def main():
    """主函数"""
    args = parse_args()

    # 设置随机种子
    set_random_seed(args.seed)

    # 确定配置目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    config_dir = os.path.join(project_dir, 'configs')

    # 创建结果目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(args.results_dir, f'train_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'logs'), exist_ok=True)

    # 加载配置
    print("=" * 60)
    print("加载配置文件...")
    env_config, rl_params = load_configs(config_dir, args)

    print(f"结果目录: {results_dir}")
    print(f"随机种子: {args.seed}")
    print(f"WandB: {'启用' if rl_params.get('use_wandb', False) else '禁用'}")
    print("=" * 60)

    if args.eval_only:
        # 仅评估模式
        print("评估模式...")
        if args.model_path is None:
            print("错误: 评估模式需要指定 --model_path")
            return

        trainer = ContextualPPOTrainer(env_config, rl_params, results_dir)
        trainer.load(args.model_path)

        # 评估
        eval_results = trainer.evaluate(n_episodes=10)
        print("\n评估结果:")
        print(f"  平均奖励: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
        print(f"  平均回合长度: {eval_results['mean_ep_length']:.1f}")

    else:
        # 训练模式
        print("开始训练...")

        # 创建训练器
        trainer = ContextualPPOTrainer(
            env_config=env_config,
            rl_params=rl_params,
            results_dir=results_dir
        )

        # 训练
        model = trainer.train()

        print("\n训练完成!")

        # 保存配置
        config_save_path = os.path.join(results_dir, 'config.yaml')
        with open(config_save_path, 'w', encoding='utf-8') as f:
            yaml.dump({
                'env_config': env_config,
                'rl_params': rl_params,
            }, f, default_flow_style=False, allow_unicode=True)
        print(f"配置已保存: {config_save_path}")

        # 最终评估
        print("\n最终评估...")
        eval_results = trainer.evaluate(n_episodes=20)
        print(f"  平均奖励: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
        print(f"  平均回合长度: {eval_results['mean_ep_length']:.1f}")

    # 可视化（训练模式或手动指定）
    if args.viz or (not args.eval_only):
        try:
            from visualizations.experiment_viz import plot_training_from_results_dir
            print("\n--- Generating training visualizations ---")
            plot_training_from_results_dir(
                results_dir=results_dir,
                window=50,
            )
        except Exception as e:
            print(f"[WARN] 可视化失败: {e}")


if __name__ == '__main__':
    main()
