# -*- coding: utf-8 -*-
"""
训练曲线可视化脚本

使用方法:
    python visualizations/plot_training.py --log_dir results/logs
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def parse_args():
    parser = argparse.ArgumentParser(description='绘制训练曲线')
    parser.add_argument('--log_dir', type=str, default='results/logs',
                        help='日志目录')
    parser.add_argument('--save_dir', type=str, default='results/figures',
                        help='图片保存目录')
    parser.add_argument('--window', type=int, default=100,
                        help='滑动平均窗口大小')
    return parser.parse_args()


def plot_training_curves(log_dir: str, save_dir: str, window: int = 100):
    """绘制训练曲线"""
    os.makedirs(save_dir, exist_ok=True)

    # 尝试加载评估日志
    eval_csv = os.path.join(log_dir, 'evaluations.csv')
    if os.path.exists(eval_csv):
        df = pd.read_csv(eval_csv)
        print(f"加载评估日志: {len(df)} 条记录")

        # 绘制奖励曲线
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # 平均奖励
        if 'mean_reward' in df.columns:
            rewards = df['mean_reward'].values
            x = np.arange(len(rewards))

            # 滑动平均
            if len(rewards) >= window:
                smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
                x_smooth = x[window//2:-(window//2)]
            else:
                smoothed = rewards
                x_smooth = x

            axes[0, 0].plot(x, rewards, alpha=0.3, label='原始')
            axes[0, 0].plot(x_smooth, smoothed, label=f'滑动平均(window={window})')
            axes[0, 0].set_xlabel('评估次数')
            axes[0, 0].set_ylabel('平均奖励')
            axes[0, 0].set_title('训练奖励曲线')
            axes[0, 0].legend()
            axes[0, 0].grid(True)

        # 回合长度
        if 'mean_episode_length' in df.columns:
            lengths = df['mean_episode_length'].values
            axes[0, 1].plot(lengths)
            axes[0, 1].set_xlabel('评估次数')
            axes[0, 1].set_ylabel('平均回合长度')
            axes[0, 1].set_title('回合长度')
            axes[0, 1].grid(True)

        # 奖励标准差
        if 'std_reward' in df.columns:
            stds = df['std_reward'].values
            axes[1, 0].plot(stds)
            axes[1, 0].set_xlabel('评估次数')
            axes[1, 0].set_ylabel('奖励标准差')
            axes[1, 0].set_title('策略稳定性')
            axes[1, 0].grid(True)

        # 损失曲线（如果有）
        if 'train_loss' in df.columns:
            losses = df['train_loss'].values
            axes[1, 1].plot(losses)
            axes[1, 1].set_xlabel('训练步数')
            axes[1, 1].set_ylabel('损失')
            axes[1, 1].set_title('训练损失')
            axes[1, 1].grid(True)

        plt.tight_layout()
        save_path = os.path.join(save_dir, 'training_curves.png')
        plt.savefig(save_path, dpi=150)
        print(f"训练曲线已保存: {save_path}")
        plt.close()


def plot_bo_convergence(results_dir: str, save_dir: str):
    """绘制BO收敛曲线"""
    import yaml

    os.makedirs(save_dir, exist_ok=True)

    # 查找BO结果
    bo_dirs = [d for d in os.listdir(results_dir) if 'bo_' in d]
    if not bo_dirs:
        print("未找到BO结果目录")
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for bo_dir in bo_dirs:
        result_file = os.path.join(results_dir, bo_dir, 'bo_results.yaml')
        if os.path.exists(result_file):
            with open(result_file, 'r', encoding='utf-8') as f:
                result = yaml.safe_load(f)

            if 'convergence' in result:
                convergence = [-v for v in result['convergence']]  # 转换为利润
                ax.plot(convergence, label=bo_dir)

    ax.set_xlabel('BO迭代次数')
    ax.set_ylabel('目标值 (利润)')
    ax.set_title('贝叶斯优化收敛曲线')
    ax.legend()
    ax.grid(True)

    save_path = os.path.join(save_dir, 'bo_convergence.png')
    plt.savefig(save_path, dpi=150)
    print(f"BO收敛曲线已保存: {save_path}")
    plt.close()


def plot_control_trajectory(data_path: str, save_dir: str):
    """绘制控制轨迹"""
    import pandas as pd

    os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(data_path):
        print(f"轨迹文件不存在: {data_path}")
        return

    df = pd.read_csv(data_path)
    print(f"加载轨迹数据: {len(df)} 步")

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    # 温度
    if 'T' in df.columns:
        axes[0, 0].plot(df['T'], label='温度')
        axes[0, 0].axhline(y=18, color='r', linestyle='--', alpha=0.5, label='下限')
        axes[0, 0].axhline(y=28, color='r', linestyle='--', alpha=0.5, label='上限')
        axes[0, 0].set_ylabel('温度 [°C]')
        axes[0, 0].set_title('箱内温度')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

    # 湿度
    if 'RH' in df.columns:
        axes[0, 1].plot(df['RH'], label='相对湿度')
        axes[0, 1].axhline(y=60, color='r', linestyle='--', alpha=0.5)
        axes[0, 1].axhline(y=80, color='r', linestyle='--', alpha=0.5)
        axes[0, 1].set_ylabel('相对湿度 [%]')
        axes[0, 1].set_title('箱内湿度')
        axes[0, 1].grid(True)

    # CO2
    if 'C_ppm' in df.columns:
        axes[1, 0].plot(df['C_ppm'])
        axes[1, 0].set_ylabel('CO2 [ppm]')
        axes[1, 0].set_title('CO2浓度')
        axes[1, 0].grid(True)

    # 动作1和2
    if 'action_0' in df.columns and 'action_1' in df.columns:
        axes[1, 1].plot(df['action_0'], label='育苗区光强')
        axes[1, 1].plot(df['action_1'], label='定植区光强')
        axes[1, 1].set_ylabel('光强 [μmol/m²/s]')
        axes[1, 1].set_title('光照控制')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

    # 奖励
    if 'reward' in df.columns:
        axes[2, 0].plot(df['reward'].cumsum())
        axes[2, 0].set_xlabel('步数')
        axes[2, 0].set_ylabel('累计奖励')
        axes[2, 0].set_title('累计奖励曲线')
        axes[2, 0].grid(True)

    # 能耗分解（如果有）
    energy_cols = [c for c in df.columns if c.startswith('E_')]
    if energy_cols:
        cumulative = df[energy_cols].cumsum()
        cumulative.plot(ax=axes[2, 1], kind='area', stacked=True, alpha=0.7)
        axes[2, 1].set_xlabel('步数')
        axes[2, 1].set_ylabel('累计能耗 [kWh]')
        axes[2, 1].set_title('能耗分解')
        axes[2, 1].legend(loc='upper left', fontsize=8)
        axes[2, 1].grid(True)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'control_trajectory.png')
    plt.savefig(save_path, dpi=150)
    print(f"控制轨迹已保存: {save_path}")
    plt.close()


def plot_baseline_comparison(results_dir: str, save_dir: str):
    """绘制基线对比图"""
    import pandas as pd

    os.makedirs(save_dir, exist_ok=True)

    # 查找基线结果
    baseline_dirs = [d for d in os.listdir(results_dir) if 'baseline_' in d]
    if not baseline_dirs:
        print("未找到基线对比结果目录")
        return

    all_results = []
    for bd in baseline_dirs:
        csv_path = os.path.join(results_dir, bd, 'baseline_results.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            all_results.append(df)

    if not all_results:
        return

    combined = pd.concat(all_results).groupby('method').agg(['mean', 'std']).reset_index()
    print(combined)

    fig, ax = plt.subplots(figsize=(8, 5))
    methods = combined['method'].values
    means = combined[('mean_reward', 'mean')].values
    stds = combined[('mean_reward', 'std')].values

    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    x = np.arange(len(methods))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors[:len(methods)], alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel('平均奖励')
    ax.set_title('基线方法对比')
    ax.grid(True, axis='y')

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{mean:.2f}', ha='center', va='bottom')

    save_path = os.path.join(save_dir, 'baseline_comparison.png')
    plt.savefig(save_path, dpi=150)
    print(f"基线对比图已保存: {save_path}")
    plt.close()


def main():
    args = parse_args()

    print("=" * 60)
    print("训练可视化")
    print("=" * 60)

    # 训练曲线
    plot_training_curves(args.log_dir, args.save_dir, args.window)

    # BO收敛曲线
    plot_bo_convergence(args.log_dir, args.save_dir)

    # 基线对比
    plot_baseline_comparison(args.log_dir, args.save_dir)

    print("\n可视化完成!")


if __name__ == '__main__':
    main()
