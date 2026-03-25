# -*- coding: utf-8 -*-
"""
Experiment Visualization Module

提供三大实验的可视化函数：
1. 训练曲线可视化（Experiment 1）
2. 控制器对比可视化（Experiment 2）
3. BO收敛曲线可视化（Experiment 3）

使用方法:
    from visualizations.experiment_viz import (
        plot_training_curves,
        plot_controller_comparison,
        plot_bo_convergence,
        plot_environment_trajectory,
        plot_batch_growth,
    )

    # 训练曲线
    plot_training_curves(log_dir='results/train_xxx/logs',
                          save_dir='results/train_xxx/figures')

    # 控制器对比
    plot_controller_comparison(results_dir='results/exp2_controller_eval',
                              save_dir='results/exp2_controller_eval/figures')

    # BO收敛曲线
    plot_bo_convergence(results_dir='results/exp3_bo_comparison',
                        save_dir='results/exp3_bo_comparison/figures')

    # 环境轨迹
    plot_environment_trajectory(trajectory_df, save_dir='results/...')

    # 批次生长曲线
    plot_batch_growth(batch_df, save_dir='results/...')
"""

import os
import sys
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 中文支持
plt.rcParams['font.sans-serif'] = [
    'SimHei', 'WenQuanYi Micro Hei', 'DejaVu Sans',
    'Arial Unicode MS', 'Noto Sans CJK SC'
]
plt.rcParams['axes.unicode_minus'] = False

# 颜色方案
COLORS = {
    'PID':      '#e74c3c',
    'MPC':      '#3498db',
    'ContextualRL': '#2ecc71',
    'RuleCtrl': '#f39c12',
}
COLOR_LIST = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c']


# =============================================================================
# 工具函数
# =============================================================================

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _load_or_empty(csv_path: str) -> pd.DataFrame:
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return pd.DataFrame()


def _rolling_mean(series: pd.Series, window: int) -> pd.Series:
    if len(series) < window:
        return series
    return series.rolling(window, min_periods=1).mean()


# =============================================================================
# Experiment 1: 训练曲线可视化
# =============================================================================

def plot_training_curves(
    log_dir: str,
    save_dir: str,
    window: int = 50,
    figsize: tuple = (14, 10),
) -> dict:
    """
    绘制训练过程中的奖励曲线、熵、损失等指标。

    参数:
        log_dir: 训练日志目录（含 evaluations.csv）
        save_dir: 图片保存目录
        window: 滑动平均窗口大小
        figsize: 图像尺寸

    输出:
        4子图: mean_reward / ep_length / std_reward / (可选) loss
    """
    _ensure_dir(save_dir)
    eval_csv = os.path.join(log_dir, 'evaluations', 'evaluations.csv')
    if not os.path.exists(eval_csv):
        eval_csv = os.path.join(log_dir, 'evaluations.csv')

    df = _load_or_empty(eval_csv)

    if df.empty:
        warnings.warn(f"未找到评估日志: {eval_csv}")
        return {}

    n_rows = 2
    n_cols = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    # 1) 平均奖励
    ax = axes[0]
    rew_col = None
    for col in ['mean_reward', 'mean_ep_rew', 'ep_reward_mean']:
        if col in df.columns:
            rew_col = col
            break
    if rew_col:
        rewards = df[rew_col].values
        x = np.arange(len(rewards))
        smoothed = _rolling_mean(pd.Series(rewards), window)
        ax.plot(x, rewards, alpha=0.25, color=COLORS['ContextualRL'], label='Raw')
        ax.plot(x, smoothed.values, color=COLORS['ContextualRL'], lw=2,
                label=f'SMA({window})')
        ax.set_xlabel('Evaluation Round')
        ax.set_ylabel('Mean Episode Reward')
        ax.set_title('Training: Mean Reward')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No reward data found', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title('Training: Mean Reward')

    # 2) 平均回合长度
    ax = axes[1]
    len_col = None
    for col in ['mean_episode_length', 'ep_length_mean', 'ep_len_mean']:
        if col in df.columns:
            len_col = col
            break
    if len_col:
        lengths = df[len_col].values
        x = np.arange(len(lengths))
        smoothed = _rolling_mean(pd.Series(lengths), window)
        ax.plot(x, lengths, alpha=0.3, color='#3498db')
        ax.plot(x, smoothed.values, color='#3498db', lw=2)
        ax.set_xlabel('Evaluation Round')
        ax.set_ylabel('Mean Episode Length (steps)')
        ax.set_title('Training: Episode Length')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No length data found', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title('Training: Episode Length')

    # 3) 奖励标准差（策略稳定性）
    ax = axes[2]
    std_col = None
    for col in ['std_reward', 'std_ep_rew', 'ep_reward_std']:
        if col in df.columns:
            std_col = col
            break
    if std_col:
        stds = df[std_col].values
        x = np.arange(len(stds))
        smoothed = _rolling_mean(pd.Series(stds), window)
        ax.plot(x, stds, alpha=0.3, color='#9b59b6')
        ax.plot(x, smoothed.values, color='#9b59b6', lw=2)
        ax.set_xlabel('Evaluation Round')
        ax.set_ylabel('Reward Std Dev')
        ax.set_title('Training: Policy Stability (Reward Std)')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No std data found', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title('Training: Policy Stability')

    # 4) 每回合采收量（如果可用）
    ax = axes[3]
    harvest_col = None
    for col in ['mean_harvest', 'harvest_mean', 'total_harvest']:
        if col in df.columns:
            harvest_col = col
            break
    if harvest_col:
        vals = df[harvest_col].values
        x = np.arange(len(vals))
        smoothed = _rolling_mean(pd.Series(vals), window)
        ax.plot(x, vals, alpha=0.3, color='#f39c12')
        ax.plot(x, smoothed.values, color='#f39c12', lw=2)
        ax.set_xlabel('Evaluation Round')
        ax.set_ylabel('Mean Harvest (kg)')
        ax.set_title('Training: Harvest per Episode')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No harvest data found', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title('Training: Harvest')

    fig.suptitle('Contextual RL Training Curves', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_curves.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[Viz] 训练曲线已保存: {save_path}')
    return {'save_path': save_path}


def plot_training_from_results_dir(
    results_dir: str,
    window: int = 50,
) -> dict:
    """
    便捷函数：从 results/train_xxx 目录自动查找日志并绘制。

    参数:
        results_dir: 训练结果根目录（包含 logs/ 子目录）
        window: 滑动平均窗口

    返回:
        保存路径字典
    """
    log_dir = os.path.join(results_dir, 'logs')
    save_dir = os.path.join(results_dir, 'figures')
    _ensure_dir(save_dir)
    return plot_training_curves(log_dir, save_dir, window)


# =============================================================================
# Experiment 2: 控制器对比可视化
# =============================================================================

def plot_controller_comparison(
    results_dir: str,
    save_dir: str,
    figsize: tuple = (18, 12),
) -> dict:
    """
    绘制PID/MPC/ContextualRL三大控制器的性能对比图。

    子图布局 (3行 x 3列):
        Row 1: 总奖励 | 能耗成本 | 采收质量
        Row 2: 约束违反率 | 平均温度 | 平均CO2
        Row 3: 累计奖励曲线 | 环境状态雷达图 | 每控制器雷达图

    参数:
        results_dir: 包含各控制器 summary CSV 的目录
        save_dir: 图片保存目录

    期望文件结构:
        results_dir/
            pid_summary.csv
            mpc_summary.csv
            rl_summary.csv
            comparison_summary.csv   (可选)
    """
    _ensure_dir(save_dir)

    controllers = ['PID', 'MPC', 'ContextualRL']
    suffix_map = {'PID': 'pid', 'MPC': 'mpc', 'ContextualRL': 'rl'}
    label_map = {
        'PID': 'PID Controller',
        'MPC': 'Model Predictive Control',
        'ContextualRL': 'Contextual RL (PPO)',
    }

    # 加载摘要数据
    summaries = {}
    for ctrl in controllers:
        key = suffix_map[ctrl]
        path = os.path.join(results_dir, f'{key}_summary.csv')
        if os.path.exists(path):
            summaries[ctrl] = pd.read_csv(path)
        else:
            summaries[ctrl] = pd.DataFrame()

    # 汇总对比表
    comparison_data = []
    for ctrl in controllers:
        df = summaries[ctrl]
        if df.empty:
            continue
        comparison_data.append({
            'Controller': label_map[ctrl],
            'TotalReward': df['total_reward'].mean() if 'total_reward' in df.columns else 0,
            'RewardStd': df['total_reward'].std() if 'total_reward' in df.columns else 0,
            'EnergyCost': df['total_cost_yuan'].mean() if 'total_cost_yuan' in df.columns else 0,
            'ViolationRate': df['violation_rate'].mean() if 'violation_rate' in df.columns else 0,
            'HarvestKg': df['harvest_mass_kg'].mean() if 'harvest_mass_kg' in df.columns else 0,
            'Harvests': df['harvests'].mean() if 'harvests' in df.columns else 0,
            'T_mean': df['T_mean'].mean() if 'T_mean' in df.columns else 0,
            'C_ppm_mean': df['C_ppm_mean'].mean() if 'C_ppm_mean' in df.columns else 0,
        })

    comp_df = pd.DataFrame(comparison_data)

    # ── 子图1: 柱状对比 (6指标) ──────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=figsize)

    metrics = [
        ('TotalReward',   'Total Reward',          True),
        ('EnergyCost',    'Energy Cost (CNY)',       False),
        ('HarvestKg',     'Harvest Mass (kg)',       False),
        ('ViolationRate', 'Constraint Violation %',  False),
        ('T_mean',        'Mean Temperature (°C)',   False),
        ('C_ppm_mean',    'Mean CO2 (ppm)',          False),
    ]
    flat_axes = axes.flatten()

    for idx, (metric_key, metric_label, higher_better) in enumerate(metrics):
        ax = flat_axes[idx]
        if metric_key not in comp_df.columns or comp_df.empty:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(metric_label)
            continue

        means = comp_df[metric_key].values
        stds = comp_df.get('RewardStd', pd.Series([0]*len(comp_df))).values \
               if metric_key == 'TotalReward' else [0]*len(means)
        x = np.arange(len(means))
        bars = ax.bar(x, means, yerr=stds, capsize=4,
                      color=[COLORS.get(c, '#95a5a6') for c in controllers],
                      alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels([label_map[c] for c in controllers], fontsize=8)
        ax.set_ylabel(metric_label)
        ax.set_title(metric_label)
        ax.grid(True, axis='y', alpha=0.3)
        for bar, val in zip(bars, means):
            va = 'bottom' if higher_better else 'top'
            offset = 0.02 * max(means) if means.max() != 0 else 0.5
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + offset if higher_better
                              else bar.get_height() - offset,
                    f'{val:.2f}', ha='center', va=va, fontsize=8)

    # ── 子图2: 累计奖励曲线 ────────────────────────────────────────────────
    ax = axes[1, 0]
    for ctrl in controllers:
        key = suffix_map[ctrl]
        traj_path = os.path.join(results_dir, f'{key}_trajectory.csv')
        if not os.path.exists(traj_path):
            continue
        df_traj = pd.read_csv(traj_path)
        if 'total_reward' in df_traj.columns:
            cumrew = df_traj.groupby('run_id')['total_reward'].cumsum() \
                     if 'run_id' in df_traj.columns \
                     else df_traj['total_reward'].cumsum()
            mean_curve = cumrew.values if isinstance(cumrew, pd.Series) else cumrew.mean(axis=0)
            ax.plot(mean_curve, label=label_map[ctrl],
                    color=COLORS.get(ctrl, '#95a5a6'), alpha=0.8)
        elif 'step_reward' in df_traj.columns:
            if 'run_id' in df_traj.columns:
                grouped = df_traj.groupby('run_id')['step_reward'].cumsum()
                mean_curve = df_traj.groupby('run_id')['step_reward'].cumsum().groupby(
                    df_traj.groupby('run_id').cumcount()).mean()
            else:
                mean_curve = df_traj['step_reward'].cumsum()
            ax.plot(mean_curve, label=label_map[ctrl],
                    color=COLORS.get(ctrl, '#95a5a6'), alpha=0.8)

    ax.set_xlabel('Step')
    ax.set_ylabel('Cumulative Reward')
    ax.set_title('Cumulative Reward over Time')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── 子图3: 雷达图（每控制器独立）─────────────────────────────────────
    ax = axes[1, 1]
    radar_metrics = ['TotalReward', 'HarvestKg', 'T_mean', 'C_ppm_mean']
    valid_metrics = [m for m in radar_metrics if m in comp_df.columns]
    if len(valid_metrics) >= 3 and len(comp_df) > 0:
        radar_fig, radar_ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
        N = len(valid_metrics)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]

        for i, ctrl in enumerate(controllers):
            if ctrl not in comp_df['Controller'].values:
                continue
            row = comp_df[comp_df['Controller'] == label_map[ctrl]].iloc[0]
            values = [row[m] for m in valid_metrics]
            # 归一化到 [0,1]
            max_vals = [comp_df[m].max() for m in valid_metrics]
            min_vals = [comp_df[m].min() for m in valid_metrics]
            ranges = [mv - mn for mv, mn in zip(max_vals, min_vals)]
            values_norm = [(v - mn) / (r + 1e-8) for v, mn, r
                           in zip(values, min_vals, ranges)]
            values_norm += values_norm[:1]
            radar_ax.plot(angles, values_norm, 'o-', lw=2,
                          label=label_map[ctrl],
                          color=COLORS.get(ctrl, '#95a5a6'))
            radar_ax.fill(angles, values_norm, alpha=0.15,
                          color=COLORS.get(ctrl, '#95a5a6'))

        radar_ax.set_xticks(angles[:-1])
        radar_ax.set_xticklabels(valid_metrics, fontsize=8)
        radar_ax.set_title('Controller Performance Radar', fontsize=10)
        radar_ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=7)
        radar_save = os.path.join(save_dir, 'controller_radar.png')
        radar_fig.savefig(radar_save, dpi=150, bbox_inches='tight')
        plt.close(radar_fig)
        ax.axis('off')
        ax.text(0.5, 0.5, f'Radar saved:\n{os.path.basename(radar_save)}',
                ha='center', va='center', transform=ax.transAxes, fontsize=9)
    else:
        ax.text(0.5, 0.5, 'No radar data', ha='center', va='center',
                transform=ax.transAxes)

    # ── 子图4: 每控制器温度-时间 ──────────────────────────────────────────
    ax = axes[1, 2]
    for ctrl in controllers:
        key = suffix_map[ctrl]
        traj_path = os.path.join(results_dir, f'{key}_trajectory.csv')
        if not os.path.exists(traj_path):
            continue
        df_traj = pd.read_csv(traj_path)
        if 'T' in df_traj.columns:
            if 'run_id' in df_traj.columns:
                mean_T = df_traj.groupby('step')['T'].mean().values
            else:
                mean_T = df_traj['T'].values
            n = min(len(mean_T), 504)  # 最多21天
            x = np.arange(n)
            ax.plot(x, mean_T[:n],
                    label=label_map[ctrl],
                    color=COLORS.get(ctrl, '#95a5a6'),
                    alpha=0.7, lw=1.2)
    ax.axhline(y=18, color='red', linestyle='--', alpha=0.4, label='T_min')
    ax.axhline(y=28, color='red', linestyle='--', alpha=0.4, label='T_max')
    ax.set_xlabel('Step (hours)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Temperature Trajectory')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Fixed-Schedule Controller Comparison\n'
                 'PID vs MPC vs Contextual RL', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'controller_comparison.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[Viz] 控制器对比图已保存: {save_path}')

    # 保存汇总CSV
    if not comp_df.empty:
        comp_csv = os.path.join(save_dir, 'comparison_table.csv')
        comp_df.to_csv(comp_csv, index=False)
        print(f'[Viz] 对比汇总表已保存: {comp_csv}')

    return {'save_path': save_path, 'radar_path': os.path.join(save_dir, 'controller_radar.png')}


# =============================================================================
# Experiment 3: BO收敛曲线可视化
# =============================================================================

def plot_bo_convergence(
    results_dir: str,
    save_dir: str,
    figsize: tuple = (16, 10),
) -> dict:
    """
    绘制贝叶斯优化在不同控制器下的收敛曲线对比。

    布局 (1行 x 3列):
        Col 1: 三控制器的最优利润收敛曲线（+ 网格搜索基线）
        Col 2: 每次迭代的采集函数值（可选）
        Col 3: 最优排程参数随迭代变化

    参数:
        results_dir: 包含 bo_{controller}_results.yaml 的目录
        save_dir: 图片保存目录

    期望文件结构:
        results_dir/
            bo_pid_results.yaml
            bo_mpc_results.yaml
            bo_rl_results.yaml
    """
    _ensure_dir(save_dir)

    controller_files = {
        'PID':           'bo_pid_results.yaml',
        'MPC':           'bo_mpc_results.yaml',
        'ContextualRL':  'bo_rl_results.yaml',
    }
    label_map = {
        'PID':           'PID Controller',
        'MPC':           'Model Predictive Control',
        'ContextualRL':  'Contextual RL (PPO)',
    }

    import yaml

    convergence_data = {}
    best_schedules = {}

    for ctrl, fname in controller_files.items():
        path = os.path.join(results_dir, fname)
        if not os.path.exists(path):
            continue
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        conv = data.get('convergence', [])
        if conv:
            # conv 是最小化值列表（-profit），取负转为利润
            profits = [-v for v in conv]
            # 最佳-so-far
            best_so_far = [max(profits[:i+1]) for i in range(len(profits))]
            convergence_data[ctrl] = {
                'iterations': list(range(1, len(profits)+1)),
                'profit': profits,
                'best_so_far': best_so_far,
            }

        sched = data.get('best_schedule', {})
        if sched:
            best_schedules[ctrl] = sched

    if not convergence_data:
        warnings.warn(f"未找到BO结果文件于: {results_dir}")
        return {}

    # ── 图1: 收敛曲线 ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    ax = axes[0]
    for ctrl, data in convergence_data.items():
        color = COLORS.get(ctrl, '#95a5a6')
        ax.plot(data['iterations'], data['profit'], 'o-',
                alpha=0.4, ms=4, color=color, label=f'{ctrl} (iter)')
        ax.plot(data['iterations'], data['best_so_far'], 's-',
                lw=2, ms=5, color=color, label=f'{ctrl} (best)')
    ax.set_xlabel('BO Iteration')
    ax.set_ylabel('Profit (CNY)')
    ax.set_title('BO Convergence: Profit per Iteration')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── 图2: 归一化对比（所有控制器叠加）───────────────────────────────
    ax = axes[1]
    for ctrl, data in convergence_data.items():
        color = COLORS.get(ctrl, '#95a5a6')
        profits = data['profit']
        all_profits = [v for d in convergence_data.values() for v in d['profit']]
        if all_profits:
            global_min = min(all_profits)
            global_max = max(all_profits)
            rng = global_max - global_min
            if rng > 0:
                normed = [(v - global_min) / rng for v in profits]
            else:
                normed = profits
            best_normed = [max(normed[:i+1]) for i in range(len(normed))]
            ax.plot(data['iterations'], best_normed, 's-', lw=2, ms=5,
                    color=color, label=label_map[ctrl])
    ax.set_xlabel('BO Iteration')
    ax.set_ylabel('Normalized Best Profit')
    ax.set_title('BO Convergence: Normalized Best-so-Far')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── 图3: 最优排程参数柱状图 ──────────────────────────────────────────
    ax = axes[2]
    if best_schedules:
        param_keys = ['t1', 't2', 'rho2', 'A1_A2']
        x = np.arange(len(param_keys))
        width = 0.25
        for i, (ctrl, sched) in enumerate(best_schedules.items()):
            vals = [sched.get(k, 0) for k in param_keys]
            # 归一化每个参数到 [0,1]
            all_vals = {k: [] for k in param_keys}
            for s in best_schedules.values():
                for k in param_keys:
                    all_vals[k].append(s.get(k, 0))
            norms = [(v - min(all_vals[k])) / (max(all_vals[k]) - min(all_vals[k]) + 1e-8)
                     for k, v in zip(param_keys, vals)]
            ax.bar(x + i * width, norms, width,
                   label=label_map[ctrl],
                   color=COLORS.get(ctrl, '#95a5a6'), alpha=0.8)
        ax.set_xticks(x + width)
        ax.set_xticklabels(param_keys)
        ax.set_ylabel('Normalized Value')
        ax.set_title('Best Schedule Parameters')
        ax.legend(fontsize=7)
        ax.grid(True, axis='y', alpha=0.3)

    fig.suptitle('Bayesian Optimization Convergence\n'
                 'Comparing PID / MPC / Contextual RL as Lower-Level Controller',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'bo_convergence.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[Viz] BO收敛曲线已保存: {save_path}')

    # 打印最优排程
    print('\n=== BO最优排程 ===')
    for ctrl, sched in best_schedules.items():
        print(f"  [{ctrl}] t1={sched.get('t1','?')}d "
              f"t2={sched.get('t2','?')}d "
              f"rho2={sched.get('rho2',0):.1f} "
              f"A1/A2={sched.get('A1_A2',0):.3f}")

    return {'save_path': save_path, 'best_schedules': best_schedules}


# =============================================================================
# 通用轨迹可视化
# =============================================================================

def plot_environment_trajectory(
    trajectory_df: pd.DataFrame,
    save_dir: str,
    ctrl_name: str = 'Controller',
    n_days: int = 21,
    figsize: tuple = (16, 12),
) -> str:
    """
    绘制单个控制器在固定排程下的环境变量和控制动作轨迹。

    参数:
        trajectory_df: 轨迹DataFrame，需包含列：
            step, hour_of_day, T, RH, C_ppm, I1, I2, Q_HVAC,
            step_reward, total_reward, violation, elec_price
        save_dir: 保存目录
        ctrl_name: 控制器名称（用于图标题和文件名）
        n_days: 显示天数（小时数 = n_days * 24）

    返回:
        保存路径
    """
    _ensure_dir(save_dir)
    df = trajectory_df.head(n_days * 24).copy()

    fig, axes = plt.subplots(3, 2, figsize=figsize)
    hours = np.arange(len(df))

    def fmt_hour(step_arr):
        return [(s % 24) for s in step_arr]

    # 温度
    ax = axes[0, 0]
    if 'T' in df.columns:
        ax.plot(hours, df['T'].values, color='#e74c3c', lw=1.2)
        ax.axhline(18, ls='--', color='gray', alpha=0.5, label='T_min=18°C')
        ax.axhline(28, ls='--', color='gray', alpha=0.5, label='T_max=28°C')
        ax.fill_between(hours, df['T'].values,
                        where=(df['T'].values < 18) | (df['T'].values > 28),
                        alpha=0.2, color='red')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title(f'{ctrl_name}: Indoor Temperature')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # 湿度
    ax = axes[0, 1]
    if 'RH' in df.columns:
        rh_vals = df['RH'].values * 100 if df['RH'].max() <= 1.0 else df['RH'].values
        ax.plot(hours, rh_vals, color='#3498db', lw=1.2)
        ax.axhline(60, ls='--', color='gray', alpha=0.5)
        ax.axhline(80, ls='--', color='gray', alpha=0.5)
        ax.set_ylabel('Relative Humidity (%)')
        ax.set_title(f'{ctrl_name}: Indoor Relative Humidity')
        ax.grid(True, alpha=0.3)

    # CO2
    ax = axes[1, 0]
    if 'C_ppm' in df.columns:
        ax.plot(hours, df['C_ppm'].values, color='#2ecc71', lw=1.2)
        ax.axhline(400, ls='--', color='gray', alpha=0.5)
        ax.axhline(1100, ls='--', color='gray', alpha=0.5)
        ax.set_ylabel('CO2 Concentration (ppm)')
        ax.set_title(f'{ctrl_name}: Indoor CO2')
        ax.grid(True, alpha=0.3)

    # 光照
    ax = axes[1, 1]
    for col, label, color in [('I1', 'Seedling Zone', '#f39c12'),
                                ('I2', 'Transplant Zone', '#e74c3c')]:
        if col in df.columns:
            ax.plot(hours, df[col].values, label=label, color=color, lw=1.2, alpha=0.8)
    ax.set_ylabel('Light Intensity (μmol/m²/s)')
    ax.set_title(f'{ctrl_name}: LED Light Intensity')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 累计奖励
    ax = axes[2, 0]
    if 'total_reward' in df.columns:
        ax.plot(hours, df['total_reward'].values, color='#9b59b6', lw=2)
        ax.fill_between(hours, 0, df['total_reward'].values,
                        alpha=0.2, color='#9b59b6')
        ax.set_xlabel('Step (hour)')
        ax.set_ylabel('Cumulative Reward')
        ax.set_title(f'{ctrl_name}: Cumulative Reward')
        ax.grid(True, alpha=0.3)
    elif 'step_reward' in df.columns:
        ax.plot(hours, df['step_reward'].cumsum().values, color='#9b59b6', lw=2)
        ax.set_xlabel('Step (hour)')
        ax.set_ylabel('Cumulative Reward')
        ax.set_title(f'{ctrl_name}: Cumulative Reward')
        ax.grid(True, alpha=0.3)

    # 能耗成本
    ax = axes[2, 1]
    if 'elec_price' in df.columns and 'I1' in df.columns:
        ep = df['elec_price'].values
        I1 = df['I1'].values
        cost_step = ep * I1 * 0.01  # 简化的单位成本
        cumcost = np.cumsum(cost_step)
        ax.plot(hours, cumcost, color='#e67e22', lw=2)
        ax.fill_between(hours, 0, cumcost, alpha=0.2, color='#e67e22')
        ax.set_xlabel('Step (hour)')
        ax.set_ylabel('Cumulative Energy Cost (CNY)')
        ax.set_title(f'{ctrl_name}: Cumulative Energy Cost')
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Environment & Control Trajectory — {ctrl_name}\n'
                 f'First {n_days} Days ({len(df)} Steps)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'trajectory_{ctrl_name.lower().replace(" ","_")}.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[Viz] 轨迹图已保存: {save_path}')
    return save_path


def plot_batch_growth(
    batch_df: pd.DataFrame,
    save_dir: str,
    figsize: tuple = (14, 8),
) -> str:
    """
    绘制各批次(育苗区/定植区)的生物量累积曲线。

    参数:
        batch_df: per-batch详细记录DataFrame，需包含列：
            step, region, batch_id, age_h, xDn, xDs, biomass_batch, LAI
        save_dir: 保存目录

    返回:
        保存路径
    """
    _ensure_dir(save_dir)
    if batch_df.empty:
        warnings.warn("batch_df 为空，跳过批次生长图")
        return ''

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for ax, region, y_col, y_label, title in [
        (axes[0], 'seedling', 'biomass_batch', 'Biomass (kg)', 'Seedling Zone'),
        (axes[0], 'transplant', 'biomass_batch', 'Biomass (kg)', 'Transplant Zone'),
        (axes[1], 'seedling', 'LAI', 'LAI (-)', 'Seedling Zone'),
        (axes[1], 'transplant', 'LAI', 'LAI (-)', 'Transplant Zone'),
    ]:
        pass  # 图已由下面循环处理

    # 重新组织：axes[0] = biomass, axes[1] = LAI
    for ax, y_col, y_label, region_filter in [
        (axes[0], 'biomass_batch', 'Biomass (kg)', None),
        (axes[1], 'LAI', 'LAI (-)', None),
    ]:
        for region, color in [('seedling', '#27ae60'), ('transplant', '#2980b9')]:
            sub = batch_df[batch_df['region'] == region]
            if sub.empty:
                continue
            for batch_id, grp in sub.groupby('batch_id'):
                grp_sorted = grp.sort_values('step')
                ax.plot(grp_sorted['age_h'] / 24, grp_sorted[y_col],
                        label=f'{region[:4]}-{batch_id}', alpha=0.7, lw=1.5,
                        color=color)
        ax.set_xlabel('Age (days)')
        ax.set_ylabel(y_label)
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)

    axes[0].set_title('Per-Batch Biomass Accumulation')
    axes[1].set_title('Per-Batch LAI Growth')

    fig.suptitle('Batch Growth Trajectories', fontsize=12, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'batch_growth.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[Viz] 批次生长图已保存: {save_path}')
    return save_path


# =============================================================================
# 实验3专用：排程-利润热力图
# =============================================================================

def plot_schedule_profit_heatmap(
    profit_dict: dict,   # {(t1,t2,rho2,A1_A2): profit}
    save_dir: str,
    ctrl_name: str = 'Controller',
    t1_range: tuple = (10, 18),
    t2_range: tuple = (18, 26),
    figsize: tuple = (10, 8),
) -> str:
    """
    绘制 (t1, t2) 网格上的利润热力图，按控制器分组。

    参数:
        profit_dict: {(t1,t2): avg_profit} 字典
        save_dir: 保存目录
        ctrl_name: 控制器名
        t1_range: t1 范围 (min, max)
        t2_range: t2 范围 (min, max)

    返回:
        保存路径
    """
    _ensure_dir(save_dir)

    t1_vals = sorted(set(k[0] for k in profit_dict.keys()))
    t2_vals = sorted(set(k[1] for k in profit_dict.keys()))

    matrix = np.zeros((len(t2_vals), len(t1_vals)))
    for i, t2 in enumerate(t2_vals):
        for j, t1 in enumerate(t1_vals):
            key = (t1, t2)
            if key in profit_dict:
                matrix[i, j] = profit_dict[key]
            else:
                matrix[i, j] = np.nan

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, aspect='auto', cmap='viridis', origin='lower')
    ax.set_xticks(np.arange(len(t1_vals)))
    ax.set_yticks(np.arange(len(t2_vals)))
    ax.set_xticklabels(t1_vals)
    ax.set_yticklabels(t2_vals)
    ax.set_xlabel('t1 — Seedling Period (days)')
    ax.set_ylabel('t2 — Transplant Period (days)')
    ax.set_title(f'Schedule Profit Heatmap — {ctrl_name}')
    plt.colorbar(im, ax=ax, label='Profit (CNY)')

    for i in range(len(t2_vals)):
        for j in range(len(t1_vals)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                        fontsize=7, color='white' if val < matrix.mean() else 'black')

    save_path = os.path.join(save_dir, f'schedule_heatmap_{ctrl_name.lower().replace(" ","_")}.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[Viz] 排程热力图已保存: {save_path}')
    return save_path


# =============================================================================
# 一键可视化（根据实验类型自动选择）
# =============================================================================

def auto_plot(exp_type: str, results_dir: str, save_dir: str = None) -> dict:
    """
    根据实验类型自动调用对应的可视化函数。

    参数:
        exp_type: 'train' | 'eval' | 'bo'
        results_dir: 结果目录
        save_dir: 图片保存目录（默认 results_dir/figures）

    返回:
        结果字典
    """
    if save_dir is None:
        save_dir = os.path.join(results_dir, 'figures')
    _ensure_dir(save_dir)

    if exp_type == 'train':
        return plot_training_from_results_dir(results_dir)
    elif exp_type == 'eval':
        return plot_controller_comparison(results_dir, save_dir)
    elif exp_type == 'bo':
        return plot_bo_convergence(results_dir, save_dir)
    else:
        raise ValueError(f"Unknown exp_type: {exp_type}")
