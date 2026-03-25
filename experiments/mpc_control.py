# -*- coding: utf-8 -*-
"""
MPC控制器实验主入口

统一的MPC评估框架，支持多种运行模式。

【核心类】MPCExperiment（src/controllers/mpc_experiment.py）
    - 管理MPC闭环仿真
    - 自动记录 per-batch 详细数据
    - 支持与RL/Baseline对比

运行模式:
    test:       单次MPC运行（详细输出）
    mpc:        单次或多次MPC统计评估
    baseline:   默认规则控制器评估
    compare:    MPC vs Baseline 对比
    ablation:   不同Np的对比
    schedule:   不同排程参数的鲁棒性测试

使用方法:
    python experiments/mpc_control.py --mode test
    python experiments/mpc_control.py --mode mpc --n_runs 3
    python experiments/mpc_control.py --mode compare --n_runs 3
    python experiments/mpc_control.py --mode ablation --Np_list 4 8 16
    python experiments/mpc_control.py --mode schedule
    python experiments/mpc_control.py --mode mpc --save --save_dir results/mpc

来源: plant_factory_optimization项目, 论文方法部分
"""

import os
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, List

# 添加项目根目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, project_dir)

from src.envs.plant_factory_env import MultiBatchPlantFactoryEnv
from src.envs.utils import load_all_configs, create_default_schedule
from src.models import co2_density_to_ppm, _batch_indices, env_and_batch_to_mpc_state
from src.controllers import PlantFactoryMPC, MPCExperiment


# =============================================================================
# 配置加载（统一使用 load_all_configs，keys保留 _params 后缀）
# =============================================================================

def build_env_config(base_configs: Dict, schedule: Dict, seed: int) -> Dict:
    """从统一的 configs 字典构建环境配置（keys带 _params 后缀）"""
    return {
        'schedule': schedule, 'seed': seed, 'dt': 3600.0,
        'container_params': base_configs.get('container_params', {}),
        'crop_params': base_configs.get('crop_params', {}),
        'equipment_params': base_configs.get('equipment_params', {}),
        'reward_params': base_configs.get('reward_params', {}),
    }


def build_mpc_config(base_configs: Dict, schedule: Dict) -> Dict:
    """从统一的 configs 字典构建MPC配置"""
    mpc_cfg = base_configs.get('mpc_params', {})
    return {
        'mpc': mpc_cfg.get('mpc', {}),
        'soft_constraints': mpc_cfg.get('soft_constraints', {}),
        'objective': mpc_cfg.get('objective', {}),
        'equipment_limits': mpc_cfg.get('equipment_limits', {}),
        'crop_params': {**base_configs.get('crop_params', {}), **base_configs.get('container_params', {})},
        'container_params': base_configs.get('container_params', {}),
        'default_schedule': schedule,
    }


def get_configs(config_dir: Optional[str] = None) -> tuple:
    """加载并返回所有配置（使用统一的 load_all_configs）"""
    if config_dir is None:
        config_dir = os.path.join(project_dir, 'configs')
    configs = load_all_configs(config_dir)
    # 合并container和crop参数（用于某些需要同时访问两者的场景）
    full_crop = {**configs.get('container_params', {}), **configs.get('crop_params', {})}
    return configs, full_crop


# =============================================================================
# 打印辅助
# =============================================================================

def print_batch_summary_table(batch_manager, lumped: Dict, indent: str = "  "):
    """打印每个batch的详细状态表"""
    for region, batches, area_total in [
        ('seedling', batch_manager.seedling_batches, batch_manager.A1),
        ('transplant', batch_manager.transplant_batches, batch_manager.A2),
    ]:
        n = len(batches)
        if n == 0:
            continue
        area_per = area_total / n
        print(f"{indent}--- {region.capitalize()} Zone (N={n}) ---")
        print(f"{indent}  {'Idx':>3} {'Id':>3} {'Age(d)':>6} "
              f"{'xDn':>8} {'xDs':>8} {'LAI':>6} {'Biomass(kg)':>12}")
        print(f"{indent}  {'-'*3}-{'-'*3}-{'-'*6}-{'-'*8}-{'-'*8}-{'-'*6}-{'-'*12}")
        for i, b in enumerate(batches):
            bm = (b.xDn + b.xDs) * area_per
            print(f"{indent}  {i:3d} {b.batch_id:3d} {b.age_h/24:6.1f} "
                  f"{b.xDn:8.4f} {b.xDs:8.4f} {b.LAI:6.2f} {bm:12.4f}")
        total_bm = sum((b.xDn + b.xDs) * area_per for b in batches)
        key = 'lai_seedling' if region == 'seedling' else 'lai_transplant'
        print(f"{indent}  {'TOTAL':>9} zone_bm={total_bm:.4f}kg LAI={lumped.get(key, 0):.2f}")


def print_episode_summary(
    summary: Dict, stats: Dict,
    batch_summary: Optional[Dict] = None
):
    """打印Episode摘要"""
    print(f"\n{'='*80}")
    print(f"  Episode Summary")
    print(f"{'='*80}")
    print(f"  步数: {summary['n_steps']} ({summary['n_steps']/24:.1f}天) "
          f"| Seed={summary.get('seed', '?')}")
    print(f"  总奖励: {summary['total_reward']:+.3f} "
          f"| 均: {summary['avg_reward']:+.4f}±{summary.get('std_reward', 0):.4f}")
    print(f"  约束违反率: {summary['violation_rate']:.1%} "
          f"| T: {summary['T_mean']:.1f}±{summary.get('T_std', 0):.1f}°C "
          f"| RH: {summary['RH_mean']*100:.0f}% "
          f"| CO2: {summary['C_ppm_mean']:.0f}ppm")
    print(f"  DLI: 育苗={summary.get('dLI1_end', 0):.1f} "
          f"| 定植={summary.get('dLI2_end', 0):.1f}")
    print(f"  动作: I1={summary.get('I1_mean', 0):.0f} "
          f"| I2={summary.get('I2_mean', 0):.0f} "
          f"| Q_HVAC={summary.get('Q_HVAC_mean', 0):+.0f}")
    print(f"  采收: {summary.get('total_harvests', 0)}次 "
          f"| 总质量={summary.get('total_harvest_mass_kg', 0):.3f}kg "
          f"| 移栽: {summary.get('total_transplants', 0)}次")
    if stats:
        print(f"  MPC: 成功率={stats.get('success_rate', 0):.1%} "
              f"| 平均求解={stats.get('avg_solve_time', 0)*1000:.1f}ms")
    print(f"{'='*80}")


# =============================================================================
# 核心运行函数
# =============================================================================

def run_mpc_single(
    schedule: Dict,
    Np: int = 8,
    n_steps: Optional[int] = None,
    seed: int = 42,
    config_dir: Optional[str] = None,
    verbose: bool = True,
    log_interval: int = 6,
    save_dir: Optional[str] = None,
    exp_name: Optional[str] = None,
) -> MPCExperiment:
    """
    单次MPC运行（使用 MPCExperiment 框架）。

    参数:
        schedule: 排程参数
        Np: 预测步数
        n_steps: 仿真步数（默认=定植周期）
        seed: 随机种子
        verbose: 是否打印详细输出
        log_interval: batch详情打印间隔（步）
        save_dir: 保存目录（None则不保存）
        exp_name: 实验名称前缀

    返回:
        MPCExperiment 实例（包含 results 和 batch_records）
    """
    configs, full_crop = get_configs(config_dir)
    env_config = build_env_config(configs, schedule, seed)
    mpc_config = build_mpc_config(configs, schedule)

    exp = MPCExperiment(
        env_config=env_config,
        schedule=schedule,
        mpc_config=mpc_config,
        Np=Np,
        verbose=verbose,
        seed=seed,
        record_detailed=True,
    )

    if n_steps is None:
        n_steps = schedule['t2'] * 24

    if verbose:
        print(f"\n{'#'*80}")
        print(f"# MPC Run | t1={schedule['t1']}d t2={schedule['t2']}d "
              f"rho2={schedule['rho2']} A1/A2={schedule['A1_A2']:.2f}")
        print(f"# Np={Np} | Steps={n_steps} ({n_steps/24:.1f}天) | Seed={seed}")
        print(f"# A1={exp.A1:.1f}m2 | A2={exp.A2:.1f}m2")
        print(f"{'#'*80}")

    exp.run(n_steps=n_steps, use_mpc=True, log_interval=log_interval)

    summary = exp.get_summary()
    stats = exp.mpc.get_statistics()
    batch_summary = exp.get_batch_summary() if exp.batch_records else None

    if verbose:
        print_episode_summary(summary, stats, batch_summary)
        if batch_summary:
            print(f"\n  --- Per-Batch 摘要 ---")
            for key, bs in sorted(batch_summary.items()):
                print(f"    {key}: age={bs.get('final_age_h', 0)/24:.1f}d "
                      f"xDs: {bs.get('final_xDs', 0):.4f} "
                      f"LAI: {bs.get('final_LAI', 0):.2f} "
                      f"bm: {bs.get('max_biomass', 0):.4f}kg Δ={bs.get('max_biomass', 0) - bs.get('max_biomass', 0):+.4f}")

    if save_dir and exp_name:
        exp.save_results(save_dir, exp_name)
        print(f"\n  Saved: {save_dir}/{exp_name}_*.csv")

    return exp


def run_mpc_multi(
    schedule: Dict,
    Np: int = 8,
    n_runs: int = 3,
    seed_base: int = 42,
    config_dir: Optional[str] = None,
    verbose: bool = True,
    save_dir: Optional[str] = None,
) -> List[MPCExperiment]:
    """多次MPC独立运行（返回所有实验实例）"""
    exps = []
    for i in range(n_runs):
        seed = seed_base + i
        if verbose and n_runs > 1:
            print(f"\n{'#'*60} Run {i+1}/{n_runs} (seed={seed}) {'#'*60}")
        exp = run_mpc_single(
            schedule=schedule, Np=Np, n_steps=None,
            seed=seed, config_dir=config_dir,
            verbose=verbose,
            log_interval=48 if n_runs > 1 else 6,
            save_dir=save_dir,
            exp_name=f"mpc_run{i+1}" if save_dir else None,
        )
        exps.append(exp)

    if len(exps) > 1:
        rewards = [e.get_summary()['total_reward'] for e in exps]
        viols = [e.get_summary()['violation_rate'] for e in exps]
        harvest_m = [e.get_summary().get('total_harvest_mass_kg', 0) for e in exps]
        mpc_stats = [e.mpc.get_statistics() for e in exps]
        print(f"\n{'='*80}")
        print(f"  多次运行聚合 (n={n_runs})")
        print(f"{'='*80}")
        print(f"  总奖励: {np.mean(rewards):+.3f} ± {np.std(rewards):.3f} "
              f"[{np.min(rewards):+.3f}, {np.max(rewards):+.3f}]")
        print(f"  违反率: {np.mean(viols):.1%} ± {np.std(viols):.1%}")
        print(f"  采收质量: {np.mean(harvest_m):.3f} ± {np.std(harvest_m):.3f} kg")
        if mpc_stats:
            rates = [s.get('success_rate', 0) for s in mpc_stats]
            times = [s.get('avg_solve_time', 0) for s in mpc_stats]
            print(f"  MPC成功率: {np.mean(rates):.1%}")
            print(f"  平均求解时间: {np.mean(times)*1000:.1f}ms")
        print(f"{'='*80}")

    return exps


def run_baseline_single(
    schedule: Dict,
    n_steps: Optional[int] = None,
    seed: int = 42,
    config_dir: Optional[str] = None,
    verbose: bool = True,
) -> Dict:
    """
    使用默认规则控制器运行仿真（Baseline）。

    规则：白天I=300μmol/m²/s，夜间I=0，其余用默认值。
    仅做仿真不构建MPC，用于与MPC性能对比。
    """
    configs, _ = get_configs(config_dir)
    env_config = build_env_config(configs, schedule, seed)

    env = MultiBatchPlantFactoryEnv(config=env_config)
    if n_steps is None:
        n_steps = schedule['t2'] * 24

    obs, _ = env.reset(seed=seed, options={'schedule': schedule})
    total_reward = 0.0
    n_violations = 0
    records = []

    # 从配置读取约束边界（与reward_params.yaml保持一致）
    rp = configs.get('reward_params', {})
    temp_min = rp.get('temp_hard_min', 18.0)
    temp_max = rp.get('temp_hard_max', 28.0)
    rh_min = rp.get('rh_soft_min', 60.0) / 100.0  # % → [-]
    rh_max = rp.get('rh_soft_max', 80.0) / 100.0
    co2_min = rp.get('co2_min', 400.0)
    co2_max = rp.get('co2_max', 1200.0)

    ep_cfg = configs.get('equipment_params', {})
    elec_price = ep_cfg.get('p_elec_base', 0.6)

    for step in range(n_steps):
        hour = env.hour_of_day
        # 默认规则动作
        I1 = 300.0 if 6 <= hour < 22 else 0.0
        I2 = 300.0 if 6 <= hour < 22 else 0.0
        action = np.array([I1, I2, 0.0, 0.0, 0.01, 1e-5], dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        T = env.state[1]
        RH = env.state[2]
        C_kg = env.state[0]
        C_ppm = co2_density_to_ppm(C_kg, T)
        viol = (T < temp_min or T > temp_max or
                RH < rh_min or RH > rh_max or
                C_ppm < co2_min or C_ppm > co2_max)
        if viol:
            n_violations += 1

        records.append({
            'step': step, 'hour_of_day': hour,
            'T': T, 'RH': RH, 'C_ppm': C_ppm,
            'I1': I1, 'I2': I2,
            'step_reward': reward, 'total_reward': total_reward,
            'violation': int(viol), 'elec_price': elec_price,
        })

        if verbose and step % 48 == 0:
            print(f"  Step {step:4d} | T={T:.1f}C RH={RH*100:.0f}% "
                  f"I1={I1:.0f} I2={I2:.0f} "
                  f"reward={reward:.3f} viol={int(viol)}")

        if terminated or step >= n_steps - 1:
            break

    df = pd.DataFrame(records)
    summary = {
        'total_reward': total_reward,
        'avg_reward': df['step_reward'].mean(),
        'std_reward': df['step_reward'].std(),
        'violation_rate': n_violations / max(len(df), 1),
        'T_mean': df['T'].mean(),
        'T_std': df['T'].std(),
        'RH_mean': df['RH'].mean(),
        'C_ppm_mean': df['C_ppm'].mean(),
        'I1_mean': df['I1'].mean(),
        'I2_mean': df['I2'].mean(),
        'total_harvests': env.batch_manager.total_harvests,
        'total_harvest_mass_kg': env.batch_manager.total_harvest_mass,
    }

    if verbose:
        print(f"\n{'='*80}")
        print(f"  Baseline Summary: reward={summary['total_reward']:+.3f} "
              f"viol={summary['violation_rate']:.1%} "
              f"harvests={summary['total_harvests']} "
              f"mass={summary['total_harvest_mass_kg']:.3f}kg")
        print(f"{'='*80}")

    return {'summary': summary, 'trajectory': df, 'exp': env}


# =============================================================================
# 高级模式
# =============================================================================

def run_ablation(
    schedule: Dict,
    Np_list: List[int],
    n_runs: int = 3,
    seed_base: int = 42,
    config_dir: Optional[str] = None,
    verbose: bool = True,
) -> Dict[int, Dict]:
    """不同预测步数Np的性能对比"""
    results = {}
    for Np in Np_list:
        if verbose:
            print(f"\n{'#'*60} Np={Np} {'#'*60}")
        exps = run_mpc_multi(
            schedule=schedule, Np=Np, n_runs=n_runs,
            seed_base=seed_base, config_dir=config_dir,
            verbose=verbose,
        )
        summaries = [e.get_summary() for e in exps]
        stats = [e.mpc.get_statistics() for e in exps]
        results[Np] = {
            'reward_mean': np.mean([s['total_reward'] for s in summaries]),
            'reward_std': np.std([s['total_reward'] for s in summaries]),
            'viol_mean': np.mean([s['violation_rate'] for s in summaries]),
            'viol_std': np.std([s['violation_rate'] for s in summaries]),
            'energy_mean': np.mean([s.get('total_energy_kWh', 0) for s in summaries]),
            'solve_rate_mean': np.mean([s.get('success_rate', 0) for s in stats]),
            'solve_time_mean': np.mean([s.get('avg_solve_time', 0) for s in stats]),
            'per_run': summaries,
        }

    if verbose:
        print(f"\n{'='*80}")
        print(f"  Ablation Summary")
        print(f"{'='*80}")
        print(f"  {'Np':>4} | {'Reward':>10} | {'Viol%':>8} | {'Succ%':>8} | {'Time(ms)':>9}")
        print(f"  {'-'*4}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*9}")
        for Np, r in results.items():
            print(f"  {Np:>4} | {r['reward_mean']:>+10.3f} | "
                  f"{r['viol_mean']:>7.1%} | {r['solve_rate_mean']:>7.1%} | "
                  f"{r['solve_time_mean']*1000:>9.1f}")
        print(f"{'='*80}")

    return results


def run_schedule_robustness(
    Np: int = 8,
    n_runs: int = 3,
    seed_base: int = 42,
    config_dir: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Dict]:
    """不同排程参数下的MPC鲁棒性测试"""
    schedules = [
        ('Standard', {'t1': 14, 't2': 21, 'rho2': 35.0, 'A1_A2': 0.5}),
        ('FastCycle', {'t1': 10, 't2': 18, 'rho2': 50.0, 'A1_A2': 0.8}),
        ('HighDensity', {'t1': 14, 't2': 21, 'rho2': 70.0, 'A1_A2': 0.3}),
        ('LongGrowth', {'t1': 18, 't2': 26, 'rho2': 30.0, 'A1_A2': 1.0}),
    ]

    results = {}
    for name, sched in schedules:
        if verbose:
            print(f"\n{'#'*60} Schedule: {name} {'#'*60}")
        exps = run_mpc_multi(
            schedule=sched, Np=Np, n_runs=n_runs,
            seed_base=seed_base, config_dir=config_dir,
            verbose=verbose,
        )
        summaries = [e.get_summary() for e in exps]
        results[name] = {
            'reward_mean': np.mean([s['total_reward'] for s in summaries]),
            'reward_std': np.std([s['total_reward'] for s in summaries]),
            'viol_mean': np.mean([s['violation_rate'] for s in summaries]),
            'per_run': summaries,
        }

    if verbose:
        print(f"\n{'='*80}")
        print(f"  Schedule Robustness Summary")
        print(f"{'='*80}")
        print(f"  {'Schedule':>12} | {'Reward':>10} | {'Viol%':>8}")
        print(f"  {'-'*12}-+-{'-'*10}-+-{'-'*8}")
        for name, r in results.items():
            print(f"  {name:>12} | {r['reward_mean']:>+10.3f} | {r['viol_mean']:>7.1%}")
        print(f"{'='*80}")

    return results


def run_compare(
    schedule: Dict,
    Np: int = 8,
    n_runs: int = 3,
    seed_base: int = 42,
    config_dir: Optional[str] = None,
    verbose: bool = True,
) -> Dict:
    """MPC vs Baseline 对比评估"""
    print(f"\n{'#'*80}")
    print(f"# Compare: MPC vs Baseline")
    print(f"# Schedule: t1={schedule['t1']}d t2={schedule['t2']}d "
          f"rho2={schedule['rho2']} A1/A2={schedule['A1_A2']:.2f}")
    print(f"# Np={Np} | Runs={n_runs}")
    print(f"{'#'*80}")

    # Baseline
    print(f"\n--- Baseline (rule-based controller) ---")
    bl_summaries = []
    for i in range(min(n_runs, 3)):
        bl = run_baseline_single(
            schedule=schedule, seed=seed_base + i,
            config_dir=config_dir, verbose=False,
        )
        bl_summaries.append(bl['summary'])

    # MPC
    print(f"\n--- MPC (per-batch model) ---")
    mpc_exps = run_mpc_multi(
        schedule=schedule, Np=Np, n_runs=n_runs,
        seed_base=seed_base, config_dir=config_dir,
        verbose=False,
    )
    mpc_summaries = [e.get_summary() for e in mpc_exps]

    # 对比摘要
    bl_rew = [s['total_reward'] for s in bl_summaries]
    mpc_rew = [s['total_reward'] for s in mpc_summaries]
    bl_vio = [s['violation_rate'] for s in bl_summaries]
    mpc_vio = [s['violation_rate'] for s in mpc_summaries]
    bl_hm = [s.get('total_harvest_mass_kg', 0) for s in bl_summaries]
    mpc_hm = [s.get('total_harvest_mass_kg', 0) for s in mpc_summaries]

    print(f"\n{'='*80}")
    print(f"  Comparison (Baseline vs MPC, n={max(len(bl_summaries), 1)}/{n_runs} runs)")
    print(f"{'='*80}")
    print(f"  {'Controller':>12} | {'TotalReward':>12} | {'ViolRate':>9} | "
          f"{'Harvest(kg)':>11} | {'Harvester':>9}")
    print(f"  {'-'*12}-+-{'-'*12}-+-{'-'*9}-+-{'-'*11}-+-{'-'*9}")
    print(f"  {'Baseline':>12} | {np.mean(bl_rew):>+12.3f} | "
          f"{np.mean(bl_vio):>8.1%} | "
          f"{np.mean(bl_hm):>11.3f} | "
          f"{np.mean([s['total_harvests'] for s in bl_summaries]):>9.0f}")
    print(f"  {'MPC':>12} | {np.mean(mpc_rew):>+12.3f} | "
          f"{np.mean(mpc_vio):>8.1%} | "
          f"{np.mean(mpc_hm):>11.3f} | "
          f"{np.mean([s['total_harvests'] for s in mpc_summaries]):>9.0f}")
    print(f"  {'-'*12}-+-{'-'*12}-+-{'-'*9}-+-{'-'*11}-+-{'-'*9}")
    delta_rew = np.mean(mpc_rew) - np.mean(bl_rew)
    base_rew = np.mean(bl_rew)
    rel_imp = delta_rew / abs(base_rew) if abs(base_rew) > 1e-6 else 0
    print(f"  {'Improvement':>12} | {delta_rew:>+12.3f} ({rel_imp:+.1%}) | "
          f"{np.mean(bl_vio)-np.mean(mpc_vio):>+8.1%} | "
          f"{np.mean(mpc_hm)-np.mean(bl_hm):>+11.3f} | -")
    print(f"{'='*80}")

    return {
        'baseline': {'summaries': bl_summaries, 'reward_mean': np.mean(bl_rew)},
        'mpc': {'exps': mpc_exps, 'summaries': mpc_summaries,
                 'reward_mean': np.mean(mpc_rew)},
    }


# =============================================================================
# 命令行接口
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='MPC控制器评估（统一入口）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python experiments/mpc_control.py --mode test
  python experiments/mpc_control.py --mode mpc --n_runs 3
  python experiments/mpc_control.py --mode compare --n_runs 3
  python experiments/mpc_control.py --mode mpc --t1 14 --t2 21 --rho2 50 --Np 8
  python experiments/mpc_control.py --mode ablation --Np_list 4 8 16
  python experiments/mpc_control.py --mode schedule
  python experiments/mpc_control.py --mode mpc --save --save_dir results/mpc
        """
    )
    parser.add_argument('--mode', type=str, default='test',
                       choices=['test', 'mpc', 'baseline', 'compare', 'ablation', 'schedule'],
                       help='运行模式')
    # 排程参数
    parser.add_argument('--t1', type=int, default=14, help='育苗期天数')
    parser.add_argument('--t2', type=int, default=21, help='定植期天数')
    parser.add_argument('--rho2', type=float, default=35.0, help='定植区密度 [株/m²]')
    parser.add_argument('--A1_A2', type=float, default=0.5, help='育苗/定植面积比')
    # MPC参数
    parser.add_argument('--Np', type=int, default=8, help='MPC预测步数')
    parser.add_argument('--n_runs', type=int, default=1, help='独立运行次数')
    parser.add_argument('--n_steps', type=int, default=None, help='仿真步数（默认=定植周期）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子基数')
    parser.add_argument('--log_interval', type=int, default=6,
                        help='batch详情打印间隔（步，默认6）')
    # 保存
    parser.add_argument('--save', action='store_true', help='保存结果到CSV')
    parser.add_argument('--save_dir', type=str, default='results/mpc', help='保存目录')
    parser.add_argument('--quiet', action='store_true', help='静默模式')
    parser.add_argument('--config_dir', type=str, default=None, help='配置文件目录')
    # Ablation
    parser.add_argument('Np_list', nargs='*', type=int, help='Ablation的Np值列表')
    return parser.parse_args()


def main():
    args = parse_args()

    config_dir = args.config_dir or os.path.join(project_dir, 'configs')
    if not os.path.exists(config_dir):
        print(f"Error: Config directory not found: {config_dir}")
        return

    verbose = not args.quiet
    save_dir = args.save_dir if args.save else None

    schedule = {
        't1': args.t1, 't2': args.t2,
        'rho2': args.rho2, 'A1_A2': args.A1_A2,
    }

    print(f"\n{'#'*80}")
    print(f"# MPC Control | mode={args.mode} | t1={args.t1}d t2={args.t2}d "
          f"rho2={args.rho2} A1/A2={args.A1_A2:.2f} | Np={args.Np} | Runs={args.n_runs}")
    print(f"{'#'*80}")

    if args.mode == 'test':
        # 单次详细运行
        run_mpc_single(
            schedule=schedule, Np=args.Np, n_steps=args.n_steps,
            seed=args.seed, config_dir=config_dir,
            verbose=True, log_interval=args.log_interval,
            save_dir=save_dir, exp_name='mpc_test' if save_dir else None,
        )

    elif args.mode == 'mpc':
        # 多次MPC评估
        if args.n_runs == 1:
            run_mpc_single(
                schedule=schedule, Np=args.Np, n_steps=args.n_steps,
                seed=args.seed, config_dir=config_dir,
                verbose=True, log_interval=args.log_interval,
                save_dir=save_dir,
                exp_name=f"mpc_r{args.seed}" if save_dir else None,
            )
        else:
            run_mpc_multi(
                schedule=schedule, Np=args.Np, n_runs=args.n_runs,
                seed_base=args.seed, config_dir=config_dir,
                verbose=verbose, save_dir=save_dir,
            )

    elif args.mode == 'baseline':
        run_baseline_single(
            schedule=schedule, n_steps=args.n_steps,
            seed=args.seed, config_dir=config_dir, verbose=True,
        )

    elif args.mode == 'compare':
        run_compare(
            schedule=schedule, Np=args.Np, n_runs=args.n_runs,
            seed_base=args.seed, config_dir=config_dir, verbose=True,
        )

    elif args.mode == 'ablation':
        Np_values = args.Np_list or [4, 8, 12, 16, 24]
        run_ablation(
            schedule=schedule, Np_list=Np_values,
            n_runs=args.n_runs, seed_base=args.seed,
            config_dir=config_dir, verbose=verbose,
        )

    elif args.mode == 'schedule':
        run_schedule_robustness(
            Np=args.Np, n_runs=args.n_runs,
            seed_base=args.seed, config_dir=config_dir, verbose=verbose,
        )

    print(f"\n完成!")


if __name__ == '__main__':
    main()
