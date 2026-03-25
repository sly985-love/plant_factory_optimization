# -*- coding: utf-8 -*-
"""
Experiment 2: 固定排程下PID/MPC/ContextualRL控制器评估

在预定义的固定排程集合上，评估三种下层控制器的性能：
  PID   — 离散PID温度控制器 + 规则CO2/通风/除湿
  MPC   — 模型预测控制器（CasADi非线性NLP）
  CRL   — 上下文强化学习（PPO，需训练好的模型）

使用方法:
    # 完整评估（需先训练RL模型）
    python experiments/evaluate_controllers.py --mode all --n_runs 3 --save

    # 仅PID
    python experiments/evaluate_controllers.py --mode pid --n_runs 3 --save

    # 仅MPC
    python experiments/evaluate_controllers.py --mode mpc --n_runs 3 --save

    # 仅RL（需指定模型路径）
    python experiments/evaluate_controllers.py --mode rl --rl_model results/models/best_model.zip

    # 在特定排程上评估
    python experiments/evaluate_controllers.py --mode all --t1 14 --t2 21 --rho2 35 --A1_A2 0.5

来源: 论文方法部分 2.5 / 2.6 对比实验
"""

import os
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

# 添加项目根目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, project_dir)

from src.envs.plant_factory_env import MultiBatchPlantFactoryEnv
from src.envs.utils import load_all_configs
from src.models import co2_density_to_ppm
from src.controllers import (
    PlantFactoryMPC,
    MPCExperiment,
    RLClosedLoopExperiment,
    PIDController,
    RuleController,
)
from experiments.mpc_control import (
    run_mpc_single,
    run_mpc_multi,
    run_baseline_single,
    build_env_config,
    build_mpc_config,
    get_configs,
)


# =============================================================================
# PID评估
# =============================================================================

def evaluate_pid(
    schedule: Dict[str, Any],
    n_runs: int = 3,
    seed_base: int = 42,
    config_dir: Optional[str] = None,
    n_steps: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[List[Dict], List[pd.DataFrame]]:
    """
    使用PID控制器评估固定排程。

    PID控制器控制温度（离散PID），其余动作（CO2/通风/除湿）用规则策略。
    光照为固定设定点。

    参数:
        schedule: 排程参数字典
        n_runs: 独立运行次数
        seed_base: 随机种子基数
        config_dir: 配置文件目录
        n_steps: 仿真步数（默认=定植周期）
        verbose: 是否打印详细信息

    返回:
        (summaries, trajectories): 每轮摘要列表 + 轨迹DataFrame列表
    """
    configs, _ = get_configs(config_dir)
    env_config = build_env_config(configs, schedule, seed=seed_base)

    if n_steps is None:
        n_steps = schedule['t2'] * 24

    summaries = []
    trajectories = []

    for run_i in range(n_runs):
        seed = seed_base + run_i
        pid_ctrl = PIDController(config=env_config)
        rule_ctrl = RuleController(config=env_config)

        env = MultiBatchPlantFactoryEnv(config=env_config)
        obs, _ = env.reset(seed=seed, options={'schedule': schedule})

        total_reward = 0.0
        n_violations = 0
        records = []

        ep_cfg = configs.get('equipment_params', {})
        rp = configs.get('reward_params', {})
        temp_min = rp.get('temp_hard_min', 18.0)
        temp_max = rp.get('temp_hard_max', 28.0)
        rh_min = rp.get('rh_soft_min', 60.0) / 100.0
        rh_max = rp.get('rh_soft_max', 80.0) / 100.0
        co2_min = rp.get('co2_min', 400.0)
        co2_max = rp.get('co2_max', 1200.0)

        for step in range(n_steps):
            hour = env.hour_of_day

            # PID温度控制
            obs_arr = obs if isinstance(obs, np.ndarray) else np.array(obs, dtype=np.float32)
            action = pid_ctrl.predict(obs_arr)

            # 其余动作用RuleController（更完整的规则策略）
            rule_action = rule_ctrl.predict(obs_arr)
            # I1, I2 用RuleController的光照策略，PID的Q_HVAC
            action[0] = rule_action[0]   # I1
            action[1] = rule_action[1]   # I2
            action[3] = rule_action[3]   # CO2
            action[4] = rule_action[4]   # V_vent
            action[5] = rule_action[5]   # m_dehum

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            env_state = env.state
            T = env_state[1]
            RH = env_state[2]
            C_kg = env_state[0]
            C_ppm = co2_density_to_ppm(C_kg, T)

            viol = (T < temp_min or T > temp_max or
                    RH < rh_min or RH > rh_max or
                    C_ppm < co2_min or C_ppm > co2_max)
            if viol:
                n_violations += 1

            records.append({
                'run_id': run_i,
                'step': step,
                'hour_of_day': hour,
                'T': T,
                'RH': RH,
                'C_ppm': C_ppm,
                'I1': float(action[0]),
                'I2': float(action[1]),
                'Q_HVAC': float(action[2]),
                'u_CO2': float(action[3]),
                'V_vent': float(action[4]),
                'm_dehum': float(action[5]),
                'step_reward': reward,
                'total_reward': total_reward,
                'violation': int(viol),
                'elec_price': env.elec_price,
                'n_seedling_batches': len(env.batch_manager.seedling_batches),
                'n_transplant_batches': len(env.batch_manager.transplant_batches),
                'total_harvests': env.batch_manager.total_harvests,
                'total_transplants': env.batch_manager.total_transplants,
                'total_harvest_mass_kg': env.batch_manager.total_harvest_mass,
            })

            if terminated or step >= n_steps - 1:
                break

        df = pd.DataFrame(records)
        trajectories.append(df)

        # 摘要统计
        dt_hours = env.dt / 3600.0
        ep = env.equipment_params
        c_optical_eff = ep.get('c_optical_eff', 2.5)
        c_led_eff = ep.get('c_led_eff', 0.52)
        c_COP = ep.get('c_COP', 3.0)
        fan_eff = ep.get('fan_eff', 7.07)
        c_dehum_eev = ep.get('c_dehum_eev', 3.0)
        p_CO2 = ep.get('p_CO2', 0.5)

        A_total = schedule.get('A1', 13.33) + schedule.get('A2', 26.67)

        I1_mean = df['I1'].mean()
        I2_mean = df['I2'].mean()
        E_led = ((I1_mean / c_optical_eff) * schedule.get('A1', 13.33) / c_led_eff +
                 (I2_mean / c_optical_eff) * schedule.get('A2', 26.67) / c_led_eff) * \
                 df['I1'].gt(0).mean() * len(df) * dt_hours / 1000.0
        Q_HVAC_mean = df['Q_HVAC'].mean()
        E_hvac = abs(Q_HVAC_mean) * A_total / c_COP * len(df) * dt_hours / 1000.0
        V_vent_mean = df['V_vent'].mean()
        E_vent = V_vent_mean * A_total / fan_eff * len(df) * dt_hours / 1000.0
        m_dehum_mean = df['m_dehum'].mean()
        E_dehum = (m_dehum_mean * A_total) / c_dehum_eev * 1000.0 * len(df) * dt_hours / 1000.0
        u_CO2_mean = df['u_CO2'].mean()
        CO2_kg = u_CO2_mean * A_total * len(df) * dt_hours / 1000.0

        total_energy = E_led + E_hvac + E_vent + E_dehum
        total_cost = total_energy * df['elec_price'].mean() + CO2_kg * p_CO2

        summaries.append({
            'run_id': run_i,
            'seed': seed,
            'n_steps': len(df),
            'total_reward': float(total_reward),
            'avg_reward': float(df['step_reward'].mean()),
            'std_reward': float(df['step_reward'].std()),
            'violation_rate': float(n_violations / max(len(df), 1)),
            'T_mean': float(df['T'].mean()),
            'T_std': float(df['T'].std()),
            'RH_mean': float(df['RH'].mean()),
            'C_ppm_mean': float(df['C_ppm'].mean()),
            'I1_mean': float(df['I1'].mean()),
            'I2_mean': float(df['I2'].mean()),
            'Q_HVAC_mean': float(df['Q_HVAC'].mean()),
            'total_energy_kWh': float(total_energy),
            'total_cost_yuan': float(total_cost),
            'harvests': int(env.batch_manager.total_harvests),
            'harvest_mass_kg': float(env.batch_manager.total_harvest_mass),
            'transplants': int(env.batch_manager.total_transplants),
        })

        if verbose:
            s = summaries[-1]
            print(f"  PID run {run_i+1}/{n_runs}: "
                  f"reward={s['total_reward']:+.2f} "
                  f"viol={s['violation_rate']:.1%} "
                  f"T={s['T_mean']:.1f}°C "
                  f"harvest={s['harvest_mass_kg']:.3f}kg")

    return summaries, trajectories


# =============================================================================
# MPC评估（复用mpc_control.py）
# =============================================================================

def evaluate_mpc(
    schedule: Dict[str, Any],
    Np: int = 8,
    n_runs: int = 3,
    seed_base: int = 42,
    config_dir: Optional[str] = None,
    n_steps: Optional[int] = None,
    verbose: bool = False,
    save_dir: Optional[str] = None,
    exp_name: Optional[str] = None,
) -> Tuple[List[Dict], List[pd.DataFrame], List[Dict]]:
    """
    使用MPC控制器评估固定排程。

    复用 experiments/mpc_control.py 中的 run_mpc_multi 函数。

    参数:
        schedule: 排程参数字典
        Np: MPC预测步数
        n_runs: 独立运行次数
        seed_base: 随机种子基数
        config_dir: 配置文件目录
        n_steps: 仿真步数
        verbose: 是否打印详细信息
        save_dir: 保存目录
        exp_name: 实验名称

    返回:
        (summaries, trajectories, batch_records): 每轮摘要 + 轨迹 + 批次记录
    """
    if n_steps is None:
        n_steps = schedule['t2'] * 24

    configs, _ = get_configs(config_dir)

    exps = run_mpc_multi(
        schedule=schedule,
        Np=Np,
        n_runs=n_runs,
        seed_base=seed_base,
        config_dir=config_dir,
        verbose=False,
        save_dir=save_dir,
    )

    summaries = []
    trajectories = []
    batch_records_list = []

    for run_i, exp in enumerate(exps):
        summary = exp.get_summary()
        # 添加seed信息
        summary['run_id'] = run_i
        summary['seed'] = seed_base + run_i
        summaries.append(summary)

        if exp.results is not None:
            traj = exp.results.copy()
            traj.insert(0, 'run_id', run_i)
            trajectories.append(traj)

        if exp.batch_records:
            batch_df = exp.get_batch_dataframe()
            batch_df.insert(0, 'run_id', run_i)
            batch_records_list.append(batch_df)

        if verbose:
            s = summary
            print(f"  MPC run {run_i+1}/{n_runs}: "
                  f"reward={s.get('total_reward', 0):+.2f} "
                  f"viol={s.get('violation_rate', 0):.1%} "
                  f"T={s.get('T_mean', 0):.1f}°C "
                  f"harvest={s.get('harvest_mass_kg', 0):.3f}kg "
                  f"solve={exp.mpc.get_statistics().get('avg_solve_time', 0)*1000:.1f}ms")

    return summaries, trajectories, batch_records_list


# =============================================================================
# RL评估
# =============================================================================

def evaluate_rl(
    schedule: Dict[str, Any],
    model_path: str,
    n_runs: int = 3,
    seed_base: int = 42,
    config_dir: Optional[str] = None,
    n_steps: Optional[int] = None,
    verbose: bool = False,
    deterministic: bool = True,
) -> Tuple[List[Dict], List[pd.DataFrame]]:
    """
    使用上下文强化学习（PPO）评估固定排程。

    参数:
        schedule: 排程参数字典
        model_path: 训练好的PPO模型路径
        n_runs: 独立运行次数
        seed_base: 随机种子基数
        config_dir: 配置文件目录
        n_steps: 仿真步数
        verbose: 是否打印详细信息
        deterministic: 是否使用确定性策略

    返回:
        (summaries, trajectories): 每轮摘要 + 轨迹DataFrame列表
    """
    if not os.path.exists(model_path):
        print(f"[WARN] RL模型未找到: {model_path}，跳过RL评估")
        return [], []

    try:
        from stable_baselines3 import PPO
    except ImportError:
        print("[WARN] stable-baselines3 未安装，跳过RL评估")
        return [], []

    configs, _ = get_configs(config_dir)
    env_config = build_env_config(configs, schedule, seed=seed_base)

    if n_steps is None:
        n_steps = schedule['t2'] * 24

    model = PPO.load(model_path)

    summaries = []
    trajectories = []

    for run_i in range(n_runs):
        seed = seed_base + run_i

        env = MultiBatchPlantFactoryEnv(config=env_config)
        obs, _ = env.reset(seed=seed, options={'schedule': schedule})

        total_reward = 0.0
        n_violations = 0
        records = []

        rp = configs.get('reward_params', {})
        temp_min = rp.get('temp_hard_min', 18.0)
        temp_max = rp.get('temp_hard_max', 28.0)
        rh_min = rp.get('rh_soft_min', 60.0) / 100.0
        rh_max = rp.get('rh_soft_max', 80.0) / 100.0
        co2_min = rp.get('co2_min', 400.0)
        co2_max = rp.get('co2_max', 1200.0)

        for step in range(n_steps):
            hour = env.hour_of_day

            action, _ = model.predict(obs, deterministic=deterministic)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            env_state = env.state
            T = env_state[1]
            RH = env_state[2]
            C_kg = env_state[0]
            C_ppm = co2_density_to_ppm(C_kg, T)

            viol = (T < temp_min or T > temp_max or
                    RH < rh_min or RH > rh_max or
                    C_ppm < co2_min or C_ppm > co2_max)
            if viol:
                n_violations += 1

            records.append({
                'run_id': run_i,
                'step': step,
                'hour_of_day': hour,
                'T': T,
                'RH': RH,
                'C_ppm': C_ppm,
                'I1': float(action[0]),
                'I2': float(action[1]),
                'Q_HVAC': float(action[2]),
                'u_CO2': float(action[3]),
                'V_vent': float(action[4]),
                'm_dehum': float(action[5]),
                'step_reward': reward,
                'total_reward': total_reward,
                'violation': int(viol),
                'elec_price': env.elec_price,
                'n_seedling_batches': len(env.batch_manager.seedling_batches),
                'n_transplant_batches': len(env.batch_manager.transplant_batches),
                'total_harvests': env.batch_manager.total_harvests,
                'total_transplants': env.batch_manager.total_transplants,
                'total_harvest_mass_kg': env.batch_manager.total_harvest_mass,
            })

            if terminated or step >= n_steps - 1:
                break

        df = pd.DataFrame(records)
        trajectories.append(df)

        # 摘要
        dt_hours = env.dt / 3600.0
        ep = env.equipment_params
        c_optical_eff = ep.get('c_optical_eff', 2.5)
        c_led_eff = ep.get('c_led_eff', 0.52)
        c_COP = ep.get('c_COP', 3.0)
        fan_eff = ep.get('fan_eff', 7.07)
        c_dehum_eev = ep.get('c_dehum_eev', 3.0)
        p_CO2 = ep.get('p_CO2', 0.5)

        A_total = schedule.get('A1', 13.33) + schedule.get('A2', 26.67)

        I1_mean = df['I1'].mean()
        I2_mean = df['I2'].mean()
        E_led = ((I1_mean / c_optical_eff) * schedule.get('A1', 13.33) / c_led_eff +
                 (I2_mean / c_optical_eff) * schedule.get('A2', 26.67) / c_led_eff) * \
                 df['I1'].gt(0).mean() * len(df) * dt_hours / 1000.0
        Q_HVAC_mean = df['Q_HVAC'].mean()
        E_hvac = abs(Q_HVAC_mean) * A_total / c_COP * len(df) * dt_hours / 1000.0
        V_vent_mean = df['V_vent'].mean()
        E_vent = V_vent_mean * A_total / fan_eff * len(df) * dt_hours / 1000.0
        m_dehum_mean = df['m_dehum'].mean()
        E_dehum = (m_dehum_mean * A_total) / c_dehum_eev * 1000.0 * len(df) * dt_hours / 1000.0
        u_CO2_mean = df['u_CO2'].mean()
        CO2_kg = u_CO2_mean * A_total * len(df) * dt_hours / 1000.0

        total_energy = E_led + E_hvac + E_vent + E_dehum
        total_cost = total_energy * df['elec_price'].mean() + CO2_kg * p_CO2

        summaries.append({
            'run_id': run_i,
            'seed': seed,
            'n_steps': len(df),
            'total_reward': float(total_reward),
            'avg_reward': float(df['step_reward'].mean()),
            'std_reward': float(df['step_reward'].std()),
            'violation_rate': float(n_violations / max(len(df), 1)),
            'T_mean': float(df['T'].mean()),
            'T_std': float(df['T'].std()),
            'RH_mean': float(df['RH'].mean()),
            'C_ppm_mean': float(df['C_ppm'].mean()),
            'I1_mean': float(df['I1'].mean()),
            'I2_mean': float(df['I2'].mean()),
            'Q_HVAC_mean': float(df['Q_HVAC'].mean()),
            'total_energy_kWh': float(total_energy),
            'total_cost_yuan': float(total_cost),
            'harvests': int(env.batch_manager.total_harvests),
            'harvest_mass_kg': float(env.batch_manager.total_harvest_mass),
            'transplants': int(env.batch_manager.total_transplants),
        })

        if verbose:
            s = summaries[-1]
            print(f"  RL run {run_i+1}/{n_runs}: "
                  f"reward={s['total_reward']:+.2f} "
                  f"viol={s['violation_rate']:.1%} "
                  f"T={s['T_mean']:.1f}°C "
                  f"harvest={s['harvest_mass_kg']:.3f}kg")

    return summaries, trajectories


# =============================================================================
# 主评估函数
# =============================================================================

def run_fixed_schedule_evaluation(
    schedule: Dict[str, Any],
    modes: List[str],
    Np: int = 8,
    n_runs: int = 3,
    seed_base: int = 42,
    config_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    rl_model_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    在固定排程上评估多种控制器。

    参数:
        schedule: 排程参数字典
        modes: 评估模式列表 ['pid', 'mpc', 'rl', 'all']
        Np: MPC预测步数
        n_runs: 独立运行次数
        seed_base: 随机种子基数
        config_dir: 配置文件目录
        save_dir: 保存目录
        rl_model_path: RL模型路径
        verbose: 是否打印详细信息

    返回:
        评估结果字典
    """
    if 'all' in modes:
        modes = ['pid', 'mpc', 'rl']

    results = {}
    all_trajectories = {}
    all_summaries = {}

    for mode in modes:
        if verbose:
            print(f"\n{'='*60}")
            print(f"# Evaluating [{mode.upper()}] on schedule: "
                  f"t1={schedule['t1']}d t2={schedule['t2']}d "
                  f"rho2={schedule['rho2']} A1/A2={schedule['A1_A2']:.2f}")
            print(f"{'='*60}")

        if mode == 'pid':
            summaries, trajectories = evaluate_pid(
                schedule=schedule, n_runs=n_runs, seed_base=seed_base,
                config_dir=config_dir, verbose=verbose,
            )
            results['pid'] = {'summaries': summaries}
            all_trajectories['pid'] = trajectories
            all_summaries['pid'] = summaries

        elif mode == 'mpc':
            summaries, trajectories, batch_records = evaluate_mpc(
                schedule=schedule, Np=Np, n_runs=n_runs, seed_base=seed_base,
                config_dir=config_dir, verbose=verbose,
                save_dir=save_dir, exp_name='mpc_eval',
            )
            results['mpc'] = {'summaries': summaries}
            all_trajectories['mpc'] = trajectories
            all_summaries['mpc'] = summaries

        elif mode == 'rl':
            if rl_model_path is None:
                print("[WARN] 未指定RL模型路径，跳过RL评估。")
                print("       请先训练RL模型：python experiments/train.py")
            else:
                summaries, trajectories = evaluate_rl(
                    schedule=schedule, model_path=rl_model_path,
                    n_runs=n_runs, seed_base=seed_base,
                    config_dir=config_dir, verbose=verbose,
                )
                results['rl'] = {'summaries': summaries}
                all_trajectories['rl'] = trajectories
                all_summaries['rl'] = summaries

    # 保存结果
    if save_dir:
        _save_evaluation_results(save_dir, all_trajectories, all_summaries)

    # 打印对比摘要
    if verbose:
        _print_comparison_table(all_summaries)

    return results


def _save_evaluation_results(
    save_dir: str,
    trajectories: Dict[str, List[pd.DataFrame]],
    summaries: Dict[str, List[Dict]],
):
    """保存评估结果到CSV"""
    os.makedirs(save_dir, exist_ok=True)

    for ctrl, traj_list in trajectories.items():
        if not traj_list:
            continue
        combined_traj = pd.concat(traj_list, ignore_index=True)
        traj_path = os.path.join(save_dir, f'{ctrl}_trajectory.csv')
        combined_traj.to_csv(traj_path, index=False)
        print(f"  [{ctrl}] 轨迹已保存: {traj_path}")

    for ctrl, summ_list in summaries.items():
        if not summ_list:
            continue
        df = pd.DataFrame(summ_list)
        summ_path = os.path.join(save_dir, f'{ctrl}_summary.csv')
        df.to_csv(summ_path, index=False)
        print(f"  [{ctrl}] 摘要已保存: {summ_path}")

    # 汇总对比表
    rows = []
    for ctrl, summ_list in summaries.items():
        if not summ_list:
            continue
        df = pd.DataFrame(summ_list)
        row = {
            'controller': ctrl.upper(),
            'n_runs': len(summ_list),
            'total_reward_mean': df['total_reward'].mean(),
            'total_reward_std': df['total_reward'].std(),
            'violation_rate_mean': df['violation_rate'].mean(),
            'violation_rate_std': df['violation_rate'].std(),
            'energy_kWh_mean': df['total_energy_kWh'].mean(),
            'cost_yuan_mean': df['total_cost_yuan'].mean(),
            'harvest_kg_mean': df['harvest_mass_kg'].mean(),
            'harvests_mean': df['harvests'].mean(),
            'T_mean_mean': df['T_mean'].mean(),
            'C_ppm_mean_mean': df['C_ppm_mean'].mean(),
        }
        rows.append(row)

    if rows:
        comp_df = pd.DataFrame(rows)
        comp_path = os.path.join(save_dir, 'comparison_summary.csv')
        comp_df.to_csv(comp_path, index=False)
        print(f"  [ALL] 对比汇总已保存: {comp_path}")


def _print_comparison_table(summaries: Dict[str, List[Dict]]):
    """打印控制器对比表"""
    if not summaries:
        return

    print(f"\n{'='*90}")
    print(f"  固定排程控制器对比汇总")
    print(f"{'='*90}")
    print(f"  {'Controller':>16} | {'Reward':>10} | {'Viol%':>7} | "
          f"{'Energy(kWh)':>12} | {'Cost(CNY)':>10} | {'Harvest(kg)':>11}")
    print(f"  {'-'*16}-+-{'-'*10}-+-{'-'*7}-+-{'-'*12}-+-{'-'*10}-+-{'-'*11}")

    for ctrl, summ_list in summaries.items():
        if not summ_list:
            continue
        df = pd.DataFrame(summ_list)
        rew_m = df['total_reward'].mean()
        rew_s = df['total_reward'].std()
        viol_m = df['violation_rate'].mean() * 100
        viol_s = df['violation_rate'].std() * 100
        eng_m = df['total_energy_kWh'].mean()
        cost_m = df['total_cost_yuan'].mean()
        harv_m = df['harvest_mass_kg'].mean()

        print(f"  {ctrl.upper():>16} | {rew_m:>+10.2f} | {viol_m:>6.1f}% | "
              f"{eng_m:>12.2f} | {cost_m:>10.2f} | {harv_m:>11.3f}")

    print(f"{'='*90}")


# =============================================================================
# 命令行接口
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='固定排程控制器评估（PID vs MPC vs Contextual RL）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 完整三控制器评估（需先训练RL）
  python experiments/evaluate_controllers.py --mode all --n_runs 3 --save

  # 仅MPC评估
  python experiments/evaluate_controllers.py --mode mpc --n_runs 3 --save

  # 指定排程和MPC参数
  python experiments/evaluate_controllers.py --mode all --t1 14 --t2 21 --rho2 50 --Np 12

  # RL评估（需指定模型路径）
  python experiments/evaluate_controllers.py --mode rl --rl_model results/models/best_model.zip
        """
    )

    # 排程参数
    parser.add_argument('--t1', type=int, default=14, help='育苗期天数')
    parser.add_argument('--t2', type=int, default=21, help='定植期天数')
    parser.add_argument('--rho2', type=float, default=35.0, help='定植区密度 [株/m²]')
    parser.add_argument('--A1_A2', type=float, default=0.5, help='育苗/定植面积比')

    # 评估模式
    parser.add_argument('--mode', type=str, default='all',
                       choices=['pid', 'mpc', 'rl', 'all'],
                       help='评估模式')

    # 运行参数
    parser.add_argument('--n_runs', type=int, default=3, help='独立运行次数')
    parser.add_argument('--Np', type=int, default=8, help='MPC预测步数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子基数')
    parser.add_argument('--n_steps', type=int, default=None, help='仿真步数')

    # RL模型
    parser.add_argument('--rl_model', type=str, default=None,
                       help='训练好的PPO模型路径 (.zip)')

    # 保存
    parser.add_argument('--save', action='store_true', help='保存结果')
    parser.add_argument('--save_dir', type=str, default='results/exp2_controller_eval',
                       help='结果保存目录')
    parser.add_argument('--quiet', action='store_true', help='静默模式')
    parser.add_argument('--config_dir', type=str, default=None, help='配置文件目录')

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

    # 计算面积
    A_total = 40.0
    A1 = A_total / (1.0 + args.A1_A2)
    A2 = A_total - A1
    schedule['A1'] = A1
    schedule['A2'] = A2

    modes = [args.mode]

    if verbose:
        print(f"\n{'#'*80}")
        print(f"# Experiment 2: 固定排程控制器评估")
        print(f"# Schedule: t1={args.t1}d t2={args.t2}d "
              f"rho2={args.rho2} A1/A2={args.A1_A2:.2f}")
        print(f"# Modes: {modes} | Np={args.Np} | Runs={args.n_runs} | Seed={args.seed}")
        print(f"# RL Model: {args.rl_model or '(not specified)'}")
        print(f"# Save: {save_dir or '(not saving)'}")
        print(f"{'#'*80}")

    # 执行评估
    run_fixed_schedule_evaluation(
        schedule=schedule,
        modes=modes,
        Np=args.Np,
        n_runs=args.n_runs,
        seed_base=args.seed,
        config_dir=config_dir,
        save_dir=save_dir,
        rl_model_path=args.rl_model,
        verbose=verbose,
    )

    # 可视化
    if save_dir:
        try:
            from visualizations.experiment_viz import plot_controller_comparison
            print(f"\n--- Generating visualizations ---")
            plot_controller_comparison(
                results_dir=save_dir,
                save_dir=os.path.join(save_dir, 'figures'),
            )
        except Exception as e:
            print(f"[WARN] 可视化失败: {e}")

    print(f"\n完成!")


if __name__ == '__main__':
    main()
