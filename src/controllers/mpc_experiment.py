# -*- coding: utf-8 -*-
"""
MPC闭环仿真实验框架

实现MPC控制器与植物工厂仿真环境的闭环运行，
支持结果记录、统计分析、与RL控制器的对比。

来源: plant_factory_optimization项目

作者: Plant Factory Optimization Team
"""

import numpy as np
import pandas as pd
import os
import time as wall_time
from typing import Dict, Any, Optional, List, Tuple
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.envs.plant_factory_env import MultiBatchPlantFactoryEnv
from src.models.mpc_model import (
    env_and_batch_to_mpc_state,
    env_state_to_mpc_state,
    mpc_state_to_env_state,
    generate_disturbance_profile,
    co2_density_to_ppm,
    _batch_indices,
)
from .mpc_controller import PlantFactoryMPC


class MPCExperiment:
    """
    MPC闭环仿真实验管理器

    在植物工厂仿真环境中运行MPC控制器，记录完整轨迹，
    支持与RL策略的对比评估。

    Attributes:
        mpc: MPC控制器实例
        env: 仿真环境实例
        config: 环境配置
        schedule: 排程参数
        results: 仿真结果DataFrame
        batch_records: per-batch详细记录列表（当 record_detailed=True 时）
    """

    def __init__(
        self,
        mpc_controller: Optional[PlantFactoryMPC] = None,
        env_config: Optional[Dict[str, Any]] = None,
        schedule: Optional[Dict[str, Any]] = None,
        mpc_config: Optional[Dict[str, Any]] = None,
        Np: int = 8,
        verbose: bool = False,
        seed: int = 42,
        record_detailed: bool = True,
    ):
        """
        初始化MPC实验。

        参数:
            mpc_controller: 已有MPC控制器实例（可选）
            env_config: 环境配置字典（传递给MultiBatchPlantFactoryEnv）
            schedule: 排程参数字典 {t1, t2, rho2, A1_A2}
            mpc_config: MPC配置字典
            Np: 预测步数
            verbose: 是否打印详细信息
            seed: 随机种子
            record_detailed: 是否记录每个batch的详细生长信息
        """
        self.verbose = verbose
        self.seed = seed
        self.record_detailed = record_detailed

        # ========== 排程参数 ==========
        if schedule is not None:
            self.schedule = schedule
        else:
            self.schedule = {
                't1': 14, 't2': 21,
                'rho2': 35.0, 'A1_A2': 0.5,
            }

        # ========== 环境配置 ==========
        if env_config is None:
            env_config = {}

        # 合并默认配置
        default_env_config = {
            'schedule': self.schedule,
            'seed': seed,
        }
        self.env_config = {**default_env_config, **env_config}

        # ========== 初始化环境 ==========
        self.env = MultiBatchPlantFactoryEnv(config=self.env_config)
        self.dt = self.env.dt
        self.episode_length = self.env.episode_length

        # 计算面积
        A_total = self.env.container_params.get('c_total_plant_area', 40.0)
        self.A1 = A_total / (1.0 + self.schedule['A1_A2'])
        self.A2 = A_total - self.A1

        # ========== 初始化MPC控制器 ==========
        if mpc_controller is not None:
            self.mpc = mpc_controller
        else:
            if mpc_config is not None:
                config = mpc_config
            else:
                config = None
            self.mpc = PlantFactoryMPC(
                config=config,
                crop_params=self.env.crop_params,
                container_params=self.env.container_params,
                equipment_params=self.env.equipment_params,
                reward_params=self.env.reward_params,
                schedule=self.schedule,
                Np=Np,
                verbose=verbose,
            )

        # ========== 结果记录 ==========
        self.results = None
        self.episode_stats = {}
        self.batch_records: List[Dict[str, Any]] = []  # per-batch详细记录

    def _record_batch_details(self, step: int, hour: int, record: Dict):
        """
        记录每个batch的详细生长信息。

        为每个育苗区和定植区的batch单独记录：
        - 批次ID、年龄、干物质密度、LAI
        - 区域总干物质和负荷
        用于事后分析和MPC物理模型验证。

        参数:
            step: 当前步数
            hour: 当前小时
            record: 当前步的基本记录字典
        """
        bm = self.env.batch_manager
        area_per_seedling = bm.A1 / max(1, len(bm.seedling_batches))
        area_per_transplant = bm.A2 / max(1, len(bm.transplant_batches))

        # --- 育苗区 per-batch 记录 ---
        for i, batch in enumerate(bm.seedling_batches):
            batch_record = {
                'step': step,
                'hour_of_day': hour,
                'region': 'seedling',
                'batch_idx': i,
                'batch_id': batch.batch_id,
                'age_h': batch.age_h,
                # 干物质密度
                'xDn': batch.xDn,
                'xDs': batch.xDs,
                'xDn_xDs': batch.xDn + batch.xDs,
                # LAI
                'LAI': batch.LAI,
                # 该batch面积
                'area_batch': area_per_seedling,
                # 该batch干物质 [kg]
                'biomass_batch': (batch.xDn + batch.xDs) * area_per_seedling,
                # 环境状态（来自record）
                'T': record['T'],
                'RH': record['RH'],
                'C_ppm': record['C_ppm'],
                # 光照
                'I_seedling': record['I1'],
                # 当前步奖励
                'step_reward': record['step_reward'],
                # 约束违反
                'violation': record['violation'],
            }
            self.batch_records.append(batch_record)

        # --- 定植区 per-batch 记录 ---
        for i, batch in enumerate(bm.transplant_batches):
            batch_record = {
                'step': step,
                'hour_of_day': hour,
                'region': 'transplant',
                'batch_idx': i,
                'batch_id': batch.batch_id,
                'age_h': batch.age_h,
                'xDn': batch.xDn,
                'xDs': batch.xDs,
                'xDn_xDs': batch.xDn + batch.xDs,
                'LAI': batch.LAI,
                'area_batch': area_per_transplant,
                'biomass_batch': (batch.xDn + batch.xDs) * area_per_transplant,
                'T': record['T'],
                'RH': record['RH'],
                'C_ppm': record['C_ppm'],
                'I_transplant': record['I2'],
                'step_reward': record['step_reward'],
                'violation': record['violation'],
            }
            self.batch_records.append(batch_record)

    def get_batch_dataframe(self) -> pd.DataFrame:
        """
        获取per-batch详细记录的DataFrame。

        返回:
            每行对应一个batch在某个step的详细数据：
            [step, hour_of_day, region, batch_idx, batch_id, age_h,
             xDn, xDs, xDn_xDs, LAI, area_batch, biomass_batch,
             T, RH, C_ppm, I_seedling/I_transplant, step_reward, violation]
        """
        if not self.batch_records:
            return pd.DataFrame()
        return pd.DataFrame(self.batch_records)

    def get_batch_summary(self) -> Dict[str, Dict[str, float]]:
        """
        获取每个batch的最终统计摘要。

        返回:
            {batch_id: {'final_xDn', 'final_xDs', 'final_LAI',
                        'max_biomass', 'total_reward', 'n_violations'}}
        """
        if not self.batch_records:
            return {}

        df = self.get_batch_dataframe()
        summaries = {}

        for (region, batch_id), grp in df.groupby(['region', 'batch_id']):
            key = f"{region}_{batch_id}"
            summaries[key] = {
                'final_xDn': grp['xDn'].iloc[-1],
                'final_xDs': grp['xDs'].iloc[-1],
                'final_LAI': grp['LAI'].iloc[-1],
                'max_biomass': grp['biomass_batch'].max(),
                'total_reward': grp['step_reward'].sum(),
                'n_violations': grp['violation'].sum(),
                'n_steps': len(grp),
                'final_age_h': grp['age_h'].iloc[-1],
            }

        return summaries

    def run(
        self,
        n_steps: Optional[int] = None,
        use_mpc: bool = True,
        save_trajectory: bool = True,
        log_interval: int = 24,
    ) -> pd.DataFrame:
        """
        运行MPC闭环仿真。

        参数:
            n_steps: 运行步数（默认=episode_length，即一个定植周期）
            use_mpc: 是否使用MPC控制器（False=使用默认控制器）
            save_trajectory: 是否保存完整轨迹
            log_interval: 日志打印间隔（步数）

        返回:
            results: DataFrame，包含所有时刻的状态/动作/成本/约束违反
        """
        if n_steps is None:
            n_steps = self.episode_length

        # ========== 重置环境和MPC ==========
        obs, info = self.env.reset(seed=self.seed, options={
            'schedule': self.schedule,
        })

        self.mpc.reset()
        env_state = self.env.state.copy()

        # ========== 初始化跟踪变量 ==========
        day_dli1 = 0.0  # 育苗区当日DLI累计
        day_dli2 = 0.0  # 定植区当日DLI累计
        prev_hour = self.env.hour_of_day

        # ========== 结果记录列表 ==========
        step_records = []

        # ========== 主仿真循环 ==========
        total_reward = 0.0
        total_growth = 0.0
        total_cost = 0.0
        n_violations = 0
        mpc_solve_times = []

        for step in range(n_steps):
            hour = self.env.hour_of_day

            # 检测新的一天（小时从23→0时重置DLI累计）
            if hour < prev_hour:
                day_dli1 = 0.0
                day_dli2 = 0.0
            prev_hour = hour

            # ========== 获取MPC状态（实际batch数量用于奖励） ==========
            x_mpc, N1_actual, N2_actual = env_and_batch_to_mpc_state(
                env_state=env_state,
                batch_manager=self.env.batch_manager,
                A1=self.A1, A2=self.A2,
                day_dli1=day_dli1, day_dli2=day_dli2,
            )

            # ========== 获取扰动信息 ==========
            external = self.env.external.copy()  # [T_out, RH_out, C_out]
            T_out, RH_out, C_out = external[0], external[1], external[2]
            C_out_ppm = co2_density_to_ppm(C_out, T_out)
            elec_price = self.env.elec_price

            # ========== MPC求解（带事件触发NLP重建）==========
            if use_mpc:
                mpc_solve_start = wall_time.time()
                u_opt_seq, x_traj, J_opt, solve_time, exit_msg = self.mpc.solve(
                    x0=x_mpc,
                    N1_actual=N1_actual,
                    N2_actual=N2_actual,
                    hour_of_day=hour,
                    day_of_period=self.env.day_of_period,
                    external=np.array([T_out, RH_out, C_out_ppm]),
                    elec_price=elec_price,
                    batch_manager=self.env.batch_manager,
                )
                mpc_solve_times.append(solve_time)
                u_action = u_opt_seq[:6, 0]
            else:
                # 使用环境默认动作
                u_action = np.array([
                    self.env.container_params.get('default_I1', 200.0),
                    self.env.container_params.get('default_I2', 200.0),
                    0.0, 0.0, 0.01, 1e-5,
                ], dtype=np.float32)
                exit_msg = 'Default'

            # ========== 动作裁剪 ==========
            al, ah = self.env._get_action_bounds_from_config()
            u_action = np.clip(u_action, al, ah).astype(np.float32)

            # ========== 更新DLI累计（使用预测动作） ==========
            I1_action = float(u_action[0])
            I2_action = float(u_action[1])
            dt_hours = self.dt / 3600.0
            # DLI增量: μmol/m²/s * s/h * mol/1e6 μmol = mol/m²
            day_dli1 += I1_action * dt_hours * 1e-6 * 3600.0
            day_dli2 += I2_action * dt_hours * 1e-6 * 3600.0

            # ========== 环境step ==========
            obs, reward, terminated, truncated, info = self.env.step(u_action)
            env_state = self.env.state.copy()
            total_reward += reward

            # ========== 检查约束违反 ==========
            T = env_state[1]
            RH = env_state[2]
            C_ppm = co2_density_to_ppm(env_state[0], T)

            violation = False
            if T < 16.0 or T > 30.0:
                violation = True
            if RH < 0.55 or RH > 0.85:
                violation = True
            if C_ppm < 400 or C_ppm > 1100:
                violation = True
            if violation:
                n_violations += 1

            # ========== 记录 ==========
            record = {
                'step': step,
                'hour_of_day': hour,
                'day_of_period': self.env.day_of_period,

                # 状态
                'T': T,
                'RH': RH,
                'C_ppm': C_ppm,
                'C_kgm3': env_state[0],

                # 动作
                'I1': u_action[0],
                'I2': u_action[1],
                'Q_HVAC': u_action[2],
                'u_CO2': u_action[3],
                'V_vent': u_action[4],
                'm_dehum': u_action[5],

                # MPC
                'mpc_exit_msg': exit_msg,
                'solve_time': mpc_solve_times[-1] if use_mpc else 0.0,
                'mpc_cost': J_opt if use_mpc else 0.0,

                # DLI
                'dLI1': day_dli1,
                'dLI2': day_dli2,

                # 奖励分解
                'step_reward': reward,
                'total_reward': total_reward,

                # 外部环境
                'T_out': T_out,
                'RH_out': RH_out,
                'elec_price': elec_price,

                # 批次信息
                'n_seedling_batches': len(self.env.batch_manager.seedling_batches),
                'n_transplant_batches': len(self.env.batch_manager.transplant_batches),
                'n_seedling_actual': N1_actual,
                'n_transplant_actual': N2_actual,
                'total_harvests': self.env.batch_manager.total_harvests,
                'total_transplants': self.env.batch_manager.total_transplants,
                'total_harvest_mass_kg': self.env.batch_manager.total_harvest_mass,

                # 约束违反
                'violation': int(violation),
            }
            step_records.append(record)

            # ========== 【per-batch详细记录】==========
            if self.record_detailed:
                self._record_batch_details(step, hour, record)

            # ========== Episode边界处理 ==========
            # 模拟自然在RL环境的episode边界结束
            # （一个定植周期 t2*24 步）
            if terminated or step >= n_steps - 1:
                if self.verbose:
                    print(f"\n  [Episode ended at step {step}] MPC solver reset for next episode")
                self.mpc.reset()
                break

            # ========== 日志打印 ==========
            if self.verbose and step % log_interval == 0:
                lumped = self.env.batch_manager._extract_lumped_features()
                print(f"  Step {step:4d} | T={T:.1f}°C RH={RH*100:.0f}% "
                      f"C={C_ppm:.0f}ppm | I1={u_action[0]:.0f} I2={u_action[1]:.0f} "
                      f"| reward={reward:.3f} | viol={violation}")

            if terminated or step >= n_steps - 1:
                break

        # ========== 保存结果 ==========
        self.results = pd.DataFrame(step_records)

        # ========== Episode统计 ==========
        self.episode_stats = {
            'n_steps': len(step_records),
            'total_reward': total_reward,
            'total_violations': n_violations,
            'violation_rate': n_violations / max(len(step_records), 1),
            'avg_mpc_solve_time': np.mean(mpc_solve_times) if mpc_solve_times else 0.0,
            'max_mpc_solve_time': np.max(mpc_solve_times) if mpc_solve_times else 0.0,
            'total_harvests': self.env.batch_manager.total_harvests,
            'total_harvest_mass_kg': self.env.batch_manager.total_harvest_mass,
            'mpc_success_rate': self.mpc.get_statistics().get('success_rate', 0.0),
        }

        # ========== 补充elec_price（用于后续成本计算）==========
        if self.results is not None:
            self.results['elec_price'] = self.env.elec_price

        if self.verbose:
            print(f"\n  Episode Summary:")
            print(f"    Steps: {self.episode_stats['n_steps']}")
            print(f"    Total Reward: {self.episode_stats['total_reward']:.3f}")
            print(f"    Violation Rate: {self.episode_stats['violation_rate']:.1%}")
            print(f"    MPC Success Rate: {self.episode_stats['mpc_success_rate']:.1%}")
            print(f"    Avg Solve Time: {self.episode_stats['avg_mpc_solve_time']:.3f}s")
            print(f"    Harvests: {self.episode_stats['total_harvests']}, Mass: {self.episode_stats['total_harvest_mass_kg']:.3f}kg")

        return self.results

    def get_summary(self) -> Dict[str, Any]:
        """获取episode摘要统计"""
        if self.results is None:
            return {}

        df = self.results

        # 成本计算
        dt_hours = self.dt / 3600.0
        ep = self.env.equipment_params
        c_optical_eff = ep.get('c_optical_eff', 2.5)
        c_led_eff = ep.get('c_led_eff', 0.52)
        c_COP = ep.get('c_COP', 3.0)
        fan_eff = ep.get('fan_eff', 7.07)
        c_dehum_eev = ep.get('c_dehum_eev', 3.0)
        p_CO2 = ep.get('p_CO2', 0.5)
        A_total = self.A1 + self.A2

        # LED能耗
        I1_mean = df['I1'].mean()
        I2_mean = df['I2'].mean()
        E_led = ((I1_mean / c_optical_eff) * self.A1 / c_led_eff +
                  (I2_mean / c_optical_eff) * self.A2 / c_led_eff) * \
                 df['I1'].gt(0).mean() * self.episode_length * dt_hours / 1000.0

        # HVAC能耗（简化）
        Q_HVAC_mean = df['Q_HVAC'].mean()
        E_hvac = abs(Q_HVAC_mean) * A_total / c_COP * self.episode_length * dt_hours / 1000.0

        # 通风能耗
        V_vent_mean = df['V_vent'].mean()
        E_vent = V_vent_mean * A_total / fan_eff * self.episode_length * dt_hours / 1000.0

        # 除湿能耗
        m_dehum_mean = df['m_dehum'].mean()
        E_dehum = (m_dehum_mean * A_total) / c_dehum_eev * 1000.0 * self.episode_length * dt_hours / 1000.0

        # CO2成本
        u_CO2_mean = df['u_CO2'].mean()
        CO2_kg = u_CO2_mean * A_total * self.episode_length * dt_hours / 1000.0

        total_energy_kWh = E_led + E_hvac + E_vent + E_dehum
        total_cost = (E_led + E_hvac + E_vent + E_dehum) * df['elec_price'].mean() + \
                     CO2_kg * p_CO2

        return {
            'n_steps': len(df),
            'total_reward': df['step_reward'].sum(),
            'avg_reward': df['step_reward'].mean(),
            'std_reward': df['step_reward'].std(),
            'violation_rate': df['violation'].mean(),
            'T_mean': df['T'].mean(),
            'T_std': df['T'].std(),
            'RH_mean': df['RH'].mean(),
            'C_ppm_mean': df['C_ppm'].mean(),
            'I1_mean': df['I1'].mean(),
            'I2_mean': df['I2'].mean(),
            'Q_HVAC_mean': df['Q_HVAC'].mean(),
            'total_energy_kWh': total_energy_kWh,
            'total_cost_yuan': total_cost,
            'dLI1_end': df['dLI1'].iloc[-1] if len(df) > 0 else 0,
            'dLI2_end': df['dLI2'].iloc[-1] if len(df) > 0 else 0,
            'mpc_success_rate': self.episode_stats.get('mpc_success_rate', 0),
            'avg_mpc_solve_time': self.episode_stats.get('avg_mpc_solve_time', 0),
            'harvests': self.episode_stats.get('total_harvests', 0),
            'harvest_mass_kg': self.episode_stats.get('total_harvest_mass_kg', 0),
        }

    def save_results(self, save_path: str, experiment_name: str = "mpc_experiment"):
        """保存实验结果到CSV"""
        if self.results is None:
            return

        os.makedirs(save_path, exist_ok=True)

        # 保存轨迹
        traj_path = os.path.join(save_path, f"{experiment_name}_trajectory.csv")
        self.results.to_csv(traj_path, index=False)

        # 保存摘要
        summary = self.get_summary()
        summary_path = os.path.join(save_path, f"{experiment_name}_summary.csv")
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(summary_path, index=False)

        # 保存per-batch详细记录
        if self.batch_records:
            batch_df = self.get_batch_dataframe()
            batch_path = os.path.join(save_path, f"{experiment_name}_batch_details.csv")
            batch_df.to_csv(batch_path, index=False)

            # 保存batch摘要
            batch_summary = self.get_batch_summary()
            if batch_summary:
                import json
                summary_json_path = os.path.join(save_path, f"{experiment_name}_batch_summary.json")
                with open(summary_json_path, 'w', encoding='utf-8') as f:
                    json.dump(batch_summary, f, indent=2, ensure_ascii=False)

        if self.verbose:
            print(f"\n  Results saved to:")
            print(f"    {traj_path}")
            print(f"    {summary_path}")
            if self.batch_records:
                print(f"    {batch_path}")

    def compare_with_rl(
        self,
        rl_model_path: Optional[str] = None,
        n_episodes: int = 3,
    ) -> Dict[str, Dict[str, Any]]:
        """
        对比MPC与RL控制器的性能。

        参数:
            rl_model_path: 训练好的RL模型路径（可选）
            n_episodes: 对比episodes数

        返回:
            对比结果字典
        """
        results = {}

        # MPC性能
        print("\n  Running MPC comparison...")
        mpc_results = []
        for i in range(n_episodes):
            seed = 42 + i
            exp = MPCExperiment(
                schedule=self.schedule,
                env_config={'seed': seed},
                Np=8,
                verbose=False,
                seed=seed,
            )
            exp.run(n_steps=self.episode_length, use_mpc=True, save_trajectory=False)
            mpc_results.append(exp.get_summary())

        results['mpc'] = {
            'mean_reward': np.mean([r['total_reward'] for r in mpc_results]),
            'std_reward': np.std([r['total_reward'] for r in mpc_results]),
            'mean_energy': np.mean([r['total_energy_kWh'] for r in mpc_results]),
            'mean_violation': np.mean([r['violation_rate'] for r in mpc_results]),
            'mean_solve_time': np.mean([r['avg_mpc_solve_time'] for r in mpc_results]),
            'per_episode': mpc_results,
        }

        if rl_model_path and os.path.exists(rl_model_path):
            print("  Running RL comparison...")
            from stable_baselines3 import PPO
            from ..controllers.rl_controller import RLController

            rl_results = []
            for i in range(n_episodes):
                seed = 42 + i
                try:
                    model = PPO.load(rl_model_path)
                    rl_exp = RLClosedLoopExperiment(
                        model=model,
                        schedule=self.schedule,
                        env_config={'seed': seed},
                        verbose=False,
                        seed=seed,
                    )
                    rl_exp.run(n_steps=self.episode_length)
                    rl_results.append(rl_exp.get_summary())
                except Exception as e:
                    print(f"    RL experiment {i} failed: {e}")

            if rl_results:
                results['rl'] = {
                    'mean_reward': np.mean([r['total_reward'] for r in rl_results]),
                    'std_reward': np.std([r['total_reward'] for r in rl_results]),
                    'mean_energy': np.mean([r['total_energy_kWh'] for r in rl_results]),
                    'mean_violation': np.mean([r['violation_rate'] for r in rl_results]),
                    'per_episode': rl_results,
                }

        # 打印对比
        print("\n  === MPC vs RL Comparison ===")
        print(f"  MPC:  reward={results['mpc']['mean_reward']:.3f}±{results['mpc']['std_reward']:.3f}, "
              f"energy={results['mpc']['mean_energy']:.1f}kWh, "
              f"viol={results['mpc']['mean_violation']:.1%}, "
              f"solve_time={results['mpc']['mean_solve_time']:.3f}s")
        if 'rl' in results:
            print(f"  RL:   reward={results['rl']['mean_reward']:.3f}±{results['rl']['std_reward']:.3f}, "
                  f"energy={results['rl']['mean_energy']:.1f}kWh, "
                  f"viol={results['rl']['mean_violation']:.1%}")

        return results


class RLClosedLoopExperiment:
    """
    RL策略闭环仿真（用于与MPC对比）

    使用训练好的PPO模型在仿真环境中运行。
    """

    def __init__(
        self,
        model,  # stable-baselines3 PPO model
        env_config: Optional[Dict[str, Any]] = None,
        schedule: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
        seed: int = 42,
    ):
        self.verbose = verbose
        self.model = model

        if schedule is not None:
            self.schedule = schedule
        else:
            self.schedule = {'t1': 14, 't2': 21, 'rho2': 35.0, 'A1_A2': 0.5}

        if env_config is None:
            env_config = {}
        default_env_config = {
            'schedule': self.schedule,
            'seed': seed,
        }
        self.env_config = {**default_env_config, **env_config}

        self.env = MultiBatchPlantFactoryEnv(config=self.env_config)
        self.dt = self.env.dt
        self.episode_length = self.env.episode_length

        A_total = self.env.container_params.get('c_total_plant_area', 40.0)
        self.A1 = A_total / (1.0 + self.schedule['A1_A2'])
        self.A2 = A_total - self.A1

        self.results = None
        self.episode_stats = {}

    def run(self, n_steps: Optional[int] = None, save_trajectory: bool = True) -> pd.DataFrame:
        if n_steps is None:
            n_steps = self.episode_length

        obs, info = self.env.reset(seed=42, options={'schedule': self.schedule})

        step_records = []
        total_reward = 0.0
        n_violations = 0
        prev_hour = self.env.hour_of_day

        for step in range(n_steps):
            hour = self.env.hour_of_day
            if hour < prev_hour:
                # 新的一天
                pass
            prev_hour = hour

            # RL预测动作
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.env.step(action)

            total_reward += reward
            env_state = self.env.state
            T, RH, C_kg = env_state[1], env_state[2], env_state[0]
            C_ppm = co2_density_to_ppm(C_kg, T)

            violation = (T < 16.0 or T > 30.0 or RH < 0.55 or RH > 0.85 or C_ppm < 400 or C_ppm > 1100)
            if violation:
                n_violations += 1

            record = {
                'step': step,
                'hour_of_day': hour,
                'T': T,
                'RH': RH,
                'C_ppm': C_ppm,
                'I1': action[0],
                'I2': action[1],
                'Q_HVAC': action[2],
                'u_CO2': action[3],
                'V_vent': action[4],
                'm_dehum': action[5],
                'step_reward': reward,
                'total_reward': total_reward,
                'violation': int(violation),
                'elec_price': self.env.elec_price,
            }
            step_records.append(record)

            if terminated or step >= n_steps - 1:
                break

        self.results = pd.DataFrame(step_records)
        self.episode_stats = {
            'n_steps': len(step_records),
            'total_reward': total_reward,
            'violation_rate': n_violations / max(len(step_records), 1),
            'total_harvests': self.env.batch_manager.total_harvests,
            'total_harvest_mass_kg': self.env.batch_manager.total_harvest_mass,
        }
        return self.results

    def get_summary(self) -> Dict[str, Any]:
        if self.results is None:
            return {}

        df = self.results
        dt_hours = self.dt / 3600.0
        ep = self.env.equipment_params
        c_optical_eff = ep.get('c_optical_eff', 2.5)
        c_led_eff = ep.get('c_led_eff', 0.52)
        c_COP = ep.get('c_COP', 3.0)
        fan_eff = ep.get('fan_eff', 7.07)
        c_dehum_eev = ep.get('c_dehum_eev', 3.0)
        p_CO2 = ep.get('p_CO2', 0.5)
        A_total = self.A1 + self.A2

        I1_mean = df['I1'].mean()
        I2_mean = df['I2'].mean()
        E_led = ((I1_mean / c_optical_eff) * self.A1 / c_led_eff +
                  (I2_mean / c_optical_eff) * self.A2 / c_led_eff) * \
                 df['I1'].gt(0).mean() * self.episode_length * dt_hours / 1000.0

        Q_HVAC_mean = df['Q_HVAC'].mean()
        E_hvac = abs(Q_HVAC_mean) * A_total / c_COP * self.episode_length * dt_hours / 1000.0

        V_vent_mean = df['V_vent'].mean()
        E_vent = V_vent_mean * A_total / fan_eff * self.episode_length * dt_hours / 1000.0

        m_dehum_mean = df['m_dehum'].mean()
        E_dehum = (m_dehum_mean * A_total) / c_dehum_eev * 1000.0 * self.episode_length * dt_hours / 1000.0

        u_CO2_mean = df['u_CO2'].mean()
        CO2_kg = u_CO2_mean * A_total * self.episode_length * dt_hours / 1000.0

        total_energy_kWh = E_led + E_hvac + E_vent + E_dehum
        elec_price_mean = df['elec_price'].mean() if 'elec_price' in df.columns else ep.get('p_elec_base', 0.6)
        total_cost = (E_led + E_hvac + E_vent + E_dehum) * elec_price_mean + \
                     CO2_kg * p_CO2

        return {
            'n_steps': len(df),
            'total_reward': df['step_reward'].sum(),
            'avg_reward': df['step_reward'].mean(),
            'violation_rate': df['violation'].mean(),
            'T_mean': df['T'].mean(),
            'RH_mean': df['RH'].mean(),
            'C_ppm_mean': df['C_ppm'].mean(),
            'total_energy_kWh': total_energy_kWh,
            'total_cost_yuan': total_cost,
            'harvests': self.episode_stats.get('total_harvests', 0),
            'harvest_mass_kg': self.episode_stats.get('total_harvest_mass_kg', 0),
        }
