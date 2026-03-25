# -*- coding: utf-8 -*-
"""
控制方法结果记录器

为所有控制方法（Rule/MPC/SMPC/RL）提供统一的结果保存机制。

保存内容：
1. 动态环控轨迹（每步保存为 CSV）
2. 能耗分解详情（每步保存）
3. 奖励与惩罚详情（每步保存）
4. 排程参数（每个实验保存一次）
5. 汇总统计（每个实验保存一次）

使用方法:
    from src.utils.result_logger import ControllerResultLogger

    logger = ControllerResultLogger(
        controller_name='mpc',
        results_dir='results/controller_comparison',
        experiment_id='exp_001'
    )

    # 每步记录
    logger.log_step(
        step=0,
        state={'T': 22.0, 'RH': 75.0, 'CO2_ppm': 1000.0},
        action={'I1': 200.0, 'I2': 200.0, ...},
        reward=0.5,
        cost_info={'cost_electric': 0.1, 'cost_CO2': 0.05},
        solver_info={'solver_time': 0.05, 'solver_status': 'Solved'},
        extra={'constraint_violation': 0.0, ...}
    )

    # 实验结束时保存
    logger.finalize(
        schedule={'t1': 14, 't2': 21, 'rho2': 35.0, 'A1_A2': 0.5},
        summary_stats={'total_reward': 100.0, ...}
    )

来源: 参考 RL-SMPC/common/results.py, 论文方法部分 2.5
"""

import os
import csv
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict


@dataclass
class StepRecord:
    """单步记录数据类"""
    step: int = 0
    sim_time_days: float = 0.0
    sim_time_hours: float = 0.0

    # 环境状态
    T_inside: float = 0.0      # 箱内温度 [°C]
    RH_inside: float = 0.0      # 箱内相对湿度 [%]
    CO2_inside: float = 0.0     # 箱内CO2 [ppm]
    T_outside: float = 0.0      # 外界温度 [°C]
    RH_outside: float = 0.0     # 外界相对湿度 [%]
    CO2_outside: float = 0.0   # 外界CO2 [ppm]
    elec_price: float = 0.0     # 电价 [元/kWh]

    # 作物状态
    lai_total: float = 0.0      # 总LAI
    M_seedling: float = 0.0     # 育苗区干重 [g]
    M_transplant: float = 0.0   # 定植区干重 [g]
    M_total: float = 0.0        # 总干重 [g]
    days_left: float = 0.0      # 剩余天数
    lai_seedling: float = 0.0   # 育苗区LAI
    lai_transplant: float = 0.0 # 定植区LAI

    # 动作 (6维)
    action_I1: float = 0.0      # 育苗区光强 [μmol/m²/s]
    action_I2: float = 0.0      # 定植区光强 [μmol/m²/s]
    action_Q_HVAC: float = 0.0  # HVAC功率 [W/m²]
    action_u_CO2: float = 0.0   # CO2注入速率 [g/m²/h]
    action_V_vent: float = 0.0  # 通风率 [m³/m²/s]
    action_m_dehum: float = 0.0 # 除湿速率 [kg/m²/s]

    # 能耗分项 [W/m²]
    power_led_I1: float = 0.0
    power_led_I2: float = 0.0
    power_led_total: float = 0.0
    power_hvac: float = 0.0
    power_vent: float = 0.0
    power_dehum: float = 0.0
    power_total: float = 0.0

    # 能耗累计 [kWh/m²]
    energy_led_cumulative: float = 0.0
    energy_hvac_cumulative: float = 0.0
    energy_vent_cumulative: float = 0.0
    energy_dehum_cumulative: float = 0.0
    energy_total_cumulative: float = 0.0

    # 成本 [元/m²]
    cost_electric: float = 0.0
    cost_CO2: float = 0.0
    cost_total: float = 0.0

    # 奖励
    reward_growth: float = 0.0
    reward_penalty: float = 0.0
    reward_total: float = 0.0

    # 约束违反
    temp_violation: float = 0.0  # 温度违反量 [°C]
    rh_violation: float = 0.0    # 湿度违反量 [%]
    co2_violation: float = 0.0   # CO2违反量 [ppm]

    # 控制器特定信息
    solver_time: float = -1.0    # 求解器时间 [s] (MPC/SMPC)
    solver_status: str = ''       # 求解状态
    solver_iterations: int = -1  # 迭代次数
    constraint_violation: float = 0.0  # 软约束违反量
    objective_value: float = 0.0  # 优化目标值

    # 排程参数 (每个episode固定, 重复记录方便分析)
    schedule_t1: float = 0.0
    schedule_t2: float = 0.0
    schedule_rho2: float = 0.0
    schedule_A1_A2: float = 0.0


@dataclass
class ExperimentSummary:
    """实验汇总数据类"""
    experiment_id: str = ''
    controller_name: str = ''
    timestamp: str = ''

    # 排程参数
    schedule_t1: float = 0.0
    schedule_t2: float = 0.0
    schedule_rho2: float = 0.0
    schedule_A1_A2: float = 0.0

    # 仿真参数
    n_episodes: int = 0
    n_steps_per_episode: int = 0
    dt: float = 0.0

    # 产量
    final_M_total_mean: float = 0.0
    final_M_total_std: float = 0.0
    harvest_success_rate: float = 0.0

    # 能耗 (每批次 [kWh/m²])
    energy_led_mean: float = 0.0
    energy_led_std: float = 0.0
    energy_hvac_mean: float = 0.0
    energy_hvac_std: float = 0.0
    energy_vent_mean: float = 0.0
    energy_vent_std: float = 0.0
    energy_dehum_mean: float = 0.0
    energy_dehum_std: float = 0.0
    energy_total_mean: float = 0.0
    energy_total_std: float = 0.0

    # 成本 (每批次 [元/m²])
    cost_electric_mean: float = 0.0
    cost_CO2_mean: float = 0.0
    cost_total_mean: float = 0.0

    # 奖励
    reward_mean: float = 0.0
    reward_std: float = 0.0
    reward_min: float = 0.0
    reward_max: float = 0.0

    # 环境控制质量
    temp_mean: float = 0.0
    temp_std: float = 0.0
    temp_mae: float = 0.0  # Mean Absolute Error
    rh_mean: float = 0.0
    rh_std: float = 0.0
    rh_mae: float = 0.0
    co2_mean: float = 0.0
    co2_std: float = 0.0
    co2_mae: float = 0.0

    # 约束违反
    temp_violation_rate: float = 0.0  # 违反率
    temp_violation_mean: float = 0.0  # 平均违反量
    rh_violation_rate: float = 0.0
    rh_violation_mean: float = 0.0
    co2_violation_rate: float = 0.0
    co2_violation_mean: float = 0.0

    # 控制器性能 (MPC/SMPC)
    solver_success_rate: float = 0.0
    solver_time_mean: float = 0.0
    solver_time_std: float = 0.0
    solver_time_max: float = 0.0

    # 综合指标
    profit_mean: float = 0.0  # 收益 - 成本
    profit_std: float = 0.0
    yield_per_energy: float = 0.0  # 产量/能耗比


class ControllerResultLogger:
    """
    控制方法结果记录器

    为每个控制器提供完整的运行时记录和结果保存功能。
    支持滚动追加、episode 合并、汇总统计。

    使用方法:
        logger = ControllerResultLogger(
            controller_name='mpc',
            results_dir='results/comparison',
            experiment_id='exp_001'
        )

        for episode in range(n_episodes):
            logger.start_episode(episode)
            for step in range(n_steps):
                logger.log_step(...)
            logger.end_episode()

        logger.finalize(schedule=schedule)
    """

    TRAJECTORY_COLUMNS = [
        # 时间
        'step', 'sim_time_days', 'sim_time_hours',

        # 环境状态
        'T_inside', 'RH_inside', 'CO2_inside',
        'T_outside', 'RH_outside', 'CO2_outside', 'elec_price',

        # 作物状态
        'lai_total', 'M_seedling', 'M_transplant', 'M_total',
        'days_left', 'lai_seedling', 'lai_transplant',

        # 动作
        'action_I1', 'action_I2', 'action_Q_HVAC',
        'action_u_CO2', 'action_V_vent', 'action_m_dehum',

        # 能耗分项
        'power_led_I1', 'power_led_I2', 'power_led_total',
        'power_hvac', 'power_vent', 'power_dehum', 'power_total',
        'energy_led_cumulative', 'energy_hvac_cumulative',
        'energy_vent_cumulative', 'energy_dehum_cumulative',
        'energy_total_cumulative',

        # 成本
        'cost_electric', 'cost_CO2', 'cost_total',

        # 奖励
        'reward_growth', 'reward_penalty', 'reward_total',

        # 约束违反
        'temp_violation', 'rh_violation', 'co2_violation',

        # 控制器特定
        'solver_time', 'solver_status', 'solver_iterations',
        'constraint_violation', 'objective_value',

        # 排程
        'schedule_t1', 'schedule_t2', 'schedule_rho2', 'schedule_A1_A2',
    ]

    ENERGY_COLUMNS = [
        'step', 'sim_time_days',
        'power_led_I1', 'power_led_I2', 'power_led_total',
        'power_hvac', 'power_vent', 'power_dehum', 'power_total',
        'energy_led_cumulative', 'energy_hvac_cumulative',
        'energy_vent_cumulative', 'energy_dehum_cumulative',
        'energy_total_cumulative',
        'elec_price',
        'schedule_t1', 'schedule_t2',
    ]

    def __init__(
        self,
        controller_name: str,
        results_dir: str = 'results/controller_comparison',
        experiment_id: Optional[str] = None,
        save_trajectory: bool = True,
        save_energy: bool = True,
    ):
        """
        初始化结果记录器

        参数:
            controller_name: 控制器名称 (rule/mpc/smpc/rl/contextual_rl)
            results_dir: 结果保存根目录
            experiment_id: 实验唯一标识，默认自动生成
            save_trajectory: 是否保存完整轨迹 CSV
            save_energy: 是否保存能耗分解 CSV
        """
        self.controller_name = controller_name
        self.results_dir = results_dir
        self.experiment_id = experiment_id or datetime.now().strftime('%Y%m%d_%H%M%S')
        self.save_trajectory = save_trajectory
        self.save_energy = save_energy

        # 创建目录结构
        self.exp_dir = os.path.join(
            results_dir,
            f'{controller_name}_{self.experiment_id}'
        )
        os.makedirs(self.exp_dir, exist_ok=True)

        self.trajectory_dir = os.path.join(self.exp_dir, 'trajectories')
        self.energy_dir = os.path.join(self.exp_dir, 'energy')
        if save_trajectory:
            os.makedirs(self.trajectory_dir, exist_ok=True)
        if save_energy:
            os.makedirs(self.energy_dir, exist_ok=True)

        # 运行时数据
        self.current_episode = -1
        self.current_step = 0
        self.episode_trajectory: List[StepRecord] = []
        self.all_episodes: List[List[StepRecord]] = []
        self.episode_rewards: List[float] = []
        self.episode_summaries: List[Dict] = []

        # 排程参数（每个实验固定）
        self.schedule: Dict[str, float] = {}

        # 累计能耗
        self.cumulative_energy = {
            'led': 0.0,
            'hvac': 0.0,
            'vent': 0.0,
            'dehum': 0.0,
            'total': 0.0,
        }

    def start_episode(self, episode_idx: int):
        """开始一个新的 episode"""
        self.current_episode = episode_idx
        self.current_step = 0
        self.episode_trajectory = []
        self.cumulative_energy = {k: 0.0 for k in self.cumulative_energy}

    def log_step(
        self,
        step: int,
        state: Dict[str, float],
        action: Dict[str, float],
        reward: float = 0.0,
        reward_breakdown: Optional[Dict[str, float]] = None,
        cost_info: Optional[Dict[str, float]] = None,
        crop_state: Optional[Dict[str, float]] = None,
        power_breakdown: Optional[Dict[str, float]] = None,
        solver_info: Optional[Dict[str, Any]] = None,
        constraint_violation: Optional[Dict[str, float]] = None,
        dt: float = 3600.0,
    ):
        """
        记录单步数据

        参数:
            step: 当前步数
            state: 环境状态 {'T': float, 'RH': float, 'CO2_ppm': float, ...}
            action: 动作 {'I1': float, 'I2': float, 'Q_HVAC': float, ...}
            reward: 总奖励
            reward_breakdown: 奖励分解 {'growth': float, 'penalty': float}
            cost_info: 成本信息 {'electric': float, 'CO2': float, 'total': float}
            crop_state: 作物状态 {'lai_total': float, 'M_total': float, ...}
            power_breakdown: 功率分解 {'led_I1': float, 'led_I2': float, ...}
            solver_info: 求解器信息 {'time': float, 'status': str, ...}
            constraint_violation: 约束违反量 {'temp': float, 'rh': float, ...}
            dt: 时间步长 [s]
        """
        record = StepRecord()

        # 时间
        record.step = step
        record.sim_time_days = step * dt / 86400.0
        record.sim_time_hours = step * dt / 3600.0

        # 环境状态
        record.T_inside = state.get('T', 0.0)
        record.RH_inside = state.get('RH', 0.0)
        record.CO2_inside = state.get('CO2_ppm', 0.0)
        record.T_outside = state.get('T_out', 0.0)
        record.RH_outside = state.get('RH_out', 0.0)
        record.CO2_outside = state.get('CO2_out', 0.0)
        record.elec_price = state.get('elec_price', 0.0)

        # 作物状态
        cs = crop_state or {}
        record.lai_total = cs.get('lai_total', 0.0)
        record.M_seedling = cs.get('M_seedling', 0.0)
        record.M_transplant = cs.get('M_transplant', 0.0)
        record.M_total = cs.get('M_total', 0.0)
        record.days_left = cs.get('days_left', 0.0)
        record.lai_seedling = cs.get('lai_seedling', 0.0)
        record.lai_transplant = cs.get('lai_transplant', 0.0)

        # 动作
        record.action_I1 = action.get('I1', action.get('action_I1', 0.0))
        record.action_I2 = action.get('I2', action.get('action_I2', 0.0))
        record.action_Q_HVAC = action.get('Q_HVAC', action.get('action_Q_HVAC', 0.0))
        record.action_u_CO2 = action.get('u_CO2', action.get('action_u_CO2', 0.0))
        record.action_V_vent = action.get('V_vent', action.get('action_V_vent', 0.0))
        record.action_m_dehum = action.get('m_dehum', action.get('action_m_dehum', 0.0))

        # 功率分解 (keys must match equipment.calculate_total_power output)
        pb = power_breakdown or {}
        record.power_led_I1 = pb.get('P_led1', 0.0)        # [W] 育苗区LED功率
        record.power_led_I2 = pb.get('P_led2', 0.0)        # [W] 定植区LED功率
        record.power_led_total = pb.get('P_led_total', 0.0) # [W] 总LED功率
        record.power_hvac = pb.get('P_hvac_total', 0.0)   # [W] HVAC总功率
        record.power_vent = pb.get('P_vent', 0.0)          # [W] 通风功率
        record.power_dehum = pb.get('P_dehum', 0.0)       # [W] 除湿功率
        record.power_total = pb.get('P_total', 0.0)        # [W] 总功率

        # 累计能耗 (W·s → kWh: W * s / 3.6e6)
        dt_hours = dt / 3600.0
        self.cumulative_energy['led'] += record.power_led_total * dt_hours / 1000.0
        self.cumulative_energy['hvac'] += max(0, record.power_hvac) * dt_hours / 1000.0
        self.cumulative_energy['vent'] += record.power_vent * dt_hours / 1000.0
        self.cumulative_energy['dehum'] += record.power_dehum * dt_hours / 1000.0
        self.cumulative_energy['total'] += record.power_total * dt_hours / 1000.0

        record.energy_led_cumulative = self.cumulative_energy['led']
        record.energy_hvac_cumulative = self.cumulative_energy['hvac']
        record.energy_vent_cumulative = self.cumulative_energy['vent']
        record.energy_dehum_cumulative = self.cumulative_energy['dehum']
        record.energy_total_cumulative = self.cumulative_energy['total']

        # 成本
        ci = cost_info or {}
        record.cost_electric = ci.get('electric', 0.0)
        record.cost_CO2 = ci.get('CO2', 0.0)
        record.cost_total = ci.get('total', 0.0)

        # 奖励
        record.reward_total = reward
        rb = reward_breakdown or {}
        record.reward_growth = rb.get('growth', reward)
        record.reward_penalty = rb.get('penalty', 0.0)

        # 约束违反
        cv = constraint_violation or {}
        record.temp_violation = cv.get('temp', 0.0)
        record.rh_violation = cv.get('rh', 0.0)
        record.co2_violation = cv.get('co2', 0.0)

        # 求解器信息
        si = solver_info or {}
        record.solver_time = si.get('time', -1.0)
        record.solver_status = si.get('status', '')
        record.solver_iterations = si.get('iterations', -1)
        record.constraint_violation = si.get('constraint_violation', 0.0)
        record.objective_value = si.get('objective_value', 0.0)

        # 排程参数
        record.schedule_t1 = self.schedule.get('t1', 0.0)
        record.schedule_t2 = self.schedule.get('t2', 0.0)
        record.schedule_rho2 = self.schedule.get('rho2', 0.0)
        record.schedule_A1_A2 = self.schedule.get('A1_A2', 0.0)

        self.episode_trajectory.append(record)
        self.current_step += 1

    def end_episode(self, total_reward: float = 0.0):
        """结束当前 episode"""
        self.all_episodes.append(self.episode_trajectory)
        self.episode_rewards.append(total_reward)

        # 保存 episode 轨迹
        if self.save_trajectory:
            self._save_episode_trajectory()

        # 保存 episode 能耗
        if self.save_energy:
            self._save_episode_energy()

        # 计算 episode 汇总
        ep_summary = self._compute_episode_summary(total_reward)
        self.episode_summaries.append(ep_summary)

        self.episode_trajectory = []

    def finalize(
        self,
        schedule: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None,
        n_episodes: int = 0,
        dt: float = 3600.0,
    ):
        """
        实验结束，保存所有结果

        参数:
            schedule: 排程参数
            config: 实验配置
            n_episodes: 评估的 episode 数
            dt: 时间步长 [s]
        """
        if schedule:
            self.schedule = schedule

        # 保存汇总 CSV
        self._save_summary_csv()

        # 保存汇总 JSON
        self._save_summary_json(config, n_episodes, dt)

        # 保存每个 episode 的汇总
        self._save_episode_summaries()

        print(f"[{self.controller_name}] 结果已保存至: {self.exp_dir}")

    def _save_episode_trajectory(self):
        """保存单 episode 完整轨迹"""
        if not self.episode_trajectory:
            return

        filename = f'episode_{self.current_episode:03d}_trajectory.csv'
        filepath = os.path.join(self.trajectory_dir, filename)

        df = pd.DataFrame([asdict(r) for r in self.episode_trajectory])
        df.to_csv(filepath, index=False)

    def _save_episode_energy(self):
        """保存单 episode 能耗分解"""
        if not self.episode_trajectory:
            return

        energy_data = []
        for r in self.episode_trajectory:
            energy_data.append({
                'step': r.step,
                'sim_time_days': r.sim_time_days,
                'power_led_I1': r.power_led_I1,
                'power_led_I2': r.power_led_I2,
                'power_led_total': r.power_led_total,
                'power_hvac': r.power_hvac,
                'power_vent': r.power_vent,
                'power_dehum': r.power_dehum,
                'power_total': r.power_total,
                'energy_led_cumulative': r.energy_led_cumulative,
                'energy_hvac_cumulative': r.energy_hvac_cumulative,
                'energy_vent_cumulative': r.energy_vent_cumulative,
                'energy_dehum_cumulative': r.energy_dehum_cumulative,
                'energy_total_cumulative': r.energy_total_cumulative,
                'elec_price': r.elec_price,
                'schedule_t1': r.schedule_t1,
                'schedule_t2': r.schedule_t2,
            })

        filename = f'episode_{self.current_episode:03d}_energy.csv'
        filepath = os.path.join(self.energy_dir, filename)

        df = pd.DataFrame(energy_data)
        df.to_csv(filepath, index=False)

    def _compute_episode_summary(self, total_reward: float) -> Dict[str, float]:
        """计算单 episode 汇总统计"""
        if not self.episode_trajectory:
            return {}

        steps = len(self.episode_trajectory)
        records = self.episode_trajectory

        summary = {
            'episode': self.current_episode,
            'n_steps': steps,
            'total_reward': total_reward,
            'mean_reward': total_reward / steps if steps > 0 else 0.0,
            'final_M_total': records[-1].M_total if records else 0.0,
            'energy_total': records[-1].energy_total_cumulative if records else 0.0,
            'cost_total': sum(r.cost_total for r in records),
            'temp_mean': np.mean([r.T_inside for r in records]),
            'temp_std': np.std([r.T_inside for r in records]),
            'rh_mean': np.mean([r.RH_inside for r in records]),
            'rh_std': np.std([r.RH_inside for r in records]),
            'co2_mean': np.mean([r.CO2_inside for r in records]),
            'co2_std': np.std([r.CO2_inside for r in records]),
            'temp_violation_rate': np.mean([r.temp_violation > 0.01 for r in records]),
            'rh_violation_rate': np.mean([r.rh_violation > 0.01 for r in records]),
            'solver_time_mean': np.mean([r.solver_time for r in records if r.solver_time >= 0]),
            'solver_success_rate': np.mean([r.solver_status == 'Solved' for r in records if r.solver_status]),
        }

        return summary

    def _save_summary_csv(self):
        """保存跨 episode 的汇总 CSV"""
        if not self.episode_summaries:
            return

        # 跨 episode 统计
        rewards = [s['total_reward'] for s in self.episode_summaries]
        temps = [s['temp_mean'] for s in self.episode_summaries]
        rhs = [s['rh_mean'] for s in self.episode_summaries]
        co2s = [s['co2_mean'] for s in self.episode_summaries]
        energies = [s['energy_total'] for s in self.episode_summaries]
        costs = [s['cost_total'] for s in self.episode_summaries]

        summary_data = {
            'controller': self.controller_name,
            'n_episodes': len(self.episode_summaries),
            'n_steps_per_episode': [s['n_steps'] for s in self.episode_summaries],
            'reward_mean': np.mean(rewards),
            'reward_std': np.std(rewards),
            'reward_min': np.min(rewards),
            'reward_max': np.max(rewards),
            'temp_mean': np.mean(temps),
            'temp_std': np.std(temps),
            'rh_mean': np.mean(rhs),
            'rh_std': np.std(rhs),
            'co2_mean': np.mean(co2s),
            'co2_std': np.std(co2s),
            'energy_mean': np.mean(energies),
            'energy_std': np.std(energies),
            'cost_mean': np.mean(costs),
            'cost_std': np.std(costs),
            'profit_mean': np.mean([r - c for r, c in zip(rewards, costs)]),
            'temp_violation_rate_mean': np.mean([s['temp_violation_rate'] for s in self.episode_summaries]),
            'rh_violation_rate_mean': np.mean([s['rh_violation_rate'] for s in self.episode_summaries]),
            'solver_success_rate': np.mean([s['solver_success_rate'] for s in self.episode_summaries]) if any(s['solver_success_rate'] >= 0 for s in self.episode_summaries) else -1.0,
            'solver_time_mean': np.mean([s['solver_time_mean'] for s in self.episode_summaries if s['solver_time_mean'] >= 0]),
        }

        filepath = os.path.join(self.exp_dir, 'summary_statistics.csv')
        df = pd.DataFrame([summary_data])
        df.to_csv(filepath, index=False)

    def _save_summary_json(self, config: Optional[Dict], n_episodes: int, dt: float):
        """保存汇总 JSON"""
        if not self.episode_summaries:
            return

        summary = ExperimentSummary(
            experiment_id=self.experiment_id,
            controller_name=self.controller_name,
            timestamp=datetime.now().isoformat(),
            schedule_t1=self.schedule.get('t1', 0.0),
            schedule_t2=self.schedule.get('t2', 0.0),
            schedule_rho2=self.schedule.get('rho2', 0.0),
            schedule_A1_A2=self.schedule.get('A1_A2', 0.0),
            n_episodes=n_episodes or len(self.episode_summaries),
            n_steps_per_episode=self.episode_summaries[0]['n_steps'] if self.episode_summaries else 0,
            dt=dt,
        )

        # 填充数值字段
        rewards = [s['total_reward'] for s in self.episode_summaries]
        energies = [s['energy_total'] for s in self.episode_summaries]
        costs = [s['cost_total'] for s in self.episode_summaries]
        temps = [s['temp_mean'] for s in self.episode_summaries]
        rhs = [s['rh_mean'] for s in self.episode_summaries]
        co2s = [s['co2_mean'] for s in self.episode_summaries]

        summary.reward_mean = np.mean(rewards)
        summary.reward_std = np.std(rewards)
        summary.reward_min = np.min(rewards)
        summary.reward_max = np.max(rewards)
        summary.energy_total_mean = np.mean(energies)
        summary.energy_total_std = np.std(energies)
        summary.cost_total_mean = np.mean(costs)
        summary.temp_mean = np.mean(temps)
        summary.temp_std = np.std(temps)
        summary.rh_mean = np.mean(rhs)
        summary.rh_std = np.std(rhs)
        summary.co2_mean = np.mean(co2s)
        summary.co2_std = np.std(co2s)
        summary.profit_mean = np.mean([r - c for r, c in zip(rewards, costs)])

        filepath = os.path.join(self.exp_dir, 'experiment_summary.json')
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(asdict(summary), f, indent=2, ensure_ascii=False)

    def _save_episode_summaries(self):
        """保存每个 episode 的汇总"""
        if not self.episode_summaries:
            return

        filepath = os.path.join(self.exp_dir, 'episode_summaries.csv')
        df = pd.DataFrame(self.episode_summaries)
        df.to_csv(filepath, index=False)

    def get_all_trajectories_df(self) -> pd.DataFrame:
        """获取所有 episode 的轨迹 DataFrame（便于分析）"""
        all_records = []
        for ep_idx, ep_records in enumerate(self.all_episodes):
            for r in ep_records:
                r_dict = asdict(r)
                r_dict['episode'] = ep_idx
                all_records.append(r_dict)

        if not all_records:
            return pd.DataFrame()

        return pd.DataFrame(all_records)

    def get_summary_stats(self) -> Dict[str, float]:
        """获取跨 episode 的汇总统计"""
        if not self.episode_summaries:
            return {}

        rewards = [s['total_reward'] for s in self.episode_summaries]
        energies = [s['energy_total'] for s in self.episode_summaries]
        costs = [s['cost_total'] for s in self.episode_summaries]
        final_ms = [s['final_M_total'] for s in self.episode_summaries]

        return {
            'reward_mean': np.mean(rewards),
            'reward_std': np.std(rewards),
            'reward_min': np.min(rewards),
            'reward_max': np.max(rewards),
            'energy_mean': np.mean(energies),
            'energy_std': np.std(energies),
            'cost_mean': np.mean(costs),
            'cost_std': np.std(costs),
            'profit_mean': np.mean([r - c for r, c in zip(rewards, costs)]),
            'profit_std': np.std([r - c for r, c in zip(rewards, costs)]),
            'yield_mean': np.mean(final_ms),
            'yield_std': np.std(final_ms),
        }


def merge_controller_results(
    results_dirs: List[str],
    output_path: str,
    schedule: Optional[Dict[str, float]] = None,
):
    """
    合并多个控制器的结果，生成对比分析 CSV

    参数:
        results_dirs: 各个控制器结果目录列表
        output_path: 输出文件路径
        schedule: 排程参数
    """
    all_summaries = []

    for res_dir in results_dirs:
        # 读取 summary_statistics.csv
        summary_path = os.path.join(res_dir, 'summary_statistics.csv')
        if os.path.exists(summary_path):
            df = pd.read_csv(summary_path)
            all_summaries.append(df)

    if all_summaries:
        merged = pd.concat(all_summaries, ignore_index=True)
        merged.to_csv(output_path, index=False)
        print(f"合并结果已保存: {output_path}")
    else:
        print("未找到可合并的结果文件")
