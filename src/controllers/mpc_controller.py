# -*- coding: utf-8 -*-
"""
MPC控制器类

实现植物工厂非线性MPC控制器，基于CasADi符号优化。

【关键设计变化】
1. MPC物理模型使用 per-batch 干物质状态（与RL BatchManager完全对齐）
2. 状态维度动态：NX = 3 + 2 + 2*N1 + 2*N2 + 2（N1/N2变化时NLP需重建）
3. 每个batch独立计算生理速率后按面积加权汇总（修正之前的zone平均错误）
4. 控制器在每次 solve() 时传入实际 N1/N2
5. L1软约束惩罚处理环境条件约束
6. Warm-starting加速NLP求解
7. Episode边界重置NLP以支持连续模拟
8. 所有参数从YAML配置文件读取

来源: plant_factory_optimization项目, 参考RL-SMPC mpc.py
"""

import casadi as ca
import numpy as np
from typing import Tuple, Dict, Any, Optional
import yaml
import os
import time as wall_time

from ..models.mpc_model import (
    IDX_C, IDX_T, IDX_RH, IDX_DLI1, IDX_DLI2,
    _batch_indices,
    define_mpc_model,
    env_and_batch_to_mpc_state,
    env_state_to_mpc_state,
    generate_disturbance_profile,
    compute_step_reward_mpc,
)


NU = 6   # 动作维度
ND = 4   # 扰动维度


class PlantFactoryMPC:
    """
    植物工厂MPC控制器

    基于CasADi非线性预测控制的植物工厂气候控制器。
    使用per-batch干物质状态MPC模型（与RL BatchManager完全对齐）。

    【核心修正】每个batch独立计算生理速率：
    - 育苗区: total_phi_phot_c = sum_i(phi_phot_c_i * area_batch1)
    - 定植区: total_phi_phot_c = sum_i(phi_phot_c_i * area_batch2)
    - 同理蒸腾、热负荷等

    Attributes:
        Np: 预测步数
        dt: 控制时间步长 [秒]
        N1_schedule: 育苗区标称batch数（由schedule推导，用于模型构建）
        N2_schedule: 定植区标称batch数（由schedule推导，用于模型构建）
        A1, A2: 育苗/定植区面积 [m²]
        F: 离散状态转移函数（per-batch模型）
        g: 输出函数
        model_info: 模型信息 {'N1', 'N2', 'NX', 'offs', ...}
        opti: NLP优化问题（NX变化时需重建）
        u_prev: 上一步最优动作 [NU]
        solver_stats: 求解统计信息
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        mpc_params_path: Optional[str] = None,
        crop_params: Optional[Dict] = None,
        container_params: Optional[Dict] = None,
        equipment_params: Optional[Dict] = None,
        reward_params: Optional[Dict] = None,
        schedule: Optional[Dict] = None,
        Np: int = 8,
        verbose: bool = False
    ):
        """
        初始化MPC控制器。

        所有参数从YAML配置文件读取。
        """
        self.verbose = verbose
        self.Np = Np

        # ========== 加载配置 ==========
        if config is not None:
            self.config = config
        else:
            self.config = self._load_default_config(mpc_params_path)

        self._parse_config()

        # ========== 接收外部参数字典（从YAML加载后传入） ==========
        if crop_params is not None:
            self.crop_params = crop_params
        if container_params is not None:
            self.container_params = container_params
        if equipment_params is not None:
            self.equipment_params = equipment_params

        # ========== 排程参数 → 面积和标称批次数 ==========
        if schedule is not None:
            self.schedule = schedule
        else:
            self.schedule = {
                't1': self.config['default_schedule']['t1'],
                't2': self.config['default_schedule']['t2'],
                'rho2': self.config['default_schedule']['rho2'],
                'A1_A2': self.config['default_schedule']['A1_A2'],
            }

        # 育苗区/定植区总面积
        A_total = self.container_params['c_total_plant_area']
        self.A1 = A_total / (1.0 + self.schedule['A1_A2'])
        self.A2 = A_total - self.A1
        self.A_total = A_total

        # 标称batch数（由schedule推导，用于初始模型构建）
        import math
        delta_t = math.gcd(self.schedule['t1'], self.schedule['t2'])
        self.N1_schedule = self.schedule['t1'] // delta_t
        self.N2_schedule = self.schedule['t2'] // delta_t

        # ========== 构建CasADi模型（使用标称N1/N2） ==========
        self._build_model()

        # ========== 构建NLP问题 ==========
        self._build_nlp()

        # ========== 初始化求解状态 ==========
        self._init_solver_state()

        # ========== 统计信息 ==========
        self.solver_stats = {
            'n_solves': 0, 'n_success': 0, 'n_fail': 0,
            'total_time': 0.0, 'solve_times': [], 'exit_messages': [],
        }

        # ========== 追踪当前NLP的N1/N2（用于懒重建检测）==========
        self._current_nlp_N1 = self.N1_schedule
        self._current_nlp_N2 = self.N2_schedule

    def _load_default_config(self, mpc_params_path: Optional[str]) -> Dict[str, Any]:
        """从YAML文件加载MPC配置"""
        if mpc_params_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            mpc_params_path = os.path.join(base_dir, 'configs', 'mpc_params.yaml')

        if os.path.exists(mpc_params_path):
            with open(mpc_params_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        else:
            raise FileNotFoundError(
                f"MPC配置文件不存在: {mpc_params_path}。"
                "所有参数必须从YAML配置文件读取。"
            )

    def _parse_config(self):
        """解析配置字典（全部来自YAML）"""
        cfg = self.config

        # MPC参数
        mpc_cfg = cfg.get('mpc', {})
        self.dt = float(mpc_cfg.get('dt', 3600.0))
        self.action_rate_limit = mpc_cfg.get('action_rate_limit', {})

        # NLP选项
        nlp_cfg = mpc_cfg.get('nlp_opts', {})
        self.nlp_opts = {
            'max_iter': int(nlp_cfg.get('max_iter', 500)),
            'warm_start': bool(nlp_cfg.get('warm_start', True)),
            'tol': float(nlp_cfg.get('tol', 1e-4)),
            'acceptable_tol': float(nlp_cfg.get('acceptable_tol', 1e-2)),
            'constr_viol_tol': float(nlp_cfg.get('constr_viol_tol', 1e-2)),
            'print_level': int(nlp_cfg.get('print_level', 0)),
        }

        # 软约束
        soft = cfg.get('soft_constraints', {})
        self.soft_CO2 = soft.get('CO2', {})
        self.soft_T = soft.get('temperature', {})
        self.soft_RH = soft.get('RH', {})
        self.slack_penalty = 100.0

        # 经济参数
        obj = cfg.get('objective', {})
        self.price_growth = float(obj.get('price_growth', 0.25))
        self.price_energy = float(obj.get('price_energy', 0.6))
        self.price_CO2 = float(obj.get('price_CO2', 0.5))
        self.seedling_discount = float(obj.get('seedling_discount', 0.5))

        # 设备限制
        equip = cfg.get('equipment_limits', {})
        self.u_min = np.array([
            equip['I1_min'], equip['I2_min'],
            equip['Q_HVAC_min'], equip['u_CO2_min'],
            equip['V_vent_min'], equip['m_dehum_min'],
        ], dtype=np.float64)
        self.u_max = np.array([
            equip['I1_max'], equip['I2_max'],
            equip['Q_HVAC_max'], equip['u_CO2_max'],
            equip['V_vent_max'], equip['m_dehum_max'],
        ], dtype=np.float64)

        # 动作变化率限制
        rl = self.action_rate_limit
        self.du_max = np.array([
            rl['I1'], rl['I2'], rl['Q_HVAC'],
            rl['u_CO2'], rl['V_vent'], rl['m_dehum'],
        ], dtype=np.float64)

    def _build_model(self, N1: int = None, N2: int = None):
        """构建CasADi预测模型（使用指定N1/N2）"""
        if N1 is None:
            N1 = self.N1_schedule
        if N2 is None:
            N2 = self.N2_schedule

        self.F, self.g, self.p_crop, self.p_cont, self.model_info = define_mpc_model(
            dt=self.dt,
            crop_params=self.crop_params,
            container_params=self.container_params,
            A1=self.A1, A2=self.A2,
            N1=N1, N2=N2,
            x_min=self._make_x_min(N1, N2),
            x_max=self._make_x_max(N1, N2),
        )

    def _make_x_min(self, N1: int, N2: int) -> np.ndarray:
        """生成per-batch状态向量的下界"""
        offs = _batch_indices(N1, N2)
        NX = offs['NX']
        x_min = np.zeros(NX, dtype=np.float64)

        # 环境状态
        x_min[IDX_C] = 1.0e-4
        x_min[IDX_T] = 10.0
        x_min[IDX_RH] = 0.1
        x_min[IDX_DLI1] = 0.0
        x_min[IDX_DLI2] = 0.0

        # per-batch干物质密度
        x_min[offs['IDX_SEEDLING_DN']:offs['IDX_SEEDLING_DN'] + N1] = 0.0
        x_min[offs['IDX_SEEDLING_DS']:offs['IDX_SEEDLING_DS'] + N1] = 0.001
        x_min[offs['IDX_TRANSPLANT_DN']:offs['IDX_TRANSPLANT_DN'] + N2] = 0.0
        x_min[offs['IDX_TRANSPLANT_DS']:offs['IDX_TRANSPLANT_DS'] + N2] = 0.001

        # biomass
        x_min[offs['IDX_BM1']] = 0.0
        x_min[offs['IDX_BM2']] = 0.0

        return x_min

    def _make_x_max(self, N1: int, N2: int) -> np.ndarray:
        """生成per-batch状态向量的上界"""
        offs = _batch_indices(N1, N2)
        NX = offs['NX']
        x_max = np.zeros(NX, dtype=np.float64)

        # 环境状态
        x_max[IDX_C] = 3.0e-3
        x_max[IDX_T] = 40.0
        x_max[IDX_RH] = 1.0
        x_max[IDX_DLI1] = 20.0
        x_max[IDX_DLI2] = 20.0

        # per-batch干物质密度
        x_max[offs['IDX_SEEDLING_DN']:offs['IDX_SEEDLING_DN'] + N1] = 0.5
        x_max[offs['IDX_SEEDLING_DS']:offs['IDX_SEEDLING_DS'] + N1] = 0.6
        x_max[offs['IDX_TRANSPLANT_DN']:offs['IDX_TRANSPLANT_DN'] + N2] = 0.5
        x_max[offs['IDX_TRANSPLANT_DS']:offs['IDX_TRANSPLANT_DS'] + N2] = 0.6

        # biomass
        x_max[offs['IDX_BM1']] = 50.0
        x_max[offs['IDX_BM2']] = 50.0

        return x_max

    def _maybe_rebuild_nlp(self, N1_actual: int, N2_actual: int,
                            event_info: Optional[Dict[str, Any]] = None):
        """
        懒重建NLP（状态维度变化或事件触发）。

        两种情况需要重建NLP：
        1. N1/N2 变化 → NX变化 → NLP决策变量维度不匹配
        2. 预测范围内有移栽/采收事件 → 连续ODE无法处理离散跳变

        参数:
            N1_actual: 育苗区实际batch数量
            N2_actual: 定植区实际batch数量
            event_info: batch_manager.predict_next_event() 的返回值
        """
        need_rebuild = False
        reason = ""

        # 1. 状态维度变化检测
        if N1_actual != self._current_nlp_N1 or N2_actual != self._current_nlp_N2:
            need_rebuild = True
            reason = f"N1 {self._current_nlp_N1}→{N1_actual}, N2 {self._current_nlp_N2}→{N2_actual}"

        # 2. 事件触发检测
        if event_info is not None and event_info.get('event_trigger', False):
            need_rebuild = True
            if not reason:
                reason = f"event trigger (t={event_info.get('first_transplant_h', -1):.1f}h, h={event_info.get('first_harvest_h', -1):.1f}h)"

        if need_rebuild:
            if self.verbose:
                print(f"  [NLP rebuild] {reason}")
            self._build_model(N1_actual, N2_actual)
            self._build_nlp()
            self._current_nlp_N1 = N1_actual
            self._current_nlp_N2 = N2_actual

    def _build_nlp(self):
        """构建MPC NLP优化问题（基于当前model_info的动态NX）"""
        F_mx = self.F
        g_mx = self.g

        offs = self.model_info['offs']
        NX = self.model_info['NX']
        idx_bm1 = offs['IDX_BM1']
        idx_bm2 = offs['IDX_BM2']

        self.opti = ca.Opti()

        # 决策变量（动态维度）
        self.us = self.opti.variable(NU, self.Np)
        self.xs = self.opti.variable(NX, self.Np + 1)
        self.slack = self.opti.variable(3, self.Np)

        # 参数
        self.x0 = self.opti.parameter(NX, 1)
        self.u_prev = self.opti.parameter(NU, 1)
        self.ds = self.opti.parameter(ND, self.Np)

        # 经济参数
        self.p_price_growth = self.opti.parameter(1, 1)
        self.p_price_CO2 = self.opti.parameter(1, 1)
        self.p_seedling_disc = self.opti.parameter(1, 1)
        self.p_dt = self.opti.parameter(1, 1)
        self.p_A1 = self.opti.parameter(1, 1)
        self.p_A2 = self.opti.parameter(1, 1)
        self.p_optical_eff = self.opti.parameter(1, 1)
        self.p_led_eff = self.opti.parameter(1, 1)
        self.p_COP = self.opti.parameter(1, 1)
        self.p_fan_eff = self.opti.parameter(1, 1)
        self.p_dehum_eev = self.opti.parameter(1, 1)
        self.p_y_min = self.opti.parameter(9, 1)
        self.p_y_max = self.opti.parameter(9, 1)
        self.p_slack_pen = self.opti.parameter(1, 1)

        # ========== 约束定义 ==========

        # 1. 初始状态
        self.opti.subject_to(self.xs[:, 0] == self.x0)

        # 2. 动作边界
        for k in range(self.Np):
            self.opti.subject_to(self.u_min <= self.us[:, k])
            self.opti.subject_to(self.us[:, k] <= self.u_max)

        # 3. 动作变化率
        for k in range(self.Np):
            if k == 0:
                self.opti.subject_to(-self.du_max <= self.us[:, k] - self.u_prev[:, 0])
                self.opti.subject_to(self.us[:, k] - self.u_prev[:, 0] <= self.du_max)
            else:
                self.opti.subject_to(-self.du_max <= self.us[:, k] - self.us[:, k - 1])
                self.opti.subject_to(self.us[:, k] - self.us[:, k - 1] <= self.du_max)

        # 4. 动力学约束
        for k in range(self.Np):
            x_next_pred = F_mx(
                self.xs[:, k],
                self.us[:, k],
                self.ds[:, k]
            )
            self.opti.subject_to(self.xs[:, k + 1] == x_next_pred)

        # 5. 软约束
        for k in range(self.Np):
            y_k = g_mx(self.xs[:, k + 1])
            self.opti.subject_to(self.slack[0, k] >= 0)
            self.opti.subject_to(y_k[0] >= self.p_y_min[0] - self.slack[0, k])
            self.opti.subject_to(y_k[0] <= self.p_y_max[0] + self.slack[0, k])
            self.opti.subject_to(self.slack[1, k] >= 0)
            self.opti.subject_to(y_k[1] >= self.p_y_min[1] - self.slack[1, k])
            self.opti.subject_to(y_k[1] <= self.p_y_max[1] + self.slack[1, k])
            self.opti.subject_to(self.slack[2, k] >= 0)
            self.opti.subject_to(y_k[2] >= self.p_y_min[2] - self.slack[2, k])
            self.opti.subject_to(y_k[2] <= self.p_y_max[2] + self.slack[2, k])

        # ========== 目标函数（基于biomass增量，与RL reward对齐）==========
        J = 0.0

        for k in range(self.Np):
            x_k = self.xs[:, k]
            x_next = self.xs[:, k + 1]
            u_k = self.us[:, k]

            dt_h = self.p_dt / 3600.0
            A1 = self.p_A1; A2 = self.p_A2

            # --- 育苗区生长收益（biomass增量 × 面积 × 半价折扣）---
            d_bm1 = x_next[idx_bm1] - x_k[idx_bm1]
            seedling_reward = d_bm1 * 1000.0 * self.p_seedling_disc * self.p_price_growth

            # --- 定植区生长收益 ---
            d_bm2 = x_next[idx_bm2] - x_k[idx_bm2]
            transplant_reward = d_bm2 * 1000.0 * self.p_price_growth

            growth_reward = seedling_reward + transplant_reward

            # --- 能耗成本 ---
            I1_k, I2_k = u_k[0], u_k[1]
            Q_HVAC_k = u_k[2]; u_CO2_k = u_k[3]
            V_vent_k = u_k[4]; m_dehum_k = u_k[5]

            A_total = A1 + A2
            c_opt = self.p_optical_eff
            c_led = self.p_led_eff

            P_led_k = (I1_k / c_opt) * A1 / c_led + \
                      (I2_k / c_opt) * A2 / c_led

            c_COP = self.p_COP
            P_hvac_k = ca.if_else(
                Q_HVAC_k > 0,
                Q_HVAC_k * A_total / c_COP,
                -Q_HVAC_k * A_total / c_COP
            )

            P_vent_k = V_vent_k * A_total / self.p_fan_eff
            P_dehum_k = (m_dehum_k * A_total) / self.p_dehum_eev * 1000.0

            E_kWh = (P_led_k + P_hvac_k + P_vent_k + P_dehum_k) * dt_h / 1000.0
            elec_price_k = self.ds[3, k]
            cost_energy = E_kWh * elec_price_k

            # CO2成本
            u_CO2_kg_s = u_CO2_k / 3600.0
            CO2_kg = u_CO2_kg_s * A_total * self.p_dt
            cost_CO2 = CO2_kg * self.p_price_CO2

            # 软约束惩罚
            slack_pen = self.p_slack_pen * (
                self.slack[0, k] + self.slack[1, k] + self.slack[2, k]
            )

            J += -growth_reward + cost_energy + cost_CO2 + slack_pen

        self.opti.minimize(J)

        # ========== 求解器配置 ==========
        ipopt_opts = {
            'max_iter': self.nlp_opts['max_iter'],
            'warm_start_entire_iterate': 'yes' if self.nlp_opts['warm_start'] else 'no',
            'nlp_scaling_method': 'gradient-based',
            'constr_viol_tol': self.nlp_opts['constr_viol_tol'],
            'acceptable_constr_viol_tol': self.nlp_opts['constr_viol_tol'] * 10,
            'acceptable_tol': self.nlp_opts['acceptable_tol'],
            'print_level': self.nlp_opts['print_level'],
        }
        self.opti.solver('ipopt', ipopt_opts)

    def _init_solver_state(self):
        """初始化求解器状态"""
        ep = self.equipment_params
        self.u_prev = np.array([
            ep.get('I1_max', 200.0) * 0.5,
            ep.get('I2_max', 200.0) * 0.5,
            0.0, 0.0,
            ep.get('vent_min', 0.0) + ep.get('vent_max', 0.1) * 0.1,
            ep.get('dehum_min', 0.0) + ep.get('dehum_max', 2e-5) * 0.5,
        ], dtype=np.float64)
        self.default_d = np.array([20.0, 0.7, 4.0e-4, 0.6], dtype=np.float64)

    def reset(self):
        """
        重置求解器状态（连续模拟时，在每个episode边界调用）。

        保留NLP结构（CasADi函数不重建），只重置动作历史和统计。
        """
        ep = self.equipment_params
        self.u_prev = np.array([
            ep.get('I1_max', 200.0) * 0.5,
            ep.get('I2_max', 200.0) * 0.5,
            0.0, 0.0,
            ep.get('vent_min', 0.0) + ep.get('vent_max', 0.1) * 0.1,
            ep.get('dehum_min', 0.0) + ep.get('dehum_max', 2e-5) * 0.5,
        ], dtype=np.float64)
        self.solver_stats = {
            'n_solves': 0, 'n_success': 0, 'n_fail': 0,
            'total_time': 0.0, 'solve_times': [], 'exit_messages': [],
        }

    def solve(
        self,
        x0: np.ndarray,
        N1_actual: int,
        N2_actual: int,
        hour_of_day: int,
        day_of_period: int = 0,
        external: Optional[np.ndarray] = None,
        elec_price: Optional[float] = None,
        u_guess: Optional[np.ndarray] = None,
        batch_manager=None,
    ) -> Tuple[np.ndarray, np.ndarray, float, float, str]:
        """
        求解MPC优化问题（Practical Hybrid MPC）。

        参数:
            x0: 初始状态 [NX=动态]
            N1_actual: 育苗区实际batch数量（来自batch_manager）
            N2_actual: 定植区实际batch数量（来自batch_manager）
            hour_of_day: 当前小时 [0-23]
            day_of_period: 周期第几天
            external: 外部环境 [T_out, RH_out, C_out_ppm]
            elec_price: 当前电价 [元/kWh]
            u_guess: 动作初始猜测 [NU, Np]
            batch_manager: BatchManager实例（用于事件预测和NLP重建）

        返回:
            u_opt, x_traj, J_opt, solve_time, exit_msg
        """
        # ========== 事件触发的NLP懒重建（Practical Hybrid MPC）==========
        event_info = None
        if batch_manager is not None:
            horizon_h = self.Np * self.dt / 3600.0
            event_info = batch_manager.predict_next_event(horizon_h)

        # 懒重建：N1/N2变化 或 事件触发
        self._maybe_rebuild_nlp(N1_actual, N2_actual, event_info)

        # 重建后需要重新获取最新的offs
        offs = self.model_info['offs']
        NX = self.model_info['NX']
        idx_bm1 = offs['IDX_BM1']
        idx_bm2 = offs['IDX_BM2']

        if external is not None:
            T_out, RH_out, C_out_ppm = external
        else:
            T_out = self.default_d[0]; RH_out = self.default_d[1]; C_out_ppm = 400.0
        if elec_price is None:
            elec_price = 0.6

        # 扰动序列
        ds = np.zeros((ND, self.Np), dtype=np.float64)
        for k in range(self.Np):
            h = (hour_of_day + k) % 24
            d = generate_disturbance_profile(
                hour_of_day=h, day_of_year=day_of_period + 1,
                T_out_base=T_out, RH_out_base=RH_out,
                C_out_ppm=C_out_ppm, elec_price_base=elec_price,
                elec_price_min=0.3, elec_price_max=1.0,
            )
            ds[:, k] = d

        # 设置参数
        self.opti.set_value(self.x0, x0.reshape((NX, 1)))
        self.opti.set_value(self.u_prev, self.u_prev.reshape((NU, 1)))
        self.opti.set_value(self.ds, ds)
        self.opti.set_value(self.p_price_growth, self.price_growth)
        self.opti.set_value(self.p_price_CO2, self.price_CO2)
        self.opti.set_value(self.p_seedling_disc, self.seedling_discount)
        self.opti.set_value(self.p_dt, self.dt)
        self.opti.set_value(self.p_A1, self.A1)
        self.opti.set_value(self.p_A2, self.A2)
        self.opti.set_value(self.p_optical_eff, self.crop_params['c_optical_eff'])
        self.opti.set_value(self.p_led_eff, self.container_params['c_led_eff'])
        self.opti.set_value(self.p_COP, self.equipment_params['c_COP'])
        self.opti.set_value(self.p_fan_eff, self.equipment_params['fan_eff'])
        self.opti.set_value(self.p_dehum_eev, self.equipment_params['c_dehum_eev'])
        self.opti.set_value(self.p_y_min, self.y_min)
        self.opti.set_value(self.p_y_max, self.y_max)
        self.opti.set_value(self.p_slack_pen, self.slack_penalty)

        # 初始猜测
        if u_guess is None:
            u_guess = np.tile(self.u_prev.reshape((NU, 1)), (1, self.Np))
        x_guess = np.zeros((NX, self.Np + 1))
        x_guess[:, 0] = x0
        for k in range(self.Np):
            x_guess[:, k + 1] = self.F(
                x_guess[:, k], u_guess[:, k], ds[:, k]
            ).toarray().ravel()

        self.opti.set_initial(self.us, u_guess)
        self.opti.set_initial(self.xs, x_guess)

        # 求解
        solve_start = wall_time.time()
        try:
            sol = self.opti.solve()
            exit_msg = sol.stats().get('return_status', 'Solved')
            u_opt = sol.value(self.us)
            x_traj = sol.value(self.xs)
            J_opt = float(sol.value(self.opti.f))
            success = True
        except RuntimeError as e:
            if self.verbose:
                print(f"  MPC solve failed: {e}")
            sol = self.opti.debug
            exit_msg = 'SolveFailed'
            try:
                u_opt = sol.value(self.us)
                x_traj = sol.value(self.xs)
                J_opt = float(sol.value(self.opti.f))
            except Exception:
                u_opt = np.tile(self.u_prev.reshape((NU, 1)), (1, self.Np))
                x_traj = np.zeros((NX, self.Np + 1)); x_traj[:, 0] = x0
                J_opt = 0.0
            success = False
        except Exception as e:
            if self.verbose:
                print(f"  MPC solve error: {e}")
            u_opt = np.tile(self.u_prev.reshape((NU, 1)), (1, self.Np))
            x_traj = np.zeros((NX, self.Np + 1)); x_traj[:, 0] = x0
            J_opt = 0.0; exit_msg = 'Error'; success = False

        solve_time = wall_time.time() - solve_start
        self.u_prev = u_opt[:, 0].copy()
        self.solver_stats['n_solves'] += 1
        self.solver_stats['total_time'] += solve_time
        self.solver_stats['solve_times'].append(solve_time)
        self.solver_stats['exit_messages'].append(exit_msg)
        if success:
            self.solver_stats['n_success'] += 1
        else:
            self.solver_stats['n_fail'] += 1

        return u_opt, x_traj, J_opt, solve_time, exit_msg

    def predict_open_loop(
        self,
        x0: np.ndarray,
        u_seq: np.ndarray,
        d_seq: np.ndarray,
        N1_actual: int = None,
        N2_actual: int = None,
    ) -> np.ndarray:
        """开环预测"""
        offs = self.model_info['offs']
        NX = self.model_info['NX']
        x_traj = np.zeros((NX, self.Np + 1))
        x_traj[:, 0] = x0
        for k in range(self.Np):
            x_traj[:, k + 1] = self.F(
                x_traj[:, k], u_seq[:, k], d_seq[:, k]
            ).toarray().ravel()
        return x_traj

    def compute_reward(
        self,
        x_curr: np.ndarray,
        x_next: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        N1_actual: int = 1,
        N2_actual: int = 1,
    ) -> Tuple[float, Dict[str, float]]:
        """计算MPC步奖励（基于biomass增量）"""
        return compute_step_reward_mpc(
            x_curr=x_curr, x_next=x_next, u=u, d=d,
            A1=self.A1, A2=self.A2,
            N1=N1_actual, N2=N2_actual,
            crop_params=self.crop_params,
            equipment_params=self.equipment_params,
            dt=self.dt,
            elec_price=d[3] if len(d) > 3 else 0.6,
            price_growth=self.price_growth,
            price_CO2=self.price_CO2,
            seedling_discount=self.seedling_discount,
        )

    def get_statistics(self) -> Dict[str, Any]:
        """获取求解器统计信息"""
        stats = self.solver_stats.copy()
        if stats['n_solves'] > 0:
            stats['success_rate'] = stats['n_success'] / stats['n_solves']
            stats['avg_solve_time'] = stats['total_time'] / stats['n_solves']
        else:
            stats['success_rate'] = 0.0
            stats['avg_solve_time'] = 0.0
        return stats

    @property
    def y_min(self) -> np.ndarray:
        """输出约束下界"""
        return np.array([
            self.soft_CO2['lb'] * 1000.0,
            self.soft_T['lb'],
            self.soft_RH['lb'],
            0.0, 0.0, 0.0, 0.0,   # LAI1, LAI2, dLI1, dLI2
            0.0, 0.0,              # biomass1, biomass2
        ], dtype=np.float64)

    @property
    def y_max(self) -> np.ndarray:
        """输出约束上界"""
        return np.array([
            self.soft_CO2['ub'] * 1000.0,
            self.soft_T['ub'],
            self.soft_RH['ub'],
            10.0, 10.0, 20.0, 20.0,
            50.0, 50.0,
        ], dtype=np.float64)

    def __repr__(self) -> str:
        return (f"PlantFactoryMPC(Np={self.Np}, "
                f"N1={self._current_nlp_N1}(A1={self.A1:.1f}m2), "
                f"N2={self._current_nlp_N2}(A2={self.A2:.1f}m2), "
                f"solves={self.solver_stats['n_solves']}, "
                f"success_rate={self.get_statistics().get('success_rate', 0):.1%})")
