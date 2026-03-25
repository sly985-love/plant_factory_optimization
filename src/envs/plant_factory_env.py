# -*- coding: utf-8 -*-
"""
植物工厂多批次仿真环境

基于PFAL-DRL PFALEnv扩展的多批次、分区光照、上下文强化学习环境。

主要特性:
1. 多批次连续生产（育苗区+定植区）
2. 分区光照控制（育苗区/定植区独立光强）
3. 上下文信息嵌入（排程参数作为额外观测）
4. 固定维度观测（28维，避免维度爆炸）

观测空间 (28维):
    - 环境7: 箱内温度、箱内相对湿度、箱内CO2浓度，外界温度、外界相对湿度、外界CO2浓度、实时电价
    - 作物7: 总LAI、育苗区总干重、定植区总干重、最老批次剩余天数、最老批次干重、育苗区LAI、定植区LAI
    - 上步动作6: 育苗区光强、定植区光强、加热/制冷功率、CO2注入速率、通风速率、除湿速率
    - 时间2: 小时归一化、周期内天数归一化
    - 上下文4: t1, t2, rho2, A1/A2（归一化）
    - 负荷统计2: 当前总蒸腾、当前总光合

奖励函数:
    r_t = r_t^growth_delta + r_t^event + r_t^cost + r_t^penalty
    - 生长收益: α * ΔM_total (α=0.25元/g)
    - 事件收益: α * harvest_mass (采收时实现)
    - 能耗成本: -p_elec * P_total - p_CO2 * u_CO2
    - 约束惩罚: 温度/DLI/湿度/暗期/采收达标

动作空间 (6维连续):
    [I1, I2, Q_HVAC, u_CO2, V_vent, m_dehum]
    单位: μmol/m²/s, μmol/m²/s, W/m², g/m²/h, m³/m²/s, kg/m²/s

【重要】所有物理参数从配置文件读取，不硬编码。

来源: 基于PFAL-DRL PFALEnv扩展，论文方法部分 2.2, 2.3

作者: Plant Factory Optimization Team
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Dict, Any, Tuple
from copy import deepcopy
import sys
import os
import yaml  # 用于配置文件加载

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    BatchManager, simulate_environment_step,
    co2_ppm_to_density, co2_density_to_ppm,
    relative_humidity_to_absolute, calculate_saturation_vapor_pressure,
    calculate_total_power, calculate_energy_cost
)


class MultiBatchPlantFactoryEnv(gym.Env):
    """
    植物工厂多批次仿真环境

    基于Van Henten作物模型和PFAL-DRL物理模型，实现多批次连续生产仿真。

    来源: 论文方法部分 2.2, 2.3
    """

    metadata = {'render_modes': []}

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化环境

        参数:
            config: 配置参数字典，包含所有模型参数。
                   从配置文件加载，无则使用 _default_config()。
        """
        super().__init__()

        self.config = config if config is not None else self._default_config()

        # 提取各模块配置
        self.schedule = self.config['schedule']
        self.container_params = self.config['container_params']
        self.crop_params = self.config['crop_params']
        self.equipment_params = self.config['equipment_params']
        self.reward_params = self.config['reward_params']
        self.dt = float(self.config.get('dt', 3600.0))

        # 提取排程参数
        self.t1 = self.schedule['t1']
        self.t2 = self.schedule['t2']
        self.rho2 = self.schedule['rho2']
        self.A1_A2 = self.schedule['A1_A2']

        # 派生参数
        A_total = self.container_params.get('c_total_plant_area', 40.0)
        self.A1 = A_total / (1 + self.A1_A2)
        self.A2 = A_total - self.A1

        # 将面积写入container_params，供environment_model使用
        self.container_params['A1'] = self.A1
        self.container_params['A2'] = self.A2

        # 初始化批次管理器（传入稳态初始化参数）
        rng = np.random.default_rng(self.config.get('seed', 42))
        steady_state_params = self.config.get('steady_state_params', None)
        self.batch_manager = BatchManager(
            self.schedule,
            self.container_params,
            self.crop_params,
            rng,
            steady_state_params,
            self.reward_params
        )

        # ========== 观测空间定义 (28维) ==========
        # 从配置读取边界，无则用合理默认值
        self._init_observation_space()

        # ========== 动作空间定义 (6维连续) ==========
        self._init_action_space()

        # ========== 环境状态 ==========
        self.state = None
        self.prev_action = None
        self.time_step = 0
        self.episode_length = self.t2 * 24
        self.total_steps = 0

        # 光周期追踪
        self.hour_of_day = 0
        self.day_of_period = 0
        self.dark_period_hours = 0

        # 统计量
        self.episode_reward = 0.0
        self.total_revenue = 0.0
        self.total_cost = 0.0
        self.total_labor_cost = 0.0

        # 【新增】上步 biomass 追踪（用于 biomass-delta 奖励）
        self.prev_seedling_M_g = 0.0
        self.prev_transplant_M_g = 0.0

        # 负荷（从batch_manager更新）
        self.total_E = 0.0  # 总蒸腾速率 [kg water/s]
        self.total_P = 0.0  # 总光合速率 [kg CO2/s]

    def _init_observation_space(self):
        """初始化观测空间边界"""
        cp = self.container_params
        sp = self.schedule

        # 环境观测 (7维)
        # RH: YAML中为小数(0-1范围) → 乘100转为百分比(0-100)，与_get_observation()返回值保持一致
        self.obs_env_low = np.array([
            cp.get('obs_temp_min', 10.0),      # 箱内温度 [°C]
            cp.get('obs_rh_min', 0.0) * 100,  # 箱内RH [%]  ← YAML用小数(0-1)，×100得百分比
            cp.get('obs_co2_min', 300.0),       # 箱内CO2 [ppm]
            cp.get('obs_out_temp_min', -10.0), # 外界温度 [°C]
            40.0,                               # 外界RH [%]
            300.0,                              # 外界CO2 [ppm]
            cp.get('elec_price_min', 0.3),     # 电价 [元/kWh]
        ], dtype=np.float32)
        self.obs_env_high = np.array([
            cp.get('obs_temp_max', 35.0),
            cp.get('obs_rh_max', 1.0) * 100,   # 箱内RH [%]  ← YAML用小数(0-1)，×100得百分比
            cp.get('obs_co2_max', 2000.0),
            cp.get('obs_out_temp_max', 40.0),
            95.0,
            600.0,
            cp.get('elec_price_max', 1.0),
        ], dtype=np.float32)

        # 作物观测 (7维)
        self.obs_crop_low = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.obs_crop_high = np.array([
            cp.get('obs_lai_max', 6.0),      # 总LAI
            cp.get('obs_M_seedling_max', 5000.0),  # 育苗区干重 [g]
            cp.get('obs_M_transplant_max', 5000.0), # 定植区干重 [g]
            cp.get('obs_days_left_max', 30.0),     # 剩余天数
            cp.get('obs_M_oldest_max', 30.0),      # 最老批次干重 [g]
            cp.get('obs_lai_regional_max', 3.0),   # 育苗区LAI
            cp.get('obs_lai_regional_max', 3.0),   # 定植区LAI
        ], dtype=np.float32)

        # 上步动作 (6维) - 从设备参数读取
        ep = self.equipment_params
        al, ah = self._get_action_bounds_from_config()
        self.obs_action_low = al.astype(np.float32)
        self.obs_action_high = ah.astype(np.float32)

        # 时间 (2维)
        self.obs_time_low = np.array([0.0, 0.0], dtype=np.float32)
        self.obs_time_high = np.array([1.0, 1.0], dtype=np.float32)

        # 上下文 (4维) - 从schedule_params或合理范围读取
        self.obs_context_low = np.array([
            sp.get('obs_t1_min', 10.0),
            sp.get('obs_t2_min', 18.0),
            sp.get('obs_rho2_min', 20.0),
            sp.get('obs_A1_A2_min', 0.1),
        ], dtype=np.float32)
        self.obs_context_high = np.array([
            sp.get('obs_t1_max', 18.0),
            sp.get('obs_t2_max', 26.0),
            sp.get('obs_rho2_max', 80.0),
            sp.get('obs_A1_A2_max', 5.0),
        ], dtype=np.float32)

        # 负荷统计 (2维)
        self.obs_load_low = np.array([0.0, 0.0], dtype=np.float32)
        self.obs_load_high = np.array([1.0, 1.0], dtype=np.float32)

        # 合并所有观测空间
        self.obs_low = np.concatenate([
            self.obs_env_low, self.obs_crop_low,
            self.obs_action_low, self.obs_time_low,
            self.obs_context_low, self.obs_load_low
        ])
        self.obs_high = np.concatenate([
            self.obs_env_high, self.obs_crop_high,
            self.obs_action_high, self.obs_time_high,
            self.obs_context_high, self.obs_load_high
        ])

        self.observation_space = spaces.Box(
            low=self.obs_low,
            high=self.obs_high,
            dtype=np.float32
        )

    def _init_action_space(self):
        """初始化动作空间（从配置读取）"""
        al, ah = self._get_action_bounds_from_config()
        self.action_space = spaces.Box(
            low=al,
            high=ah,
            dtype=np.float32
        )

    def _get_action_bounds_from_config(self):
        """从配置文件读取动作空间边界"""
        ep = self.equipment_params

        I_max = ep.get('I_max', 600.0)
        Q_HVAC_max = ep.get('hvac_max_power_density', 212.0)
        Q_HVAC_min = ep.get('hvac_min_power_density', -212.0)
        co2_supply_max = ep.get('co2_supply_max', 0.5)
        vent_max = ep.get('c_vent_fan_cap', 0.5)
        dehum_max = ep.get('c_dehum_cap', 0.002)

        # 动作顺序: [I1, I2, Q_HVAC, u_CO2, V_vent, m_dehum]
        action_low = np.array([
            0.0,        # I1 [μmol/m²/s]
            0.0,        # I2 [μmol/m²/s]
            Q_HVAC_min, # Q_HVAC [W/m²] (从配置读取)
            0.0,        # u_CO2 [g/m²/h]
            0.0,        # V_vent [m³/m²/s]
            0.0,        # m_dehum [kg/m²/s]
        ], dtype=np.float32)
        action_high = np.array([
            I_max,         # I1
            I_max,         # I2
            Q_HVAC_max,    # Q_HVAC
            co2_supply_max,# u_CO2
            vent_max,      # V_vent
            dehum_max,     # m_dehum
        ], dtype=np.float32)

        return action_low, action_high

    def _default_config(self) -> Dict[str, Any]:
        """返回默认配置（从配置文件加载）"""
        config_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'configs'
        )

        config = {
            'schedule': {
                't1': 14, 't2': 21, 'rho2': 35.0, 'A1_A2': 0.5
            },
            'seed': 42,
            'dt': 3600.0,
        }

        try:
            with open(os.path.join(config_dir, 'container_params.yaml'), 'r', encoding='utf-8') as f:
                config['container_params'] = yaml.safe_load(f)
            with open(os.path.join(config_dir, 'crop_params.yaml'), 'r', encoding='utf-8') as f:
                config['crop_params'] = yaml.safe_load(f)
            with open(os.path.join(config_dir, 'equipment_params.yaml'), 'r', encoding='utf-8') as f:
                config['equipment_params'] = yaml.safe_load(f)
            with open(os.path.join(config_dir, 'reward_params.yaml'), 'r', encoding='utf-8') as f:
                config['reward_params'] = yaml.safe_load(f)
            with open(os.path.join(config_dir, 'schedule_params.yaml'), 'r', encoding='utf-8') as f:
                sp = yaml.safe_load(f)
                config['schedule_params'] = sp
                # schedule_params中的连续范围也作为schedule的观测边界
                config['schedule']['obs_t1_min'] = sp.get('t1_min', 10)
                config['schedule']['obs_t1_max'] = sp.get('t1_max', 18)
                config['schedule']['obs_t2_min'] = sp.get('t2_min', 18)
                config['schedule']['obs_t2_max'] = sp.get('t2_max', 26)
                config['schedule']['obs_rho2_min'] = sp.get('rho2_min', 20.0)
                config['schedule']['obs_rho2_max'] = sp.get('rho2_max', 80.0)
                config['schedule']['obs_A1_A2_min'] = sp.get('A1_A2_min', 0.1)
                config['schedule']['obs_A1_A2_max'] = sp.get('A1_A2_max', 5.0)
            # 稳态初始化参数（从container_params读取）
            cp = config['container_params']
            config['steady_state_params'] = {
                'I_standard': cp.get('I_standard', 200.0),
                'T_standard': cp.get('T_standard', 22.0),
                'C_standard_ppm': cp.get('C_standard_ppm', 1000.0),
                'RH_standard': cp.get('RH_standard', 0.75),
                'dt': cp.get('dt_steady', 3600.0),
                'disturb_factor_max': cp.get('disturb_factor_max', 0.05),
                'seedling_nonstruct_ratio': cp.get('seedling_nonstruct_ratio', 0.1),
                'initial_seedling_mass': cp.get('initial_seedling_mass', 0.72e-3),
                'I_standard_umol': cp.get('I_standard_umol', True),
            }

        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"配置文件缺失: {e}. "
                "请确保 configs/ 目录下包含所有必要 YAML 文件，或检查 configs_path 配置。"
            )

        return config
    def _hardcoded_equipment_params(self) -> Dict:
        """硬编码的设备参数（仅在配置文件缺失时使用）"""
        return {
            'c_led_eff': 0.52, 'c_optical_eff': 2.5,
            'led_max_power_density_seedling': 200.0,
            'led_max_power_density_transplant': 300.0,
            'c_COP': 3.0, 'hvac_max_power_density': 212.0,
            'hvac_min_power_density': -212.0,
            'co2_supply_max': 0.5, 'co2_injection_efficiency': 0.5, 'p_CO2': 0.5,
            'c_vent_fan_cap': 0.5, 'fan_eff': 7.07,
            'c_dehum_cap': 0.002, 'c_dehum_eev': 3.0,
            'p_elec_base': 0.6, 'p_elec_min': 0.3, 'p_elec_max': 1.0,
            'I_max': 600.0,
        }

    def _hardcoded_reward_params(self) -> Dict:
        """硬编码的奖励参数（仅在配置文件缺失时使用）"""
        return {
            'alpha_growth': 0.25,
            'seedling_growth_discount': 0.5,
            'p_CO2_cost': 0.5,
            'harvest_fail_penalty': -200.0,
            'transplant_fail_penalty': -100.0,
            'temp_violation_penalty': -10.0,
            'dark_period_penalty': -10.0,
            'min_dark_period_hours': 4.0,
            'dli_deviation_penalty': -1.0,
            'light_change_penalty_coef': 0.001,
            'rh_deviation_penalty': -0.1,
            'temp_hard_min': 18.0, 'temp_hard_max': 28.0,
            'rh_soft_min': 60.0, 'rh_soft_max': 80.0,
            'dli_target_min': 8.0, 'dli_target_max': 15.0,
            'harvest_min_dry_mass': 25.0,
            'transplant_min_dry_mass': 5.0,
            'reward_scale_factor': 100.0,
            'obs_lai_max': 6.0,
            'obs_M_seedling_max': 5000.0,
            'obs_M_transplant_max': 5000.0,
            'obs_M_oldest_max': 30.0,
            'obs_lai_regional_max': 3.0,
        }

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """重置环境到初始状态"""
        super().reset(seed=seed)

        # 处理排程参数
        if options and 'schedule' in options:
            self.schedule = options['schedule']
            self.t1 = self.schedule['t1']
            self.t2 = self.schedule['t2']
            self.rho2 = self.schedule['rho2']
            self.A1_A2 = self.schedule['A1_A2']

            rng = np.random.default_rng(seed if seed is not None else 42)
            self.batch_manager = BatchManager(
                self.schedule,
                self.container_params,
                self.crop_params,
                rng,
                self.config.get('steady_state_params', None),
                self.reward_params
            )

            A_total = self.container_params.get('c_total_plant_area', 40.0)
            self.A1 = A_total / (1 + self.A1_A2)
            self.A2 = A_total - self.A1
            self.container_params['A1'] = self.A1
            self.container_params['A2'] = self.A2

            self.episode_length = self.t2 * 24

        # 处理外部环境
        # 从container_params读取标准条件作为默认值
        cp = self.container_params
        if options and 'external' in options:
            self.external = np.array(options['external'])
        else:
            # 外部环境从container_params读取，无则用合理默认值
            T_out = cp.get('ext_temp_summer', cp.get('ext_temp_winter', 15.0))
            RH_out = cp.get('ext_rh_summer', cp.get('ext_rh_winter', 0.7))
            C_out_ppm = cp.get('ext_co2', 400.0)
            C_out = co2_ppm_to_density(C_out_ppm)
            self.external = np.array([T_out, RH_out, C_out])

        # 处理电价
        if options and 'elec_price' in options:
            self.elec_price = options['elec_price']
        else:
            # 电价从equipment_params读取
            self.elec_price = self.equipment_params.get('p_elec_base', 0.6)

        # 初始化环境状态（从container_params读取标准条件）
        T_target = cp.get('T_standard', 22.0)   # °C
        RH_target = cp.get('RH_standard', 0.75)  # [-]
        C_target_ppm = cp.get('C_standard_ppm', 1000.0)
        C_target = co2_ppm_to_density(C_target_ppm)
        self.state = np.array([C_target, T_target, RH_target])

        # 初始化动作（从container_params读取默认值）
        self.prev_action = np.array([
            cp.get('default_I1', 200.0),
            cp.get('default_I2', 200.0),
            0.0,
            0.0,
            self.equipment_params.get('c_vent_fan_cap', 0.5) * 0.5,
            self.equipment_params.get('c_dehum_cap', 0.002) * 0.5,
        ], dtype=np.float32)

        # 重置时间
        self.time_step = 0
        self.hour_of_day = np.random.randint(0, 24)
        self.day_of_period = 0
        self.dark_period_hours = 0

        # 重置统计量
        self.episode_reward = 0.0
        self.total_revenue = 0.0
        self.total_cost = 0.0
        self.total_labor_cost = 0.0

        # 重置 biomass 追踪
        self.prev_seedling_M_g = 0.0
        self.prev_transplant_M_g = 0.0

        # 初始化负荷
        self.total_E = 0.0
        self.total_P = 0.0

        # 更新批次状态（使用标准条件获取初始负荷）
        # 光强和温度从container_params读取
        I_init = cp.get('I_standard', 200.0)
        batch_info = self.batch_manager.update(
            self.dt, I_init, I_init, T_target, C_target, RH_target
        )
        self.total_E = batch_info['total_E_rate']
        self.total_P = batch_info['total_P_rate']
        # 初始化 prev_biomass = 当前 biomass（第一步奖励的 delta 起点）
        self.prev_seedling_M_g = batch_info['M_seedling']
        self.prev_transplant_M_g = batch_info['M_transplant']

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """执行一步仿真"""
        # 动作裁剪（使用配置的动作空间边界）
        al, ah = self._get_action_bounds_from_config()
        action = np.clip(action, al, ah).astype(np.float32)

        I1, I2, Q_HVAC, u_CO2, V_vent, m_dehum = action
        T_current = self.state[1]
        C_current = self.state[0]
        RH_current = self.state[2]

        # u_CO2 单位是 g/m²/h → kg/m²/s (密度)
        u_CO2_kg = u_CO2 * 1000.0 / 3600.0

        # 保存上步 biomass（用于计算奖励中的 delta）
        prev_seedling_M_g = self.prev_seedling_M_g
        prev_transplant_M_g = self.prev_transplant_M_g

        # 更新批次状态（传入RH）
        batch_info = self.batch_manager.update(
            self.dt, I1, I2, T_current, C_current, RH_current
        )

        # batch_manager.update() 返回的总速率 [kg/s]
        # = sum(batch_i的phi_transp * batch_i的面积)
        total_E = batch_info['total_E_rate']  # [kg water/s]
        total_P = batch_info['total_P_rate']  # [kg CO2/s]

        # 环境动力学更新
        actions_for_env = np.array([I1, I2, Q_HVAC, u_CO2_kg, V_vent, m_dehum])
        next_state, status = simulate_environment_step(
            self.state,
            actions_for_env,
            self.external,
            total_E,
            total_P,
            self.container_params,
            dt=self.dt
        )

        if status != 0:
            pass

        self.state = next_state

        # 计算奖励
        reward, cost_info = self._compute_reward(
            action, batch_info, prev_seedling_M_g, prev_transplant_M_g
        )

        # 更新 prev_biomass
        self.prev_seedling_M_g = batch_info['M_seedling']
        self.prev_transplant_M_g = batch_info['M_transplant']

        # 更新统计
        self.episode_reward += reward
        self.total_cost += cost_info['cost_electric'] + cost_info.get('cost_CO2', 0.0)
        self.total_labor_cost += cost_info.get('labor_cost', 0.0)

        self.prev_action = action.copy()

        self.time_step += 1
        self.total_steps += 1
        self.hour_of_day = (self.hour_of_day + 1) % 24
        if self.hour_of_day == 0:
            self.day_of_period += 1

        # 更新暗期追踪
        photoperiod_config = self.config.get('photoperiod', {})
        dark_start = photoperiod_config.get('dark_hour_start', 0)
        dark_end = photoperiod_config.get('dark_hour_end', 8)
        if dark_start <= self.hour_of_day < dark_end:
            self.dark_period_hours += 1
        else:
            self.dark_period_hours = 0

        terminated = self.time_step >= self.episode_length
        truncated = False

        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """构建28维观测向量"""
        C_density = self.state[0]
        T = self.state[1]
        RH = self.state[2]

        C_ppm = co2_density_to_ppm(C_density, T)

        T_out = self.external[0]
        RH_out = self.external[1]
        C_out_ppm = co2_density_to_ppm(self.external[2], T_out)

        # 环境观测
        obs_env = np.array([
            T, RH * 100, C_ppm,
            T_out, RH_out * 100, C_out_ppm,
            self.elec_price
        ], dtype=np.float32)

        # 作物集总特征
        lumped = self.batch_manager._extract_lumped_features()

        # 作物观测
        obs_crop = np.array([
            lumped['lai_total'],
            lumped['M_seedling'],
            lumped['M_transplant'],
            lumped['days_left'],
            lumped['M_oldest'],
            lumped['lai_seedling'],
            lumped['lai_transplant'],
        ], dtype=np.float32)

        # 上步动作
        obs_action = self.prev_action.astype(np.float32)

        # 时间
        obs_time = np.array([
            self.hour_of_day / 24.0,
            self.day_of_period / max(1, self.t2)
        ], dtype=np.float32)

        # 上下文
        obs_context = np.array([
            self.t1 / 30.0,
            self.t2 / 30.0,
            self.rho2 / 100.0,
            self.A1_A2 / 10.0,
        ], dtype=np.float32)

        # 负荷统计
        obs_load = np.array([
            self.total_E / 10.0,
            self.total_P / 10.0,
        ], dtype=np.float32)

        obs = np.concatenate([
            obs_env, obs_crop, obs_action,
            obs_time, obs_context, obs_load
        ])

        return obs

    def _compute_reward(
        self,
        action: np.ndarray,
        batch_info: Dict[str, Any],
        prev_seedling_M_g: float,
        prev_transplant_M_g: float,
    ) -> Tuple[float, Dict[str, float]]:
        """
        计算奖励函数（基于 biomass-delta + 事件价值）。

        r_t = r_t^growth + r_t^event + r_t^cost + r_t^penalty

        与旧版本的根本区别：
        - 旧：r_t^growth = alpha * standing_crop（绝对 biomass，重复收取库存价值）
        - 新：r_t^growth = alpha * ΔM（biomass 增量，物理正确）
        - 采收经济价值在采收时已通过 ΔM 体现在生长收益中，无需额外奖励
        - 不达标采收（单株干重 < harvest_min_dry_mass）施加强惩罚
        """
        rp = self.reward_params
        ep = self.equipment_params

        I1, I2, Q_HVAC, u_CO2, V_vent, m_dehum = action

        # ========== 生长收益（biomass 增量）==========
        # batch_manager.update() 后，current biomass 已在 batch_info 中
        curr_seedling_M_g = batch_info['M_seedling']
        curr_transplant_M_g = batch_info['M_transplant']

        d_seedling_g = curr_seedling_M_g - prev_seedling_M_g
        d_transplant_g = curr_transplant_M_g - prev_transplant_M_g

        alpha_growth = rp.get('alpha_growth', 0.25)
        discount = rp.get('seedling_growth_discount', 0.5)
        growth_reward = alpha_growth * (
            d_transplant_g + d_seedling_g * discount
        )

        # ========== 能耗成本 ==========
        u_CO2_kg = u_CO2 * 1000.0 / 3600.0

        actions_scaled = np.array([I1, I2, Q_HVAC, u_CO2_kg, V_vent, m_dehum])
        power_dict = calculate_total_power(
            actions_scaled, self.A1, self.A2, self.equipment_params
        )

        dt_hours = self.dt / 3600.0

        E_total = (power_dict['P_led_total'] + power_dict['P_hvac_total'] +
                   power_dict['P_vent'] + power_dict['P_dehum']) * dt_hours / 1000
        cost_electric = E_total * self.elec_price

        A_total = self.A1 + self.A2
        total_CO2_kg = u_CO2 * A_total * dt_hours / 1000.0
        p_CO2 = ep.get('p_CO2', 0.5)
        cost_CO2 = total_CO2_kg * p_CO2

        cost_info = {
            'cost_electric': cost_electric,
            'cost_CO2': cost_CO2,
            'labor_cost': 0.0,
            'growth_reward': growth_reward,
            'd_seedling_g': d_seedling_g,
            'd_transplant_g': d_transplant_g,
        }

        # ========== 约束惩罚 ==========
        penalty = 0.0

        T = self.state[1]
        if T < rp.get('temp_hard_min', 18.0) or T > rp.get('temp_hard_max', 28.0):
            penalty += rp.get('temp_violation_penalty', -10.0)

        if self.dark_period_hours > 0 and self.dark_period_hours < rp.get('min_dark_period_hours', 4.0):
            penalty += rp.get('dark_period_penalty', -10.0)

        RH = self.state[2] * 100
        if RH < rp.get('rh_soft_min', 60.0):
            penalty += rp.get('rh_deviation_penalty', -0.1) * (rp['rh_soft_min'] - RH)
        elif RH > rp.get('rh_soft_max', 80.0):
            penalty += rp.get('rh_deviation_penalty', -0.1) * (RH - rp['rh_soft_max'])

        I1_prev = self.prev_action[0] if self.prev_action is not None else I1
        I2_prev = self.prev_action[1] if self.prev_action is not None else I2
        light_change = (I1 - I1_prev)**2 + (I2 - I2_prev)**2
        penalty -= rp.get('light_change_penalty_coef', 0.001) * light_change

        # ========== 采收质量惩罚 ==========
        harvest_fail = batch_info.get('harvest_fail', False)
        if harvest_fail:
            penalty += rp.get('harvest_fail_penalty', -200.0)

        # ========== 总奖励 ==========
        reward = (growth_reward - cost_electric - cost_CO2 + penalty)
        reward /= rp.get('reward_scale_factor', 100.0)

        return float(reward), cost_info

    def _get_info(self) -> Dict[str, Any]:
        """获取附加信息"""
        lumped = self.batch_manager._extract_lumped_features()

        info = {
            'time_step': self.time_step,
            'hour_of_day': self.hour_of_day,
            'day_of_period': self.day_of_period,
            'T': self.state[1],
            'RH': self.state[2] * 100,
            'C_ppm': co2_density_to_ppm(self.state[0], self.state[1]),
            'total_revenue': self.total_revenue,
            'total_cost': self.total_cost,
            'episode_reward': self.episode_reward,
            'lai_total': lumped['lai_total'],
            'M_total': lumped['M_seedling'] + lumped['M_transplant'],
            'batch_summary': self.batch_manager.get_state_summary(),
        }

        return info

    def render(self):
        """渲染环境（当前不支持）"""
        pass

    def close(self):
        """关闭环境"""
        pass
