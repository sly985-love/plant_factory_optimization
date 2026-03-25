# -*- coding: utf-8 -*-
"""
规则控制器模块

基于农艺经验的固定控制策略。

设计:
- 固定光强（白天按配置值，夜间关）
- HVAC 按温度设定点比例控制
- CO2 维持目标浓度
- 通风、除湿按湿度和CO2阈值控制

【重要】所有控制参数从 controller_params.yaml 配置文件读取，
不再硬编码。默认值仅在配置缺失时使用。

来源: 论文方法部分 2.5 基线对比方法
"""

import numpy as np
import os
import yaml
from typing import Optional, Dict, Any
from .base_controller import BaseController


class RuleController(BaseController):
    """
    规则控制器

    基于固定规则的简单控制器，作为基线方法。
    所有参数从配置文件读取。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._load_controller_params()

    def _load_controller_params(self):
        """从配置文件加载控制器参数"""
        cfg = self.config
        if cfg is None:
            cfg = {}

        # 尝试加载 controller_params.yaml
        controller_yaml = cfg.get('controller_params_yaml', None)
        if controller_yaml is not None and os.path.exists(controller_yaml):
            with open(controller_yaml, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)
        else:
            yaml_config = {}

        rule_cfg = yaml_config.get('rule_controller', {})
        action_limits = yaml_config.get('action_limits', {})

        # ========== 光照参数 ==========
        self.light_intensity = rule_cfg.get('light_intensity', 200.0)  # μmol/m²/s
        self.light_on_hour = rule_cfg.get('light_on_hour', 8)
        self.light_off_hour = rule_cfg.get('light_off_hour', 24)

        # ========== 温度控制参数 ==========
        self.temp_setpoint = rule_cfg.get('temp_setpoint', 22.0)  # °C
        self.kp_temp = rule_cfg.get('kp_temp', 10.0)  # 温度P增益
        self.temp_deadband = rule_cfg.get('temp_deadband', 1.0)  # 死区

        # ========== CO2控制参数 ==========
        self.co2_setpoint = rule_cfg.get('co2_setpoint', 1000.0)  # ppm
        self.co2_low_threshold = rule_cfg.get('co2_low_threshold', 800.0)  # ppm
        self.co2_high_threshold = rule_cfg.get('co2_high_threshold', 1200.0)  # ppm
        self.co2_injection_rate = rule_cfg.get('co2_injection_rate', 0.2)  # g/m²/h

        # ========== 通风控制参数 ==========
        self.vent_rate = rule_cfg.get('vent_rate', 0.25)  # 中值通风率
        self.vent_high_threshold = rule_cfg.get('vent_high_threshold', 0.4)  # 高通风率
        self.vent_trigger_co2 = rule_cfg.get('vent_trigger_co2', 1200.0)  # ppm
        self.vent_trigger_temp = rule_cfg.get('vent_trigger_temp', 25.0)  # °C

        # ========== 除湿控制参数 ==========
        self.dehum_rate = rule_cfg.get('dehum_rate', 0.001)  # 中值
        self.dehum_high_rh = rule_cfg.get('dehum_high_rh', 0.80)
        self.dehum_mid_rh = rule_cfg.get('dehum_mid_rh', 0.75)
        self.dehum_max_rate = rule_cfg.get('dehum_max_rate', 0.002)

        # ========== 动作限制（来自action_limits或设备参数）==========
        ep = self.config.get('equipment_params', {}) if self.config else {}
        self.Q_HVAC_max = ep.get('hvac_max_power_density',
            action_limits.get('Q_HVAC_max', 212.0))
        self.Q_HVAC_min = ep.get('hvac_min_power_density',
            action_limits.get('Q_HVAC_min', -212.0))

        # PID控制增益（用于备用）
        self.co2_injection_rate = rule_cfg.get('co2_injection_rate', 0.2)  # g/m²/h
        self.temp_bias = 0.0  # 温度偏差累积

    def predict(self, obs: np.ndarray, context: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        根据观测预测动作

        参数:
            obs: 观测向量 [28维]
                索引0: 箱内温度 [°C]
                索引1: 箱内相对湿度 [%]
                索引2: 箱内CO2 [ppm]
                ...

        返回:
            action: [I1, I2, Q_HVAC, u_CO2, V_vent, m_dehum]
        """
        # 解包观测
        T = obs[0]  # 箱内温度 [°C]
        RH = obs[1] / 100.0  # 转为 [-]
        C_ppm = obs[2]  # CO2 [ppm]

        # ========== 光照控制 ==========
        # 固定光强（白天按配置值，夜间关）
        hour_norm = obs[12] if len(obs) > 12 else 0.5  # 小时归一化
        hour = int(hour_norm * 24)
        if self.light_on_hour <= hour < self.light_off_hour:  # 光期
            I1 = self.light_intensity
            I2 = self.light_intensity
        else:  # 暗期
            I1 = 0.0
            I2 = 0.0

        # ========== 温度控制 ==========
        # 比例控制
        temp_error = self.temp_setpoint - T
        Q_HVAC = np.clip(temp_error * self.kp_temp, self.Q_HVAC_min, self.Q_HVAC_max)

        # ========== CO2控制 ==========
        # 简单开关控制
        if C_ppm < self.co2_low_threshold:
            u_CO2 = self.co2_injection_rate
        elif C_ppm > self.co2_high_threshold:
            u_CO2 = 0.0
        else:
            u_CO2 = self.co2_injection_rate * 0.5

        # ========== 通风控制 ==========
        # CO2过高或温度过高时增加通风
        if C_ppm > self.vent_trigger_co2 or T > self.vent_trigger_temp:
            V_vent = self.vent_high_threshold
        else:
            V_vent = self.vent_rate

        # ========== 除湿控制 ==========
        # 湿度过高时除湿
        if RH > self.dehum_high_rh:
            m_dehum = self.dehum_max_rate
        elif RH > self.dehum_mid_rh:
            m_dehum = self.dehum_rate
        else:
            m_dehum = self.dehum_rate * 0.5

        action = np.array([I1, I2, Q_HVAC, u_CO2, V_vent, m_dehum], dtype=np.float32)

        return action


class PIDController(BaseController):
    """
    PID控制器（备用）

    简单的离散PID控制器。
    所有参数从配置文件读取。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._load_controller_params()

    def _load_controller_params(self):
        """从配置文件加载控制器参数"""
        cfg = self.config
        if cfg is None:
            cfg = {}

        controller_yaml = cfg.get('controller_params_yaml', None)
        if controller_yaml is not None and os.path.exists(controller_yaml):
            with open(controller_yaml, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)
        else:
            yaml_config = {}

        pid_cfg = yaml_config.get('pid_controller', {})
        rule_cfg = yaml_config.get('rule_controller', {})
        action_limits = yaml_config.get('action_limits', {})

        self.temp_setpoint = pid_cfg.get('temp_setpoint',
            rule_cfg.get('temp_setpoint', 22.0))
        self.kp = pid_cfg.get('kp', 20.0)
        self.ki = pid_cfg.get('ki', 0.5)
        self.kd = pid_cfg.get('kd', 5.0)

        # 动作限制
        ep = cfg.get('equipment_params', {}) if cfg else {}
        self.Q_HVAC_max = ep.get('hvac_max_power_density',
            action_limits.get('Q_HVAC_max', 212.0))
        self.Q_HVAC_min = ep.get('hvac_min_power_density',
            action_limits.get('Q_HVAC_min', -212.0))
        self.vent_rate = rule_cfg.get('vent_rate', 0.25)
        self.co2_injection_rate = rule_cfg.get('co2_injection_rate', 0.2)
        self.dehum_rate = rule_cfg.get('dehum_rate', 0.001)
        self.light_intensity = rule_cfg.get('light_intensity', 200.0)
        self.light_on_hour = rule_cfg.get('light_on_hour', 8)
        self.light_off_hour = rule_cfg.get('light_off_hour', 24)

        self.integral = 0.0
        self.prev_error = 0.0

    def predict(self, obs: np.ndarray, context: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """PID控制"""
        T = obs[0]

        error = self.temp_setpoint - T
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error

        Q_HVAC = self.kp * error + self.ki * self.integral + self.kd * derivative
        Q_HVAC = np.clip(Q_HVAC, self.Q_HVAC_min, self.Q_HVAC_max)

        # 光照
        hour_norm = obs[12] if len(obs) > 12 else 0.5
        hour = int(hour_norm * 24)
        I1 = self.light_intensity if self.light_on_hour <= hour < self.light_off_hour else 0.0
        I2 = I1

        action = np.array([I1, I2, Q_HVAC, self.co2_injection_rate,
                          self.vent_rate, self.dehum_rate], dtype=np.float32)
        return action

    def reset(self):
        """重置PID状态"""
        self.integral = 0.0
        self.prev_error = 0.0
