# -*- coding: utf-8 -*-
"""
环境工具模块

提供环境相关的辅助函数，包括：
1. 配置加载
2. 归一化/反归一化
3. 动作空间定义

【重要】本模块中的函数应尽可能从配置文件读取参数，
避免在代码中硬编码数值常量。
"""

import numpy as np
import yaml
import os
from typing import Dict, Any, Optional


def load_all_configs(config_dir: str) -> Dict[str, Any]:
    """
    加载所有配置文件

    参数:
        config_dir: 配置文件目录路径

    返回:
        包含所有配置的字典
    """
    config = {}

    config_files = [
        'crop_params.yaml',
        'container_params.yaml',
        'equipment_params.yaml',
        'reward_params.yaml',
        'schedule_params.yaml',
        'rl_params.yaml',
        'bo_params.yaml',
        'experiment_params.yaml',
        'mpc_params.yaml',
        'controller_params.yaml',
    ]

    for fname in config_files:
        fpath = os.path.join(config_dir, fname)
        if os.path.exists(fpath):
            with open(fpath, 'r', encoding='utf-8') as f:
                key = fname.replace('.yaml', '')
                config[key] = yaml.safe_load(f)

    return config


def create_default_schedule(
    schedule_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    创建默认排程参数

    所有值从 schedule_params 字典获取，无则用合理默认值。

    参数:
        schedule_params: schedule_params.yaml 加载的字典

    返回:
        排程参数字典
    """
    if schedule_params is None:
        schedule_params = {}

    return {
        't1': int(schedule_params.get('t1_min', 10) +
                   (schedule_params.get('t1_max', 18) - schedule_params.get('t1_min', 10)) * 0.5),
        't2': int(schedule_params.get('t2_min', 18) +
                   (schedule_params.get('t2_max', 26) - schedule_params.get('t2_min', 18)) * 0.5),
        'rho2': (schedule_params.get('rho2_min', 20.0) +
                 (schedule_params.get('rho2_max', 80.0) - schedule_params.get('rho2_min', 20.0)) * 0.5),
        'A1_A2': (schedule_params.get('A1_A2_min', 0.1) +
                  (schedule_params.get('A1_A2_max', 5.0) - schedule_params.get('A1_A2_min', 0.1)) * 0.5),
    }


def normalize_observation(
    obs: np.ndarray,
    obs_low: np.ndarray,
    obs_high: np.ndarray,
    method: str = 'linear'
) -> np.ndarray:
    """
    归一化观测向量

    参数:
        obs: 原始观测
        obs_low: 观测下界
        obs_high: 观测上界
        method: 归一化方法 ('linear', 'minmax')

    返回:
        归一化后的观测
    """
    if method == 'linear':
        # 线性缩放到 [-1, 1] 或 [0, 1]
        return 2.0 * (obs - obs_low) / (obs_high - obs_low + 1e-8) - 1.0
    elif method == 'minmax':
        return (obs - obs_low) / (obs_high - obs_low + 1e-8)
    else:
        return obs


def denormalize_action(
    action_norm: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray
) -> np.ndarray:
    """
    反归一化动作

    参数:
        action_norm: 归一化动作 [-1, 1]
        action_low: 动作下界
        action_high: 动作上界

    返回:
        物理动作
    """
    # 假设动作空间是 [-1, 1]
    return (action_norm + 1.0) / 2.0 * (action_high - action_low) + action_low


def get_action_bounds(
    equipment_params: Optional[Dict[str, Any]] = None
) -> tuple:
    """
    获取动作空间边界（物理单位）

    所有值从 equipment_params 字典获取，无则用合理默认值。

    参数:
        equipment_params: equipment_params.yaml 加载的字典

    返回:
        (action_low, action_high) 两个numpy数组
    """
    if equipment_params is None:
        equipment_params = {}

    I_max = equipment_params.get('I_max', 600.0)  # μmol/m²/s
    Q_HVAC_max = equipment_params.get('hvac_max_power_density', 212.0)  # W/m²
    Q_HVAC_min = equipment_params.get('hvac_min_power_density', -212.0)  # W/m²
    co2_supply_max = equipment_params.get('co2_supply_max', 0.5)  # g/m²/h
    vent_max = equipment_params.get('c_vent_fan_cap', 0.5)  # m³/m²/s
    dehum_max = equipment_params.get('c_dehum_cap', 0.002)  # kg/m²/s

    action_low = np.array([
        0.0,           # I1 [μmol/m²/s]
        0.0,           # I2 [μmol/m²/s]
        Q_HVAC_min,    # Q_HVAC [W/m²] (可加热/制冷)
        0.0,           # u_CO2 [g/m²/h]
        0.0,           # V_vent [m³/m²/s]
        0.0,           # m_dehum [kg/m²/s]
    ], dtype=np.float32)

    action_high = np.array([
        I_max,
        I_max,
        Q_HVAC_max,
        co2_supply_max,
        vent_max,
        dehum_max,
    ], dtype=np.float32)

    return action_low, action_high
