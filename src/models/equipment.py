# -*- coding: utf-8 -*-
"""
设备功率计算模块

实现各设备的功率和能耗计算，包括：
1. LED照明系统
2. HVAC系统（加热/制冷）
3. CO2注入系统
4. 通风系统
5. 除湿系统

所有能耗用于奖励函数和分项能耗统计。

来源: 论文方法部分 2.2.3, PFAL-DRL, RL-SMPC

作者: Plant Factory Optimization Team
"""

import numpy as np
from typing import Dict, Tuple, Optional


def calculate_led_power(
    I1: float,
    I2: float,
    A1: float,
    A2: float,
    equipment_params: Dict
) -> Tuple[float, float, float]:
    """
    计算LED照明系统功率

    LED电功率 = (I1/A1 + I2/A2) / η_opt
    分别计算两个区域。

    参数:
        I1: 育苗区光强 [μmol/m²/s]
        I2: 定植区光强 [μmol/m²/s]
        A1: 育苗区面积 [m²]
        A2: 定植区面积 [m²]
        equipment_params: 设备参数字典

    返回:
        P_led_total: 总LED电功率 [W]
        P_led1: 育苗区LED功率 [W]
        P_led2: 定植区LED功率 [W]
    """
    c_optical_eff = equipment_params.get('c_optical_eff', 2.5)
    c_led_eff = equipment_params.get('c_led_eff', 0.52)

    # μmol/m²/s → W/m² (PAR): I_W = I / c_optical_eff
    # 电功率 (PAR / η) = I / c_optical_eff / c_led_eff
    I1_W = I1 / c_optical_eff  # [W/m²]
    I2_W = I2 / c_optical_eff

    # 计算电功率 [W] = PAR功率密度 [W/m²] × 面积 [m²] / η_led
    P_led1 = I1_W * A1 / c_led_eff  # [W]
    P_led2 = I2_W * A2 / c_led_eff  # [W]
    P_led_total = P_led1 + P_led2

    return P_led_total, P_led1, P_led2


def calculate_hvac_power(
    Q_HVAC: float,
    A_total: float,
    equipment_params: Dict
) -> Tuple[float, float, float]:
    """
    计算HVAC系统功率（加热和制冷分开计算）

    HVAC功率 = |Q_HVAC| * A / COP

    参数:
        Q_HVAC: 加热/制冷功率密度 [W/m²]
                正值 = 加热，负值 = 制冷
        A_total: 总面积 [m²]
        equipment_params: 设备参数字典

    返回:
        P_hvac_total: 总HVAC电功率 [W]
        P_heating: 制热电功率 [W]
        P_cooling: 制冷电功率 [W]
    """
    c_COP = equipment_params.get('c_COP', 3.0)

    if Q_HVAC > 0:
        # 加热
        P_heating = Q_HVAC * A_total / c_COP  # [W]
        P_cooling = 0.0
    else:
        # 制冷
        P_heating = 0.0
        P_cooling = abs(Q_HVAC) * A_total / c_COP  # [W]

    P_hvac_total = P_heating + P_cooling

    return P_hvac_total, P_heating, P_cooling


def calculate_co2_power(
    u_CO2: float,
    A_total: float,
    equipment_params: Dict
) -> float:
    """
    计算CO2注入系统功率

    CO2注入不直接消耗电功率，但需要计入运行成本。
    这里返回等效电功率用于成本计算。

    参数:
        u_CO2: CO2注入速率密度 [kg/m²/s]
        A_total: 总面积 [m²]
        equipment_params: 设备参数字典

    返回:
        P_CO2: CO2等效电功率 [W] (用于成本计算)
    """
    # CO2注入的电功率与注入速率成正比
    # 这里简化处理，假设为线性关系
    # 实际中CO2成本单独计算

    # 返回0，因为CO2成本在奖励函数中单独处理
    return 0.0


def calculate_vent_power(
    V_vent: float,
    A_total: float,
    equipment_params: Dict
) -> float:
    """
    计算通风系统功率

    通风功率 = V_vent * A_total / fan_eff

    参数:
        V_vent: 通风率 [m³/m²/s]
        A_total: 总面积 [m²]
        equipment_params: 设备参数字典

    返回:
        P_vent: 通风电功率 [W]
    """
    fan_eff = equipment_params.get('fan_eff', 7.07)

    # 通风体积流量 [m³/s]
    V_total = V_vent * A_total

    # 通风电功率 [W] = 体积流量 / 风机效率
    P_vent = V_total / fan_eff  # [W]

    return P_vent


def calculate_dehum_power(
    m_dehum: float,
    A_total: float,
    equipment_params: Dict
) -> float:
    """
    计算除湿系统功率

    除湿功率 = (m_dehum * A_total) / dehum_eev

    参数:
        m_dehum: 除湿速率密度 [kg/m²/s]
        A_total: 总面积 [m²]
        equipment_params: 设备参数字典

    返回:
        P_dehum: 除湿电功率 [W]
    """
    c_dehum_eev = equipment_params.get('c_dehum_eev', 3.0)

    # 除湿质量流量 [kg/s]
    m_total = m_dehum * A_total

    # 除湿电功率 [W] = 质量流量 / 能效比 * 1000 (转换为 kW 再转 W)
    # c_dehum_eev 单位是 kg/kWh
    P_dehum = (m_total / c_dehum_eev) * 1000  # [W]

    return P_dehum


def calculate_total_power(
    actions: np.ndarray,
    A1: float,
    A2: float,
    equipment_params: Dict
) -> Dict[str, float]:
    """
    计算所有设备的总功率

    这是最常用的接口，一次计算所有分项能耗。

    参数:
        actions: 动作数组 [I1, I2, Q_HVAC, u_CO2, V_vent, m_dehum]
                I1, I2: [μmol/m²/s]
                Q_HVAC: [W/m²]
                u_CO2: [kg/m²/s]
                V_vent: [m³/m²/s]
                m_dehum: [kg/m²/s]
        A1: 育苗区面积 [m²]
        A2: 定植区面积 [m²]
        equipment_params: 设备参数字典

    返回:
        power_dict: 包含各分项功率的字典 [W]
            - P_led_total: 总LED功率
            - P_led1: 育苗区LED功率
            - P_led2: 定植区LED功率
            - P_hvac_total: 总HVAC功率
            - P_heating: 制热功率
            - P_cooling: 制冷功率
            - P_vent: 通风功率
            - P_dehum: 除湿功率
            - P_CO2: CO2等效功率
            - P_total: 总功率
    """
    I1, I2, Q_HVAC, u_CO2, V_vent, m_dehum = actions

    A_total = A1 + A2

    # LED功率
    P_led_total, P_led1, P_led2 = calculate_led_power(
        I1, I2, A1, A2, equipment_params
    )

    # HVAC功率（冷热分离）
    P_hvac_total, P_heating, P_cooling = calculate_hvac_power(
        Q_HVAC, A_total, equipment_params
    )

    # 通风功率
    P_vent = calculate_vent_power(V_vent, A_total, equipment_params)

    # 除湿功率
    P_dehum = calculate_dehum_power(m_dehum, A_total, equipment_params)

    # CO2等效功率
    P_CO2 = calculate_co2_power(u_CO2, A_total, equipment_params)

    # 总功率
    P_total = P_led_total + P_hvac_total + P_vent + P_dehum

    return {
        'P_led_total': P_led_total,
        'P_led1': P_led1,
        'P_led2': P_led2,
        'P_hvac_total': P_hvac_total,
        'P_heating': P_heating,
        'P_cooling': P_cooling,
        'P_vent': P_vent,
        'P_dehum': P_dehum,
        'P_CO2': P_CO2,
        'P_total': P_total,
    }


def calculate_energy_cost(
    power_dict: Dict[str, float],
    dt: float,
    elec_price: float,
    u_CO2_density: float = 0.0,
    A_total: float = 40.0,
    p_CO2: float = 0.5
) -> Dict[str, float]:
    """
    计算能源成本

    参数:
        power_dict: 各分项功率字典 [W]
        dt: 时间步长 [秒]
        elec_price: 电价 [元/kWh]
        u_CO2_density: CO2注入速率密度 [kg/m²/s]
        A_total: 总种植面积 [m²]
        p_CO2: CO2价格 [元/kg]

    返回:
        cost_dict: 包含各分项成本的字典 [元]
    """
    # 时间转换
    dt_hours = dt / 3600.0  # 秒 → 小时

    # 分项能耗 [kWh] = 功率 [W] * 时间 [h] / 1000
    E_led1 = power_dict.get('P_led1', 0) * dt_hours / 1000  # [kWh]
    E_led2 = power_dict.get('P_led2', 0) * dt_hours / 1000
    E_heating = power_dict.get('P_heating', 0) * dt_hours / 1000
    E_cooling = power_dict.get('P_cooling', 0) * dt_hours / 1000
    E_vent = power_dict.get('P_vent', 0) * dt_hours / 1000
    E_dehum = power_dict.get('P_dehum', 0) * dt_hours / 1000

    # 分项电费 [元]
    cost_led = (E_led1 + E_led2) * elec_price
    cost_hvac = (E_heating + E_cooling) * elec_price
    cost_vent = E_vent * elec_price
    cost_dehum = E_dehum * elec_price
    cost_electric = cost_led + cost_hvac + cost_vent + cost_dehum

    # CO2成本 [元] = 注入速率密度 * 面积 * dt * 单价
    total_CO2_kg = u_CO2_density * A_total * dt
    cost_CO2 = total_CO2_kg * p_CO2

    return {
        'E_led1_kWh': E_led1,
        'E_led2_kWh': E_led2,
        'E_heating_kWh': E_heating,
        'E_cooling_kWh': E_cooling,
        'E_vent_kWh': E_vent,
        'E_dehum_kWh': E_dehum,
        'cost_electric': cost_electric,
        'cost_led': cost_led,
        'cost_hvac': cost_hvac,
        'cost_vent': cost_vent,
        'cost_dehum': cost_dehum,
        'cost_CO2': cost_CO2,
    }


def calculate_power_with_bounds(
    actions: np.ndarray,
    A1: float,
    A2: float,
    equipment_params: Optional[Dict] = None
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    计算设备功率，并返回裁剪后的动作

    确保动作在设备物理限制内。所有限制值从equipment_params获取。

    参数:
        actions: 原始动作数组
        A1: 育苗区面积 [m²]
        A2: 定植区面积 [m²]
        equipment_params: 设备参数字典

    返回:
        power_dict: 各分项功率字典
        clipped_actions: 裁剪后的动作数组
    """
    if equipment_params is None:
        equipment_params = {}

    # 动作限制（从配置获取，无则用默认值）
    I_max = equipment_params.get('I_max', 600.0)
    Q_HVAC_max = equipment_params.get('hvac_max_power_density', 212.0)
    Q_HVAC_min = equipment_params.get('hvac_min_power_density', -212.0)
    co2_supply_max = equipment_params.get('co2_supply_max', 0.5)  # g/m²/h
    vent_max = equipment_params.get('c_vent_fan_cap', 0.5)  # m³/m²/s
    dehum_max = equipment_params.get('c_dehum_cap', 0.002)  # kg/m²/s

    # 裁剪
    clipped = np.copy(actions)
    clipped[0] = np.clip(clipped[0], 0, I_max)  # I1
    clipped[1] = np.clip(clipped[1], 0, I_max)  # I2
    clipped[2] = np.clip(clipped[2], Q_HVAC_min, Q_HVAC_max)  # Q_HVAC (从配置读取)
    clipped[3] = np.clip(clipped[3], 0, co2_supply_max)  # u_CO2
    clipped[4] = np.clip(clipped[4], 0, vent_max)  # V_vent
    clipped[5] = np.clip(clipped[5], 0, dehum_max)  # m_dehum

    # 计算功率
    power_dict = calculate_total_power(clipped, A1, A2, equipment_params)

    return power_dict, clipped
