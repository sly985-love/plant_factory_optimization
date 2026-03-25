# -*- coding: utf-8 -*-
"""
集装箱/植物工厂环境动力学模型

实现集装箱环境（温度、湿度、CO2）的集总动态模型。

模型基于质量平衡和能量平衡方程：
1. CO2动态: dC/dt = (注入 - 光合吸收 + 通风交换) / 体积
2. 温度动态: dT/dt = (HVAC热交换 + LED热辐射 + 墙体传热 + 通风热交换 - 蒸腾潜热) / 热容
3. 湿度动态: dH/dt = (蒸腾 - 通风除湿 - 除湿机) / 体积

【重要】所有动态方程中的热流/质量流均以整个种植区（A1+A2）的总量为基准，
单位均为 [W] 或 [kg/s]，不除以面积。
这与PFAL-DRL PFALEnv.ode() 的处理方式一致。

来源: PFAL-DRL PFALEnv.ode(), 论文方法部分 2.2.3

作者: Plant Factory Optimization Team
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Tuple, Optional


def environment_dynamics(
    t: float,
    state: np.ndarray,
    actions: np.ndarray,
    external: np.ndarray,
    total_E: float,
    total_P: float,
    container_params: dict,
    batch_manager=None
) -> np.ndarray:
    """
    环境动力学微分方程

    【重要】参数说明：
    - total_E: 整个种植区的总蒸腾速率 [kg water/s]（不是密度！）
    - total_P: 整个种植区的总光合速率 [kg CO2/s]（不是密度！）
    - A1, A2: 育苗区和定植区面积 [m²]，由container_params传入
    - 所有动态方程以 [W] 或 [kg/s] 为基准，不除以面积

    这与PFAL-DRL一致：PFALEnv.ode() 中所有热流均为 W/m²，
    但在乘以 c_grow_area 后变为总 W。

    参数:
        t: 当前时间 [秒] (unused, 但solve_ivp需要)
        state: 环境状态 [C, T, RH]
            C: CO2浓度 [kg/m³]
            T: 温度 [°C]
            RH: 相对湿度 [-]
        actions: 控制动作 [I1, I2, Q_HVAC, u_CO2, V_vent, m_dehum]
            I1: 育苗区光强 [μmol/m²/s] (由I_in_umol参数指定单位)
            I2: 定植区光强 [μmol/m²/s]
            Q_HVAC: HVAC热功率密度 [W/m²] (正=加热, 负=制冷)
            u_CO2: CO2注入速率密度 [kg/m²/s]
            V_vent: 通风率 [m³/m²/s]
            m_dehum: 除湿速率密度 [kg/m²/s]
        external: 外部环境 [T_out, RH_out, C_out]
            T_out: 外部温度 [°C]
            RH_out: 外部相对湿度 [-]
            C_out: 外部CO2浓度 [kg/m³]
        total_E: 总蒸腾速率 [kg water/s] (整个种植区)
        total_P: 总光合速率 [kg CO2/s] (整个种植区)
        container_params: 集装箱参数字典
        batch_manager: 批次管理器 (可选, 用于计算实时负荷)

    返回:
        d_state/dt: 状态导数 [dC/dt, dT/dt, dRH/dt]

    单位说明:
        - CO2动态: dC/dt [kg/m³/s]
        - 温度动态: dT/dt [°C/s]
        - 湿度动态: dH/dt [kg/m³/s]
    """
    # 解包状态
    C = state[0]  # CO2浓度 [kg/m³]
    T = state[1]  # 温度 [°C]
    RH = state[2]  # 相对湿度 [-]

    # 解包动作
    I1, I2, Q_HVAC, u_CO2, V_vent, m_dehum = actions

    # 解包外部环境
    T_out = external[0]
    RH_out = external[1]
    C_out = external[2]

    # 提取参数
    c_volume = container_params.get('c_volume', 91.5)  # [m³]
    A1 = container_params.get('A1', 20.0)  # 育苗区面积 [m²]
    A2 = container_params.get('A2', 20.0)  # 定植区面积 [m²]
    A_total = A1 + A2  # 总种植面积 [m²]
    c_cap_q = container_params.get('c_cap_q', 30000.0)  # [J/m²/°C]
    c_cap_q_v = container_params.get('c_cap_q_v', 1290.0)  # [J/m³/°C]
    c_surface_area = container_params.get('c_surface_area', 143.2)  # [m²]
    c_U = container_params.get('c_U', 0.3)  # [W/m²/°C]
    c_lat_water = container_params.get('c_lat_water', 2256.4)  # [kJ/kg]
    c_led_eff = container_params.get('c_led_eff', 0.52)  # [-]
    c_optical_eff = container_params.get('c_optical_eff', 2.5)  # [μmol/J]
    I_in_umol = container_params.get('I_in_umol', True)  # 光强单位标志

    # 事件函数阈值（从参数获取，不再硬编码）
    event_temp_hi = container_params.get('event_temp_hi', 50.0)  # 温度上限 [°C]
    event_temp_lo = container_params.get('event_temp_lo', -5.0)  # 温度下限 [°C]

    # 计算饱和水汽压 (使用PFAL-DRL版本)
    xH_sat = calculate_saturation_vapor_pressure(T, container_params)
    xH = RH * xH_sat  # 实际水汽压 [kg/m³]

    # 计算外部饱和水汽压
    xH_sat_out = calculate_saturation_vapor_pressure(T_out, container_params)
    xH_out = RH_out * xH_sat_out

    # ========== CO2动态 ==========
    # total_P 是总光合速率 [kg CO2/s]
    # 转换为密度: phi_phot_c = total_P / A_total [kg CO2/m²/s]
    if A_total > 0:
        phi_phot_c = total_P / A_total
    else:
        phi_phot_c = 0.0
    phi_vent_c = V_vent * (C - C_out)  # CO2通风交换密度 [kg/m²/s]

    # dC/dt = (u_CO2 - phi_phot_c - phi_vent_c) * A_total / V
    dC_dt = (u_CO2 - phi_phot_c - phi_vent_c) * A_total / c_volume

    # ========== LED热辐射 ==========
    # 【重要】I1和I2可能来自不同面积区域，分别计算
    if I_in_umol:
        # μmol/m²/s → W/m² (PAR): I_W = I / c_optical_eff
        # 电功率 = PAR功率 / LED效率 = (I / c_optical_eff) / c_led_eff
        # 热功率 = 电功率 * (1 - c_led_eff) = I / c_optical_eff * (1 - c_led_eff)
        I1_W = I1 / c_optical_eff  # 育苗区PAR [W/m²]
        I2_W = I2 / c_optical_eff  # 定植区PAR [W/m²]
    else:
        I1_W = I1  # 已经是W/m² (PAR)
        I2_W = I2

    # LED热辐射 [W]:
    # 每区热 = I_W * (1 - η_led) * A_area
    Q_led1 = I1_W * (1 - c_led_eff) * A1  # 育苗区LED热 [W]
    Q_led2 = I2_W * (1 - c_led_eff) * A2  # 定植区LED热 [W]
    Q_led = Q_led1 + Q_led2  # 总LED热 [W]

    # ========== 墙体传热 ==========
    # Q_wall = U * (表面积/种植面积) * ΔT * 种植面积
    #         = U * 表面积 * ΔT  [W]
    Q_wall = c_U * c_surface_area * (T_out - T)  # [W]

    # ========== 通风热交换 ==========
    # Q_vent = ρ * c_p * V_vent * A_total * ΔT [W]
    # 简化: 使用 c_cap_q_v (体积热容) * V_vent * A_total * ΔT
    Q_vent = c_cap_q_v * V_vent * A_total * (T_out - T)  # [W]

    # ========== 蒸腾潜热 ==========
    # total_E 是总蒸腾 [kg water/s]
    # Q_transp = total_E * λ * 1000 [W] (kJ → J)
    Q_transp = total_E * c_lat_water * 1000.0  # [W]

    # ========== HVAC热交换 ==========
    # Q_HVAC 是密度 [W/m²]，乘以面积得到总量 [W]
    Q_HVAC_total = Q_HVAC * A_total  # [W]

    # ========== 温度变化率 ==========
    # 【重要】分母用 c_cap_q * A_total (J/°C)，不是 c_cap_q (J/m²/°C)
    # 这样所有热流项都用 [W]，保持一致性
    # dT/dt [°C/s] = Q [W] / (c_cap_q [J/m²/°C] * A [m²]) = Q [W] / C [J/°C]
    dT_dt = (Q_HVAC_total + Q_led - Q_wall - Q_vent - Q_transp) / (c_cap_q * A_total)

    # ========== 湿度动态 ==========
    # xH [kg/m³] 的变化
    # 通风水汽交换: phi_vent_h [kg/m³/s] = V_vent * (xH - xH_out)
    # 除湿: m_dehum_total [kg/s] = m_dehum * A_total
    # dH/dt [kg/m³/s] = (total_E - phi_vent_h*A_total - m_dehum_total) / V
    # 简化: total_E - V_vent*(xH-xH_out)*A_total - m_dehum*A_total 已是 kg/s
    phi_vent_h = V_vent * (xH - xH_out) * A_total  # 通风水汽交换 [kg/s]
    m_dehum_total = m_dehum * A_total  # 总除湿速率 [kg/s]

    dH_dt = (total_E - phi_vent_h - m_dehum_total) / c_volume

    return np.array([dC_dt, dT_dt, dH_dt])


def calculate_saturation_vapor_pressure(T: float, params: dict) -> float:
    """
    计算饱和水汽压 (Antoine方程, 与PFAL-DRL PFALEnv.xH_sat一致)

    参数:
        T: 温度 [°C]
        params: 参数字典
            - c_v_0: 水汽压常数系数 [-], 默认0.85
            - c_v_1: Antoine系数 [Pa]
            - c_v_2: Antoine系数 [-]
            - c_v_3: Antoine系数 [°C]
            - mw_water: 水分子量 [kg/kmol]
            - c_R: 通用气体常数 [J/K/kmol]
            - c_T_abs: 绝对温度偏移 [K]

    返回:
        xH_sat: 饱和水汽压 [kg/m³]
    """
    c_v_0 = params.get('c_v_0', 0.85)
    c_v_1 = params.get('c_v_1', 611.0)
    c_v_2 = params.get('c_v_2', 17.4)
    c_v_3 = params.get('c_v_3', 239.0)
    mw_water = params.get('mw_water', 18.0)
    c_R = params.get('c_R', 8314.0)
    c_T_abs = params.get('c_T_abs', 273.0)

    # PFAL-DRL版本 (包含c_v_0系数)
    xH_sat = (
        (c_v_0 * c_v_1 * mw_water / (c_R * (T + c_T_abs))) *
        np.exp(c_v_2 * T / (T + c_v_3))
    )

    return xH_sat


def simulate_environment_step(
    state_current: np.ndarray,
    actions: np.ndarray,
    external: np.ndarray,
    total_E: float,
    total_P: float,
    container_params: dict,
    dt: float = 3600.0,
    method: str = 'Radau'
) -> Tuple[np.ndarray, int]:
    """
    单步环境仿真

    使用scipy.integrate.solve_ivp进行数值积分。

    参数:
        state_current: 当前状态 [C, T, RH]
        actions: 控制动作
        external: 外部环境
        total_E: 总蒸腾速率 [kg water/s] (整个种植区)
        total_P: 总光合速率 [kg CO2/s] (整个种植区)
        container_params: 集装箱参数
        dt: 积分步长 [秒]
        method: 积分方法 ('Radau', 'RK45', 'RK4')

    返回:
        state_next: 下一时刻状态
        status: 积分状态 (0=成功, 其他=失败)
    """
    # 定义ODE函数（包装以匹配solve_ivp接口）
    def ode_func(t, y):
        return environment_dynamics(
            t, y, actions, external, total_E, total_P, container_params
        )

    # 【重要】事件函数阈值从参数获取，不再硬编码
    event_temp_hi = container_params.get('event_temp_hi', 50.0)
    event_temp_lo = container_params.get('event_temp_lo', -5.0)

    def event_hi(t, y):
        return y[1] - event_temp_hi  # 温度上限
    event_hi.terminal = True

    def event_lo(t, y):
        return event_temp_lo - y[1]  # 温度下限
    event_lo.terminal = True

    try:
        sol = solve_ivp(
            ode_func,
            [0, dt],
            state_current,
            method=method,
            events=[event_hi, event_lo],
            vectorized=False,
            rtol=1e-6,
            atol=1e-9
        )

        if sol.status == 0:
            return sol.y[:, -1], 0
        else:
            # 事件触发或积分失败
            return state_current, sol.status

    except Exception as e:
        return state_current, -1


def solve_environment_steady_state(
    external: np.ndarray,
    container_params: dict,
    actions_guess: Optional[np.ndarray] = None,
    T_target: float = 22.0,
    RH_target: float = 0.7,
    default_I1: float = 200.0,
    default_I2: float = 200.0,
    default_Q_HVAC: float = 0.0,
    default_u_CO2: float = 0.0,
    default_V_vent: float = 0.01,
    default_m_dehum: float = 1e-5
) -> np.ndarray:
    """
    求解环境稳态

    给定外部条件和控制动作，求解环境平衡状态。
    所有默认值从container_params获取，container_params中无则用函数参数。

    参数:
        external: 外部环境 [T_out, RH_out, C_out]
        container_params: 集装箱参数
        actions_guess: 动作猜测值 [I1, I2, Q_HVAC, u_CO2, V_vent, m_dehum]
        T_target: 目标温度 [°C]
        RH_target: 目标相对湿度 [-]
        default_*: 各动作的默认值（当container_params中无配置时使用）

    返回:
        state_ss: 稳态环境状态 [C, T, RH]
    """
    if actions_guess is None:
        # 从container_params获取默认值，无则用函数参数
        actions_guess = np.array([
            container_params.get('default_I1', default_I1),
            container_params.get('default_I2', default_I2),
            container_params.get('default_Q_HVAC', default_Q_HVAC),
            container_params.get('default_u_CO2', default_u_CO2),
            container_params.get('default_V_vent', default_V_vent),
            container_params.get('default_m_dehum', default_m_dehum),
        ])

    # 默认总负荷
    total_E = container_params.get('default_total_E', 1e-4)  # [kg water/s]
    total_P = container_params.get('default_total_P', 1e-6)  # [kg CO2/s]

    # 初始猜测
    C_guess = external[2] if len(external) > 2 else container_params.get('default_C', 1e-3)
    state_guess = np.array([C_guess, T_target, RH_target])

    # 简化稳态求解（忽略动态，取准稳态）
    # 在实际中，这里应该用数值方法求解代数方程
    # 这里使用简化方法

    C_ss = external[2] if len(external) > 2 else container_params.get('default_C', 1e-3)
    T_ss = T_target
    RH_ss = RH_target

    return np.array([C_ss, T_ss, RH_ss])


def relative_humidity_to_absolute(T: float, RH: float, params: dict) -> float:
    """
    相对湿度转换为绝对湿度

    参数:
        T: 温度 [°C]
        RH: 相对湿度 [-]
        params: 参数字典

    返回:
        xH: 绝对湿度 [kg/m³]
    """
    xH_sat = calculate_saturation_vapor_pressure(T, params)
    return RH * xH_sat


def absolute_humidity_to_relative(T: float, xH: float, params: dict) -> float:
    """
    绝对湿度转换为相对湿度

    参数:
        T: 温度 [°C]
        xH: 绝对湿度 [kg/m³]
        params: 参数字典

    返回:
        RH: 相对湿度 [-]
    """
    xH_sat = calculate_saturation_vapor_pressure(T, params)
    if xH_sat > 0:
        return np.clip(xH / xH_sat, 0, 1)
    else:
        return 0.5


def co2_ppm_to_density(ppm: float, T: float = 22.0) -> float:
    """
    CO2浓度 ppm 转换为 kg/m³

    使用理想气体方程。

    参数:
        ppm: CO2浓度 [ppm]
        T: 温度 [°C] (默认22°C)

    返回:
        C: CO2浓度 [kg/m³]
    """
    # 物理常数 (通用气体常数，与container_params.yaml中的c_R=8314.0 J/K/kmol等价，R=8.314 J/mol/K)
    # 这些是通用物理常数，不需要也不应该在配置文件中修改
    P = 101325.0     # 标准大气压 [Pa]
    M_CO2 = 44.01e-3 # CO2 分子量 [kg/mol]
    R = 8.314        # 通用气体常数 [J/mol/K]

    # 转换公式验证: ppm = C * R*(T+273.15) / (M_CO2 * P) * 1e6
    # 量纲: (kg/m³) * (J/mol/K) * (K) / (kg/mol) / (Pa) = (J/m³) / (N/m²) = (N·m/m³) / (N/m²) = m⁰ = [-]
    # 乘以1e6得到ppm值

    # 转换
    C = ppm * 1e-6 * M_CO2 * P / (R * (T + 273.15))

    return C


def co2_density_to_ppm(C: float, T: float = 22.0) -> float:
    """
    CO2浓度 kg/m³ 转换为 ppm

    参数:
        C: CO2浓度 [kg/m³]
        T: 温度 [°C] (默认22°C)

    返回:
        ppm: CO2浓度 [ppm]
    """
    P = 101325.0
    M_CO2 = 44.01e-3
    R = 8.314

    ppm = C * R * (T + 273.15) / (M_CO2 * P) * 1e6

    return ppm
