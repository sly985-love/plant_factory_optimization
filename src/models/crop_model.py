# -*- coding: utf-8 -*-
"""
作物生长模型模块

基于 Van Henten (1994) 植物生长模型，实现光合作用、呼吸作用和蒸腾作用计算。
本模块对原始 Van Henten 模型进行了以下扩展：
1. 增加密度修正（Beer-Lambert光截获修正）
2. 支持多批次分区计算
3. 单位转换和接口适配

来源:
    - 作物参数: Van Henten 1994 "A greenhouse model with seasonal daylight"
    - 部分参数调整: PFAL-DRL 项目
    - 密度修正: 论文方法部分 2.2.2

作者: Plant Factory Optimization Team
"""

import numpy as np
from typing import Tuple, Optional


def calculate_saturation_vapor_pressure(T: float, params: dict) -> float:
    """
    计算饱和水汽压 (Antoine方程, 与PFAL-DRL PFALEnv.xH_sat一致)

    参数:
        T: 空气温度 [°C]
        params: 包含 Antoine 方程参数的字典
            - c_v_0: 水汽压常数系数 [-]
            - c_v_1: Antoine系数 [Pa]
            - c_v_2: Antoine系数 [-]
            - c_v_3: Antoine系数 [°C]
            - mw_water: 水分子量 [kg/kmol]
            - c_R: 通用气体常数 [J/K/kmol]
            - c_T_abs: 绝对温度偏移 [K]

    返回:
        xH_sat: 饱和水汽压 [kg/m³]

    公式来源: PFAL-DRL PFALEnv.xH_sat()
        xH_sat = (c_v_0 * c_v_1 * mw_water / (c_R * (T + c_T_abs)))
                  * exp(c_v_2 * T / (T + c_v_3))
    注意: 与原 crop_model.py 中旧公式的差异在于多了 c_v_0 系数(0.85)，
          该系数来自PFAL-DRL原始实现，此处采用PFAL-DRL版本以保持一致。
    """
    c_v_0 = params.get('c_v_0', 0.85)  # [-] PFAL-DRL水汽压系数
    c_v_1 = params['c_v_1']  # Pa
    c_v_2 = params['c_v_2']  # -
    c_v_3 = params['c_v_3']  # °C
    mw_water = params['mw_water']  # kg/kmol
    c_R = params['c_R']  # J/K/kmol
    c_T_abs = params['c_T_abs']  # K

    xH_sat = (
        (c_v_0 * c_v_1 * mw_water / (c_R * (T + c_T_abs))) *
        np.exp(c_v_2 * T / (T + c_v_3))
    )

    return xH_sat


def photosynthesis(
    xDn: float,
    xDs: float,
    I: float,
    T: float,
    C: float,
    rho: float,
    params: dict,
    I_in_umol: bool = True
) -> Tuple[float, float, float]:
    """
    计算光合作用速率（考虑密度修正）

    基于 Van Henten (1994) 光合作用模型，增加密度修正因子。
    原始模型假设单位面积上的作物，本函数扩展为支持任意种植密度。

    参数:
        xDn: 非结构干物质密度 [kg/m²]
        xDs: 结构干物质密度 [kg/m²]
        I: 光合有效光强
            - 当 I_in_umol=True 时: [μmol/m²/s] (光子通量密度)
            - 当 I_in_umol=False 时: [W/m²] (PAR能量通量)
        T: 空气温度 [°C]
        C: CO2浓度 [kg/m³]
        rho: 种植密度 [株/m²]
        params: 作物模型参数字典
        I_in_umol: 光强I是否为μmol单位 (默认True)

            以下参数用于单位转换:
            - c_optical_eff: LED光子效率 [μmol/J], 默认2.5
            以下参数用于光合计算:
            - c_alpha, c_beta, c_Gamma, c_Q10_Gamma: 光合效率参数
            - c_eps: 光合效率 [kg/J]
            - c_k: 光衰减系数 [-]
            - c_lar_s: 比叶面积 [m²/kg]
            - c_par: PAR比例 [-]
            - c_rad_rf: 辐射比例因子 [-]
            - c_tau: 分配系数 [-]
            - c_bnd, c_stm: 边界层/气孔导度 [m/s]
            - c_car_1, c_car_2, c_car_3: 叶肉导度系数

    返回:
        phi_phot: 总光合速率 [kg CO2/m²/s] (整个种植区)
        phi_phot_per_plant: 单株光合速率 [kg CO2/plant/s]
        f_abs: 光截获率 [-]
    """
    c_optical_eff = params.get('c_optical_eff', 2.5)

    # ========== 单位转换 ==========
    # μmol/m²/s → W/m² (PAR)
    # 1 W_m²_PAR ≈ c_optical_eff μmol/m²/s
    if I_in_umol:
        I_W = I / c_optical_eff  # 转换为 W/m² (PAR)
    else:
        I_W = I  # 已经是 W/m²

    # ========== 温度修正 ==========
    c_Gamma = params['c_Gamma']
    c_Q10_Gamma = params['c_Q10_Gamma']
    Gamma = c_Gamma * c_Q10_Gamma ** ((T - 20) / 10)

    c_eps = params['c_eps']
    eps = c_eps * ((C - Gamma) / (C + 2 * Gamma))

    # ========== CO2导度计算 ==========
    c_car_1 = params['c_car_1']
    c_car_2 = params['c_car_2']
    c_car_3 = params['c_car_3']
    sigma_car = c_car_1 * T**2 + c_car_2 * T + c_car_3

    c_bnd = params['c_bnd']
    c_stm = params['c_stm']
    sigma_CO2 = 1.0 / ((1.0 / c_bnd) + (1.0 / c_stm) + (1.0 / (sigma_car + 1e-10)))

    # ========== 光截获计算（Beer-Lambert定律）==========
    # LAI = c_lar_s * (1 - c_tau) * xDs [m²/m²]
    # 其中 xDs 是结构干物质密度 [kg/m²]
    # Beer-Lambert定律: f_abs = 1 - exp(-c_k * LAI)
    # 密度效应已隐含在 xDs 中：密度↑ → xDs↑ → LAI↑ → f_abs↑ → 光合↑
    # 这与PFAL-DRL的实现方式一致:
    #   phi_phot = phi_phot_max * (1 - exp(-c_k*c_lar_s*(1-c_tau)*xDs))
    c_k = params['c_k']
    c_lar_s = params['c_lar_s']
    c_tau = params['c_tau']
    LAI = c_lar_s * (1 - c_tau) * xDs  # 总LAI [m²/m²]

    # Beer-Lambert定律（密度效应已隐含在LAI中）
    f_abs = 1.0 - np.exp(-c_k * LAI)

    # 吸收光强 [W/m²]
    I_abs = I_W * f_abs

    # ========== 最大光合速率 (Michaelis-Menten型) ==========
    c_par = params['c_par']
    c_rad_rf = params.get('c_rad_rf', 1.0)

    numerator = eps * c_par * c_rad_rf * I_abs * sigma_CO2 * (C - Gamma)
    denominator = eps * c_par * c_rad_rf * I_abs + sigma_CO2 * (C - Gamma)
    phi_phot_max = numerator / (denominator + 1e-10)

    # 实际光合速率（Beer-Lambert光截获饱和，与PFAL-DRL一致）
    phi_phot = max(0, phi_phot_max * (1 - np.exp(-c_k * LAI)))

    # ========== 单株光合速率 ==========
    if rho > 0:
        phi_phot_per_plant = phi_phot / rho  # [kg CO2/plant/s]
    else:
        phi_phot_per_plant = 0.0

    return phi_phot, phi_phot_per_plant, f_abs


def respiration(
    xDs: float,
    T: float,
    params: dict
) -> float:
    """
    计算呼吸作用速率

    基于 Van Henten (1994) 呼吸模型，包括维持呼吸和生长呼吸。

    参数:
        xDs: 结构干物质密度 [kg/m²]
        T: 空气温度 [°C]
        params: 作物模型参数字典
            - c_resp_s: 茎/叶维持呼吸系数 [1/s]
            - c_resp_r: 根呼吸系数 [1/s]
            - c_tau: 分配系数 [-]
            - c_Q10_resp: Q10温度响应 [-]

    返回:
        phi_resp: 呼吸速率 [kg CO2/m²/s]
    """
    c_resp_s = params['c_resp_s']
    c_resp_r = params['c_resp_r']
    c_tau = params['c_tau']
    c_Q10_resp = params['c_Q10_resp']

    Q10_factor = c_Q10_resp ** ((T - 25) / 10)
    phi_resp = (
        (c_resp_s * (1 - c_tau) + c_resp_r * c_tau) *
        xDs * Q10_factor
    )

    return phi_resp


def transpiration(
    xDs: float,
    T: float,
    RH: float,
    rho: float,
    params: dict
) -> Tuple[float, float, float]:
    """
    计算蒸腾速率（考虑密度修正）

    基于 Van Henten (1994) 蒸腾模型。

    参数:
        xDs: 结构干物质密度 [kg/m²]
        T: 空气温度 [°C]
        RH: 相对湿度 [-] (0-1)
        rho: 种植密度 [株/m²]
        params: 作物模型参数字典
            - c_a_pl: 蒸腾叶面积系数 [m²/kg]
            - c_v_pl_ai: 气孔-空气水汽流系数 [m/s]
            - c_tau: 分配系数 [-]
            - c_v_0, c_v_1, c_v_2, c_v_3, mw_water, c_R, c_T_abs: 饱和水汽压参数

    返回:
        phi_transp: 总蒸腾速率 [kg water/m²/s]
        phi_transp_per_plant: 单株蒸腾速率 [kg water/plant/s]
        xH_sat: 饱和水汽压 [kg/m³]
    """
    c_a_pl = params['c_a_pl']
    c_v_pl_ai = params['c_v_pl_ai']
    c_tau = params['c_tau']

    # 饱和水汽压 (使用PFAL-DRL版本)
    xH_sat = calculate_saturation_vapor_pressure(T, params)

    # 实际水汽压 [kg/m³]
    xH = RH * xH_sat

    # 蒸腾叶面积: LAI_transp = c_a_pl * (1 - c_tau) * xDs [m²/m²]
    # 直接用 xDs 参与 Beer-Lambert 公式（与PFAL-DRL一致）
    # 密度效应已隐含在 xDs 中：密度↑ → xDs↑ → LAI↑ → 蒸腾↑
    LAI_transp = c_a_pl * (1 - c_tau) * xDs  # [m²/m²]

    # 蒸腾速率 (Beer-Lambert形式，与PFAL-DRL一致)
    # PFAL-DRL: phi_transp_h = (1-exp(-c_a_pl*xDs)) * c_v_pl_ai * (xH_sat - xH)
    phi_transp = (
        (1 - np.exp(-LAI_transp)) *
        c_v_pl_ai *
        (xH_sat - xH)
    )

    # 单株蒸腾速率
    if rho > 0:
        phi_transp_per_plant = phi_transp / rho
    else:
        phi_transp_per_plant = 0.0

    return phi_transp, phi_transp_per_plant, xH_sat


def growth_rate(
    xDn: float,
    xDs: float,
    T: float,
    params: dict
) -> float:
    """
    计算相对生长率

    基于 Van Henten (1994) 生长模型。

    参数:
        xDn: 非结构干物质密度 [kg/m²]
        xDs: 结构干物质密度 [kg/m²]
        T: 空气温度 [°C]
        params: 作物模型参数字典
            - c_r_gr_max: 最大相对生长率 [1/s]
            - c_Q10_gr: Q10温度响应 [-]

    返回:
        r_gr: 相对生长率 [1/s]
    """
    c_r_gr_max = params['c_r_gr_max']
    c_Q10_gr = params['c_Q10_gr']

    if xDs + xDn > 1e-10:
        r_gr = c_r_gr_max * (xDn / (xDs + xDn)) * c_Q10_gr ** ((T - 20) / 10)
    else:
        r_gr = 0.0

    return r_gr


def net_carbon_assimilation(
    xDn: float,
    xDs: float,
    I: float,
    T: float,
    C: float,
    RH: float,
    rho: float,
    params: dict,
    I_in_umol: bool = True
) -> Tuple[float, float, float, float]:
    """
    计算净碳同化速率（综合光合、呼吸、生长）

    这是最常用的接口函数，返回所有关键的生理速率。

    参数:
        xDn: 非结构干物质密度 [kg/m²]
        xDs: 结构干物质密度 [kg/m²]
        I: 光合有效光强 (单位由I_in_umol决定)
        T: 空气温度 [°C]
        C: CO2浓度 [kg/m³]
        RH: 相对湿度 [-] (0-1) 【重要: 此参数现已传入，不再硬编码!】
        rho: 种植密度 [株/m²]
        params: 作物模型参数字典
        I_in_umol: 光强I是否为μmol单位 (默认True)

    返回:
        phi_phot_c: 净CO2固定速率 [kg CO2/m²/s] (正值=固定)
        phi_phot: 总光合速率 [kg CO2/m²/s]
        phi_resp: 呼吸速率 [kg CO2/m²/s]
        phi_transp: 蒸腾速率 [kg water/m²/s]
    """
    c_alpha = params['c_alpha']
    c_beta = params['c_beta']

    # 光合作用
    phi_phot, phi_phot_per_plant, f_abs = photosynthesis(
        xDn, xDs, I, T, C, rho, params, I_in_umol
    )

    # 呼吸作用
    phi_resp = respiration(xDs, T, params)

    # 生长率
    r_gr = growth_rate(xDn, xDs, T, params)

    # ========== 净CO2同化率 (PFAL-DRL版本) ==========
    # PFAL-DRL PFALEnv.ode():
    # phi_phot_c = phi_phot - (1/c_alpha)*phi_resp - ((1-c_beta)/(c_alpha*c_beta))*r_gr*xDs
    phi_phot_c = (
        phi_phot -
        (1.0 / c_alpha) * phi_resp -
        ((1.0 - c_beta) / (c_alpha * c_beta)) * r_gr * xDs
    )

    # 蒸腾作用 (使用传入的RH，不再硬编码!)
    phi_transp, _, _ = transpiration(xDs, T, RH, rho, params)

    return phi_phot_c, phi_phot, phi_resp, phi_transp


def dry_mass_per_plant(
    xDn: float,
    xDs: float,
    rho: float
) -> float:
    """
    计算单株干物质质量

    参数:
        xDn: 非结构干物质密度 [kg/m²]
        xDs: 结构干物质密度 [kg/m²]
        rho: 种植密度 [株/m²]

    返回:
        M: 单株干物质质量 [g/plant]
    """
    if rho > 0:
        M = (xDn + xDs) / rho * 1000  # kg → g
    else:
        M = 0.0

    return M


def lai_per_plant(
    xDs: float,
    rho: float,
    params: dict
) -> float:
    """
    计算单株叶面积

    参数:
        xDs: 结构干物质密度 [kg/m²]
        rho: 种植密度 [株/m²]
        params: 作物模型参数字典
            - c_lar_s: 比叶面积 [m²/kg]
            - c_tau: 分配系数 [-]

    返回:
        LAI_p: 单株叶面积 [m²/plant]
    """
    c_lar_s = params['c_lar_s']
    c_tau = params['c_tau']

    if rho > 0:
        LAI_total = c_lar_s * (1 - c_tau) * xDs
        LAI_p = LAI_total / rho
    else:
        LAI_p = 0.0

    return LAI_p


def growth_update(
    xDn: float,
    xDs: float,
    phi_phot_c: float,
    T: float,
    dt: float,
    params: dict
) -> Tuple[float, float]:
    """
    更新干物质状态（ODE积分一步）

    使用简单的欧拉积分更新干物质。

    参数:
        xDn: 当前非结构干物质密度 [kg/m²]
        xDs: 当前结构干物质密度 [kg/m²]
        phi_phot_c: 净CO2固定速率 [kg CO2/m²/s]
        T: 当前温度 [°C] (用于计算生长率)
        dt: 时间步长 [s]
        params: 作物模型参数字典

    返回:
        xDn_new: 更新后的非结构干物质密度 [kg/m²]
        xDs_new: 更新后的结构干物质密度 [kg/m²]
    """
    c_alpha = params['c_alpha']
    c_beta = params['c_beta']

    # 计算相对生长率 (使用当前温度，不再硬编码22°C)
    r_gr = growth_rate(xDn, xDs, T, params)

    # 非结构干物质变化 (PFAL-DRL版本)
    # d(xDn)/dt = alpha*phi_phot - r_gr*xDs - phi_resp - ((1-beta)/beta)*r_gr*xDs
    # 简化: 直接使用 phi_phot_c
    d_xDn = phi_phot_c * dt

    # 结构干物质变化
    d_xDs = r_gr * xDs * dt

    # 更新
    xDn_new = max(0, xDn + d_xDn)
    xDs_new = max(0, xDs + d_xDs)

    # 确保非结构干物质不会变为负，且不超过合理上限
    # 非结构物质通常不超过总干物质的50%
    total = xDn_new + xDs_new
    if total > 1e-10:
        xDn_new = min(xDn_new, total * 0.5)

    return xDn_new, xDs_new


def simulate_crop_growth(
    initial_xDn: float,
    initial_xDs: float,
    I_values: np.ndarray,
    T_values: np.ndarray,
    C_values: np.ndarray,
    RH_values: np.ndarray,
    rho: float,
    params: dict,
    dt: float = 3600.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    模拟作物生长轨迹

    给定一系列环境条件，模拟作物干物质和LAI的变化。

    参数:
        initial_xDn: 初始非结构干物质密度 [kg/m²]
        initial_xDs: 初始结构干物质密度 [kg/m²]
        I_values: 光强序列 [μmol/m²/s], 长度 N
        T_values: 温度序列 [°C], 长度 N
        C_values: CO2浓度序列 [kg/m³], 长度 N
        RH_values: 相对湿度序列 [-], 长度 N
        rho: 种植密度 [株/m²]
        params: 作物模型参数字典
        dt: 时间步长 [s]

    返回:
        xDn_seq: 非结构干物质密度序列 [kg/m²]
        xDs_seq: 结构干物质密度序列 [kg/m²]
        LAI_seq: 叶面积指数序列 [-]
        M_seq: 单株干物质序列 [g/plant]
    """
    N = len(I_values)
    xDn_seq = np.zeros(N)
    xDs_seq = np.zeros(N)
    LAI_seq = np.zeros(N)
    M_seq = np.zeros(N)

    xDn = initial_xDn
    xDs = initial_xDs

    c_lar_s = params['c_lar_s']
    c_tau = params['c_tau']

    for i in range(N):
        I = I_values[i]
        T = T_values[i]
        C = C_values[i]
        RH = RH_values[i]

        # 计算生理速率 (注意传入RH，不再使用默认值!)
        phi_phot_c, _, _, phi_transp = net_carbon_assimilation(
            xDn, xDs, I, T, C, RH, rho, params, I_in_umol=True
        )

        # 更新状态 (使用当前温度)
        xDn, xDs = growth_update(xDn, xDs, phi_phot_c, T, dt, params)

        # 存储
        xDn_seq[i] = xDn
        xDs_seq[i] = xDs

        # 计算集总量
        LAI = c_lar_s * (1 - c_tau) * xDs
        LAI_seq[i] = LAI

        M = dry_mass_per_plant(xDn, xDs, rho)
        M_seq[i] = M

    return xDn_seq, xDs_seq, LAI_seq, M_seq
