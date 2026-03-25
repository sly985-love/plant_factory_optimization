# -*- coding: utf-8 -*-
"""
MPC用CasADi符号预测模型

将Van Henten作物生长模型和PFAL-DRL集装箱环境动力学
转化为CasADi符号表达式，用于MPC的NLP优化求解。

【关键物理修正】每个批次的生长阶段不同，必须按批次独立计算生理速率：
1. phi_phot_c_i = f(xDn_i, xDs_i, I, T, C, RH)  — 每个batch独立计算
2. phi_phot_c_total = sum_i(phi_phot_c_i * area_batch_i) / A_total  — 按面积加权汇总
3. 同理蒸腾、热负荷等也必须按batch汇总

状态设计（per-batch干物质密度用于生理速率，zone均值用于环境反馈）：
- 环境状态(3): [C, T, RH]
- DLI累计(2): [dLI1, dLI2]
- 育苗区per-batch干物质密度(2*N1): [xDn1_0, xDs1_0, ..., xDn1_{N1-1}, xDs1_{N1-1}]
- 定植区per-batch干物质密度(2*N2): [xDn2_0, xDs2_0, ..., xDn2_{N2-1}, xDs2_{N2-1}]
- zone聚合biomass(2): [biomass1, biomass2]  — 用于奖励计算

与RL BatchManager对齐：
- BatchManager.update() 中每个batch独立计算 net_carbon_assimilation()
- 汇总: total_P = sum_i(phi_phot_i * area_batch_i)
- MPC ODE 中必须做完全相同的操作

来源: 基于PFAL-DRL PFALEnv.ode()和Van Henten作物模型, 论文方法部分 2.2.1
"""

import casadi as ca
import numpy as np
from typing import Tuple, Dict, Any, List

from .environment_model import co2_ppm_to_density, co2_density_to_ppm


# =============================================================================
# 符号常量定义（动态，由 define_mpc_model 初始化）
# =============================================================================

# NX 动态: 3 + 2 + 2*N1 + 2*N2 + 2
# 前缀索引
IDX_C = 0
IDX_T = 1
IDX_RH = 2
IDX_DLI1 = 3
IDX_DLI2 = 4
# per-batch 从 IDX_BATCH_START 开始
IDX_BATCH_START = 5
# biomass 在末尾: [biomass1, biomass2]


def _batch_indices(N1: int, N2: int) -> Dict[str, int]:
    """计算各区域的索引偏移量"""
    n_env = 5  # C, T, RH, dLI1, dLI2
    n_batch = 2 * N1 + 2 * N2
    n_bm = 2   # biomass1, biomass2
    NX = n_env + n_batch + n_bm

    offs = {
        'NX': NX,
        'IDX_BM1': NX - 2,
        'IDX_BM2': NX - 1,
        'IDX_SEEDLING_DN': n_env,                           # 育苗区xDn起始
        'IDX_SEEDLING_DS': n_env + N1,                     # 育苗区xDs起始
        'IDX_TRANSPLANT_DN': n_env + 2 * N1,               # 定植区xDn起始
        'IDX_TRANSPLANT_DS': n_env + 2 * N1 + N2,          # 定植区xDs起始
        'N_ENV': n_env,
        'N_BATCH': n_batch,
        'N_BM': n_bm,
    }
    return offs


def _build_params(crop_params: Dict) -> ca.DM:
    """将作物参数字典转换为CasADi DM向量"""
    return ca.DM([
        crop_params['c_alpha'],
        crop_params['c_beta'],
        crop_params['c_Gamma'],
        crop_params['c_Q10_Gamma'],
        crop_params['c_k'],
        crop_params['c_lar_s'],
        crop_params['c_tau'],
        crop_params['c_eps'],
        crop_params['c_par'],
        crop_params['c_Q10_resp'],
        crop_params['c_resp_s'],
        crop_params['c_resp_r'],
        crop_params['c_r_gr_max'],
        crop_params['c_Q10_gr'],
        crop_params['c_bnd'],
        crop_params['c_stm'],
        crop_params['c_car_1'],
        crop_params['c_car_2'],
        crop_params['c_car_3'],
        crop_params['c_a_pl'],
        crop_params['c_v_pl_ai'],
        crop_params['c_optical_eff'],
    ])


def _build_container_params(container_params: Dict) -> ca.DM:
    """将容器参数字典转换为CasADi DM向量"""
    return ca.DM([
        container_params['c_volume'],
        container_params['c_cap_q'],
        container_params['c_cap_q_v'],
        container_params['c_surface_area'],
        container_params['c_U'],
        container_params['c_lat_water'],
        container_params['c_v_0'],
        container_params['c_v_1'],
        container_params['c_v_2'],
        container_params['c_v_3'],
        container_params['mw_water'],
        container_params['c_R'],
        container_params['c_T_abs'],
        container_params['c_led_eff'],
        container_params['c_optical_eff'],
        container_params['c_total_plant_area'],
    ])


# =============================================================================
# CasADi 符号 ODE 模型（per-batch生理速率计算）
# =============================================================================

def mpc_ode(
    x: ca.SX,
    u: ca.SX,
    d: ca.SX,
    p_crop: ca.SX,
    p_cont: ca.SX,
    A1: float, A2: float,
    N1: int, N2: int
) -> ca.SX:
    """
    MPC ODE（CasADi符号表达式）。

    【核心修正】per-batch独立生理速率计算：
    - 每个batch的 xDn_i, xDs_i 独立 → phi_phot_c_i 独立
    - phi_phot_c_total = sum_i(phi_phot_c_i * area_batch_i) / A_total
    - 同理 phi_transp_total, d_biomass_i 等

    状态 x[NX]:
        x[0:3]:   [C, T, RH] — 环境状态
        x[3:5]:   [dLI1, dLI2] — 当日DLI累计
        x[5:5+2*N1]:   育苗区每batch干物质密度 [xDn1_0, xDs1_0, ..., xDn1_{N1-1}, xDs1_{N1-1}]
        x[5+2*N1:5+2*N1+2*N2]: 定植区每batch干物质密度
        x[NX-2:NX]: [biomass1, biomass2] — zone实际总干物质

    与RL BatchManager.update() 完全对齐：
    - 育苗区: area_per_batch1 = A1 / N1
    - 定植区: area_per_batch2 = A2 / N2
    - total_P = sum_i(phi_phot_i * area_batch_i)
    - total_E = sum_i(phi_transp_i * area_batch_i)
    - d_biomass_i = (phi_phot_c_i + r_gr_i * xDs_i) * area_batch_i
    """
    offs = _batch_indices(N1, N2)
    NX = offs['NX']
    n_env = offs['N_ENV']
    idx_bm1 = offs['IDX_BM1']
    idx_bm2 = offs['IDX_BM2']

    # --- 解包环境参数 ---
    c_vol = p_cont[0]; c_cap_q = p_cont[1]; c_cap_q_v = p_cont[2]
    c_surf = p_cont[3]; c_U = p_cont[4]; c_lat = p_cont[5]
    c_v_0 = p_cont[6]; c_v_1 = p_cont[7]; c_v_2 = p_cont[8]; c_v_3 = p_cont[9]
    mw_w = p_cont[10]; c_Rg = p_cont[11]; c_Tabs = p_cont[12]
    c_led = p_cont[13]; c_optical = p_cont[14]; A_total = p_cont[15]

    # --- 解包环境状态 ---
    C = x[IDX_C]; T = x[IDX_T]; RH = x[IDX_RH]

    # --- 解包扰动 ---
    T_out = d[0]; RH_out = d[1]; C_out = d[2]

    # --- 解包动作 ---
    I1, I2 = u[0], u[1]
    Q_HVAC = u[2]
    u_CO2_kg = u[3] / 3600.0
    V_vent = u[4]; m_dehum = u[5]

    # --- per-batch面积 ---
    area_per_batch1 = A1 / max(1, N1)
    area_per_batch2 = A2 / max(1, N2)

    # =========================================================================
    # Step 1: 【核心修正】per-batch独立生理速率计算
    # 与RL BatchManager.update() 中的逻辑完全一致
    #
    # 【CasADi for循环说明】
    # CasADi SX 类型支持 Python for 循环，但每次迭代会创建新的符号节点。
    # 对于少量batch（N1≤14, N2≤21），当前实现功能正确且足够快。
    # 若追求极致效率，可用 CasADi 的向量化/多维SX操作替代：
    #   - 将 xDn/xDs 组成 (N1,) 和 (N2,) 向量
    #   - 用 ca.SX记法一次性计算所有batch的LAI、phi_phot等
    #   - 用 ca.sum1() 汇总
    # 但向量化的代码可读性差，当前版本优先保证正确性和可维护性。
    # =========================================================================

    # --- 共享中间变量（与CO2浓度/温度相关的全局项）---
    Gamma = p_crop[2] * p_crop[3] ** ((T - 20.0) / 10.0)
    sigma_car = p_crop[16] * T ** 2 + p_crop[17] * T + p_crop[18]
    sigma_CO2 = 1.0 / (
        1.0 / p_crop[14] + 1.0 / p_crop[15] + 1.0 / (sigma_car + 1e-30)
    )

    # 蒸腾相关饱和蒸汽压
    xH_sat = (c_v_0 * c_v_1 * mw_w / (c_Rg * (T + c_Tabs))) * ca.exp(c_v_2 * T / (T + c_v_3))
    xH = RH * xH_sat
    xH_sat_out = (c_v_0 * c_v_1 * mw_w / (c_Rg * (T_out + c_Tabs))) * ca.exp(c_v_2 * T_out / (T_out + c_v_3))
    xH_out = RH_out * xH_sat_out

    # --- 育苗区per-batch计算 ---
    total_phi_phot_c_seedling = 0.0  # sum_i(phi_phot_c_i * area_batch1)
    total_phi_transp_seedling = 0.0  # sum_i(phi_transp_i * area_batch1)
    total_r_gr_xDs_seedling = 0.0    # sum_i(r_gr_i * xDs_i * area_batch1)

    for i in range(N1):
        xDn_i = x[offs['IDX_SEEDLING_DN'] + i]
        xDs_i = x[offs['IDX_SEEDLING_DS'] + i]

        # --- 光合作用速率（与RL crop_model.photosynthesis一致）---
        LAI_i = p_crop[5] * (1.0 - p_crop[6]) * xDs_i
        f_abs_i = 1.0 - ca.exp(-p_crop[4] * LAI_i)
        I1_W = I1 / c_optical
        I1_abs = I1_W * f_abs_i
        eps_i = p_crop[7] * (C - Gamma) / (C + 2.0 * Gamma + 1e-30)

        phi_phot_i = p_crop[0] * eps_i * p_crop[8] * I1_abs * sigma_CO2 * (C - Gamma) / \
                     (eps_i * p_crop[8] * I1_abs + sigma_CO2 * (C - Gamma) + 1e-30)

        # --- 呼吸速率（与RL crop_model.respiration一致）---
        phi_resp_i = (p_crop[10] * (1.0 - p_crop[6]) + p_crop[11] * p_crop[6]) * xDs_i * \
                     p_crop[9] ** ((T - 25.0) / 10.0)

        # --- 相对生长率（与RL crop_model.growth_rate一致）---
        r_gr_i = p_crop[12] * (xDn_i / (xDn_i + xDs_i + 1e-30)) * \
                  p_crop[13] ** ((T - 20.0) / 10.0)

        # --- 净CO2同化率（与RL net_carbon_assimilation一致）---
        phi_phot_c_i = phi_phot_i - (1.0 / p_crop[1]) * phi_resp_i - \
                       ((1.0 - p_crop[0]) / (p_crop[0] * p_crop[1])) * r_gr_i * xDs_i

        # --- 蒸腾速率（与RL crop_model.transpiration一致）---
        LAI_transp_i = p_crop[19] * (1.0 - p_crop[6]) * xDs_i
        phi_transp_i = (1.0 - ca.exp(-LAI_transp_i)) * p_crop[20] * (xH_sat - xH)

        # --- 累加育苗区总速率（密度×面积）---
        total_phi_phot_c_seedling += phi_phot_c_i * area_per_batch1
        total_phi_transp_seedling += phi_transp_i * area_per_batch1
        total_r_gr_xDs_seedling += r_gr_i * xDs_i * area_per_batch1

    # --- 定植区per-batch计算 ---
    total_phi_phot_c_transplant = 0.0
    total_phi_transp_transplant = 0.0
    total_r_gr_xDs_transplant = 0.0

    for i in range(N2):
        xDn_i = x[offs['IDX_TRANSPLANT_DN'] + i]
        xDs_i = x[offs['IDX_TRANSPLANT_DS'] + i]

        LAI_i = p_crop[5] * (1.0 - p_crop[6]) * xDs_i
        f_abs_i = 1.0 - ca.exp(-p_crop[4] * LAI_i)
        I2_W = I2 / c_optical
        I2_abs = I2_W * f_abs_i
        eps_i = p_crop[7] * (C - Gamma) / (C + 2.0 * Gamma + 1e-30)

        phi_phot_i = p_crop[0] * eps_i * p_crop[8] * I2_abs * sigma_CO2 * (C - Gamma) / \
                     (eps_i * p_crop[8] * I2_abs + sigma_CO2 * (C - Gamma) + 1e-30)

        phi_resp_i = (p_crop[10] * (1.0 - p_crop[6]) + p_crop[11] * p_crop[6]) * xDs_i * \
                     p_crop[9] ** ((T - 25.0) / 10.0)

        r_gr_i = p_crop[12] * (xDn_i / (xDn_i + xDs_i + 1e-30)) * \
                  p_crop[13] ** ((T - 20.0) / 10.0)

        phi_phot_c_i = phi_phot_i - (1.0 / p_crop[1]) * phi_resp_i - \
                       ((1.0 - p_crop[0]) / (p_crop[0] * p_crop[1])) * r_gr_i * xDs_i

        LAI_transp_i = p_crop[19] * (1.0 - p_crop[6]) * xDs_i
        phi_transp_i = (1.0 - ca.exp(-LAI_transp_i)) * p_crop[20] * (xH_sat - xH)

        total_phi_phot_c_transplant += phi_phot_c_i * area_per_batch2
        total_phi_transp_transplant += phi_transp_i * area_per_batch2
        total_r_gr_xDs_transplant += r_gr_i * xDs_i * area_per_batch2

    # =========================================================================
    # Step 2: 环境动力学（使用per-batch汇总的总速率）
    # =========================================================================

    # --- CO2动态 ---
    # phi_phot_c_total = 总CO2固定速率 [kg CO2/s]
    # = sum_i(phi_phot_c_i * area_batch_i)
    phi_phot_c_total = total_phi_phot_c_seedling + total_phi_phot_c_transplant
    phi_vent_c_density = V_vent * (C - C_out)
    dC_dt = (u_CO2_kg - phi_phot_c_total - phi_vent_c_density) * A_total / c_vol

    # --- 温度动态 ---
    I1_W_led = I1_W * (1.0 - c_led) * A1
    I2_W_led = I2_W * (1.0 - c_led) * A2
    Q_led = I1_W_led + I2_W_led
    Q_wall = c_U * c_surf * (T_out - T)
    Q_vent = c_cap_q_v * V_vent * A_total * (T_out - T)

    # 总蒸腾速率 [kg water/s] = sum_i(phi_transp_i * area_batch_i)
    E_total = total_phi_transp_seedling + total_phi_transp_transplant
    Q_transp = E_total * c_lat * 1000.0
    Q_HVAC_total = Q_HVAC * A_total
    dT_dt = (Q_HVAC_total + Q_led - Q_wall - Q_vent - Q_transp) / (c_cap_q * A_total)

    # --- 湿度动态 ---
    phi_vent_h = V_vent * (xH - xH_out) * A_total
    m_dehum_total = m_dehum * A_total
    dH_dt = (E_total - phi_vent_h - m_dehum_total) / c_vol

    # =========================================================================
    # Step 3: per-batch干物质状态更新（ODE连续部分）
    # =========================================================================
    dx_batch = []

    # 育苗区: d(xDn_i)/dt = phi_phot_c_i, d(xDs_i)/dt = r_gr_i * xDs_i
    for i in range(N1):
        xDn_i = x[offs['IDX_SEEDLING_DN'] + i]
        xDs_i = x[offs['IDX_SEEDLING_DS'] + i]

        r_gr_i = p_crop[12] * (xDn_i / (xDn_i + xDs_i + 1e-30)) * \
                  p_crop[13] ** ((T - 20.0) / 10.0)

        dx_batch.append(phi_phot_c_from_state(xDn_i, xDs_i, I1, T, C, p_crop, c_optical))
        dx_batch.append(r_gr_i * xDs_i)

    # 定植区
    for i in range(N2):
        xDn_i = x[offs['IDX_TRANSPLANT_DN'] + i]
        xDs_i = x[offs['IDX_TRANSPLANT_DS'] + i]

        r_gr_i = p_crop[12] * (xDn_i / (xDn_i + xDs_i + 1e-30)) * \
                  p_crop[13] ** ((T - 20.0) / 10.0)

        dx_batch.append(phi_phot_c_from_state(xDn_i, xDs_i, I2, T, C, p_crop, c_optical))
        dx_batch.append(r_gr_i * xDs_i)

    # =========================================================================
    # Step 4: biomass连续更新（zone内所有batch的干物质变化之和）
    # =========================================================================
    d_biomass1 = (total_phi_phot_c_seedling + total_r_gr_xDs_seedling)
    d_biomass2 = (total_phi_phot_c_transplant + total_r_gr_xDs_transplant)

    # =========================================================================
    # 组装ODE向量
    # =========================================================================
    dx_env = [dC_dt, dT_dt, dH_dt]
    dx_dli = [I1 * 1e-6, I2 * 1e-6]

    return ca.vertcat(
        dx_env[0],        # C
        dx_env[1],        # T
        dx_env[2],        # RH
        dx_dli[0],        # dLI1
        dx_dli[1],        # dLI2
        *dx_batch,        # per-batch干物质
        d_biomass1,       # biomass1
        d_biomass2,       # biomass2
    )


def phi_phot_c_from_state(
    xDn: ca.SX, xDs: ca.SX,
    I: ca.SX, T: ca.SX, C: ca.SX,
    p_crop: ca.SX, c_optical: ca.SX
) -> ca.SX:
    """
    从状态计算单个batch的净CO2同化率 phi_phot_c。

    与RL crop_model.net_carbon_assimilation() 完全一致。
    """
    Gamma = p_crop[2] * p_crop[3] ** ((T - 20.0) / 10.0)
    sigma_car = p_crop[16] * T ** 2 + p_crop[17] * T + p_crop[18]
    sigma_CO2 = 1.0 / (
        1.0 / p_crop[14] + 1.0 / p_crop[15] + 1.0 / (sigma_car + 1e-30)
    )

    LAI = p_crop[5] * (1.0 - p_crop[6]) * xDs
    f_abs = 1.0 - ca.exp(-p_crop[4] * LAI)
    I_W = I / c_optical
    I_abs = I_W * f_abs
    eps = p_crop[7] * (C - Gamma) / (C + 2.0 * Gamma + 1e-30)

    phi_phot = p_crop[0] * eps * p_crop[8] * I_abs * sigma_CO2 * (C - Gamma) / \
               (eps * p_crop[8] * I_abs + sigma_CO2 * (C - Gamma) + 1e-30)

    phi_resp = (p_crop[10] * (1.0 - p_crop[6]) + p_crop[11] * p_crop[6]) * xDs * \
               p_crop[9] ** ((T - 25.0) / 10.0)

    r_gr = p_crop[12] * (xDn / (xDn + xDs + 1e-30)) * p_crop[13] ** ((T - 20.0) / 10.0)

    phi_phot_c = phi_phot - (1.0 / p_crop[1]) * phi_resp - \
                 ((1.0 - p_crop[0]) / (p_crop[0] * p_crop[1])) * r_gr * xDs

    return phi_phot_c


def define_mpc_model(
    dt: float,
    crop_params: Dict,
    container_params: Dict,
    A1: float,
    A2: float,
    N1: int,
    N2: int,
    x_min: np.ndarray,
    x_max: np.ndarray
) -> Tuple[ca.Function, ca.Function, ca.DM, ca.DM, Dict]:
    """
    构建MPC预测模型（离散化）。

    参数:
        dt: 时间步长 [秒]
        crop_params: 作物参数字典
        container_params: 容器参数字典
        A1, A2: 育苗/定植区总面积 [m²]
        N1, N2: 育苗/定植区batch数量（由batch_manager实时提供）
        x_min, x_max: 状态变量边界 [NX]

    返回:
        F: 离散状态转移函数
        g: 输出函数
        p_crop: 作物参数向量
        p_cont: 容器参数向量
        info: {'N1': N1, 'N2': N2, 'NX': NX, 'offs': offs}
    """
    p_crop = _build_params(crop_params)
    p_cont = _build_container_params(container_params)
    offs = _batch_indices(N1, N2)
    NX = offs['NX']

    x = ca.SX.sym("x", NX)
    u = ca.SX.sym("u", 6)
    d = ca.SX.sym("d", 4)

    dx_dt = mpc_ode(x, u, d, p_crop, p_cont, A1, A2, N1, N2)

    opts = {"simplify": True, "number_of_finite_elements": 4}
    integrator = ca.integrator(
        "F_int", "rk",
        {"x": x, "p": ca.vertcat(u, d), "ode": dx_dt},
        0.0, dt, opts
    )

    xmin_ca = ca.DM(x_min).reshape((NX, 1))
    xmax_ca = ca.DM(x_max).reshape((NX, 1))

    res = integrator(x0=x, p=ca.vertcat(u, d))
    x_next_ub = res["xf"]
    x_next_lim = ca.fmin(ca.fmax(x_next_ub, xmin_ca), xmax_ca)

    F = ca.Function("F", [x, u, d], [x_next_lim], ["x", "u", "d"], ["xnext"])
    F.expand()

    # 输出函数: [C_ppm, T, RH, LAI1, LAI2, dLI1, dLI2, biomass1, biomass2]
    C_density = x[IDX_C]
    T_x = x[IDX_T]
    c_lar_s_s = p_crop[5]; c_tau_s = p_crop[6]

    # zone平均LAI（用于软约束和观测）
    if N1 > 0:
        lai_seedling_list = [c_lar_s_s * (1.0 - c_tau_s) * x[offs['IDX_SEEDLING_DS'] + i]
                              for i in range(N1)]
        LAI1 = ca.sum1(ca.vertcat(*lai_seedling_list)) / N1
    else:
        LAI1 = 0.0

    if N2 > 0:
        lai_transplant_list = [c_lar_s_s * (1.0 - c_tau_s) * x[offs['IDX_TRANSPLANT_DS'] + i]
                                for i in range(N2)]
        LAI2 = ca.sum1(ca.vertcat(*lai_transplant_list)) / N2
    else:
        LAI2 = 0.0

    M_CO2 = 44.01e-3; P_atm = 101325.0; R_gas = 8.314
    T_K = T_x + 273.15
    C_ppm = C_density * R_gas * T_K / (M_CO2 * P_atm) * 1e6

    y = ca.vertcat(C_ppm, T_x, x[IDX_RH], LAI1, LAI2,
                   x[IDX_DLI1], x[IDX_DLI2], x[offs['IDX_BM1']], x[offs['IDX_BM2']])
    g = ca.Function("g", [x], [y], ["x"], ["y"])
    g.expand()

    info = {
        'N1': N1, 'N2': N2, 'NX': NX,
        'offs': offs,
        'A1': A1, 'A2': A2,
        'A_total': A1 + A2,
        'area_per_batch1': A1 / max(1, N1),
        'area_per_batch2': A2 / max(1, N2),
    }

    return F, g, p_crop, p_cont, info


# =============================================================================
# 辅助函数
# =============================================================================

def env_and_batch_to_mpc_state(
    env_state: np.ndarray,
    batch_manager,
    A1: float, A2: float,
    day_dli1: float = 0.0, day_dli2: float = 0.0
) -> Tuple[np.ndarray, int, int]:
    """
    将RL环境状态和BatchManager转换为MPC per-batch状态向量。

    【关键设计】
    - per-batch干物质密度作为MPC状态（用于精确的生理速率计算）
    - biomass来自batch_manager.get_aggregated_biomass()（与RL奖励一致）

    参数:
        env_state: 环境状态 [C, T, RH]
        batch_manager: BatchManager实例
        A1, A2: 育苗/定植区面积 [m²]
        day_dli1, day_dli2: 当日已累计DLI [mol/m²]

    返回:
        x_mpc: MPC状态向量（per-batch干物质密度）
        N1: 育苗区实际batch数量
        N2: 定植区实际batch数量
    """
    seedling_batches = batch_manager.seedling_batches
    transplant_batches = batch_manager.transplant_batches
    N1 = len(seedling_batches)
    N2 = len(transplant_batches)

    offs = _batch_indices(N1, N2)
    NX = offs['NX']
    x = np.zeros(NX, dtype=np.float64)

    # --- 环境状态 ---
    x[IDX_C] = env_state[0]
    x[IDX_T] = env_state[1]
    x[IDX_RH] = env_state[2]
    x[IDX_DLI1] = day_dli1
    x[IDX_DLI2] = day_dli2

    # --- per-batch干物质密度 ---
    for i, batch in enumerate(seedling_batches):
        x[offs['IDX_SEEDLING_DN'] + i] = batch.xDn
        x[offs['IDX_SEEDLING_DS'] + i] = batch.xDs

    for i, batch in enumerate(transplant_batches):
        x[offs['IDX_TRANSPLANT_DN'] + i] = batch.xDn
        x[offs['IDX_TRANSPLANT_DS'] + i] = batch.xDs

    # --- zone实际总干物质（来自RL BatchManager，与RL奖励完全一致）---
    total_M, seedling_M, transplant_M = batch_manager.get_aggregated_biomass()
    x[offs['IDX_BM1']] = float(seedling_M)
    x[offs['IDX_BM2']] = float(transplant_M)

    return x, N1, N2


def env_state_to_mpc_state(
    env_state: np.ndarray,
    batch_manager=None,
    batch_data: Dict = None,
    hour_of_day: int = 0,
    A1: float = 0.0, A2: float = 0.0,
    N1: int = 1, N2: int = 1,
    day_dli1: float = 0.0, day_dli2: float = 0.0
) -> np.ndarray:
    """兼容接口"""
    if batch_manager is not None:
        x, _, _ = env_and_batch_to_mpc_state(
            env_state, batch_manager, A1, A2, day_dli1, day_dli2
        )
        return x
    else:
        offs = _batch_indices(N1, N2)
        NX = offs['NX']
        x = np.zeros(NX, dtype=np.float64)
        x[IDX_C] = env_state[0]; x[IDX_T] = env_state[1]
        x[IDX_RH] = env_state[2]
        x[IDX_DLI1] = day_dli1; x[IDX_DLI2] = day_dli2
        return x


def mpc_state_to_env_state(x_mpc: np.ndarray, N1: int = 0, N2: int = 0) -> np.ndarray:
    """从MPC状态提取环境状态 [C, T, RH]"""
    return np.array([x_mpc[IDX_C], x_mpc[IDX_T], x_mpc[IDX_RH]], dtype=np.float64)


def compute_step_reward_mpc(
    x_curr: np.ndarray, x_next: np.ndarray,
    u: np.ndarray, d: np.ndarray,
    A1: float, A2: float,
    N1: int, N2: int,
    crop_params: Dict, equipment_params: Dict,
    dt: float, elec_price: float,
    price_growth: float = 0.25,
    price_CO2: float = 0.5,
    seedling_discount: float = 0.5
) -> Tuple[float, Dict]:
    """
    计算MPC单步奖励。

    与RL环境 _compute_reward() 完全对齐:
    - 生长收益 = alpha * (transplant_biomass_delta + seedling_biomass_delta * discount)
    - biomass变化直接从MPC per-batch状态中的 biomass1/biomass2 状态提取
    - 能量成本与RL equipment.py 一致
    """
    offs = _batch_indices(N1, N2)

    # biomass增量（与RL get_aggregated_biomass() 一致）
    d_biomass1_kg = x_next[offs['IDX_BM1']] - x_curr[offs['IDX_BM1']]
    d_biomass2_kg = x_next[offs['IDX_BM2']] - x_curr[offs['IDX_BM2']]

    # 育苗区半价折扣
    seedling_growth_g = d_biomass1_kg * 1000.0 * seedling_discount
    transplant_growth_g = d_biomass2_kg * 1000.0

    growth_reward = price_growth * (transplant_growth_g + seedling_growth_g)

    # 能量成本
    I1, I2, Q_HVAC, u_CO2, V_vent, m_dehum = u
    A_total = A1 + A2
    c_opt = equipment_params.get('c_optical_eff', 2.5)
    c_led = equipment_params.get('c_led_eff', 0.52)
    c_COP = equipment_params.get('c_COP', 3.0)
    fan_eff = equipment_params.get('fan_eff', 7.07)
    c_dehum_eev = equipment_params.get('c_dehum_eev', 3.0)

    P_led = (I1 / c_opt) * A1 / c_led + (I2 / c_opt) * A2 / c_led
    P_hvac = (abs(Q_HVAC) * A_total / c_COP)
    P_vent = V_vent * A_total / fan_eff
    P_dehum = (m_dehum * A_total) / c_dehum_eev * 1000.0

    dt_h = dt / 3600.0
    E_kWh = (P_led + P_hvac + P_vent + P_dehum) * dt_h / 1000.0
    cost_electric = E_kWh * elec_price

    total_CO2_kg = u_CO2 * A_total * dt_h / 1000.0
    cost_CO2 = total_CO2_kg * price_CO2

    return growth_reward - cost_electric - cost_CO2, {
        'growth_reward': growth_reward,
        'cost_electric': cost_electric,
        'cost_CO2': cost_CO2,
        'seedling_growth_g': seedling_growth_g,
        'transplant_growth_g': transplant_growth_g,
        'd_biomass1_kg': d_biomass1_kg,
        'd_biomass2_kg': d_biomass2_kg,
        'total_reward': growth_reward - cost_electric - cost_CO2,
    }


def generate_disturbance_profile(
    hour_of_day: int, day_of_year: int = 1,
    T_out_base: float = 20.0, RH_out_base: float = 0.7,
    C_out_ppm: float = 400.0,
    elec_price_base: float = 0.6,
    elec_price_min: float = 0.3, elec_price_max: float = 1.0
) -> np.ndarray:
    """生成典型扰动曲线"""
    T_delta = 5.0 * np.sin((hour_of_day - 6) * np.pi / 12.0)
    T_out = T_out_base + T_delta
    RH_delta = -0.1 * np.sin((hour_of_day - 6) * np.pi / 12.0)
    RH_out = np.clip(RH_out_base + RH_delta, 0.3, 0.95)
    C_out_kgm3 = co2_ppm_to_density(C_out_ppm, T_out)
    if 8 <= hour_of_day < 12 or 18 <= hour_of_day < 22:
        elec_price = elec_price_max
    elif 0 <= hour_of_day < 8:
        elec_price = elec_price_min
    else:
        elec_price = elec_price_base
    return np.array([T_out, RH_out, C_out_kgm3, elec_price], dtype=np.float64)
