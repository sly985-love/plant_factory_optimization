# -*- coding: utf-8 -*-
"""
批次管理器模块

管理植物工厂中的多批次生产系统，包括：
1. 批次创建、状态更新、移栽/采收
2. 多批次负荷聚合
3. 稳态初始化
4. 集总特征提取

【重要】面积说明：
- 育苗区和定植区的总面积 A_total = A1 + A2
- 但在计算干物质时，batch的xDn/xDs是密度 [kg/m²]，对应的是其所在区的面积
- 因此：批次干物质 = (xDn + xDs) * 批次所在区面积

来源: 论文方法部分 2.2.1, 2.2.4, 2.2.5

作者: Plant Factory Optimization Team
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
import math

from .crop_model import net_carbon_assimilation, growth_rate, simulate_crop_growth


@dataclass
class Batch:
    """
    单个批次的数据类

    属性:
        batch_id: 批次唯一标识
        age_h: 批次年龄 [小时]
        region: 所属区域 ("seedling" 或 "transplant")
        xDn: 非结构干物质密度 [kg/m²]
        xDs: 结构干物质密度 [kg/m²]
        LAI: 叶面积指数 [-]
        rho: 种植密度 [株/m²]
        harvest_ready: 是否可以采收
    """
    batch_id: int
    age_h: float
    region: str  # "seedling" 或 "transplant"
    xDn: float   # 非结构干物质 [kg/m²]
    xDs: float   # 结构干物质 [kg/m²]
    LAI: float   # 叶面积指数 [-]
    rho: float   # 种植密度 [株/m²]
    harvest_ready: bool = False


class BatchManager:
    """
    多批次管理系统

    负责管理育苗区和定植区的所有批次，实现：
    - 批次队列的创建和更新
    - 移栽和采收事件的处理
    - 多批次负荷的聚合
    - 集总特征的提取

    来源: 论文方法部分 2.2.1, 2.2.4, 2.2.5
    """

    def __init__(
        self,
        schedule: Dict[str, Any],
        container_params: Dict[str, Any],
        crop_params: Dict[str, Any],
        rng: Optional[np.random.Generator] = None,
        steady_state_params: Optional[Dict[str, Any]] = None,
        reward_params: Optional[Dict[str, Any]] = None
    ):
        """
        初始化批次管理器

        参数:
            schedule: 排程参数字典，包含 t1, t2, rho2, A1_A2
            container_params: 集装箱参数字典
            crop_params: 作物参数字典
            rng: 随机数生成器，用于稳态初始化的随机扰动
            steady_state_params: 稳态初始化参数字典
            reward_params: 奖励参数字典（包含采收/移栽干物质阈值）
        """
        self.schedule = schedule
        self.container_params = container_params
        # NOTE:
        # crop_model.* 计算依赖的参数在项目中被拆分在 container_params.yaml 与 crop_params.yaml。
        # BatchManager 内部统一使用 crop_params 字典，因此需要在这里合并两者，
        # 否则可能出现 KeyError（例如 c_a_pl）。
        self.crop_params = {**container_params, **crop_params}
        self.rng = rng if rng is not None else np.random.default_rng(42)

        # 采收干物质阈值（从reward_params读取）
        self.harvest_min_dry_mass = (
            reward_params.get('harvest_min_dry_mass', 25.0) if reward_params else 25.0
        )  # 单株干重 [g]

        # 稳态初始化参数
        self._init_steady_state_params(steady_state_params)

        # 提取排程参数
        self.t1 = schedule['t1']      # 育苗期天数
        self.t2 = schedule['t2']      # 定植期天数
        self.rho2 = schedule['rho2']  # 定植区密度 [株/m²]
        self.A1_A2 = schedule['A1_A2']  # 面积比

        # 计算派生参数
        self._compute_derived_params()

        # 初始化批次队列
        self.seedling_batches: List[Batch] = []  # 育苗区批次队列
        self.transplant_batches: List[Batch] = []  # 定植区批次队列

        # 批次ID计数器
        self.next_batch_id = 0

        # 统计信息
        self.total_transplants = 0  # 累计移栽次数
        self.total_harvests = 0     # 累计采收次数
        self.total_harvest_mass = 0.0  # 累计采收干物质 [kg]

        # 初始化稳态批次队列
        self._initialize_steady_state()

    def _init_steady_state_params(self, steady_state_params: Optional[Dict[str, Any]] = None):
        """初始化稳态参数，从传入字典或container_params获取，无则用默认值"""
        if steady_state_params is None:
            steady_state_params = {}

        cp = self.container_params
        sp = steady_state_params

        self.I_standard = sp.get('I_standard', cp.get('I_standard', 200.0))
        self.T_standard = sp.get('T_standard', cp.get('T_standard', 22.0))
        C_ppm = sp.get('C_standard_ppm', cp.get('C_standard_ppm', 1000.0))
        self.C_standard = sp.get('C_standard', cp.get('C_standard',
            self._ppm_to_density(C_ppm)))
        self.RH_standard = sp.get('RH_standard', cp.get('RH_standard', 0.75))
        self.dt_steady = sp.get('dt', cp.get('dt_steady', 3600.0))
        self.disturb_factor_max = sp.get('disturb_factor_max', cp.get('disturb_factor_max', 0.05))
        self.seedling_nonstruct_ratio = sp.get('seedling_nonstruct_ratio',
            cp.get('seedling_nonstruct_ratio', 0.1))
        self.initial_seedling_mass = sp.get('initial_seedling_mass',
            cp.get('initial_seedling_mass', 0.72e-3))
        self.I_standard_umol = sp.get('I_standard_umol', cp.get('I_standard_umol', True))

    def _compute_derived_params(self):
        """
        计算派生参数

        包括:
        - Δt = gcd(t1, t2) 移栽间隔
        - N1 = t1/Δt 育苗区批次数
        - N2 = t2/Δt 定植区批次数
        - A1, A2 各区面积
        - rho1 育苗区密度（由密度约束决定）
        """
        # 移栽间隔 [天]
        self.delta_t = math.gcd(self.t1, self.t2)

        # 各区批次数
        self.N1 = self.t1 // self.delta_t  # 育苗区批次数
        self.N2 = self.t2 // self.delta_t  # 定植区批次数

        # 各区面积 [m²]
        A_total = self.container_params.get('c_total_plant_area', 40.0)
        self.A1 = A_total / (1 + self.A1_A2)
        self.A2 = A_total - self.A1

        # 育苗区密度（由连续生产约束推导）
        # ρ1 * A1 * t1 = ρ2 * A2 * t2 → ρ1 = ρ2 * A2 * t2 / (A1 * t1)
        self.rho1 = self.rho2 * self.A2 * self.t2 / (self.A1 * self.t1)

        # 育苗期和定植期小时数
        self.t1_hours = self.t1 * 24  # [小时]
        self.t2_hours = self.t2 * 24  # [小时]

    def _initialize_steady_state(self):
        """
        稳态初始化

        创建均匀分布年龄的批次队列，每个批次的初始干物质
        通过预模拟计算，并施加 ±扰动 随机扰动。

        来源: 论文方法部分 2.2.5
        """
        # ========== 初始化育苗区批次 ==========
        seedling_ages = np.array([i * self.delta_t * 24 for i in range(self.N1)])

        for age_h in seedling_ages:
            # 计算该年龄的干物质
            xDn, xDs = self._get_initial_M_LAI(age_h, "seedling")

            # ±扰动 随机扰动
            disturb_factor = 1.0 + self.rng.uniform(
                -self.disturb_factor_max, self.disturb_factor_max
            )
            xDn *= disturb_factor
            xDs *= disturb_factor

            # 计算LAI
            c_lar_s = self.crop_params['c_lar_s']
            c_tau = self.crop_params['c_tau']
            LAI = c_lar_s * (1 - c_tau) * xDs

            batch = Batch(
                batch_id=self.next_batch_id,
                age_h=age_h,
                region="seedling",
                xDn=xDn,
                xDs=xDs,
                LAI=LAI,
                rho=self.rho1,
                harvest_ready=False
            )
            self.seedling_batches.append(batch)
            self.next_batch_id += 1

        # ========== 初始化定植区批次 ==========
        transplant_ages = np.array([i * self.delta_t * 24 for i in range(self.N2)])

        for age_h in transplant_ages:
            # 计算该年龄的干物质
            xDn, xDs = self._get_initial_M_LAI(age_h, "transplant")

            # ±扰动 随机扰动
            disturb_factor = 1.0 + self.rng.uniform(
                -self.disturb_factor_max, self.disturb_factor_max
            )
            xDn *= disturb_factor
            xDs *= disturb_factor

            # 计算LAI
            c_lar_s = self.crop_params['c_lar_s']
            c_tau = self.crop_params['c_tau']
            LAI = c_lar_s * (1 - c_tau) * xDs

            batch = Batch(
                batch_id=self.next_batch_id,
                age_h=age_h,
                region="transplant",
                xDn=xDn,
                xDs=xDs,
                LAI=LAI,
                rho=self.rho2,
                harvest_ready=False
            )
            self.transplant_batches.append(batch)
            self.next_batch_id += 1

    def _get_initial_M_LAI(self, age_h: float, region: str) -> Tuple[float, float]:
        """
        获取给定年龄的初始干物质

        使用预模拟表或生长模型前向积分计算。

        参数:
            age_h: 批次年龄 [小时]
            region: 区域 ("seedling" 或 "transplant")

        返回:
            xDn: 非结构干物质密度 [kg/m²]
            xDs: 结构干物质密度 [kg/m²]
        """

        # 确定密度
        rho = self.rho1 if region == "seedling" else self.rho2

        # 确定初始干物质（使用配置的参数）
        if age_h < 24:  # 刚移栽的幼苗
            # 从稳态参数获取
            total_mass = self.initial_seedling_mass
            xDn = self.seedling_nonstruct_ratio * total_mass
            xDs = (1 - self.seedling_nonstruct_ratio) * total_mass
        else:
            # 使用已有的最近批次干物质作为参考
            # 这里简化处理，使用经验公式
            period_hours = self.t1_hours if region == "seedling" else self.t2_hours
            growth_factor = min(age_h / period_hours, 1.0)
            xDn = 0.18e-3 * (1 + growth_factor * 3)
            xDs = 0.54e-3 * (1 + growth_factor * 3)

        # 模拟到指定年龄
        n_steps = max(int(age_h), 1)
        I_seq = np.full(n_steps, self.I_standard)
        T_seq = np.full(n_steps, self.T_standard)
        C_seq = np.full(n_steps, self.C_standard)
        RH_seq = np.full(n_steps, self.RH_standard)

        _, xDs_seq, _, _ = simulate_crop_growth(
            xDn, xDs, I_seq, T_seq, C_seq, RH_seq,
            rho, self.crop_params, self.dt_steady
        )

        xDn_out = xDn
        xDs_out = xDs_seq[-1] if len(xDs_seq) > 0 else xDs

        # 确保合理的干物质值
        xDs_out = max(0.001, xDs_out)  # 最小值 1 g/m²
        xDn_out = min(xDn_out, xDs_out)  # 非结构不超过结构

        return xDn_out, xDs_out

    def _ppm_to_density(self, ppm: float) -> float:
        """
        CO2浓度 ppm 转换为 kg/m³

        使用理想气体方程:
        C (kg/m³) = ppm * 1e-6 * M_CO2 * P / (R * (T + 273.15))
        """
        # 标准大气压 [Pa]
        P = 101325.0

        # CO2 分子量 [kg/mol]
        M_CO2 = 44.01e-3

        # 通用气体常数 [J/mol/K]
        R = 8.314

        # 温度假设 22°C
        T = 22.0

        C = ppm * 1e-6 * M_CO2 * P / (R * (T + 273.15))

        return C

    def update(
        self,
        dt: float,
        I1: float,
        I2: float,
        T: float,
        C: float,
        RH: float
    ) -> Dict[str, Any]:
        """
        更新所有批次状态

        参数:
            dt: 时间步长 [秒]
            I1: 育苗区光强 [μmol/m²/s]
            I2: 定植区光强 [μmol/m²/s]
            T: 箱内温度 [°C]
            C: 箱内CO2浓度 [kg/m³]
            RH: 箱内相对湿度 [-]

        返回:
            info: 包含负荷和集总特征的字典
        """
        dt_hours = dt / 3600.0  # 转换为小时

        # 负荷聚合为总速率 [kg/s]（环境模型需要总量，不是密度）
        # 每个batch的phi_*是密度[kg/m²/s]，乘以其批次面积[m²]得到[kg/s]
        total_E_rate = 0.0  # 总蒸腾速率 [kg water/s]
        total_P_rate = 0.0  # 总光合速率 [kg CO2/s]

        # ========== 更新育苗区批次 ==========
        harvest_list = []  # 待移栽的批次
        area_per_seedling_batch = self.A1 / max(1, self.N1)  # 育苗区每批次面积 [m²]

        for batch in self.seedling_batches:
            # 更新年龄
            batch.age_h += dt_hours

            # 计算生理速率
            phi_phot_c, phi_phot, phi_resp, phi_transp = net_carbon_assimilation(
                batch.xDn, batch.xDs,
                I1 if batch.age_h > 0 else 0,
                T, C, RH, batch.rho, self.crop_params,
                I_in_umol=True
            )

            # 更新干物质 (使用PFAL-DRL版本)
            c_alpha = self.crop_params['c_alpha']
            c_beta = self.crop_params['c_beta']
            c_tau = self.crop_params['c_tau']
            c_lar_s = self.crop_params['c_lar_s']
            c_r_gr_max = self.crop_params['c_r_gr_max']
            c_Q10_gr = self.crop_params['c_Q10_gr']

            # 相对生长率
            if batch.xDs + batch.xDn > 1e-10:
                r_gr = c_r_gr_max * (batch.xDn / (batch.xDs + batch.xDn)) * c_Q10_gr ** ((T - 20) / 10)
            else:
                r_gr = 0.0

            # 结构干物质变化: d(xDs)/dt = r_gr * xDs
            d_xDs = r_gr * batch.xDs * dt

            # 非结构干物质变化 (PFAL-DRL版本)
            d_xDn = phi_phot_c * dt

            batch.xDs = max(0.001, batch.xDs + d_xDs)
            batch.xDn = max(0, batch.xDn + d_xDn)

            # 确保非结构不超过总干物质50%
            total = batch.xDn + batch.xDs
            if total > 1e-10:
                batch.xDn = min(batch.xDn, total * 0.5)

            # 更新LAI
            batch.LAI = c_lar_s * (1 - c_tau) * batch.xDs

            # 检查移栽条件
            if batch.age_h >= self.t1_hours:
                harvest_list.append(batch)

            # 累加负荷（密度[kg/m²/s] × 批次面积[m²] = 速率[kg/s]）
            total_E_rate += phi_transp * area_per_seedling_batch
            total_P_rate += phi_phot * area_per_seedling_batch

        # 执行移栽
        transplant_mass_this_step = 0.0  # 本步移栽的干物质 [g]
        area_per_seedling_batch = self.A1 / max(1, self.N1)  # 育苗区每批次面积 [m²]

        for batch in reversed(harvest_list):
            if batch in self.seedling_batches:
                self.seedling_batches.remove(batch)
                self.total_transplants += 1

                # 【关键】将育苗区批次转为定植区批次，加入定植区
                # 物理原理：株数变化(ρ1→ρ2)但物理面积不变 → 干物质密度需按比例缩放
                # 单位: xDn/xDs 是 [kg/m²] 密度，株均干物质量 [kg/plant] 不变
                # 所以: xDs_new = xDs_old * (rho2/rho1)，xDn 同理
                if self.rho1 > 1e-10:
                    xDs_transplant = batch.xDs * (self.rho2 / self.rho1)
                    xDn_transplant = batch.xDn * (self.rho2 / self.rho1)
                else:
                    xDs_transplant = batch.xDs
                    xDn_transplant = batch.xDn

                # 追踪移栽干物质（育苗区面积 → 定植区面积，单位不变）
                transplant_mass_kg = (xDn_transplant + xDs_transplant) * area_per_seedling_batch
                transplant_mass_this_step += transplant_mass_kg * 1000.0  # kg → g

                # 更新LAI（用新密度下的xDs）
                c_lar_s = self.crop_params['c_lar_s']
                c_tau = self.crop_params['c_tau']
                LAI_transplant = c_lar_s * (1 - c_tau) * xDs_transplant

                transplant_batch = Batch(
                    batch_id=self.next_batch_id,
                    age_h=0.0,  # 归零，进入定植期
                    region="transplant",
                    xDn=xDn_transplant,
                    xDs=xDs_transplant,
                    LAI=LAI_transplant,
                    rho=self.rho2,    # 变为定植区密度
                    harvest_ready=False
                )
                self.transplant_batches.append(transplant_batch)
                self.next_batch_id += 1

        # ========== 更新定植区批次 ==========
        harvest_list = []  # 待采收的批次
        area_per_transplant_batch = self.A2 / max(1, self.N2)  # 定植区每批次面积 [m²]

        for batch in self.transplant_batches:
            # 更新年龄
            batch.age_h += dt_hours

            # 计算生理速率
            phi_phot_c, phi_phot, phi_resp, phi_transp = net_carbon_assimilation(
                batch.xDn, batch.xDs,
                I2 if batch.age_h > 0 else 0,
                T, C, RH, batch.rho, self.crop_params,
                I_in_umol=True
            )

            # 更新干物质 (使用PFAL-DRL版本)
            c_alpha = self.crop_params['c_alpha']
            c_beta = self.crop_params['c_beta']
            c_tau = self.crop_params['c_tau']
            c_lar_s = self.crop_params['c_lar_s']
            c_r_gr_max = self.crop_params['c_r_gr_max']
            c_Q10_gr = self.crop_params['c_Q10_gr']

            if batch.xDs + batch.xDn > 1e-10:
                r_gr = c_r_gr_max * (batch.xDn / (batch.xDs + batch.xDn)) * c_Q10_gr ** ((T - 20) / 10)
            else:
                r_gr = 0.0

            d_xDs = r_gr * batch.xDs * dt
            d_xDn = phi_phot_c * dt

            batch.xDs = max(0.001, batch.xDs + d_xDs)
            batch.xDn = max(0, batch.xDn + d_xDn)

            total = batch.xDn + batch.xDs
            if total > 1e-10:
                batch.xDn = min(batch.xDn, total * 0.5)

            # 更新LAI
            batch.LAI = c_lar_s * (1 - c_tau) * batch.xDs

            # 检查采收条件
            if batch.age_h >= self.t2_hours:
                harvest_list.append(batch)

            # 累加负荷（密度[kg/m²/s] × 批次面积[m²] = 速率[kg/s]）
            total_E_rate += phi_transp * area_per_transplant_batch
            total_P_rate += phi_phot * area_per_transplant_batch

        # 执行采收
        harvest_fail_this_step = False
        harvest_mass_this_step = 0.0  # [g]
        harvest_area = self.A2 / max(1, self.N2)  # 定植区每批次面积 [m²]

        for batch in reversed(harvest_list):
            if batch in self.transplant_batches:
                self.transplant_batches.remove(batch)
                self.total_harvests += 1
                harvest_mass_kg = (batch.xDn + batch.xDs) * harvest_area
                harvest_mass_g = harvest_mass_kg * 1000.0
                self.total_harvest_mass += harvest_mass_kg
                harvest_mass_this_step += harvest_mass_g
                harvest_per_plant_g = (batch.xDn + batch.xDs) * 1000.0 / max(batch.rho, 1e-10)
                if harvest_per_plant_g < self.harvest_min_dry_mass:
                    harvest_fail_this_step = True

        # ========== 提取集总特征 ==========
        lumped_features = self._extract_lumped_features()

        info = {
            # 负荷总速率 [kg/s]（环境模型需要总量）
            'total_E_rate': total_E_rate,  # 总蒸腾速率 [kg water/s]
            'total_P_rate': total_P_rate,  # 总光合速率 [kg CO2/s]

            # 集总特征（用于RL观测）
            **lumped_features,

            # 事件统计
            'transplants': self.total_transplants,
            'harvests': self.total_harvests,

            # 采收/移栽信息（供奖励函数使用）
            'harvest_mass': harvest_mass_this_step,  # 本步采收干物质 [g]
            'transplant_mass': transplant_mass_this_step,  # 本步移栽干物质 [g]
            'harvest_fail': harvest_fail_this_step,  # 是否有不达标采收
        }

        return info

    def _extract_lumped_features(self) -> Dict[str, float]:
        """
        提取集总特征（用于RL观测空间）

        【重要】单位说明：
        batch的xDn/xDs是密度 [kg/m²]，每个批次占地面积 = 区域总面积 / 批次数
        - 育苗区每批次面积 = A1 / N1 [m²]
        - 定植区每批次面积 = A2 / N2 [m²]
        批次总干物质 = (xDn + xDs) * 批次面积 [kg]
        即：
        - 育苗区 = sum_i[(xDn+xDs)_i * (A1/N1)]
        - 定植区 = sum_j[(xDn+xDs)_j * (A2/N2)]

        来源: 论文方法部分 2.2.4

        返回:
            包含以下特征的字典:
            - lai_total: 总叶面积指数 [-]
            - M_seedling: 育苗区总干物质 [g]
            - M_transplant: 定植区总干物质 [g]
            - days_left: 最老批次剩余天数 [天]
            - M_oldest: 最老批次干物质 [g]
            - lai_seedling: 育苗区LAI [-]
            - lai_transplant: 定植区LAI [-]
        """
        # ========== 育苗区 ==========
        # LAI = 总叶面积 / 占地面积 [m²/m²]
        # 育苗区总叶面积 = sum(LAI_i * 批次面积_i) = sum(LAI_i) * (A1/N1)
        # 育苗区LAI = 总叶面积 / A1 = sum(LAI_i) / N1
        area_per_seedling_batch = self.A1 / max(1, self.N1)
        lai_seedling = sum(b.LAI for b in self.seedling_batches) * area_per_seedling_batch / self.A1
        # 育苗区总干物质 = sum(密度_i * 批次面积_i) [kg]
        M_seedling_g = sum(
            (b.xDn + b.xDs) * area_per_seedling_batch for b in self.seedling_batches
        ) * 1000  # kg → g

        # ========== 定植区 ==========
        area_per_transplant_batch = self.A2 / max(1, self.N2)
        lai_transplant = sum(b.LAI for b in self.transplant_batches) * area_per_transplant_batch / self.A2
        M_transplant_g = sum(
            (b.xDn + b.xDs) * area_per_transplant_batch for b in self.transplant_batches
        ) * 1000

        # ========== 总LAI ==========
        lai_total = lai_seedling + lai_transplant

        # ========== 最老批次信息 ==========
        if self.transplant_batches:
            oldest = max(self.transplant_batches, key=lambda b: b.age_h)
            days_left = (self.t2_hours - oldest.age_h) / 24.0
            # 最老批次的干物质 (g) = 密度 * 批次面积 * 1000
            M_oldest = (oldest.xDn + oldest.xDs) * area_per_transplant_batch * 1000
        else:
            days_left = 0.0
            M_oldest = 0.0

        return {
            'lai_total': lai_total,
            'M_seedling': M_seedling_g,
            'M_transplant': M_transplant_g,
            'days_left': max(0, days_left),
            'M_oldest': M_oldest,
            'lai_seedling': lai_seedling,
            'lai_transplant': lai_transplant,
        }

    def get_aggregated_biomass(self) -> Tuple[float, float, float]:
        """
        获取聚合的干物质信息

        【重要】计算方式：密度 × 面积 = 总干物质

        返回:
            total_M: 总干物质 [kg]
            seedling_M: 育苗区干物质 [kg]
            transplant_M: 定植区干物质 [kg]
        """
        seedling_M = sum(
            (b.xDn + b.xDs) * (self.A1 / max(1, self.N1))
            for b in self.seedling_batches
        )
        transplant_M = sum(
            (b.xDn + b.xDs) * (self.A2 / max(1, self.N2))
            for b in self.transplant_batches
        )
        total_M = seedling_M + transplant_M

        return total_M, seedling_M, transplant_M

    def get_batch_ages(self) -> Dict[str, Any]:
        """
        获取所有 batch 的年龄信息（用于 MPC 事件预测）。

        返回:
            {
                'seedling_ages_h': [age_h, ...] 育苗区每batch年龄
                'transplant_ages_h': [age_h, ...] 定植区每batch年龄
                'seedling_count': int
                'transplant_count': int
            }
        """
        return {
            'seedling_ages_h': [b.age_h for b in self.seedling_batches],
            'transplant_ages_h': [b.age_h for b in self.transplant_batches],
            'seedling_count': len(self.seedling_batches),
            'transplant_count': len(self.transplant_batches),
        }

    def predict_next_event(self, horizon_h: float) -> Dict[str, Any]:
        """
        预测在给定时间范围内是否会发生移栽/采收事件。

        这是 MPC 事件触发的核心方法：告诉控制器"预测范围内是否有离散事件"。

        参数:
            horizon_h: 预测时间范围 [小时]

        返回:
            {
                'transplant_in_horizon': bool,  # 范围内是否有移栽
                'harvest_in_horizon': bool,   # 范围内是否有采收
                'first_transplant_h': float,    # 距第一次移栽的小时数（-1=无）
                'first_harvest_h': float,       # 距第一次采收的小时数（-1=无）
                'n_transplants': int,             # 范围内移栽次数
                'n_harvests': int,              # 范围内采收次数
                'event_trigger': bool,            # 是否触发NLP重建（任意事件）
            }
        """
        result = {
            'transplant_in_horizon': False,
            'harvest_in_horizon': False,
            'first_transplant_h': -1.0,
            'first_harvest_h': -1.0,
            'n_transplants': 0,
            'n_harvests': 0,
            'event_trigger': False,
        }

        # 育苗区：检查移栽事件（batch.age_h >= t1_hours）
        t1_h = self.t1_hours  # 育苗期长度
        for batch in self.seedling_batches:
            time_to_transplant = t1_h - batch.age_h  # 距移栽的小时数
            if 0 <= time_to_transplant <= horizon_h:
                result['transplant_in_horizon'] = True
                result['n_transplants'] += 1
                if result['first_transplant_h'] < 0 or time_to_transplant < result['first_transplant_h']:
                    result['first_transplant_h'] = time_to_transplant

        # 定植区：检查采收事件（batch.age_h >= t2_hours）
        t2_h = self.t2_hours  # 定植期长度
        for batch in self.transplant_batches:
            time_to_harvest = t2_h - batch.age_h  # 距采收的小时数
            if 0 <= time_to_harvest <= horizon_h:
                result['harvest_in_horizon'] = True
                result['n_harvests'] += 1
                if result['first_harvest_h'] < 0 or time_to_harvest < result['first_harvest_h']:
                    result['first_harvest_h'] = time_to_harvest

        result['event_trigger'] = (
            result['transplant_in_horizon'] or result['harvest_in_horizon']
        )
        return result

    def get_mpc_state(self, A1: float = None, A2: float = None) -> Dict[str, Any]:
        """
        提取供MPC控制器使用的batch-level状态。

        【重要】返回每个batch的独立干物质密度，而非zone平均。
        与MPC ODE中的batch-level状态完全对齐。

        参数:
            A1: 育苗区面积 [m²]（默认使用self.A1）
            A2: 定植区面积 [m²]（默认使用self.A2）

        返回:
            dict，键:
            - 'xDn_seedling': 育苗区每批次xDn列表
            - 'xDs_seedling': 育苗区每批次xDs列表
            - 'xDn_transplant': 定植区每批次xDn列表
            - 'xDs_transplant': 定植区每批次xDs列表
            - 'LAI_seedling': 育苗区每批次LAI列表
            - 'LAI_transplant': 定植区每批次LAI列表
            - 'N1': 育苗区实际批次数
            - 'N2': 定植区实际批次数
        """
        if A1 is None:
            A1 = self.A1
        if A2 is None:
            A2 = self.A2

        xDn_seedling = [b.xDn for b in self.seedling_batches]
        xDs_seedling = [b.xDs for b in self.seedling_batches]
        LAI_seedling = [b.LAI for b in self.seedling_batches]

        xDn_transplant = [b.xDn for b in self.transplant_batches]
        xDs_transplant = [b.xDs for b in self.transplant_batches]
        LAI_transplant = [b.LAI for b in self.transplant_batches]

        return {
            'xDn_seedling': xDn_seedling,
            'xDs_seedling': xDs_seedling,
            'xDn_transplant': xDn_transplant,
            'xDs_transplant': xDs_transplant,
            'LAI_seedling': LAI_seedling,
            'LAI_transplant': LAI_transplant,
            'N1': len(self.seedling_batches),
            'N2': len(self.transplant_batches),
        }

    def get_state_summary(self) -> Dict[str, Any]:
        """
        获取状态摘要

        返回:
            包含批次数量、各区状态等信息的字典
        """
        return {
            'n_seedling_batches': len(self.seedling_batches),
            'n_transplant_batches': len(self.transplant_batches),
            'total_transplants': self.total_transplants,
            'total_harvests': self.total_harvests,
            'total_harvest_mass_kg': self.total_harvest_mass,
            'schedule': self.schedule.copy(),
            'rho1': self.rho1,
            'rho2': self.rho2,
            'A1': self.A1,
            'A2': self.A2,
            'A_total': self.A1 + self.A2,
        }
