# -*- coding: utf-8 -*-
"""
模型模块初始化文件

导出主要类和函数供外部使用。
"""

from .crop_model import (
    photosynthesis,
    respiration,
    transpiration,
    growth_rate,
    net_carbon_assimilation,
    dry_mass_per_plant,
    lai_per_plant,
    growth_update,
    simulate_crop_growth,
    calculate_saturation_vapor_pressure,
)

from .batch_manager import Batch, BatchManager

from .environment_model import (
    environment_dynamics,
    simulate_environment_step,
    solve_environment_steady_state,
    calculate_saturation_vapor_pressure,
    relative_humidity_to_absolute,
    absolute_humidity_to_relative,
    co2_ppm_to_density,
    co2_density_to_ppm,
)

from .equipment import (
    calculate_led_power,
    calculate_hvac_power,
    calculate_co2_power,
    calculate_vent_power,
    calculate_dehum_power,
    calculate_total_power,
    calculate_energy_cost,
    calculate_power_with_bounds,
)

from .schedule_utils import (
    check_schedule_feasibility,
    sample_valid_schedule,
)

from .mpc_model import (
    IDX_C, IDX_T, IDX_RH, IDX_DLI1, IDX_DLI2,
    IDX_BATCH_START,
    _batch_indices,
    mpc_ode,
    define_mpc_model,
    env_state_to_mpc_state,
    env_and_batch_to_mpc_state,
    mpc_state_to_env_state,
    compute_step_reward_mpc,
    generate_disturbance_profile,
    co2_ppm_to_density,
    co2_density_to_ppm,
)

__all__ = [
    # 作物模型
    'photosynthesis',
    'respiration',
    'transpiration',
    'growth_rate',
    'net_carbon_assimilation',
    'dry_mass_per_plant',
    'lai_per_plant',
    'growth_update',
    'simulate_crop_growth',
    'calculate_saturation_vapor_pressure',
    # 批次管理
    'Batch',
    'BatchManager',
    # 环境模型
    'environment_dynamics',
    'simulate_environment_step',
    'solve_environment_steady_state',
    'relative_humidity_to_absolute',
    'absolute_humidity_to_relative',
    'co2_ppm_to_density',
    'co2_density_to_ppm',
    # 设备模型
    'calculate_led_power',
    'calculate_hvac_power',
    'calculate_co2_power',
    'calculate_vent_power',
    'calculate_dehum_power',
    'calculate_total_power',
    'calculate_energy_cost',
    'calculate_power_with_bounds',
    # 排程工具
    'check_schedule_feasibility',
    'sample_valid_schedule',
    # MPC模型（Per-Batch独立生理速率，与RL BatchManager对齐）
    # NX/NU/ND 动态，IDX_BM1/IDX_BM2 等需通过 _batch_indices() 获取
    'IDX_C', 'IDX_T', 'IDX_RH', 'IDX_DLI1', 'IDX_DLI2',
    'IDX_BATCH_START',
    '_batch_indices',
    'mpc_ode',
    'define_mpc_model',
    'env_state_to_mpc_state',
    'env_and_batch_to_mpc_state',
    'mpc_state_to_env_state',
    'compute_step_reward_mpc',
    'generate_disturbance_profile',
]
