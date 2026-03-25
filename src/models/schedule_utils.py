# -*- coding: utf-8 -*-
"""
排程可行性检查工具

独立的排程可行性检查函数，供其他模块调用。
不在YAML文件中定义（YAML只存储数据，不存储函数）。

来源: 论文方法部分 2.2.1
"""

import numpy as np


def check_schedule_feasibility(
    t1: float,
    t2: float,
    rho2: float,
    A1_A2: float,
    schedule_params: dict
) -> tuple:
    """
    检查排程参数是否满足物理约束

    给定排程 c = (t1, t2, rho2, A1_A2)，计算 rho1 = rho2 * A2 * t2 / (A1 * t1)
    并检查: rho1_min <= rho1 <= rho1_max

    参数:
        t1: 育苗期时长 [天]
        t2: 定植期时长 [天]
        rho2: 定植区密度 [株/m²]
        A1_A2: 面积比 A1/A2 [-]
        schedule_params: schedule_params.yaml 加载的字典
            需包含: rho1_min, rho1_max, A_total

    返回:
        (is_feasible: bool, rho1: float)
            is_feasible: 是否满足约束
            rho1: 计算得到的育苗区密度 [株/m²]
    """
    A_total = schedule_params.get('A_total', 40.0)
    rho1_min = schedule_params.get('rho1_min', 30.0)
    rho1_max = schedule_params.get('rho1_max', 60.0)

    # A1 = A_total / (1 + A1_A2) (注意: A1_A2 = A1/A2)
    # A2 = A_total - A1
    if abs(A1_A2) < 1e-10:
        return False, 0.0

    A1 = A_total / (1 + A1_A2)
    A2 = A_total - A1

    # rho1 = rho2 * A2 * t2 / (A1 * t1)
    if abs(t1) < 1e-10 or abs(A1) < 1e-10:
        return False, 0.0

    rho1 = rho2 * A2 * t2 / (A1 * t1)

    is_feasible = rho1_min <= rho1 <= rho1_max

    return is_feasible, rho1


def sample_valid_schedule(
    schedule_params: dict,
    rng=None
) -> dict:
    """
    从排程空间采样一个有效的随机排程

    参数:
        schedule_params: schedule_params.yaml 加载的字典
        rng: 随机数生成器

    返回:
        有效的排程字典 {'t1', 't2', 'rho2', 'A1_A2'}
    """
    if rng is None:
        rng = np.random.default_rng()

    t1_min = int(schedule_params.get('t1_min', 10))
    t1_max = int(schedule_params.get('t1_max', 18))
    t2_min = int(schedule_params.get('t2_min', 18))
    t2_max = int(schedule_params.get('t2_max', 26))
    rho2_min = schedule_params.get('rho2_min', 20.0)
    rho2_max = schedule_params.get('rho2_max', 80.0)
    A1_A2_min = schedule_params.get('A1_A2_min', 0.1)
    A1_A2_max = schedule_params.get('A1_A2_max', 5.0)

    # 最大重试次数
    max_attempts = 100
    for _ in range(max_attempts):
        t1 = rng.integers(t1_min, t1_max + 1)
        t2 = rng.integers(t2_min, t2_max + 1)
        rho2 = rng.uniform(rho2_min, rho2_max)
        A1_A2 = rng.uniform(A1_A2_min, A1_A2_max)

        is_feasible, _ = check_schedule_feasibility(t1, t2, rho2, A1_A2, schedule_params)
        if is_feasible:
            return {'t1': t1, 't2': t2, 'rho2': rho2, 'A1_A2': A1_A2}

    # 如果找不到有效排程，返回一个默认值
    return {
        't1': 14,
        't2': 21,
        'rho2': 35.0,
        'A1_A2': 0.5,
    }
