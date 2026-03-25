# -*- coding: utf-8 -*-
"""
通用工具模块
"""

import numpy as np
import os
import csv
from typing import Dict, Any, List, Optional


def save_trajectory_to_csv(
    trajectory: List[Dict[str, Any]],
    filename: str,
    results_dir: str = 'results/data'
):
    """
    保存轨迹到CSV文件

    参数:
        trajectory: 轨迹列表，每个元素是一个字典
        filename: 保存的文件名
        results_dir: 保存目录
    """
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, filename)

    if not trajectory:
        return

    keys = trajectory[0].keys()

    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(trajectory)


def load_trajectory_from_csv(
    filename: str,
    results_dir: str = 'results/data'
) -> List[Dict[str, Any]]:
    """
    从CSV加载轨迹
    """
    filepath = os.path.join(results_dir, filename)
    trajectory = []

    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            trajectory.append(row)

    return trajectory


def set_random_seed(seed: int):
    """
    设置全局随机种子

    参数:
        seed: 随机种子
    """
    import random
    import numpy as np

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except:
        pass

    random.seed(seed)
    np.random.seed(seed)


def load_weather_data(
    filepath: str,
    sample_interval: int = 3600
) -> Dict[str, np.ndarray]:
    """
    加载天气数据

    参数:
        filepath: 天气数据文件路径
        sample_interval: 采样间隔 [秒]

    返回:
        weather_dict: 包含天气变量的字典
    """
    import pandas as pd

    df = pd.read_csv(filepath)

    # 提取列
    weather = {
        'time': df['time'].values if 'time' in df.columns else np.arange(len(df)),
        'radiation': df['radiation'].values if 'radiation' in df.columns else np.zeros(len(df)),
        'temperature': df['temperature'].values if 'temperature' in df.columns else np.zeros(len(df)),
        'humidity': df['humidity'].values if 'humidity' in df.columns else np.zeros(len(df)),
        'co2': df['co2'].values if 'co2' in df.columns else np.ones(len(df)) * 400,
    }

    return weather


def compute_electricity_price(
    hour: int,
    price_model: str = 'time_of_use',
    time_of_use_periods: Optional[Dict[str, List[int]]] = None,
    time_of_use_prices: Optional[Dict[str, float]] = None,
    constant_price: float = 0.6
) -> float:
    """
    计算电价

    参数:
        hour: 小时 [0-23]
        price_model: 电价模型
        time_of_use_periods: 分时时段
        time_of_use_prices: 分时价格
        constant_price: 固定电价

    返回:
        price: 电价 [元/kWh]
    """
    if price_model == 'constant':
        return constant_price

    elif price_model == 'time_of_use':
        if time_of_use_periods is None:
            time_of_use_periods = {
                'peak': [8, 9, 10, 11, 18, 19, 20, 21],
                'off_peak': [6, 7, 12, 17, 22, 23],
                'valley': [0, 1, 2, 3, 4, 5, 13, 14, 15, 16],
            }

        if time_of_use_prices is None:
            time_of_use_prices = {'peak': 1.0, 'off_peak': 0.6, 'valley': 0.3}

        for period_name, hours in time_of_use_periods.items():
            if hour in hours:
                return time_of_use_prices.get(period_name, constant_price)

        return constant_price

    else:
        return constant_price


def normalize(data: np.ndarray, low: float, high: float) -> np.ndarray:
    """
    线性归一化到 [0, 1]
    """
    return (data - low) / (high - low + 1e-8)


def denormalize(data: np.ndarray, low: float, high: float) -> np.ndarray:
    """
    反归一化
    """
    return data * (high - low) + low
