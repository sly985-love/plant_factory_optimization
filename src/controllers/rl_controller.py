# -*- coding: utf-8 -*-
"""
RL控制器模块

使用stable-baselines3训练好的PPO策略进行控制。
支持上下文RL（输入包含排程参数）和独立RL（固定排程）。
"""

import numpy as np
from typing import Optional, Dict, Any
from .base_controller import BaseController

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RLController(BaseController):
    """
    RL控制器

    包装stable-baselines3的PPO模型，提供统一的predict接口。
    支持上下文RL（观测包含排程参数）。
    """

    def __init__(
        self,
        model=None,
        config: Optional[Dict[str, Any]] = None,
        use_context: bool = True
    ):
        """
        初始化RL控制器

        参数:
            model: stable-baselines3的模型实例（如PPO）
            config: 配置参数字典
            use_context: 是否使用上下文（True=上下文RL，False=独立RL）
        """
        super().__init__(config)
        self.model = model
        self.use_context = use_context

        # 从配置读取默认动作
        ep = config.get('equipment_params', {}) if config else {}
        self.default_action = np.array([
            config.get('default_I', 200.0) if config else 200.0,
            config.get('default_I', 200.0) if config else 200.0,
            0.0,
            ep.get('co2_supply_max', 0.5) * 0.2 if ep else 0.1,
            ep.get('c_vent_fan_cap', 0.5) * 0.4 if ep else 0.2,
            ep.get('c_dehum_cap', 0.002) * 0.5 if ep else 0.001,
        ], dtype=np.float32)

    def predict(self, obs: np.ndarray, context: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        使用RL模型预测动作

        参数:
            obs: 观测向量 [n_obs]
            context: 上下文信息（如排程参数）
                  如果use_context=True，context会被嵌入obs中

        返回:
            action: 控制动作向量 [6维]
        """
        if self.model is None:
            # 无模型时返回默认动作
            return self._default_action()

        # 处理观测格式
        if len(obs.shape) == 1:
            obs = obs.reshape(1, -1)
            single = True
        else:
            single = False

        # 预测
        action, _ = self.model.predict(obs, deterministic=True)

        if single:
            action = action[0]

        return action.astype(np.float32)

    def _default_action(self) -> np.ndarray:
        """无模型时的默认动作（从配置读取）"""
        return self.default_action


class IndependentRLController(RLController):
    """
    独立RL控制器

    针对固定排程训练的单一RL策略，不使用上下文。
    """

    def __init__(
        self,
        model=None,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(model, config, use_context=False)


class ContextualRLController(RLController):
    """
    上下文RL控制器

    训练时使用排程作为上下文，可泛化至不同排程。
    """

    def __init__(
        self,
        model=None,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(model, config, use_context=True)
