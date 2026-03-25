# -*- coding: utf-8 -*-
"""
控制器基类模块

定义所有控制器的统一接口，确保不同控制器之间的可替换性。
所有控制器必须实现 predict(obs, context=None) 方法。

来源: 论文方法部分 2.5
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class BaseController(ABC):
    """
    控制器基类

    所有控制器（规则、RL、MPC、SMPC）必须继承此类并实现 predict 方法。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化控制器

        参数:
            config: 配置参数字典
        """
        self.config = config if config is not None else {}

    @abstractmethod
    def predict(self, obs: np.ndarray, context: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        根据当前观测预测控制动作

        参数:
            obs: 当前观测向量 [n_obs]
            context: 上下文信息（如排程参数）

        返回:
            action: 控制动作向量 [n_action]
        """
        pass

    def reset(self):
        """重置控制器状态（如有内部状态）"""
        pass

    def get_config(self) -> Dict[str, Any]:
        """获取控制器配置"""
        return self.config.copy()
