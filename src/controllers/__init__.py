# -*- coding: utf-8 -*-
"""
控制器模块初始化文件

包含以下控制器:
1. BaseController - 基类
2. RuleController - 规则控制器
3. PIDController - PID控制器
4. RLController - 强化学习控制器
5. PlantFactoryMPC - 模型预测控制器（CasADi非线性MPC）
6. MPCExperiment - MPC闭环仿真实验框架
"""

from .base_controller import BaseController
from .rule_controller import RuleController, PIDController
from .rl_controller import RLController, IndependentRLController, ContextualRLController
from .mpc_controller import PlantFactoryMPC
from .mpc_experiment import MPCExperiment, RLClosedLoopExperiment

__all__ = [
    # 基类
    'BaseController',
    # 规则类
    'RuleController',
    'PIDController',
    # RL类
    'RLController',
    'IndependentRLController',
    'ContextualRLController',
    # MPC类
    'PlantFactoryMPC',
    'MPCExperiment',
    'RLClosedLoopExperiment',
]
