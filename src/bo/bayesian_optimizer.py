# -*- coding: utf-8 -*-
"""
贝叶斯优化模块

实现基于高斯过程的贝叶斯优化，用于上层排程参数优化。

主要功能:
1. 连续搜索空间定义（rho2和A1_A2为连续变量）
2. 高斯过程代理模型
3. 采集函数 (EI, PI, LCB)
4. 物理可行性约束检查

【重要】rho2 和 A1_A2 现在是连续搜索变量，不再映射到离散集合。
这允许在给定范围内找到最优值。

来源: 论文方法部分 2.4
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Callable, Optional

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skopt import gp_minimize
from skopt.space import Real, Integer, Space
from skopt.utils import use_named_args
from models.schedule_utils import check_schedule_feasibility


class BayesianOptimizer:
    """
    贝叶斯优化器

    用于优化排程参数 c = (t1, t2, rho2, A1_A2)
    """

    def __init__(
        self,
        config: Dict[str, Any],
        objective_func: Callable,
        schedule_params: Dict[str, Any]
    ):
        """
        初始化贝叶斯优化器

        参数:
            config: BO参数字典
            objective_func: 目标函数 f(schedule) -> profit
            schedule_params: 排程参数空间定义
        """
        self.config = config
        self.objective_func = objective_func
        self.schedule_params = schedule_params

        # 搜索空间
        self._build_search_space()

        # 优化结果
        self.results = []
        self.best_result = None
        self.best_schedule = None

    def _build_search_space(self):
        """
        构建搜索空间

        搜索空间: [t1, t2, rho2, A1_A2]
        - t1, t2: 整数
        - rho2: 连续 [rho2_min, rho2_max]
        - A1_A2: 连续 [A1_A2_min, A1_A2_max]

        所有范围从 schedule_params 获取。
        """
        sp = self.schedule_params

        # 整数维度: t1, t2
        self.dimensions = [
            Integer(
                low=int(sp.get('t1_min', 10)),
                high=int(sp.get('t1_max', 18)),
                name='t1'
            ),
            Integer(
                low=int(sp.get('t2_min', 18)),
                high=int(sp.get('t2_max', 26)),
                name='t2'
            ),
            # 连续维度: rho2, A1_A2
            Real(
                low=sp.get('rho2_min', 20.0),
                high=sp.get('rho2_max', 80.0),
                name='rho2'
            ),
            Real(
                low=sp.get('A1_A2_min', 0.1),
                high=sp.get('A1_A2_max', 5.0),
                name='A1_A2'
            ),
        ]

        # 搜索空间对象
        self.space = Space(self.dimensions)

        # 密度约束参数
        self.rho1_min = sp.get('rho1_min', 30)
        self.rho1_max = sp.get('rho1_max', 60)
        self.A_total = sp.get('A_total', 40.0)

    def _check_feasibility(self, t1: float, t2: float, rho2: float, A1_A2: float) -> Tuple[bool, float]:
        """
        检查排程可行性

        使用独立的可行性检查函数。

        参数:
            t1, t2, rho2, A1_A2: 搜索值

        返回:
            (is_feasible, rho1): 是否可行及计算的rho1
        """
        is_feasible, rho1 = check_schedule_feasibility(
            t1, t2, rho2, A1_A2, self.schedule_params
        )
        return is_feasible, rho1

    def _validate_schedule(self, t1: int, t2: int, rho2: float, A1_A2: float) -> Dict[str, Any]:
        """
        验证并构建排程字典

        参数:
            t1, t2, rho2, A1_A2: 搜索值

        返回:
            schedule: 排程字典
        """
        A1 = self.A_total / (1 + A1_A2)
        A2 = self.A_total - A1

        return {
            't1': t1,
            't2': t2,
            'rho2': rho2,
            'A1_A2': A1_A2,
            'A1': A1,
            'A2': A2,
        }

    def optimize(self) -> Dict[str, Any]:
        """
        执行贝叶斯优化

        返回:
            result: 优化结果
        """
        @use_named_args(dimensions=self.dimensions)
        def objective(t1, t2, rho2, A1_A2):
            # 检查可行性
            is_feas, rho1 = self._check_feasibility(t1, t2, rho2, A1_A2)

            if not is_feas:
                # 不可行点返回惩罚值
                penalty = self.config.get('infeasible_penalty', -1e6)
                return penalty

            # 构建排程（连续值直接使用）
            schedule = self._validate_schedule(t1, t2, rho2, A1_A2)

            # 评估目标函数
            try:
                profit = self.objective_func(schedule)
                self.results.append({'schedule': schedule, 'profit': profit,
                                     'continuous': [t1, t2, rho2, A1_A2]})
                return -profit  # 最小化问题
            except Exception as e:
                print(f"评估失败: {e}")
                return -1e6

        # GP优化
        res = gp_minimize(
            func=objective,
            dimensions=self.dimensions,
            n_initial_points=self.config.get('n_initial_points', 10),
            n_calls=self.config.get('n_iter', 40),
            random_state=self.config.get('seed', 42),
            verbose=True,
            acq_func=self.config.get('acquisition', 'EI'),
            kappa=self.config.get('lcb_kappa', 1.96),
            xi=self.config.get('ei_xi', 0.01),
        )

        # 最佳结果
        best_idx = np.argmin(res.func_vals)
        best_cont = res.x_iters[best_idx]
        self.best_schedule = self._validate_schedule(*best_cont)
        self.best_result = res

        return {
            'best_schedule': self.best_schedule,
            'best_profit': -res.fun,
            'best_continuous': best_cont,
            'convergence': res.func_vals,
            'iterations': res.x_iters,
        }

    def get_convergence(self) -> Tuple[List[float], List[List]]:
        """获取收敛曲线"""
        if self.results:
            profits = [r['profit'] for r in self.results]
            schedules = [r['continuous'] for r in self.results]
            return profits, schedules
        return [], []
