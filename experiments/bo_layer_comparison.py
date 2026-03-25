# -*- coding: utf-8 -*-
"""
Experiment 3: 贝叶斯优化+不同下层控制器的联合优化实验

上层：贝叶斯优化搜索最优排程参数 c = (t1, t2, rho2, A1/A2)
下层：分别使用 PID / MPC / ContextualRL 作为底层控制器评估排程质量

目标：对比三种下层控制器在 BO 上层优化框架下的最优利润和收敛速度。

使用方法:
    # 使用所有三种控制器进行BO优化对比
    python experiments/bo_layer_comparison.py --modes pid mpc rl --n_iter 30 --n_eval_repeats 3

    # 仅用MPC下层控制器做BO优化
    python experiments/bo_layer_comparison.py --modes mpc --n_iter 30 --n_eval_repeats 3

    # 指定RL模型路径
    python experiments/bo_layer_comparison.py --modes rl --rl_model results/models/best_model.zip

    # 完整对比（含可视化）
    python experiments/bo_layer_comparison.py --modes pid mpc rl --n_iter 30 --n_eval_repeats 3 --save --viz

来源: 论文方法部分 2.4 / 2.5 / 2.6
"""

import os
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Callable

# 添加项目根目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, project_dir)

from src.envs.plant_factory_env import MultiBatchPlantFactoryEnv
from src.envs.utils import load_all_configs
from src.models import co2_density_to_ppm, check_schedule_feasibility
from src.controllers import PlantFactoryMPC, PIDController, RuleController
from src.bo import BayesianOptimizer


# =============================================================================
# 下层控制器评估接口
# =============================================================================

class LowerLevelEvaluator:
    """
    下层控制器评估器

    给定排程和控制器，运行仿真并返回利润（=累计奖励）。
    支持三种控制器类型：pid, mpc, rl
    """

    def __init__(
        self,
        controller_type: str,
        config_dir: str,
        rl_model_path: Optional[str] = None,
        mpc_Np: int = 4,
        n_eval_repeats: int = 3,
        n_steps: Optional[int] = None,
    ):
        """
        参数:
            controller_type: 'pid' | 'mpc' | 'rl'
            config_dir: 配置文件目录
            rl_model_path: RL模型路径（rl模式必需）
            mpc_Np: MPC预测步数（mpc模式使用，默认4加快评估）
            n_eval_repeats: 每次BO评估的仿真重复次数
            n_steps: 仿真步数（默认=一个定植周期）
        """
        self.controller_type = controller_type
        self.config_dir = config_dir
        self.rl_model_path = rl_model_path
        self.mpc_Np = mpc_Np
        self.n_eval_repeats = n_eval_repeats
        self.n_steps = n_steps
        self._load_configs()
        self._init_controller()

    def _load_configs(self):
        """加载配置文件"""
        self.configs = load_all_configs(self.config_dir)
        self.env_config = {
            'schedule': {'t1': 14, 't2': 21, 'rho2': 35.0, 'A1_A2': 0.5},
            'seed': 42,
            'dt': 3600.0,
        }
        for key in ['container_params', 'crop_params', 'equipment_params',
                    'reward_params', 'schedule_params']:
            if key in self.configs:
                self.env_config[key] = self.configs[key]

        self.schedule_params = self.configs.get('schedule_params', {})
        self.schedule_params.setdefault('rho1_min', 30)
        self.schedule_params.setdefault('rho1_max', 60)
        self.schedule_params.setdefault('A_total', 40.0)
        self.schedule_params.setdefault('rho2_min', 20.0)
        self.schedule_params.setdefault('rho2_max', 80.0)
        self.schedule_params.setdefault('A1_A2_min', 0.1)
        self.schedule_params.setdefault('A1_A2_max', 5.0)
        self.schedule_params.setdefault('t1_min', 10)
        self.schedule_params.setdefault('t1_max', 18)
        self.schedule_params.setdefault('t2_min', 18)
        self.schedule_params.setdefault('t2_max', 26)

        self.rp = self.configs.get('reward_params', {})
        self.ep = self.configs.get('equipment_params', {})

    def _init_controller(self):
        """初始化控制器"""
        if self.controller_type == 'rl':
            if not self.rl_model_path or not os.path.exists(self.rl_model_path):
                raise FileNotFoundError(f"RL模型未找到: {self.rl_model_path}")
            try:
                from stable_baselines3 import PPO
                self.rl_model = PPO.load(self.rl_model_path)
                print(f"[Evaluator] RL模型已加载: {self.rl_model_path}")
            except Exception as e:
                raise RuntimeError(f"无法加载RL模型: {e}")
        elif self.controller_type == 'mpc':
            from src.controllers import build_mpc_config
            configs = self.configs
            dummy_schedule = {'t1': 14, 't2': 21, 'rho2': 35.0, 'A1_A2': 0.5}
            mpc_cfg = build_mpc_config(configs, dummy_schedule)
            self.mpc_controller = PlantFactoryMPC(
                config=mpc_cfg,
                crop_params=self.configs.get('crop_params', {}),
                container_params=self.configs.get('container_params', {}),
                equipment_params=self.configs.get('equipment_params', {}),
                reward_params=self.configs.get('reward_params', {}),
                schedule=dummy_schedule,
                Np=self.mpc_Np,
                verbose=False,
            )
        elif self.controller_type == 'pid':
            self.pid_controller = PIDController(config=self.env_config)
            self.rule_controller = RuleController(config=self.env_config)

    def evaluate_schedule(self, schedule: Dict[str, Any]) -> float:
        """
        在给定排程下评估当前控制器，返回平均累计奖励（利润）。

        参数:
            schedule: 排程参数字典

        返回:
            平均累计奖励（n_eval_repeats次的均值）
        """
        rewards = []
        for rep in range(self.n_eval_repeats):
            seed = 42 + rep
            profit = self._run_single_simulation(schedule, seed)
            rewards.append(profit)
        return float(np.mean(rewards))

    def _run_single_simulation(self, schedule: Dict[str, Any], seed: int) -> float:
        """运行单次仿真"""
        n_steps = self.n_steps or (schedule['t2'] * 24)

        if self.controller_type == 'pid':
            return self._run_pid(schedule, seed, n_steps)
        elif self.controller_type == 'mpc':
            return self._run_mpc(schedule, seed, n_steps)
        elif self.controller_type == 'rl':
            return self._run_rl(schedule, seed, n_steps)
        else:
            raise ValueError(f"Unknown controller type: {self.controller_type}")

    def _run_pid(self, schedule: Dict[str, Any], seed: int, n_steps: int) -> float:
        """PID控制器仿真"""
        env = MultiBatchPlantFactoryEnv(config=self.env_config)
        obs, _ = env.reset(seed=seed, options={'schedule': schedule})

        total_reward = 0.0
        pid = PIDController(config=self.env_config)
        rule = RuleController(config=self.env_config)

        for step in range(n_steps):
            obs_arr = obs if isinstance(obs, np.ndarray) else np.array(obs, dtype=np.float32)
            action = pid.predict(obs_arr)
            rule_action = rule.predict(obs_arr)
            action[0] = rule_action[0]
            action[1] = rule_action[1]
            action[3] = rule_action[3]
            action[4] = rule_action[4]
            action[5] = rule_action[5]

            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            if terminated or step >= n_steps - 1:
                break

        return float(total_reward)

    def _run_mpc(self, schedule: Dict[str, Any], seed: int, n_steps: int) -> float:
        """MPC控制器仿真"""
        from experiments.mpc_control import build_env_config, build_mpc_config
        env_cfg = build_env_config(self.configs, schedule, seed=seed)
        mpc_cfg = build_mpc_config(self.configs, schedule)

        exp = MPCExperiment(
            env_config=env_cfg,
            schedule=schedule,
            mpc_config=mpc_cfg,
            Np=self.mpc_Np,
            verbose=False,
            seed=seed,
            record_detailed=False,
        )

        exp.run(n_steps=n_steps, use_mpc=True, save_trajectory=False, log_interval=9999)
        summary = exp.get_summary()
        return float(summary.get('total_reward', 0.0))

    def _run_rl(self, schedule: Dict[str, Any], seed: int, n_steps: int) -> float:
        """RL控制器仿真"""
        env = MultiBatchPlantFactoryEnv(config=self.env_config)
        obs, _ = env.reset(seed=seed, options={'schedule': schedule})

        total_reward = 0.0

        for step in range(n_steps):
            action, _ = self.rl_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            if terminated or step >= n_steps - 1:
                break

        return float(total_reward)


# =============================================================================
# BO优化器封装
# =============================================================================

class BOWithController:
    """
    带有指定下层控制器的贝叶斯优化器

    在每次BO迭代中，使用 LowerLevelEvaluator 评估排程质量。
    """

    def __init__(
        self,
        evaluator: LowerLevelEvaluator,
        n_iter: int = 40,
        n_initial_points: int = 10,
        seed: int = 42,
    ):
        """
        参数:
            evaluator: LowerLevelEvaluator 实例
            n_iter: BO最大迭代次数
            n_initial_points: 初始随机探索点数
            seed: 随机种子
        """
        self.evaluator = evaluator
        self.n_iter = n_iter
        self.n_initial_points = n_initial_points
        self.seed = seed
        self.schedule_params = evaluator.schedule_params

        self.results = []
        self.best_result = None
        self.best_schedule = None
        self.best_profit = -float('inf')

    def _build_search_space(self):
        """构建搜索空间"""
        from skopt.space import Real, Integer

        sp = self.schedule_params
        dimensions = [
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
        return dimensions

    def optimize(self, verbose: bool = True) -> Dict[str, Any]:
        """
        执行贝叶斯优化

        返回:
            {
                'best_schedule': {...},
                'best_profit': float,
                'convergence': [float, ...],
                'iterations': [[t1, t2, rho2, A1_A2], ...],
            }
        """
        from skopt import gp_minimize
        from skopt.utils import use_named_args

        dimensions = self._build_search_space()

        @use_named_args(dimensions=dimensions)
        def objective(t1, t2, rho2, A1_A2):
            # 可行性检查
            is_feas, rho1 = check_schedule_feasibility(
                t1, t2, rho2, A1_A2, self.schedule_params
            )

            if not is_feas:
                return 1e6  # 惩罚

            # 构建排程
            A_total = self.schedule_params.get('A_total', 40.0)
            A1 = A_total / (1.0 + A1_A2)
            A2 = A_total - A1

            schedule = {
                't1': int(t1),
                't2': int(t2),
                'rho2': float(rho2),
                'A1_A2': float(A1_A2),
                'A1': float(A1),
                'A2': float(A2),
            }

            # 评估
            try:
                profit = self.evaluator.evaluate_schedule(schedule)
                self.results.append({
                    'schedule': schedule,
                    'profit': profit,
                    'continuous': [t1, t2, rho2, A1_A2],
                })

                if profit > self.best_profit:
                    self.best_profit = profit
                    self.best_schedule = schedule.copy()
                    if verbose:
                        print(f"  [BO] New best: profit={profit:+.3f} "
                              f"t1={schedule['t1']} t2={schedule['t2']} "
                              f"rho2={rho2:.1f} A1/A2={A1_A2:.3f}")

                return -profit  # 最小化问题
            except Exception as e:
                if verbose:
                    print(f"  [BO] Evaluation failed: {e}")
                return 1e6

        if verbose:
            print(f"\n[BO-{self.evaluator.controller_type}] Starting optimization...")
            print(f"  n_iter={self.n_iter}, n_initial={self.n_initial_points}")

        res = gp_minimize(
            func=objective,
            dimensions=dimensions,
            n_initial_points=self.n_initial_points,
            n_calls=self.n_iter,
            random_state=self.seed,
            verbose=verbose,
            acq_func='EI',
            kappa=1.96,
            xi=0.01,
        )

        # 构建结果
        best_idx = np.argmin(res.func_vals)
        best_cont = res.x_iters[best_idx]

        A_total = self.schedule_params.get('A_total', 40.0)
        best_A1 = A_total / (1.0 + best_cont[3])
        best_A2 = A_total - best_A1

        self.best_schedule = {
            't1': int(best_cont[0]),
            't2': int(best_cont[1]),
            'rho2': float(best_cont[2]),
            'A1_A2': float(best_cont[3]),
            'A1': float(best_A1),
            'A2': float(best_A2),
        }
        self.best_profit = -res.fun
        self.convergence = [-v for v in res.func_vals]

        return {
            'best_schedule': self.best_schedule,
            'best_profit': self.best_profit,
            'best_continuous': list(best_cont),
            'convergence': self.convergence,
            'iterations': res.x_iters,
        }


# =============================================================================
# 主实验函数
# =============================================================================

def run_bo_comparison(
    modes: List[str],
    n_iter: int = 40,
    n_initial_points: int = 10,
    n_eval_repeats: int = 3,
    mpc_Np: int = 4,
    seed: int = 42,
    rl_model_path: Optional[str] = None,
    config_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    运行BO优化对比实验

    参数:
        modes: ['pid', 'mpc', 'rl'] 或其子集
        n_iter: BO迭代次数
        n_initial_points: 初始探索点数
        n_eval_repeats: 每次排程评估的重复次数
        mpc_Np: MPC预测步数（降低以加速BO）
        seed: 随机种子
        rl_model_path: RL模型路径
        config_dir: 配置文件目录
        save_dir: 保存目录
        verbose: 是否打印详细信息

    返回:
        {controller_type: result_dict}
    """
    if config_dir is None:
        config_dir = os.path.join(project_dir, 'configs')

    results = {}

    for mode in modes:
        print(f"\n{'#'*70}")
        print(f"# Experiment 3: BO + [{mode.upper()}] lower-level controller")
        print(f"{'#'*70}")

        try:
            # 创建评估器
            evaluator = LowerLevelEvaluator(
                controller_type=mode,
                config_dir=config_dir,
                rl_model_path=rl_model_path,
                mpc_Np=mpc_Np,
                n_eval_repeats=n_eval_repeats,
            )

            # 创建BO优化器
            bo = BOWithController(
                evaluator=evaluator,
                n_iter=n_iter,
                n_initial_points=n_initial_points,
                seed=seed,
            )

            # 运行优化
            result = bo.optimize(verbose=verbose)

            # 打印结果
            print(f"\n  [{mode.upper()}] Best profit: {result['best_profit']:+.3f}")
            print(f"  [{mode.upper()}] Best schedule:")
            print(f"    t1={result['best_schedule']['t1']}d "
                  f"t2={result['best_schedule']['t2']}d "
                  f"rho2={result['best_schedule']['rho2']:.1f}株/m² "
                  f"A1={result['best_schedule']['A1']:.1f}m² "
                  f"A2={result['best_schedule']['A2']:.1f}m² "
                  f"A1/A2={result['best_schedule']['A1_A2']:.3f}")

            results[mode] = result

            # 保存结果
            if save_dir:
                mode_save_dir = os.path.join(save_dir, mode)
                os.makedirs(mode_save_dir, exist_ok=True)

                result_path = os.path.join(mode_save_dir, f'bo_{mode}_results.yaml')
                with open(result_path, 'w', encoding='utf-8') as f:
                    yaml.dump(result, f, default_flow_style=False, allow_unicode=True)
                print(f"  [{mode.upper()}] 结果已保存: {result_path}")

        except Exception as e:
            print(f"\n  [ERROR] {mode.upper()} 实验失败: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            results[mode] = {'error': str(e)}

    # 汇总对比
    if len(results) > 1:
        _print_bo_comparison_table(results)

    return results


def _print_bo_comparison_table(results: Dict[str, Dict[str, Any]]):
    """打印BO优化对比表"""
    print(f"\n{'='*85}")
    print(f"  BO + Lower-Level Controller 对比汇总")
    print(f"{'='*85}")
    print(f"  {'Controller':>16} | {'BestProfit':>12} | "
          f"{'t1':>4} | {'t2':>4} | {'rho2':>6} | {'A1/A2':>6}")
    print(f"  {'-'*16}-+-{'-'*12}-+-{'-'*4}-+-{'-'*4}-+-{'-'*6}-+-{'-'*6}")

    for mode, result in results.items():
        if 'error' in result:
            print(f"  {mode.upper():>16} | {'FAILED':>12} | - | - | - | -")
            continue
        sched = result.get('best_schedule', {})
        profit = result.get('best_profit', 0)
        print(f"  {mode.upper():>16} | {profit:>+12.3f} | "
              f"{sched.get('t1','?'):>4} | {sched.get('t2','?'):>4} | "
              f"{sched.get('rho2', 0):>6.1f} | {sched.get('A1_A2', 0):>6.3f}")

    print(f"{'='*85}")

    # 保存对比汇总
    rows = []
    for mode, result in results.items():
        if 'error' in result:
            continue
        sched = result.get('best_schedule', {})
        rows.append({
            'controller': mode,
            'best_profit': result.get('best_profit', 0),
            'best_continuous': str(result.get('best_continuous', [])),
            't1': sched.get('t1', 0),
            't2': sched.get('t2', 0),
            'rho2': sched.get('rho2', 0),
            'A1_A2': sched.get('A1_A2', 0),
            'A1': sched.get('A1', 0),
            'A2': sched.get('A2', 0),
        })

    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame()


# =============================================================================
# 命令行接口
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='BO + 下层控制器联合优化对比实验',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 三种控制器完整对比
  python experiments/bo_layer_comparison.py --modes pid mpc rl --n_iter 30

  # 仅用MPC（精确评估）
  python experiments/bo_layer_comparison.py --modes mpc --n_iter 30 --mpc_Np 8

  # 指定RL模型
  python experiments/bo_layer_comparison.py --modes rl --rl_model results/models/best_model.zip

  # 完整运行 + 保存 + 可视化
  python experiments/bo_layer_comparison.py --modes pid mpc rl --n_iter 30 --save --viz
        """
    )

    # 控制器模式
    parser.add_argument('--modes', type=str, nargs='+',
                       default=['pid', 'mpc', 'rl'],
                       choices=['pid', 'mpc', 'rl'],
                       help='使用的下层控制器类型')

    # BO参数
    parser.add_argument('--n_iter', type=int, default=30,
                       help='BO迭代次数（默认30，推荐至少20）')
    parser.add_argument('--n_initial_points', type=int, default=8,
                       help='BO初始探索点数（默认8）')
    parser.add_argument('--n_eval_repeats', type=int, default=3,
                       help='每次排程评估的重复次数（默认3）')
    parser.add_argument('--mpc_Np', type=int, default=4,
                       help='MPC预测步数（BO中用小值加速，默认4）')

    # 随机种子
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    # RL模型
    parser.add_argument('--rl_model', type=str, default=None,
                       help='训练好的PPO模型路径 (.zip)')

    # 保存与可视化
    parser.add_argument('--save', action='store_true', help='保存结果')
    parser.add_argument('--save_dir', type=str, default='results/exp3_bo_comparison',
                       help='结果保存目录')
    parser.add_argument('--viz', action='store_true', help='生成可视化')
    parser.add_argument('--quiet', action='store_true', help='静默模式')
    parser.add_argument('--config_dir', type=str, default=None, help='配置文件目录')

    return parser.parse_args()


def main():
    args = parse_args()

    config_dir = args.config_dir or os.path.join(project_dir, 'configs')
    if not os.path.exists(config_dir):
        print(f"Error: Config directory not found: {config_dir}")
        return

    verbose = not args.quiet
    save_dir = args.save_dir if args.save else None

    if verbose:
        print(f"\n{'#'*80}")
        print(f"# Experiment 3: BO + Lower-Level Controller Comparison")
        print(f"# Modes: {args.modes}")
        print(f"# BO: n_iter={args.n_iter}, n_init={args.n_initial_points}")
        print(f"# Evaluation: repeats={args.n_eval_repeats}, MPC Np={args.mpc_Np}")
        print(f"# RL Model: {args.rl_model or '(none)'}")
        print(f"# Seed: {args.seed}")
        print(f"# Save: {save_dir or '(not saving)'}")
        print(f"{'#'*80}")

    results = run_bo_comparison(
        modes=args.modes,
        n_iter=args.n_iter,
        n_initial_points=args.n_initial_points,
        n_eval_repeats=args.n_eval_repeats,
        mpc_Np=args.mpc_Np,
        seed=args.seed,
        rl_model_path=args.rl_model,
        config_dir=config_dir,
        save_dir=save_dir,
        verbose=verbose,
    )

    # 保存对比汇总CSV
    if save_dir and len(results) > 1:
        rows = _print_bo_comparison_table(results)
        if not rows.empty:
            comp_path = os.path.join(save_dir, 'bo_comparison_summary.csv')
            rows.to_csv(comp_path, index=False)
            print(f"\n对比汇总已保存: {comp_path}")

    # 可视化
    if args.viz and save_dir:
        try:
            from visualizations.experiment_viz import plot_bo_convergence
            print(f"\n--- Generating BO convergence visualizations ---")
            plot_bo_convergence(
                results_dir=save_dir,
                save_dir=os.path.join(save_dir, 'figures'),
            )
        except Exception as e:
            print(f"[WARN] 可视化失败: {e}")

    print(f"\n完成!")


if __name__ == '__main__':
    main()
