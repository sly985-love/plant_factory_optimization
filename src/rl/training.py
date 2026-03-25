# -*- coding: utf-8 -*-
"""
RL训练模块

使用stable-baselines3的PPO算法训练上下文强化学习策略。

主要功能:
1. 环境向量化（多环境并行）
2. PPO训练循环
3. WandB日志集成
4. 模型保存与加载

来源: 论文方法部分 2.3.4
"""

import numpy as np
import gymnasium as gym
from typing import Optional, Dict, Any, List, Tuple

import os
import sys
import time
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import CallbackList, BaseCallback, EvalCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    STABLE_BASELINES3_AVAILABLE = True
except ImportError:
    STABLE_BASELINES3_AVAILABLE = False

try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class TrainingCallback(BaseCallback):
    """
    自定义训练回调

    用于记录额外指标和日志
    """

    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        # 记录训练步数
        if self.num_timesteps % self.log_freq == 0:
            # 计算最近episode的平均奖励
            if len(self.episode_rewards) > 0:
                mean_reward = np.mean(self.episode_rewards[-100:])
                self.logger.record('train/mean_reward_100', mean_reward)

        return True

    def _on_rollout_end(self) -> None:
        pass


class ContextualPPOTrainer:
    """
    上下文PPO训练器

    用于训练能够泛化到不同排程的RL策略
    """

    def __init__(
        self,
        env_config: Dict[str, Any],
        rl_params: Dict[str, Any],
        results_dir: str = 'results'
    ):
        """
        初始化训练器

        参数:
            env_config: 环境配置字典（包含schedule_params用于采样）
            rl_params: PPO超参数字典
            results_dir: 结果保存目录
        """
        self.env_config = env_config
        self.rl_params = rl_params
        self.results_dir = results_dir
        self.model = None
        self.train_logger = None

        # 排程采样范围（从配置读取）
        sp = env_config.get('schedule_params', {})
        self._t1_min = sp.get('t1_min', 10)
        self._t1_max = sp.get('t1_max', 18)
        self._t2_min = sp.get('t2_min', 18)
        self._t2_max = sp.get('t2_max', 26)
        self._rho2_min = sp.get('rho2_min', 20.0)
        self._rho2_max = sp.get('rho2_max', 80.0)
        self._A1_A2_min = sp.get('A1_A2_min', 0.1)
        self._A1_A2_max = sp.get('A1_A2_max', 5.0)

        self._setup_wandb()

    def _setup_wandb(self):
        """设置WandB日志"""
        if WANDB_AVAILABLE and self.rl_params.get('use_wandb', False):
            wandb.init(
                project=self.rl_params.get('wandb_project', 'plant_factory_optimization'),
                name=self.rl_params.get('wandb_run_name', 'contextual_ppo'),
                config={
                    'rl_params': self.rl_params,
                    'env_config': self.env_config,
                }
            )
            self.train_logger = wandb
        else:
            self.train_logger = None

    def _make_env(self, seed: int = 0):
        """创建单个环境"""
        from envs import MultiBatchPlantFactoryEnv
        env = MultiBatchPlantFactoryEnv(self.env_config)
        env.reset(seed=seed)
        return env

    def _sample_schedule(self) -> Dict[str, Any]:
        """
        从排程空间采样随机排程（连续范围）

        【重要】rho2 和 A1_A2 现在是连续范围，不再是离散集合。
        所有范围值从初始化时从配置读取，无则用合理默认值。
        """
        t1 = np.random.randint(self._t1_min, self._t1_max + 1)
        t2 = np.random.randint(self._t2_min, self._t2_max + 1)
        # 连续范围采样：rho2 和 A1_A2 可以在范围内取任意浮点值
        rho2 = np.random.uniform(self._rho2_min, self._rho2_max)  # [株/m²]
        A1_A2 = np.random.uniform(self._A1_A2_min, self._A1_A2_max)   # [-]

        return {'t1': t1, 't2': t2, 'rho2': rho2, 'A1_A2': A1_A2}

    def _make_contextual_env(self, seed: int = 0):
        """
        创建上下文环境

        每个环境使用不同的排程
        """
        from envs import MultiBatchPlantFactoryEnv
        env = MultiBatchPlantFactoryEnv(self.env_config)
        schedule = self._sample_schedule()
        options = {'schedule': schedule}
        env.reset(seed=seed, options=options)
        return env

    def train(self) -> PPO:
        """
        执行训练

        返回:
            model: 训练好的PPO模型
        """
        if not STABLE_BASELINES3_AVAILABLE:
            raise RuntimeError("stable-baselines3未安装")

        # 创建向量化环境
        n_envs = self.rl_params.get('n_envs', 16)
        vec_env = make_vec_env(
            self._make_contextual_env,
            n_envs=n_envs,
            seed=self.rl_params.get('seed', 42),
            vec_env_cls=SubprocVecEnv,
        )

        # 创建评估环境
        eval_env = DummyVecEnv([self._make_contextual_env])

        # 创建回调
        callbacks = []

        # WandB回调
        if WANDB_AVAILABLE and self.rl_params.get('use_wandb', False):
            wandb_cb = WandbCallback(
                model_save_path=os.path.join(self.results_dir, 'models'),
                verbose=2,
            )
            callbacks.append(wandb_cb)

        # 评估回调
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(self.results_dir, 'models/best'),
            log_path=os.path.join(self.results_dir, 'logs'),
            eval_freq=self.rl_params.get('eval_freq', 100000),
            n_eval_episodes=self.rl_params.get('n_eval_episodes', 10),
            deterministic=True,
            render=False,
        )
        callbacks.append(eval_cb)

        # 自定义回调
        custom_cb = TrainingCallback(
            log_freq=self.rl_params.get('wandb_log_interval', 1000)
        )
        callbacks.append(custom_cb)

        callback = CallbackList(callbacks)

        # 创建PPO模型
        policy_arch = tuple(self.rl_params.get('policy_net_arch', [256, 256]))

        model = PPO(
            'MlpPolicy',
            vec_env,
            learning_rate=self.rl_params.get('learning_rate', 3e-4),
            n_steps=self.rl_params.get('n_steps', 2048),
            batch_size=self.rl_params.get('batch_size', 2048),
            n_epochs=self.rl_params.get('n_epochs', 10),
            gamma=self.rl_params.get('gamma', 0.99),
            gae_lambda=self.rl_params.get('gae_lambda', 0.95),
            clip_range=self.rl_params.get('clip_range', 0.2),
            ent_coef=self.rl_params.get('ent_coef', 0.01),
            vf_coef=self.rl_params.get('vf_coef', 0.5),
            max_grad_norm=self.rl_params.get('max_grad_norm', 0.5),
            policy_kwargs=dict(net_arch=policy_arch),
            tensorboard_log=os.path.join(self.results_dir, 'tensorboard'),
            seed=self.rl_params.get('seed', 42),
            verbose=1,
        )

        # 训练
        total_timesteps = self.rl_params.get('total_timesteps', 5000000)
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=self.rl_params.get('wandb_log_interval', 1000),
            progress_bar=True,
        )

        self.model = model

        # 保存最终模型
        model.save(os.path.join(self.results_dir, 'models/final_model'))

        # 关闭WandB
        if WANDB_AVAILABLE and self.rl_params.get('use_wandb', False):
            wandb.finish()

        return model

    def save(self, path: str):
        """保存模型"""
        if self.model is not None:
            self.model.save(path)

    def load(self, path: str) -> PPO:
        """加载模型"""
        if not STABLE_BASELINES3_AVAILABLE:
            raise RuntimeError("stable-baselines3未安装")
        self.model = PPO.load(path)
        return self.model

    def evaluate(
        self,
        n_episodes: int = 10,
        schedules: Optional[List[Dict[str, Any]]] = None,
        deterministic: bool = True
    ) -> Dict[str, Any]:
        """
        评估模型性能

        参数:
            n_episodes: 评估回合数
            schedules: 评估用的排程列表
            deterministic: 是否使用确定性策略

        返回:
            评估结果字典
        """
        if self.model is None:
            raise RuntimeError("模型未训练或加载")

        from envs import MultiBatchPlantFactoryEnv

        all_rewards = []
        all_ep_lengths = []
        all_info = []

        for i in range(n_episodes):
            # 选择排程
            if schedules and i < len(schedules):
                schedule = schedules[i]
            else:
                schedule = self._sample_schedule()

            env = MultiBatchPlantFactoryEnv(self.env_config)
            obs, _ = env.reset(options={'schedule': schedule})

            done = False
            total_reward = 0
            ep_len = 0
            ep_info = []

            while not done:
                action, _ = self.model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                ep_len += 1
                ep_info.append(info)

            all_rewards.append(total_reward)
            all_ep_lengths.append(ep_len)
            all_info.append(ep_info)

        return {
            'mean_reward': np.mean(all_rewards),
            'std_reward': np.std(all_rewards),
            'mean_ep_length': np.mean(all_ep_lengths),
            'std_ep_length': np.std(all_ep_lengths),
            'rewards': all_rewards,
            'ep_lengths': all_ep_lengths,
        }
