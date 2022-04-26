from deep_hvac.spaces import MultiDiscrete

from easyrl.agents.ppo_agent import PPOAgent
from easyrl.configs import cfg
from easyrl.configs import set_config
from easyrl.engine.ppo_engine import PPOEngine
from easyrl.models.categorical_policy import CategoricalPolicy
from easyrl.models.diag_gaussian_policy import DiagGaussianPolicy
from easyrl.models.mlp import MLP
from easyrl.models.value_net import ValueNet
from easyrl.runner.nstep_runner import EpisodicRunner
from easyrl.utils.common import set_random_seed
from easyrl.utils.gym_util import make_vec_env

import torch
from torch import nn
from pathlib import Path
import gym


def train_ppo(env_name='DefaultBuilding-v0', max_steps=100000,
              policy_lr=3e-4, value_lr=1e-3, gae_lambda=0.95,
              rew_discount=0.99, seed=0):
    """
    Note that the environment name must already be registered before running
    this.
    """
    set_config('ppo')
    cfg.alg.num_envs = 1
    cfg.alg.seed = seed

    cfg.alg.episode_steps = 24 * 30
    cfg.alg.log_interval = 1
    cfg.alg.eval_interval = 20

    cfg.alg.policy_lr = policy_lr
    cfg.alg.value_lr = value_lr
    cfg.alg.gae_lambda = gae_lambda
    cfg.alg.rew_discount = rew_discount

    cfg.alg.max_steps = max_steps
    cfg.alg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.alg.env_name = env_name
    cfg.alg.save_dir = Path.cwd().absolute().joinpath('data').as_posix()
    cfg.alg.save_dir += '/' + env_name

    setattr(cfg.alg, 'diff_cfg', dict(save_dir=cfg.alg.save_dir))

    print(f'====================================')
    print(f'      Device:{cfg.alg.device}')
    print(f'      Total number of steps:{cfg.alg.max_steps}')
    print(f'====================================')

    set_random_seed(cfg.alg.seed)

    env = make_vec_env(cfg.alg.env_name,
                       cfg.alg.num_envs,
                       seed=cfg.alg.seed)
    env.reset()
    ob_size = env.observation_space.shape[0]

    if isinstance(env.action_space, MultiDiscrete):
        action_categorical = True
    elif isinstance(env.action_space, gym.spaces.Box):
        action_categorical = False
    else:
        raise TypeError(f'Unknown action space type: {env.action_space}')
    actor = make_actor(ob_size, env.envs[0].action_size,
                       categorical=action_categorical)

    critic = make_critic(ob_size)
    agent = PPOAgent(actor=actor, critic=critic, env=env)
    runner = EpisodicRunner(agent=agent, env=env)
    engine = PPOEngine(agent=agent,
                       runner=runner)
    engine.train()

    return agent, cfg.alg.save_dir


def load_agent(modelpath, env_name):
    env = make_vec_env(env_name, 1)
    set_config('ppo')
    cfg.alg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.alg.num_envs = 1

    actor = make_actor(
        observation_size=env.observation_space.shape[0],
        action_size=env.envs[0].action_size,
        categorical=env.envs[0].config.categorical_action)
    critic = make_critic(env.observation_space.shape[0])
    agent = PPOAgent(actor=actor, critic=critic, env=env)
    agent.load_model(pretrain_model=modelpath)
    return agent


def make_critic(observation_size, in_features=64):
    body = MLP(input_size=observation_size,
               hidden_sizes=[64, 64],
               output_size=64,
               hidden_act=nn.Tanh,
               output_act=nn.Tanh)
    return ValueNet(body, in_features=in_features)


def make_actor(observation_size, action_size,
               categorical=False):
    body = MLP(input_size=observation_size,
               hidden_sizes=[64, 64],
               output_size=64,
               hidden_act=nn.Tanh,
               output_act=nn.Tanh)
    if not categorical:
        return DiagGaussianPolicy(
            body_net=body,
            in_features=64,
            action_dim=action_size,
            tanh_on_dist=cfg.alg.tanh_on_dist,
            std_cond_in=cfg.alg.std_cond_in
        )
    else:
        return CategoricalPolicy(
            body_net=body,
            in_features=64,
            action_dim=action_size
        )

