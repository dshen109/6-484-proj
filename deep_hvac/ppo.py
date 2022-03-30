from easyrl.agents.ppo_agent import PPOAgent
from easyrl.configs import cfg
from easyrl.configs import set_config
from easyrl.configs.command_line import cfg_from_cmd
from easyrl.engine.ppo_engine import PPOEngine
from easyrl.models.categorical_policy import CategoricalPolicy
from easyrl.models.diag_gaussian_policy import DiagGaussianPolicy
from easyrl.models.mlp import MLP
from easyrl.models.value_net import ValueNet
from easyrl.runner.nstep_runner import EpisodicRunner
from easyrl.utils.common import set_random_seed
from easyrl.utils.gym_util import make_vec_env
from easyrl.utils.common import load_from_json

import torch
from torch import nn
from pathlib import Path
import gym

def train_ppo(env_name='DefaultBuilding', max_steps=100000):    
    set_config('ppo')
    cfg.alg.num_envs = 1

    # number of hours in a year
    cfg.alg.episode_steps = 1024
    cfg.alg.log_interval = 1
    cfg.alg.eval_interval = 20
    
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
    ob_size = env.observation_space

    actor_body = MLP(input_size=ob_size,
                     hidden_sizes=[64, 64],
                     output_size=64,
                     hidden_act=nn.Tanh,
                     output_act=nn.Tanh)

    critic_body = MLP(input_size=ob_size,
                     hidden_sizes=[64, 64],
                     output_size=64,
                     hidden_act=nn.Tanh,
                     output_act=nn.Tanh)
    
    # if isinstance(env.action_space, gym.spaces.Discrete):
    #     act_size = 2
    #     actor = CategoricalPolicy(actor_body,
    #                              in_features=64,
    #                              action_dim=act_size)
    # elif isinstance(env.action_space, gym.spaces.Box):
    #     act_size = 2
    #     actor = DiagGaussianPolicy(actor_body,
    #                                in_features=64,
    #                                action_dim=act_size,
    #                                tanh_on_dist=cfg.alg.tanh_on_dist,
    #                                std_cond_in=cfg.alg.std_cond_in)
    # else:
    #     raise TypeError(f'Unknown action space type: {env.action_space}')

    act_size = 2
    actor = DiagGaussianPolicy(actor_body,
                                in_features=64,
                                action_dim=act_size,
                                tanh_on_dist=cfg.alg.tanh_on_dist,
                                std_cond_in=cfg.alg.std_cond_in)

    critic = ValueNet(critic_body, in_features=64)
    agent = PPOAgent(actor=actor, critic=critic, env=env)
    runner = EpisodicRunner(agent=agent, env=env)
    engine = PPOEngine(agent=agent,
                       runner=runner)
    engine.train()

    return agent, cfg.alg.save_dir