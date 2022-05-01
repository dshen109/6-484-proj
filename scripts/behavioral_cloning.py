from dataclasses import dataclass

from deep_hvac import agent, behavioral_clone, ppo, runner

from easyrl.utils.gym_util import make_vec_env
from easyrl.configs import cfg
import gym
import pandas as pd
import torch
from torch import nn
from easyrl.utils.torch_util import (
    action_entropy, action_from_dist, action_log_prob, move_to,
    torch_float, torch_to_np
)


@dataclass
class BasicAgent:
    actor: nn.Module
    env: gym.Env

    def __post_init__(self):
        move_to([self.actor],
                device=cfg.alg.device)

    @torch.no_grad()
    def get_action(self, ob, sample=True, *args, **kwargs):
        t_ob = torch_float(ob, device=cfg.alg.device)
        act_dist, _ = self.actor(t_ob)
        # sample from the distribution
        action = action_from_dist(act_dist,
                                  sample=sample)
        # get the log-probability of the sampled actions
        log_prob = action_log_prob(action, act_dist)
        # get the entropy of the action distribution
        entropy = action_entropy(act_dist, log_prob)
        action_info = dict(
            log_prob=torch_to_np(log_prob),
            entropy=torch_to_np(entropy),
        )
        return torch_to_np(action), action_info


def mimic(env_name):
    """Actor that is taught to mimic the expert."""
    agent, _ = ppo.train_ppo(
        env_name=env_name, save_dir='tmp', train=False
    )
    return BasicAgent(agent.actor, env=agent.env)


if __name__ == '__main__':

    _, env_name = runner.make_default_env(
        terminate_on_discomfort=False, create_expert=False,
        discrete_action=True, season='summer'
    )
    env = make_vec_env(env_name, 1, 0)
    try:
        trajectories = pd.read_pickle(
            'scripts/fixtures/expert-traj-summer.pickle')
    except FileNotFoundError:
        trajectories = None

    expertagent = agent.NaiveAgent(env=env)
    behavioral_clone.set_configs(env_name)
    if trajectories is None:
        trajectories = behavioral_clone.generate_demonstration_data(
            expertagent, env, 2)
    else:
        trajectories = trajectories
    agent_cloned, logs, _ = behavioral_clone.train_bc_agent(
        mimic(env_name), trajectories, max_epochs=1000)
    torch.save(agent_cloned.actor, 'scripts/output/cloned-agent-summer.pt')
