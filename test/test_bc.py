from dataclasses import dataclass

from deep_hvac import agent, behavioral_clone, ppo, runner

from easyrl.utils.gym_util import make_vec_env
from easyrl.configs import cfg
import gym
import numpy as np
import pandas as pd
import torch
from torch import nn
from easyrl.utils.torch_util import (
    action_entropy, action_from_dist, action_log_prob, move_to,
    torch_float, torch_to_np
)

from unittest import TestCase


class TestBehavioralClone(TestCase):

    def setUp(self):
        _, self.env_name = runner.make_default_env(
            terminate_on_discomfort=False, create_expert=False,
            discrete_action=True, season='summer'
        )
        self.env = make_vec_env(self.env_name, 1, 0)
        try:
            self.trajectories = pd.read_pickle(
                'test/fixtures/expert-traj-summer.pickle')
        except FileNotFoundError:
            self.trajectories = None

    def mimic(self):
        """Actor that is taught to mimic the expert."""
        agent, _ = ppo.train_ppo(
            env_name=self.env_name, save_dir='tmp', train=False
        )
        return BasicAgent(agent.actor, env=agent.env)

    def test_generate_demo_data(self):
        expertagent = agent.NaiveAgent(env=self.env)
        behavioral_clone.set_configs(self.env_name)
        if self.trajectories is None:
            trajectories = behavioral_clone.generate_demonstration_data(
                expertagent, self.env, 10)
            pd.to_pickle(
                trajectories, 'test/fixtures/expert-traj-summer.pickle')
        self.assertEqual(len(trajectories), 10)
        self.assertEqual(len(trajectories[0].states), 24 * 30)
        self.assertEqual(len(np.unique(trajectories[0].actions)), 4)

    def test_train_bc_agent(self):
        expertagent = agent.NaiveAgent(env=self.env)
        behavioral_clone.set_configs(self.env_name)
        if self.trajectories is None:
            trajectories = behavioral_clone.generate_demonstration_data(
                expertagent, self.env, 2)
        else:
            trajectories = self.trajectories
        agent_cloned, logs, _ = behavioral_clone.train_bc_agent(
            self.mimic(), trajectories, max_epochs=10)
        torch.save(agent_cloned.actor, 'test/output/cloned-agent-summer.pt')


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
