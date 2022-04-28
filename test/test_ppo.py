from deep_hvac import ppo
from deep_hvac.runner import make_default_env

from unittest import TestCase


class TestDiscrete(TestCase):

    def test_training(self):
        env, env_name = make_default_env(
            expert_performance='data/results-expert.pickle',
            discrete_action=True)
        ppo_agent, save_dir = ppo.train_ppo(
            env_name=env_name, max_steps=10, policy_lr=1e-3, value_lr=1e-2,
            seed=0, save_dir='tmp')
