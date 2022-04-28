from deep_hvac import ppo, runner

import numpy as np

from unittest import TestCase


class TestDefaultEnv(TestCase):

    def test_default(self):
        env, _ = runner.make_default_env()
        self.assertTrue(env.expert_performance is not None)

    def test_discrete_actions(self):
        env, env_name = runner.make_default_env(
            expert_performance='data/results-expert.pickle',
            discrete_action=True)
        path = 'fixtures/ppo_model_discrete.pt'
        agent = ppo.load_agent(path, env_name)
        results = runner.get_results(agent, env, max_steps=30 * 24,
                                     episode_steps=30 * 24)
        self.assertEqual(np.array(results['t_air']).shape, (1, 30 * 24))

    def test_continuous_actions(self):
        env, env_name = runner.make_default_env(
            expert_performance='data/results-expert.pickle',
            discrete_action=False)
        path = 'fixtures/ppo_model_continuous.pt'
        agent = ppo.load_agent(path, env_name)
        results = runner.get_results(agent, env, max_steps=30 * 24)
        self.assertEqual(np.array(results['t_air']).shape, (1, 30 * 24))
