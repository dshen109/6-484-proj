from deep_hvac import runner

from unittest import TestCase


class TestDefaultEnv(TestCase):

    def test_default(self):
        env = runner.make_default_env()
        self.assertTrue(env.expert_performance is not None)
