from deep_hvac import agent, behavioral_clone, runner
from deep_hvac.simulator import SimEnv

from easyrl.utils.gym_util import make_vec_env
import pandas as pd
import torch

from unittest import TestCase


class TestStateSubsetAgent(TestCase):

    def setUp(self):
        self.expert_traj = pd.read_pickle(
            'test/fixtures/expert-traj-summer.pickle'
        )
        _, self.env_name = runner.make_default_env(
            terminate_on_discomfort=False, create_expert=False,
            discrete_action=True, season='summer'
        )
        self.env = make_vec_env(self.env_name, 1, 0)
        behavioral_clone.set_configs(self.env_name, exp_name='tmp')

    def test_subset_agent(self):
        # Agent that takes a subset of the space.
        subset_agent = agent.BasicCategoricalAgentStateSubset(
            state_indices=(
                SimEnv.state_idx['hour'],
                SimEnv.state_idx['weekday'],
                SimEnv.state_idx['occupancy_ahead_0']
            ), env=self.env.envs[0]
        )
        env = self.env.envs[0]
        dataset = behavioral_clone.TrajDataset(self.expert_traj[0:1])
        dataloader = behavioral_clone.DataLoader(
            dataset, batch_size=16, shuffle=True)
        for _, sample in enumerate(dataloader):
            states = sample['state'].float()
            expert_actions = sample['action'].float()
            expert_actions = torch.tensor(
                env.continuous_action_to_discrete(
                    expert_actions[:, 0, 0], expert_actions[:, 0, 1]
                )
            )
            action_dist, _ = subset_agent.get_action(states)
            break
