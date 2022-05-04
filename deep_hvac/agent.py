from dataclasses import dataclass

from deep_hvac import building
from deep_hvac.models import (
    CategoricalPolicyFiltering, DiagGaussianPolicyFiltering
)
from deep_hvac.simulator import SimEnv

from easyrl.agents.base_agent import BaseAgent
from easyrl.configs import cfg
from easyrl.envs.dummy_vec_env import DummyVecEnv
from easyrl.models.mlp import MLP
from easyrl.utils.torch_util import (
    action_entropy, action_from_dist, action_log_prob, move_to,
    torch_float, torch_to_np
)
import gym
import numpy as np
import torch
from torch import nn


class NaiveAgent(BaseAgent):
    """
    Agent that does thermostat setback to set temperatures during
    unoccupied hours.
    """
    action_shift = 0

    def __init__(self, *args, **kwargs):
        if not kwargs.get('skip_parent_init'):
            super().__init__(*args, **kwargs)

    def get_action(self, observation, sample=True, *args, **kwargs):
        if isinstance(observation, (list, tuple)):
            observation = np.array(observation)
        if len(observation.shape) == 1:
            observation = np.expand_dims(observation, 0)
        is_occupied = observation[:, SimEnv.state_idx['occupancy_ahead_0']]
        is_occupied = is_occupied.astype(bool)
        action = np.zeros((observation.shape[0], 2))
        action[is_occupied, :] = [
                building.OCCUPIED_HEATING_STPT, building.OCCUPIED_COOLING_STPT
            ]
        action[~is_occupied, :] = [
                building.UNOCCUPIED_HEATING_STPT,
                building.UNOCCUPIED_COOLING_STPT
            ]

        return action, None


class AshraeComfortAgent(BaseAgent):
    """
    Agent that sets thermostat to +/- 2.5 ASHRAE comfort temperature limits
    during occupied hours and +/-3.5 outside the comfort temperature during
    unoccupied hours.
    """
    action_shift = 0

    def __init__(self, *args, **kwargs):
        if not kwargs.get('skip_parent_init'):
            super().__init__(*args, **kwargs)

    def get_action(self, observation, sample=True, *args, **kwargs):
        """
        :return array: heating and cooling stpt
        """
        if isinstance(observation, (list, tuple)):
            observation = np.array(observation)
        if len(observation.shape) == 1:
            observation = np.expand_dims(observation, 0)
        is_occupied = observation[:, SimEnv.state_idx['occupancy_ahead_0']]
        outdoor_temperature = observation[
            :, SimEnv.state_idx['outdoor_temperature']]
        is_occupied = is_occupied.astype(bool)

        if isinstance(observation, (list, tuple)):
            observation = np.array(observation)

        comfort_t = building.comfort_temperature(outdoor_temperature).item()

        if is_occupied.item():
            action = np.array([comfort_t - 2.5, comfort_t + 2.5])
        else:
            action = np.array([comfort_t - 3.5, comfort_t + 3.5])
        return action, None


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


class BasicAgentStateSubset(BaseAgent):
    """
    Agent that takes a subset of the state observations to return
    an action.
    """

    def __init__(self, state_indices, env, actor=None, hidden_size=256,
                 categorical=True, **kwargs):
        self.state_indices = state_indices
        self.hidden_size = hidden_size
        self.categorical = categorical
        super().__init__(env=env)

        # Make actor
        if isinstance(env, DummyVecEnv):
            self.action_size = env.envs[0].action_size
        else:
            self.action_size = env.action_size
        self.ob_size_env = env.observation_space.shape[0]
        self.ob_size = len(state_indices)

        if actor is None:
            self._make_actor()
        else:
            self.actor = actor

    def _make_actor(self):
        body = MLP(input_size=self.ob_size,
                   hidden_sizes=[self.hidden_size, self.hidden_size],
                   output_size=self.hidden_size,
                   hidden_act=nn.Tanh,
                   output_act=nn.Tanh)
        if self.categorical:
            self.actor = CategoricalPolicyFiltering(
                state_indices=self.state_indices,
                body_net=body,
                in_features=self.hidden_size,
                action_dim=self.action_size
            )
        else:
            self.actor = DiagGaussianPolicyFiltering(
                state_indices=self.state_indices,
                body_net=body,
                in_features=self.hidden_size,
                actio_dim=self.action_size
            )

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
