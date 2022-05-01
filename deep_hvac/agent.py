from operator import is_
from deep_hvac import building
from deep_hvac.simulator import SimEnv

from easyrl.agents.base_agent import BaseAgent
import numpy as np


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

        outdoor_temperature = observation[2]
        is_occupied = observation[SimEnv.state_idx['occupancy_ahead_0']]
        comfort_t = building.comfort_temperature(outdoor_temperature)
        if is_occupied:
            action = [comfort_t - 2.5, comfort_t + 2.5]
        else:
            action = [comfort_t - 3.5, comfort_t + 3.5]
        return action, None
