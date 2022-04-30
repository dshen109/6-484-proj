from deep_hvac import building
from deep_hvac.simulator import SimEnv

from easyrl.agents.base_agent import BaseAgent


class NaiveAgent(BaseAgent):
    """
    Agent that does thermostat setback to set temperatures during
    unoccupied hours.
    """
    action_shift = 0

    def __init__(self, *args, **kwargs):
        pass

    def get_action(self, observation):
        is_occupied = observation[SimEnv.state_idx['occupancy_ahead_0']]
        if is_occupied:
            action = [
                building.OCCUPIED_HEATING_STPT, building.OCCUPIED_COOLING_STPT
            ]
        else:
            action = [
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
        pass

    def get_action(self, observation):
        """
        :return array: heating and cooling stpt
        """
        outdoor_temperature = observation[2]
        is_occupied = observation[SimEnv.state_idx['occupancy_ahead_0']]
        comfort_t = building.comfort_temperature(outdoor_temperature)
        if is_occupied:
            action = [comfort_t - 2.5, comfort_t + 2.5]
        else:
            action = [comfort_t - 3.5, comfort_t + 3.5]
        return action, None
