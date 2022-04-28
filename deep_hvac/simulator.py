from collections import defaultdict
import datetime as dt
import torch, copy

from gym import Env, spaces
import numpy as np

from deep_hvac.agent import NaiveAgent, AshraeComfortAgent
from deep_hvac.spaces.multi_discrete import MultiDiscrete
from deep_hvac.building import comfort_temperature
from deep_hvac.util import sun_position

class SimConfig():

    def __init__(self, episode_length=24 * 30,
                 terminate_on_discomfort=False,
                 discomfort_penalty=10, action_bound_penalty=10,
                 discrete_action=False, action_change_penalty=1,
                 comfort_praise=0.1, price_indicator=False,
                 price_indicator_confidence=0.6):
        """
        :param int episode_length: Maximum number of hours for each episode
        :param bool terminate_on_discomfort: Whether to terminate the
            episode if temperature bounds are violated during occupant
            hours
        :param float discomfort_penalty: Penalty multiplier for violating
            temperature bounds
        :param bool shift_action: Shift action by some amount.
        :param int action_change_penalty: Demerit applied to reward for
            changing the HVAC setpoint by more than 0.5 degrees in each
            transition (positive)
        :param float comfort_praise: Reward applied for keeping all the
            occupants comfortable
        :param bool price_indicator: Indicator for whether price increases
            by more than 50%, decreases by more than 50%, or stays the same in
            the next state.
        :param bool price_indicator_confidence: Probability that the price
            indicator gives the correct value
        """
        # Add config to output kWh of electricity used at previous timestep
        # for MPC purposes
        self.episode_length = episode_length
        self.terminate_on_discomfort = terminate_on_discomfort
        self.discomfort_penalty = discomfort_penalty
        self.action_bound_penalty = action_bound_penalty
        self.discrete_action = discrete_action
        self.action_change_penalty = action_change_penalty
        self.comfort_praise = comfort_praise
        self.price_indicator = price_indicator
        self.price_indicator_confidence = price_indicator_confidence


class SimEnv(Env):
    """Simulation environment

    :ivar int time: Hour of the year
    """
    occupancy_lookahead = 3
    # Absolute lower and upper bounds for action
    t_low = 10
    t_high = 40

    state_idx = {
        'heating_setpoint': 0,
        'cooling_setpoint': 1,
        'heating_setpoint_previous': 2,
        'cooling_setpoint_previous': 3,
        'outdoor_temperature': 4,
        'dni': 5,
        'dhi': 6,
        'outdoor_temperature_previous': 7,
        'dni_previous': 8,
        'dhi_previous': 9,
        'indoor_temperature': 10,
        'indoor_temperature_previous': 11,
        'hour': 12,
        'day': 13,
        'month': 14,
        'weekday': 15,
        'price_previous': 16,
        'electricity_consumed': 17,
        'price_indicator': 18,
        'occupancy_ahead_0': 19,
        'occupancy_ahead_1': 20,
        'occupancy_ahead_2': 21
    }

    def __init__(self, prices, weather, agent, coords, zone, windows,
                 config=None, expert_performance=None):
        """
        :param DataFrame prices: 15m electricity prices. DatetimeIndex.
            Is in $/MWh.
        :param DataFrame weather: hourly weather. DatetimeIndex.
        :param Agent agent: RL agent.
        :param tuple coords: (latitude, longitude) of the building
        :param Zone zone: Zone object of the building
        :param Array of Windows in the building
        :param DataFrame expert_performance: DataFrame describing hourly expert
            performance, with column 'cost' describing their electricity cost.
        """
        super(SimEnv, self).__init__()
        self.prices = prices
        self.weather = weather
        self.agent = agent
        self.latitude, self.longitude = coords
        self.zone = zone
        self.windows = windows
        self.expert_performance = expert_performance
        self.config = config
        self.random = np.random.default_rng(0)

        inf = float('inf')
        # todo: Configure observation space based on the lookahead configs
        self.observation_space = spaces.Box(
            low=np.array([-inf] * len(self.state_idx)),
            high=np.array([inf] * len(self.state_idx)),
            dtype=np.float32)
        # First element is heating stpt, second element is cooling stpt.
        if not self.config.discrete_action:
            self.action_space = spaces.Box(
                low=np.array([self.t_low, self.t_low]),
                high=np.array([self.t_high, self.t_high]),
                dtype=np.float32)
        else:
            span = self.t_high - self.t_low + 1
            self.action_space = MultiDiscrete(
                nvec=[span, span], starts=[self.t_low, self.t_low]
            )

        if isinstance(agent, (NaiveAgent, AshraeComfortAgent)) or \
                agent is None:
            self.action_shift = 0
        elif self.config.discrete_action:
            self.action_shift = 0
        else:
            # Add a shift to continuous actions so it doesn't start the
            # thermostat at 0
            self.action_shift = 21

        self.reset()

    def reset(self, time=None):
        """Reset the simulator to a certain time of the year.

        :param int time:
        """
        if time is None:
            # Reset to midnight starting at a random month.
            month = self.random.integers(1, 13)
            timestamp_start = self.get_timestamp(0)
            year = timestamp_start.year
            tzinfo = timestamp_start.tzinfo
            timestamp = dt.datetime(year, month, 1, 0, 0, 0, tzinfo=tzinfo)
            time = self.get_index(timestamp)

        self.time = time
        self.timestep = 0

        # set episode return to 0
        self.ep_reward = 0

        self.zone.t_set_heating = 20
        self.zone.t_set_cooling = 26

        outdoor_air_temperature = self.get_outdoor_temperature(self.time)

        if outdoor_air_temperature > self.zone.t_set_cooling:
            t_inside = self.zone.t_set_cooling
        elif outdoor_air_temperature < self.zone.t_set_heating:
            t_inside = self.zone.t_set_heating
        else:
            t_inside = outdoor_air_temperature

        self.zone.t_air = t_inside

        self.t_air_prev = t_inside
        self.t_set_heating_prev = self.zone.t_set_heating
        self.t_set_cooling_prev = self.zone.t_set_cooling

        # reset the temperature of the building mass to mean of outside and
        # inside as initial approximation
        self.t_m_prev = (outdoor_air_temperature + t_inside) / 2
        self.step_bulk()
        timestamp, cur_weather = self.get_timestamp_and_weather()
        self.cur_state = [
            self.zone.t_set_heating,
            self.zone.t_set_cooling,
            cur_weather['Temperature'],
            self.zone.t_air,
            timestamp.hour,
            timestamp.weekday()
        ]

        # reset the episodes results
        self.results = defaultdict(list)

        return self.get_state()

    def step_bulk(self):
        """Run building simulator to update the bulk building
        mass.

        :return:
            - timestamp that was updated
            - t_out outdoor air temperature.
        """
        timestamp, cur_weather = self.get_timestamp_and_weather()

        altitude, azimuth = sun_position(
            self.latitude, self.longitude, timestamp)

        for window in self.windows:
            # Window simulator actually takes in Wm**2 not Whm**2 like the
            # documentation says.
            window.calc_solar_gains(
                sun_altitude=altitude, sun_azimuth=azimuth,
                normal_direct_radiation=cur_weather['DNI_Wm2'],
                horizontal_diffuse_radiation=cur_weather['DHI_Wm2']
            )
        t_out = self.get_outdoor_temperature(self.time)

        if self.is_occupied(timestamp):
            occupant_gains = 30 * 100
        else:
            occupant_gains = 0

        self.t_air_prev = self.zone.t_air

        self.zone.solve_energy(
            internal_gains=occupant_gains,
            solar_gains=sum([window.solar_gains for window in self.windows]),
            t_out=t_out,
            t_m_prev=self.t_m_prev
        )
        # Electricity consumed in Watts
        self.electricity_consumed = (
            self.zone.heating_sys_electricity +
            self.zone.cooling_sys_electricity
        )

        self.t_m_prev = self.zone.t_m_next

        self.time += 1

        return timestamp, t_out

    def step(self, action):
        """
        Step from time t to t + 1, action is applied to time t and returns
            state corresponding to t + 1

        If the action is outside the action, space, terminate the episode.

        :param Array action: new setpoint for heating and cooling temperature
        :return: state, reward, done, info
        """
        self.t_set_heating_prev = self.zone.t_set_heating
        self.t_set_cooling_prev = self.zone.t_set_cooling

        if np.isscalar(action):
            action = self.discrete_action_to_setpoints(action)

        t_set_heating = action[0] + self.action_shift
        t_set_cooling = action[1] + self.action_shift

        self.zone.t_set_heating = t_set_heating
        self.zone.t_set_cooling = t_set_cooling

        # timestamp is the t time
        timestamp, t_out = self.step_bulk()

        for attr in ('heating_demand', 'heating_sys_electricity',
                     'cooling_demand', 'cooling_sys_electricity',
                     'electricity_out', 't_air'):
            self.results[attr].append(getattr(self.zone, attr))
        self.results['t_inside'].append(self.zone.t_air)
        self.results['t_bulk'].append(self.t_m_prev)
        self.results['timestamp'].append(timestamp)
        self.results['t_outside'].append(t_out)
        self.results['t_comfort'].append(comfort_temperature(t_out))

        self.results['solar_gain'].append(
            sum([window.solar_gains for window in self.windows]))

        # Calculate all electricity consumed and price
        price = self.get_avg_hourly_price(timestamp)
        elec_consumed = (
            self.zone.heating_sys_electricity +
            self.zone.cooling_sys_electricity
        )
        self.results['electricity_price'].append(price)
        # Watts of electricity consumed
        self.results['electricity_consumed'].append(elec_consumed)
        electricity_cost = elec_consumed * price / 1000
        self.results['electricity_cost'].append(electricity_cost)

        # t_out, timestamp.hour and weekday are for the t+1 time.
        # t_set_heating, t_set_cooling, t_air is for the t time
        reward, info = self.get_reward(
            timestamp, electricity_cost, self.zone.t_air, t_set_heating,
            t_set_cooling, self.t_set_heating_prev, self.t_set_cooling_prev)

        action_bound_violation = info['action_bound_violation']

        self.ep_reward += reward

        self.results['reward'].append(reward)
        self.results['set_heating'].append(self.zone.t_set_heating)
        self.results['set_cooling'].append(self.zone.t_set_cooling)

        self.timestep += 1
        info['success'] = False

        terminate = (
            self.time == 365 * 24 or
            info['discomfort_termination'] or
            action_bound_violation or
            self.timestep >= self.config.episode_length)

        return self.get_state(), reward, self.time == 365 * 24 or self.timestep >= self.config.episode_length, info

    def get_state(self):
        """
        If self.time is at index t, returns:

            - t_set_heating for t - 1
            - t_set_cooling for t - 1
            - t_set_heating for t - 2
            - t_set_cooling for t - 2
            - t_out (outside air temperature) for t
            - direct normal radiation for t
            - diffuse horizontal radiation for t
            - t_out (outside air temperature) for t - 1
            - direct normal radiation for t - 1
            - diffuse horizontal radiation for t - 1
            - t_air (inside air temperature) for t - 1
            - t_air (inside air temperature) for t - 2
            - hour for t
            - day of the month for t
            - month for t
            - weekday for t
            - mean electricity price for t - 1
            - electricity consumed (W) in t - 1
            - price indicator for t as a function of t - 1
            - building occupancy boolean (1 or 0) for hours t to
                occupancy_lookahead
        """
        weather = self.get_weather(self.time)
        weather_previous = self.get_weather(self.time - 1)
        occupancy = []
        for ahead in range(self.occupancy_lookahead):
            try:
                occupancy.append(int(self.is_occupied(self.time + ahead)))
            except IndexError:
                # exceeded end of simulation, pad with zeros.
                occupancy.append(0)

        elec_consumed = (
            self.zone.heating_sys_electricity +
            self.zone.cooling_sys_electricity
        )

        return [
            self.zone.t_set_heating,
            self.zone.t_set_cooling,
            self.t_set_heating_prev,
            self.t_set_cooling_prev,
            self.get_outdoor_temperature(self.time),
            weather['DNI_Wm2'],
            weather['DHI_Wm2'],
            weather_previous['Temperature'],
            weather_previous['DNI_Wm2'],
            weather_previous['DHI_Wm2'],
            self.zone.t_air,
            self.t_air_prev,
            self.get_timestamp(self.time).hour,
            self.get_timestamp(self.time).day,
            self.get_timestamp(self.time).month,
            self.get_timestamp(self.time).weekday(),
            self.get_avg_hourly_price(self.get_timestamp(self.time - 1)),
            self.electricity_consumed,
            self.price_indicator(self.time)
        ] + occupancy

    def get_obs(self):
        return self.get_state()

    def get_avg_hourly_price(self, timestamp):
        """
        Calculates the average price of electricity for the current hour.

        :return: Price in $/kWh
        """
        if isinstance(timestamp, int):
            timestamp = self.get_timestamp(timestamp)
        return self.prices.loc[
            timestamp:timestamp + dt.timedelta(minutes=59),
            'Settlement Point Price'].mean() / 1000

    def price_indicator(self, timestamp):
        # TODO: Finish.
        return 0

    def hour_of_year(self, timestamp):
        """Get integer corresponding to the hour of the year for `timestamp`

        :return int:
        """
        year = timestamp.year
        start = dt.datetime(year, 1, 1, 0, 0, 0, tzinfo=timestamp.tzinfo)
        return int((start - timestamp).total_seconds() / 3600)

    def get_weather(self, time):
        """Return Series of weather data."""
        if not isinstance(time, int):
            time = self.index(time)
        return self.weather.iloc[self.time]

    def get_timestamp_and_weather(self):
        """
        Get timestamp and weather data represented by `self.time`

        :return: timestamp, weather data.
        """
        return self.get_timestamp(self.time), self.weather.iloc[self.time]

    def get_index(self, timestamp):
        return self.hour_of_year(timestamp)

    def get_timestamp(self, index):
        return self.weather.index[index]

    def get_outdoor_temperature(self, index):
        """Get outdoor air temperature at time *index*

        :param int index:
        """
        return self.weather.iloc[index].loc['Temperature']

    def get_reward(self, timestamp, electricity_cost, t_air,
                   t_set_heating, t_set_cooling, t_set_heating_prev,
                   t_set_cooling_prev):
        """Return tuple of reward and info dict.

        :param Timestamp timestamp:
        :param float electricity_cost: Total cost paid for electricity
        :param float t_air: Interior air temperature
        """
        # If we have an expert, the reward is how much better we did than the
        # expert
        reward_from_expert_improvement = (
            self.expert_price_paid(timestamp) - electricity_cost)
        # discomfort penalty only applies if it is a weekday and between 8am -
        # 6pm
        reward_from_discomfort = 0
        discomf_terminate = False
        discomf_score = 0
        reward_from_comfort = 0
        action_change_reward = 0
        action_bound_violation_reward = False
        action_bound_violation = False

        if self.is_occupied(timestamp):
            discomf_score = self.comfort_penalty(timestamp, t_air)
            reward_from_discomfort -= (
                discomf_score * self.config.discomfort_penalty
            )
            if discomf_score >= 1 and self.config.terminate_on_discomfort:
                discomf_terminate = True
            if discomf_score == 0:
                reward_from_comfort += self.config.comfort_praise

        if not self.action_valid(t_set_heating, t_set_cooling):
            action_bound_violation_reward -= self.config.action_bound_penalty
            action_bound_violation = True

        if (np.abs(t_set_cooling - t_set_cooling_prev) > 0.5 or
                np.abs(t_set_heating - t_set_heating_prev) > 0.5):
            action_change_reward -= self.config.action_change_penalty

        reward = (
            reward_from_expert_improvement + reward_from_discomfort +
            reward_from_comfort + action_bound_violation_reward +
            action_change_reward
        )
        info = {
            'reward': reward, 'discomfort_termination': discomf_terminate,
            'reward_from_expert_improvement': reward_from_expert_improvement,
            'reward_from_discomfort': reward_from_discomfort,
            'reward_from_comfort': reward_from_comfort,
            'discomfort_score': discomf_score,
            'reward_from_action_bound_violation':
                action_bound_violation_reward,
            'action_bound_violation': action_bound_violation,
            'action_change_reward': action_change_reward}
        return reward, info

    def comfort_penalty(self, timestamp, t_air):
        """
        Calculate ASHRAE comfort penalty.

        Is interpolated between [0.5, 1] if >2.5 or <3.5 degrees away from
        comfort temperature and |diff - 2.5| if >3.5 degrees away from comfort
        temperature.

        A score of 0 is good.

        :param float t_air: Indoor air temperature
        """
        outdoor = self.get_outdoor_temperature(self.get_index(timestamp))
        t_comf = comfort_temperature(outdoor)
        deviation = np.abs(t_air - t_comf)
        if deviation <= 2.5:
            return 0
        elif deviation <= 3.5:
            return np.interp(deviation, [2.5, 3.5], [0.5, 1])
        else:
            return deviation - 2.5

    def is_occupied(self, timestamp):
        if isinstance(timestamp, int):
            timestamp = self.get_timestamp(timestamp)
        is_weekday = timestamp.weekday() < 5
        hour = timestamp.hour

        return is_weekday and (hour >= 8 and hour <= 6 + 12)

    def expert_price_paid(self, timestamp):
        if self.expert_performance is None:
            return 0
        else:
            return self.expert_performance.loc[timestamp, 'cost']

    def discrete_action_to_setpoints(self, action):
        """Given a scalar discrete action, translate it to the setpoint.

        :return: Tuple of heating and cooling.
        """
        span = self.t_high - self.t_low + 1
        action_max = span ** 2
        if action < 0 or action >= action_max:
            raise ValueError(
                f"Discrete action of {action} cannot be mapped to setpoints")
        heating_shift = action // span
        cooling_shift = action % span
        return (self.t_low + heating_shift, self.t_low + cooling_shift)
    @property
    def action_size(self):
        """
        Number of action elements if continuous action space or number of
        categorical actions if discrete.
        """
        if self.config.discrete_action:
            return (self.t_high - self.t_low + 1) ** 2
        else:
            return 2

    def action_valid(self, t_set_heating, t_set_cooling):
        """Return true if action is valid."""
        action = [t_set_heating, t_set_cooling]
        return self.action_space.contains(action) and \
            t_set_heating < t_set_cooling
