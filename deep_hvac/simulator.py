from collections import defaultdict
import datetime as dt
import os

from gym import Env, spaces
import numpy as np
import pandas as pd

from deep_hvac import logger
from deep_hvac.building import make_default_building, comfort_temperature
from deep_hvac.util import sun_position


class SimConfig():

    def __init__(self, episode_length=24 * 30,
                 terminate_on_discomfort=False,
                 discomfort_penalty=100):
        """
        :param int episode_length: Number of hours for each episode
        :param bool terminate_on_discomfort: Whether to terminate the
            episode if temperature bounds are violated during occupant
            hours
        :param float discomfort_penalty: Penalty multiplier for violating
            temperature bounds
        """

        self.episode_length = episode_length
        self.terminate_on_discomfort = terminate_on_discomfort
        self.discomfort_penalty = discomfort_penalty


class SimEnv(Env):
    """Simulation environment

    :ivar int time: Hour of the year
    """
    occupancy_lookahead = 3

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

        inf = float('inf')
        self.observation_space = spaces.Box(
            low=np.array([-inf, -inf, -inf, -inf, -inf, -inf]),
            high=np.array([inf, inf, inf, inf, inf, inf]),
            dtype=np.float32)
        # First element is heating stpt, second element is cooling stpt.
        self.action_space = spaces.Box(low=np.array([10, 10]),
                                       high=np.array([40, 40]),
                                       dtype=np.float32)

        self.action_space
        self.prices = prices
        self.weather = weather
        self.agent = agent
        self.latitude, self.longitude = coords
        self.zone = zone
        self.windows = windows
        self.expert_performance = expert_performance
        self.config = config
        self.random = np.random.default_rng(0)

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

        # reset the temperature of the building mass
        self.t_m_prev = (self.zone.t_set_heating + self.zone.t_set_cooling) / 2
        self.step_bulk()

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
        current_weather = self.get_weather(self.time)

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
        self.zone.solve_energy(
            internal_gains=occupant_gains,
            solar_gains=sum([window.solar_gains for window in self.windows]),
            t_out=t_out,
            t_m_prev=self.t_m_prev
        )
        self.t_m_prev = self.zone.t_m_next

        self.time += 1

        return timestamp, t_out

    def step(self, action):
        """
        Step from time t to t + 1, action is applied to time t and returns
            state corresponding to t + 1

        :param Array action: new setpoint for heating and
            cooling temperature
        :return: state, reward, done, info
        """
        self.zone.t_set_heating = action[0]
        self.zone.t_set_cooling = action[1]

        # timestamp is the t time
        timestamp, t_out = self.step_bulk()

        for attr in ('heating_demand', 'heating_sys_electricity',
                     'cooling_demand', 'cooling_sys_electricity',
                     'electricity_out', 't_air'):
            self.results[attr].append(getattr(self.zone, attr))
        self.results['t_inside'].append(self.zone.t_air)
        self.results['timestamp'].append(timestamp)
        self.results['t_outside'].append(t_out)

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
            timestamp, electricity_cost, self.zone.t_air)
        self.ep_reward += reward
        self.results['reward'].append(reward)
        self.results['set_heating'].append(self.zone.t_set_heating)
        self.results['set_cooling'].append(self.zone.t_set_cooling)

        self.timestep += 1
        info['success'] = False

        terminate = (
            info['discomfort_termination'] or
            self.timestep >= self.config.episode_length)

        return self.get_state(), reward, terminate, info

    def get_state(self):
        """If self.time is at index t, returns:

            - t_set_heating for t-1
            - t_set_cooling for t-1
            - t_out (outside air temperature) for t
            - t_air (inside air tmperature) for t-1
            - hour for t
            - weekday for t
            - electricity price for t - 1
            - direct normal radiation for t
            - diffuse horizontal radiation for t
            - building occupancy boolean (1 or 0) for hours t to
                occupancy_lookahead
        """
        weather = self.get_weather(self.time)
        occupancy = []
        for ahead in range(self.occupancy_lookahead):
            occupancy.append(self.is_occupied(self.time + ahead))

        return [
            self.zone.t_set_heating,
            self.zone.t_set_cooling,
            self.get_outdoor_temperature(self.time),
            self.zone.t_air,
            self.get_timestamp(self.time).hour,
            self.get_timestamp(self.time).weekday(),
            self.get_avg_hourly_price(self.get_timestamp(self.time - 1)),
            weather['DNI_Wm2'],
            weather['DHI_Wm2'],
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
        """Get outdoor air temperature at time *index*"""
        return self.weather.iloc[index].loc['Temperature']

    def get_reward(self, timestamp, electricity_cost, t_air):
        """Return tuple of reward and info dict.

        :param Timestamp timestamp:
        :param float electricity_cost: Total cost paid for electricity
        :param float t_air: Air temperature
        :param float lam: Air temperature penalty.
        """
        # If we have an expert, the reward is how much better we did than the
        # expert
        reward_from_expert_improvement = (
            self.expert_price_paid(timestamp) - electricity_cost)
        # penalty only applies if it is a weekday and between 8am - 6pm
        reward_from_discomfort = 0
        discomf_terminate = False
        if self.is_occupied(timestamp):
            discomf_score = self.comfort_penalty(timestamp, t_air)
            reward_from_discomfort += (
                discomf_score * self.config.discomfort_penalty
            )
            if discomf_score == 1 and self.config.terminate_on_discomfort:
                discomf_terminate = True
        reward = reward_from_expert_improvement + reward_from_discomfort
        info = {
            'reward': reward, 'discomfort_termination': discomf_terminate,
            'reward_from_expert_improvement': reward_from_expert_improvement,
            'reward_from_discomfort': reward_from_discomfort,
            'discomfort_score': discomf_score}
        return reward, info

    def comfort_penalty(self, timestamp, t_air):
        """Calculate comfort penalty. Is 0.5 if if >2.5 degrees away from
        comfort temperature and 1 if >3.5 degrees away from comfort
        temperature."""
        outdoor = self.get_outdoor_temperature(self.get_index(timestamp))
        t_comf = comfort_temperature(outdoor)
        deviation = np.abs(t_air - t_comf)
        if deviation < 2.5:
            return 0
        elif deviation < 3.5:
            return 0.5
        else:
            return 1

    def is_occupied(self, timestamp):
        if isinstance(timestamp, int):
            timestamp = self.get_timestamp(timestamp)
        is_weekday = timestamp.weekday() < 5
        hour = timestamp.hour

        return is_weekday and (hour >= 8 or hour <= 6 + 12)

    def expert_price_paid(self, timestamp):
        if self.expert_performance is None:
            return 0
        else:
            return self.expert_performance.loc[timestamp, 'cost']


def make_default_env(episode_length=24 * 30, terminate_on_discomfort=True,
                     discomfort_penalty=100):
    config = SimConfig(
        episode_length=episode_length,
        terminate_on_discomfort=terminate_on_discomfort,
        discomfort_penalty=discomfort_penalty)
    datadir = os.path.join(
        os.path.split(os.path.abspath(__file__))[0],
        '..', 'data'
    )
    logger.debug("Loading NSRDB data...")
    nsrdb = NsrdbReader(os.path.join(datadir, '1704559_29.72_-95.35_2018.csv'))
    logger.debug("Finished loading NSRDB data.")
    logger.debug("Loading Houston price data...")
    ercot = pd.read_pickle(os.path.join(
        datadir, 'houston-2018-prices.pickle'))
    logger.debug("Finished loading price data.")
    # Run the "best case" agent.


def make_testing_env(episode_length=24 * 30, terminate_on_discomfort=False,
                     discomfort_penalty=100):
    """Create 2019 Houston dataset as testing environment."""
    datadir = os.path.join(
        os.path.split(os.path.abspath(__file__))[0],
        '..', 'data'
    )
    logger.debug("Loading NSRDB data...")
    nsrdb = NsrdbReader(os.path.join(datadir, '1704559_29.72_-95.35_2019.csv'))
    logger.debug("Finished loading NSRDB data.")
    logger.debug("Loading Houston price data...")
    ercot = ErcotPriceReader(os.path.join(
        datadir, 'ercot-2019-rt.xlsx'))
    logger.debug("Finished loading price data.")
    # TODO: finish.
