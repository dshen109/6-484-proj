from collections import defaultdict
import random
from types import SimpleNamespace

from gym import Env, spaces
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from deep_hvac.util import sun_position


class SimConfig(SimpleNamespace):

    def __init__(self, *args, **kwargs):
        pass


class SimEnv(Env):
    """Simulation environment

    :ivar int time: Hour of the year
    """

    def __init__(self, prices, weather, agent, coords, zone, windows,
                 ep_length=1024, discomfort_multiplier=0.2):
        """
        :param DataFrame prices: 15m electricity prices. DatetimeIndex.
        :param DataFrame weather: hourly weather. DatetimeIndex.
        :param Agent agent: RL agent.
        :param tuple coords: (latitude, longitude) of the building
        :param Zone zone: Zone object of the building
        :param Array of Windows in the building
        :param int ep_length: Total length of episode, hours
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
        self.ep_length = ep_length

        self.lambda_discomf = discomfort_multiplier

        self.reset()

    def reset(self, time=5024):
        """Reset the simulator to a certain time of the year.

        :param int time:
        """
        # reset time to a time in the year
        # TODO: Reset to a random time that guarantees a full episode
        self.time = time
        self.timestep = 0

        # set episode return to 0
        self.ep_reward = 0

        self.zone.t_set_heating = 20
        self.zone.t_set_cooling = 25

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

        t_out = self.get_temperature(self.time)

        # TODO: Get internal gains from occupants
        self.zone.solve_energy(
            internal_gains=0,
            solar_gains=sum([window.solar_gains for window in self.windows]),
            t_out=t_out,
            t_m_prev=self.t_m_prev
        )
        self.t_m_prev = self.zone.t_m_next

        self.time += 1

        return timestamp, t_out

    def step(self, action):
        """
        :param Array of size two to determine new setpoint for heating and
        cooling temperature respectively

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
        self.results['timestamp'].append(self.get_timestamp(self.time - 1))

        self.results['solar_gain'].append(
            sum([window.solar_gains for window in self.windows]))

        # Calculate all electricity consumed and price
        price = self.get_avg_hourly_price()
        elec_consumed = (
            self.zone.heating_sys_electricity +
            self.zone.cooling_sys_electricity
        )
        self.results['electricity_consumed'].append(elec_consumed)
        self.results['price'].append(elec_consumed * price)

        # t_out, timestamp.hour and weekday are for the t+1 time.
        # t_set_heating, t_set_cooling, t_air is for the t time
        reward, info = self.get_reward(elec_consumed * price, self.zone.t_air)
        self.ep_reward += reward
        self.results['reward'].append(reward)
        self.results['set_heating'].append(self.zone.t_set_heating)
        self.results['set_cooling'].append(self.zone.t_set_cooling)
        self.timestep += 1
        info['success'] = False

        return self.get_state(), reward, self.timestep >= 1024, info

    def plot_results(self, attr='t_air'):
        annual_results = pd.DataFrame(self.results)
        annual_results[[attr]].plot()
        plt.show()

    def get_state(self):
        """If self.time is at index t, returns:

            - t_set_heating for t-1
            - t_set_cooling for t-1
            - t_out for t
            - t_air for t-1
            - hour for t
            - weekday for t
        """
        return [
            self.zone.t_set_heating,
            self.zone.t_set_cooling,
            self.get_temperature(self.time),
            self.zone.t_air,
            self.get_timestamp(self.time).hour,
            self.get_timestamp(self.time).weekday()
        ]

    def get_obs(self):
        return self.get_state()

    def get_avg_hourly_price(self):
        '''
        Calculates the average price of electricity for the current hour
        '''
        hour_prices = self.prices['Settlement Point Price'][
            self.time*4: self.time*4 + 4]
        return sum(hour_prices)/4

    def get_timestamp_and_weather(self):
        """
        Get timestamp represented by `self.time`

        :return: timestamp, weather data.
        """
        return self.get_timestamp(self.time), self.weather.iloc[self.time]

    def get_timestamp(self, index):
        return self.weather.index[index]

    def get_temperature(self, index):
        return self.weather.iloc[index].loc['Temperature']

    def get_reward(self, price, t_air):
        """Return tuple of reward and info dict.

        :param float price: Total cost paid for electricity
        :param float t_air: Air temperature
        :param float lam: Air temperature penalty.
        """
        reward = - price
        timestamp, _ = self.get_timestamp_and_weather()
        # penalty only applies if it is a weekday and between 8am - 6pm
        is_weekday = timestamp.weekday() < 5
        hour = timestamp.hour
        if is_weekday and (hour < 8 or hour >= 6):
            if t_air < 20 or t_air > 25:
                reward -= 50000
        info = {'reward':  reward}
        return reward, info
