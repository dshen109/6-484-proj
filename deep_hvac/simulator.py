from gym import Env, spaces
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from util import sun_position
import random

class SimEnv(Env):
    """Simulation environment

    :ivar int time: Hour of the year
    """

    def __init__(self, prices, weather, agent, coords, zone, windows):
        """
        TODO: Add lower/upper band for comfort bounds
        TODO: Add lower/upper band for massive reward penalty

        :param DataFrame prices: 15m electricity prices. DatetimeIndex.
        :param DataFrame weather: hourly weather. DatetimeIndex.
        :param Agent agent: RL agent.
        :param tuple coords: (latitude, longitude) of the building
        :param Zone zone: Zone object of the building
        :param Array of Windows in the building
        """
        super(SimEnv, self).__init__()

        #TODO: include band settings in observation
        inf = float('inf')
        # TODO: add current heating/cooling setpoint to observation
        self.observation_space = spaces.Box(low=np.array([-inf, -inf, -inf, -inf]),
                                            high=np.array([inf, inf, inf, inf]),
                                            dtype=np.float32)
        # First element is heating setpoint, second element is cooling setpoint.
        self.action_space = spaces.Box(low=np.array([-20, -20]),
                                       high=np.array([20, 20]),
                                       dtype=np.float32)

        self.action_space
        self.prices = prices
        self.weather = weather
        self.agent = agent
        self.latitude, self.longitude = coords
        self.zone = zone
        self.windows = windows

        self.reset()

    def reset(self):
        # reset time to a random time in the year
        # TODO: Reset to a random time that guarantees a full episode
        self.time = random.randint(0, 8760)
        self.timestep = 0

        # set episode return to 0
        self.ep_reward = 0

        # reset the temperature of the building mass
        self.t_m_prev = random.randint(15, 25)
        timestamp, cur_weather = self.get_timestamp_and_weather()
        self.cur_state = [
            cur_weather['Temperature'], self.t_m_prev, self.time % 24,
            int(self.time/24) % 7
        ]

        # reset the episodes results
        self.results = defaultdict(list)

        return self.cur_state

    def step(self, action):
        """
        :param Array of size two to determine change in heating and cooling
        temperature respectively
        """
        self.zone.t_set_heating += action[0]
        self.zone.t_set_cooling += action[1]

        timestamp, cur_weather = self.get_timestamp_and_weather()

        altitude, azimuth = sun_position(self.latitude, self.longitude, timestamp)

        for window in self.windows:
            window.calc_solar_gains(
                sun_altitude=altitude, sun_azimuth=azimuth,
                normal_direct_radiation=cur_weather['DNI_Whm2'],
                horizontal_diffuse_radiation=cur_weather['DHI_Whm2']
            )

        t_out = cur_weather['Temperature']

        self.zone.solve_energy(
            internal_gains=0,
            solar_gains=sum([window.solar_gains for window in self.windows]),
            t_out=t_out,
            t_m_prev=self.t_m_prev
        )
        self.t_m_prev = self.zone.t_m_next

        for attr in ('heating_demand', 'heating_energy', 'cooling_demand',
                     'cooling_energy', 'electricity_out', 't_air'):
            self.results[attr].append(getattr(self.zone, attr))
        self.results['t_out'].append(t_out)

        self.results['solar_gain'].append(
            sum([window.solar_gains for window in self.windows]))

        # Calculate all electricity consumed and price
        price = self.get_avg_hourly_price()
        elec_consumed = self.zone.heating_energy + self.zone.cooling_energy
        self.results['electricity_consumed'].append(elec_consumed)
        self.results['price'].append(elec_consumed * price)

        self.time += 1
        # Allow for a rollover into january from December
        # TODO: We might not want to do this since the air temperatures will be
        # discontinuous
        if self.time >= 8760:
            self.time = 0

        # TODO: t_m_prev is privileged, reset to t_air
        # TODO: move to using timestamps instead of hour int for generalizability
        # to test case
        self.cur_state = [t_out, self.t_m_prev, self.time % 24, int(self.time/24) % 7]
        reward, info = self.get_reward(self.t_m_prev, t_out)
        self.ep_reward += reward
        self.timestep += 1
        info['success'] = False

        return self.cur_state, reward, self.timestep >= 1024, info

    def plot_results(self, attr='t_air'):
        annual_results = pd.DataFrame(self.results)
        annual_results[[attr]].plot()
        plt.show()

    def get_obs(self):
        return self.cur_state

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
        """
        row = self.weather.reset_index().iloc[self.time]
        return row[0], row[1:]

    def get_reward(self, price, t_air, lam=0.2):
        """Return tuple of reward and info dict."""
        # TODO: test lam as a hyperparameter / better reward design
        # TODO: deadband for comfort penalty
        info = {'reward':  -(price + lam*(abs(t_air-21.1)))}
        return -(price + lam*(abs(t_air-21.1))), info
