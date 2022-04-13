from collections import defaultdict
import random

from gym import Env, spaces
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from deep_hvac.util import sun_position


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
        self.observation_space = spaces.Box(low=np.array([-inf, -inf, -inf, -inf, -inf, -inf]),
                                            high=np.array([inf, inf, inf, inf, inf, inf]),
                                            dtype=np.float32)
        # First element is heating setpoint, second element is cooling setpoint.
        self.action_space = spaces.Box(low=np.array([-10, -10]),
                                       high=np.array([10, 10]),
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
        # reset time to a random time in the year
        # TODO: Reset to a random time that guarantees a full episode
        self.time = time
        self.timestep = 0

        # set episode return to 0
        self.ep_reward = 0

        self.zone.t_set_heating = 20
        self.zone.t_set_cooling = 25

        # reset the temperature of the building mass
        self.t_m_prev = random.randint(15, 25)
        self.step_bulk()
        timestamp, cur_weather = self.get_timestamp_and_weather()
        self.cur_state = [
            self.zone.t_set_heating,
            self.zone.t_set_cooling,
            cur_weather['Temperature'],
            self.t_m_prev,
            timestamp.hour,
            timestamp.weekday()
        ]

        # reset the episodes results
        self.results = defaultdict(list)

        return self.cur_state

    def step_bulk(self):
        """Step through the building simulator and update the bulk building
        mass."""

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

        self.time += 1

        return timestamp, t_out

    def step(self, action):
        """
        :param Array of size two to determine new setpoint for heating and
        cooling temperature respectively
        """
        self.zone.t_set_heating = action[0]
        self.zone.t_set_cooling = action[1]

        timestamp, t_out = self.step_bulk()

        for attr in ('heating_demand', 'heating_energy', 'cooling_demand',
                     'cooling_energy', 'electricity_out', 't_air'):
            self.results[attr].append(getattr(self.zone, attr))
        self.results['t_inside'].append(self.t_m_prev)

        self.results['solar_gain'].append(
            sum([window.solar_gains for window in self.windows]))

        # Calculate all electricity consumed and price
        price = self.get_avg_hourly_price()
        elec_consumed = self.zone.heating_energy + self.zone.cooling_energy
        self.results['electricity_consumed'].append(elec_consumed)
        self.results['price'].append(elec_consumed * price)

        self.cur_state = [self.zone.t_set_heating,
                         self.zone.t_set_cooling,
                         t_out,
                         self.t_m_prev,
                         timestamp.hour,
                         timestamp.weekday()]
        reward, info = self.get_reward(elec_consumed * price, self.t_m_prev)
        self.ep_reward += reward
        self.results['reward'].append(reward)
        self.results['set_heating'].append(self.zone.t_set_heating)
        self.results['set_cooling'].append(self.zone.t_set_cooling)
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

        :return: timestamp, weather data.
        """
        row = self.weather.reset_index().iloc[self.time]
        return row[0], row[1:]

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
