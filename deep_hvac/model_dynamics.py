from multiprocessing.sharedctypes import Value
from time import time
import torch, os, sys, gym
import numpy as np
import pandas as pd
import datetime as dt
import random

from deep_hvac import agent, logger, simulator
from deep_hvac.building import default_building, comfort_temperature
from deep_hvac.util import NsrdbReader, sun_position

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
zone, windows, latitude, longitude = default_building()

config = simulator.SimConfig(
    episode_length=24,
    terminate_on_discomfort=True,
    discomfort_penalty=1e4,
    discrete_action=False
)

env_args = dict(
    prices=ercot,
    weather=nsrdb.weather_hourly,
    coords=(latitude, longitude),
    zone=zone,
    windows=windows,
    expert_performance=None,
    agent=None,
    config = config
)

# Run the naive agent as a baseline.
naive_agent = agent.NaiveAgent()
env = simulator.SimEnv(**env_args)
logger.debug("Running naive baseline...")
obs = env.reset(0)
for _ in range(365 * 24 - 2):
    obs, _, terminate, _ = env.step(naive_agent.get_action(obs)[0])
costs = env.results['electricity_cost']
# pad with zeros
costs.insert(0, 0)
costs.append(0)
env_args['expert_performance'] = pd.DataFrame(
    costs, columns=['cost'], index=nsrdb.weather_hourly.index)
env.results['expert_performance'] = env_args['expert_performance']

class BuildingDynamics:

    def __init__(
        self, 
        building=default_building(),
        env_args = env_args,
        u_range = 30, 
    ):
        self.zone = building[0]
        self.windows = building[1]
        self.latitude = building[2]
        self.longitude = building[3]
        self.env_args = env_args
        self.weather = self.env_args['weather'].reset_index()
        self.prices = self.env_args['prices']
        self.u_range = u_range

        self.expert_price_last = 0

        self.discomfort_penalty = 1000
        self.comfort_praise = 0.1
        self.action_bound_penalty = 10

        self.expert_performance = self.env_args['expert_performance']

        self.u_lb = torch.tensor([10]).float()
        self.u_ub = torch.tensor([40]).float()
        self.q_shape = 9
        self.u_shape = 2

    def step(self, q, u):
        self.zone.t_set_heating = u[0]
        self.zone.t_set_cooling = u[1]

        time_ind = int(q[0])
        timestamp, cur_weather = self.get_timestamp_and_weather(time_ind)

        altitude, azimuth = sun_position(self.latitude, self.longitude, timestamp)

        for window in self.windows:
            window.calc_solar_gains(
                sun_altitude=altitude, sun_azimuth=azimuth,
                normal_direct_radiation=cur_weather['DNI_Wm2'],
                horizontal_diffuse_radiation=cur_weather['DHI_Wm2']
            )
        
        t_out = self.weather.iloc[time_ind].loc['Temperature']

        if self.is_occupied(timestamp):
            occupant_gains = 30 * 100
        else:
            occupant_gains = 0

        self.zone.solve_energy(
            internal_gains = occupant_gains,
            solar_gains=sum([window.solar_gains for window in self.windows]),
            t_out = t_out,
            t_m_prev = q[1]
        )

        elec_consumed = (
            self.zone.heating_sys_electricity +
            self.zone.cooling_sys_electricity
        )

        return [
            time_ind + 1, 
            self.zone.t_air.item(),
            t_out,
            cur_weather['DNI_Wm2'],
            cur_weather['DHI_Wm2'],
            timestamp.hour,
            timestamp.day, 
            timestamp.month,
            timestamp.weekday(),
            self.get_avg_hourly_price(timestamp),
            elec_consumed
        ]

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

    def get_timestamp_and_weather(self, time_ind):
        return self.weather['index'].iloc[time_ind], self.weather.iloc[time_ind]

    def is_occupied(self, timestamp):
        is_weekday = timestamp.weekday() < 5
        hour = timestamp.hour

        return is_weekday and (hour >= 8 and hour <= 6 + 12)

    def run_batch_of_trajectories(self, q, u):
        qs = [q]

        for t in range(u.shape[1]):
            new_qs = []
            for act in range(u.shape[0]):
                new_qs.append(self.step(qs[-1][act], u[act][t]))
            qs.append(torch.tensor(new_qs))

        return torch.stack(qs, dim=1)

    def get_q_from_time_ind(self, time_ind):
        timestamp, cur_weather = self.get_timestamp_and_weather(time_ind)

        return [
            time_ind, 
            23,
            cur_weather.loc['Temperature'],
            cur_weather['DNI_Wm2'],
            cur_weather['DHI_Wm2'],
            timestamp.hour,
            timestamp.day, 
            timestamp.month,
            timestamp.weekday(),
            self.get_avg_hourly_price(timestamp),
            0
        ]

    def reward(self, q, u):
        if isinstance(q, list):
            q = torch.tensor([q])
            answers = torch.zeros((q.shape[0]), dtype=torch.float32)
            i = 0
            for ind_q in q:
                cur_timestamp, _ = self.get_timestamp_and_weather(int(ind_q[0]))
                electricity_cost = (ind_q[-1]/1000) * ind_q[-2]
                t_air = ind_q[1]
                t_set_heating, t_set_cooling = u
                
                # If we have an expert, the reward is how much better we did than the
                # expert
                reward_from_expert_improvement = (
                    self.expert_price_paid(cur_timestamp) - electricity_cost)

                reward_from_discomfort = 0
                discomf_score = 0
                reward_from_comfort = 0
                action_change_reward = 0
                action_bound_violation_reward = False

                if self.is_occupied(cur_timestamp):
                    discomf_score = self.comfort_penalty(t_air, ind_q[2])
                    reward_from_discomfort -= (
                        discomf_score * self.discomfort_penalty
                    )
                    if discomf_score == 0:
                        reward_from_comfort += self.comfort_praise

                if not self.action_valid(t_set_heating, t_set_cooling):
                    action_bound_violation_reward -= self.action_bound_penalty

                reward = (
                    reward_from_expert_improvement + reward_from_discomfort +
                    reward_from_comfort + action_bound_violation_reward +
                    action_change_reward
                )
                answers[i] = reward

                i+=1

            return answers
            

        answers = torch.zeros((q.shape[0], q.shape[1]))

        for traj in range(q.shape[0]):
            qs, us, i = q[traj], u[traj], 0
            for ind_q, ind_u in zip(qs, us):
                cur_timestamp, _ = self.get_timestamp_and_weather(int(ind_q[0]))
                electricity_cost = (ind_q[-1]/1000) * ind_q[-2]
                t_air = ind_q[1]
                t_set_heating, t_set_cooling = ind_u
                
                # If we have an expert, the reward is how much better we did than the
                # expert
                reward_from_expert_improvement = (
                    self.expert_price_paid(cur_timestamp) - electricity_cost)

                self.expert_price_last = self.expert_price_paid

                reward_from_discomfort = 0
                discomf_score = 0
                reward_from_comfort = 0
                action_change_reward = 0
                action_bound_violation_reward = False

                if self.is_occupied(cur_timestamp):
                    discomf_score = self.comfort_penalty(t_air, ind_q[2])
                    reward_from_discomfort -= (
                        discomf_score * self.discomfort_penalty
                    )
                    if discomf_score == 0:
                        reward_from_comfort += self.comfort_praise

                if not self.action_valid(t_set_heating, t_set_cooling):
                    action_bound_violation_reward -= self.action_bound_penalty

                reward = (
                    reward_from_expert_improvement + reward_from_discomfort +
                    reward_from_comfort + action_bound_violation_reward +
                    action_change_reward
                )
                answers[traj][i] = reward

                i+=1

        return answers


    def comfort_penalty(self, t_air, outdoor):
        t_comf = comfort_temperature(outdoor)
        deviation = np.abs(t_air - t_comf)
        if deviation <= 2.5:
            return 0
        elif deviation <= 3.5:
            return np.interp(deviation, [2.5, 3.5], [0.5, 1])
        else:
            return deviation - 2.5

    def action_valid(self, t_set_heating, t_set_cooling):
        return t_set_heating < t_set_cooling

    def expert_price_paid(self, timestamp):
        if self.expert_performance is None:
            return 0
        else:
            return self.expert_performance.loc[timestamp, 'cost']

class BuildingGym(gym.Env):

    def __init__(self, timestep_limit=200):
        self.dynamics = BuildingDynamics()

        self.timestep_limit = 24
        self.reset()

    def reset(self, time=None):
        self.dynamics.zone.t_set_heating = 20
        self.dynamics.zone.t_set_cooling = 26

        if time is None:
            # Reset to midnight starting at a random month.
            month = random.randint(1, 12)
            timestamp_start = self.dynamics.get_timestamp_and_weather(0)[0]
            year = timestamp_start.year
            tzinfo = timestamp_start.tzinfo
            timestamp = dt.datetime(year, month, 1, 0, 0, 0, tzinfo=tzinfo)
            time = self.hour_of_year(timestamp)

        self.time = time
        self.timestep = 0

        self.q_sim = self.dynamics.get_q_from_time_ind(self.time)

        self.traj = [self.get_observation()]

        return self.traj[-1]

    def get_observation(self):
        return self.q_sim

    def step(self, action):
        new_q = self.dynamics.step(
            self.q_sim, action
        )

        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action)

        reward = self.dynamics.reward(
            new_q, action
        ).numpy()

        self.q_sim = np.array(new_q)
        done = self.is_done()

        self.timestep += 1

        self.traj.append(self.q_sim)
        
        return self.q_sim, reward, done, {'last_expert': self.dynamics.expert_price_last}

    def is_done(self):
        # Kill trial when too much time has passed
        if self.timestep >= self.timestep_limit:
            return True
                
        return False

    def hour_of_year(self, timestamp):
        """Get integer corresponding to the hour of the year for `timestamp`

        :return int:
        """
        year = timestamp.year
        start = dt.datetime(year, 1, 1, 0, 0, 0, tzinfo=timestamp.tzinfo)
        return int((start - timestamp).total_seconds() / 3600)

if __name__ == "__main__":
    building_dynamics = BuildingDynamics()
    print('\n\n\n\n\n\n')

    q = np.array([0, 20], dtype=np.float32)
    u = np.array([1, 1], dtype=np.float32)
    print(building_dynamics.step(q, u))
