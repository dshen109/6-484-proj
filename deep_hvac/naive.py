from collections import defaultdict
import os
import random
import sys

import matplotlib.pyplot as plt

from deep_hvac import logger
from deep_hvac.simulator import SimEnv
from deep_hvac.util import NsrdbReader, ErcotPriceReader

sim_dir = os.path.join(
        os.path.split(os.path.abspath(__file__))[0],
        '..', 'rc-building-sim'
    )
sys.path.insert(1, sim_dir)
from rc_simulator.building_physics import Zone
from rc_simulator.radiation import Window


def run_episode(env, ep_steps, heating_temp, cooling_temp):
    env.reset()
    for i in range(ep_steps):
        env.step([heating_temp, cooling_temp])
    return env.results


def plot_curves(data_dict, title):
    # {label: [x, y]}
    fig, ax = plt.subplots(figsize=(4, 3))
    labels = data_dict.keys()
    for label, data in data_dict.items():
        x = data[0]
        y = data[1]
        ax.plot(x, y, label=label)
    ax.set_title(title)
    ax.legend()

    plt.show()


def update_results(results, ep_results):
    for k in ep_results.keys():
        results[k].append(ep_results[k])


def make_default_env():
    datadir = os.path.join(
        os.path.split(os.path.abspath(__file__))[0],
        '..', 'data'
    )
    nsrdb = NsrdbReader(os.path.join(datadir, '1704559_29.72_-95.35_2018.csv'))
    ercot = ErcotPriceReader(os.path.join(
        datadir, 'ercot-2018-rt.xlsx'))
    window_area = 10
    office = Zone(window_area=window_area)
    south_window = Window(azimuth_tilt=0, altitude_tilt=90, area=window_area)

    latitude, longitude = 29.749907, -95.358421

    env = SimEnv(prices=ercot.prices,
                 weather=nsrdb.weather_hourly,
                 agent=None,
                 coords=[latitude, longitude],
                 zone=office,
                 windows=[south_window]
        )

    return env


def naive_agent(env_name='DefaultBuilding-v0', max_steps=100000,
                heating_temp=20, cooling_temp=24):
    """Naive agent that follows a fixed temperature schedule."""
    episode_steps = 1024
    results = defaultdict(list)
    total_steps = 0
    random.seed(12)

    env = make_default_env()

    logger.debug(
        f"Making naive agent with heating setpoint {heating_temp} and "
        f"cooling setpoint {cooling_temp}.")

    while total_steps < max_steps:
        ep_results = run_episode(
            env, episode_steps, heating_temp=heating_temp,
            cooling_temp=cooling_temp)
        update_results(results, ep_results)
        total_steps += episode_steps
        # Break after 1 run because no need to train.
        break

    return results


if __name__ == "__main__":
    print(naive_agent().keys())
