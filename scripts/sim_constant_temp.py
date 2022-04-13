import os
import sys

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from deep_hvac import logger
from deep_hvac.naive import naive_agent
from deep_hvac.simulator import SimEnv
from deep_hvac.util import NsrdbReader, ErcotPriceReader
from rc_simulator.building_physics import Zone
from rc_simulator.radiation import Window


hvac_dir = os.path.join(
        os.path.split(os.path.abspath(__file__))[0],
        '..', 'deep_hvac'
    )
sys.path.insert(2, hvac_dir)


def plot(data, title, xlabel='steps', ylabel='reward'):
    plt.figure()
    x = data[0]
    y = data[1]
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def make_env():
    datadir = os.path.join(
        os.path.split(os.path.abspath(__file__))[0],
        '..', 'data'
    )
    logger.debug("Loading NSRDB data...")
    nsrdb = NsrdbReader(os.path.join(datadir, '1704559_29.72_-95.35_2018.csv'))
    logger.debug("Finished loading NSRDB data.")
    logger.debug("Loading Houston price data...")
    prices = pd.read_pickle(
        os.path.join(datadir, 'houston-2018-prices.pickle')
    )
    logger.debug("Finished loading price data.")
    window_area = 10
    office = Zone(window_area=window_area)
    south_window = Window(azimuth_tilt=0, altitude_tilt=90, area=window_area)

    latitude, longitude = 29.749907, -95.358421

    env = SimEnv(prices=prices,
                 weather=nsrdb.weather_hourly,
                 agent=None,
                 coords=[latitude, longitude],
                 zone=office,
                 windows=[south_window]
    )

    return env


if __name__ == '__main__':
    env = make_env()
    naive_results = naive_agent(heating_temp=30, cooling_temp=35)

    prices = [[i for i in range(1024)], list(np.mean(np.array(naive_results['price']), axis=0))]
    air_temp = [[i for i in range(1024)], list(np.mean(np.array(naive_results['t_air']), axis=0))]
    cooling = [[i for i in range(1024)], list(np.mean(np.array(naive_results['set_cooling']), axis=0))]

    plot(prices, 'cost, naive', ylabel='cost')
    plot(air_temp, 'air, naive', ylabel='air temp')
    plot(cooling, 'cooling setpoint, naive', ylabel='cooling stpt')
