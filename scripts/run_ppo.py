import os, sys
import pandas as pd
from gym.envs.registration import registry, register
import random

sim_dir = os.path.join(
        os.path.split(os.path.abspath(__file__))[0],
        '..', 'rc-building-sim'
    )
sys.path.insert(1, sim_dir)
from rc_simulator.building_physics import Zone
from rc_simulator.radiation import Window


hvac_dir = os.path.join(
        os.path.split(os.path.abspath(__file__))[0],
        '..', 'deep_hvac'
    )
sys.path.insert(2, hvac_dir)
from util import NsrdbReader, ErcotPriceReader
from simulator import SimEnv
from ppo import train_ppo

def make_ppo_agent(max_steps=100000):

    random.seed(12)

    datadir = os.path.join(
        os.path.split(os.path.abspath(__file__))[0],
        '..', 'data'
    )
    nsrdb = NsrdbReader(os.path.join(datadir, '1704559_29.72_-95.35_2018.csv'))
    ercot = ErcotPriceReader(os.path.join(
        datadir, 'ercot-2018-rt.xlsx'
    ))

    window_area = 1
    office = Zone(window_area=window_area)
    south_window = Window(azimuth_tilt=0, altitude_tilt=90, area=window_area)

    latitude, longitude = 29.749907, -95.358421

    env_args = {
        'prices': ercot.prices,
        'weather': nsrdb.weather_hourly,
        'agent': None,
        'coords': [latitude, longitude],
        'zone': office,
        'windows': [south_window]
    }

    env_name = 'DefaultBuilding-v0'
    if env_name in registry.env_specs:
        del registry.env_specs[env_name]
    register(
        id=env_name,
        entry_point=f'simulator:SimEnv',
        kwargs=env_args
    )

    return train_ppo(env_name='DefaultBuilding-v0', max_steps=max_steps)

if __name__ == "__main__":
    make_ppo_agent(max_steps=600000)