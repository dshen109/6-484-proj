import os, sys
import pandas as pd

sim_dir = os.path.join(
        os.path.split(os.path.abspath(__file__))[0],
        '..', 'rc-building-sim'
    )
sys.path.insert(1, sim_dir)
from rc_simulator.building_physics import Zone
from rc_simulator import supply_system
from rc_simulator import emission_system
from rc_simulator.radiation import Window


hvac_dir = os.path.join(
        os.path.split(os.path.abspath(__file__))[0],
        '..', 'deep_hvac'
    )
sys.path.insert(2, hvac_dir)
from util import NsrdbReader, ErcotPriceReader

from simulator import SimEnv
if __name__ == "__main__":
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

    env = SimEnv(prices=ercot.prices,
                 weather=nsrdb.weather_hourly,
                 agent=None,
                 coords=[latitude, longitude],
                 zone=office,
                 windows=[south_window])

    for _ in range(600):
        env.step([0, 0])

    env.plot_results()