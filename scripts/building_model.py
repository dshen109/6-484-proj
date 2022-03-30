from collections import defaultdict
import os, sys

hvac_dir = os.path.join(
        os.path.split(os.path.abspath(__file__))[0],
        '..', 'deep_hvac'
    )
sys.path.insert(2, hvac_dir)
from util import ErcotPriceReader, NsrdbReader, sun_position

sim_dir = os.path.join(
        os.path.split(os.path.abspath(__file__))[0],
        '..', 'rc-building-sim'
    )
sys.path.insert(1, sim_dir)
from rc_simulator.building_physics import Zone
from rc_simulator import supply_system
from rc_simulator import emission_system
from rc_simulator.radiation import Window

import matplotlib.pyplot as plt
import pandas as pd


if __name__ == '__main__':
    datadir = os.path.join(
        os.path.split(os.path.abspath(__file__))[0],
        '..', 'data'
    )
    nsrdb = NsrdbReader(os.path.join(datadir, '1704559_29.72_-95.35_2018.csv'))
    # ercot = ErcotPriceReader(os.path.join(
    #     datadir, 'ercot-2018-rt.xlsx'
    # ))
    window_area = 1

    office = Zone(window_area=window_area)
    south_window = Window(azimuth_tilt=0, altitude_tilt=90, area=window_area)

    # Starting temperature of building mass
    t_m_prev = 20

    latitude, longitude = 29.749907, -95.358421

    results = defaultdict(list)

    for timestamp, weather in nsrdb.weather_hourly.iterrows():
        altitude, azimuth = sun_position(latitude, longitude, timestamp)
        south_window.calc_solar_gains(
            sun_altitude=altitude, sun_azimuth=azimuth,
            normal_direct_radiation=weather['DNI_Whm2'],
            horizontal_diffuse_radiation=weather['DHI_Whm2'])

        t_out = weather['Temperature']
        office.solve_energy(
            internal_gains=0,
            solar_gains=south_window.solar_gains, t_out=t_out,
            t_m_prev=t_m_prev
            )
        t_m_prev = office.t_m_next

        for attr in ('heating_demand', 'heating_energy', 'cooling_demand',
                     'cooling_energy', 'electricity_out', 't_air'):
            results[attr].append(getattr(office, attr))
        results['t_out'].append(t_out)
        results['solar_gain'].append(south_window.solar_gains)

    annual_results = pd.DataFrame(results)
    annual_results[['t_air']].plot()
    plt.show()
