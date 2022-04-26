import datetime as dt

from deep_hvac import simulator
from deep_hvac.util import NsrdbReader
from deep_hvac import building

import numpy as np
import pandas as pd

from unittest import TestCase


class TestSim(TestCase):

    def setUp(self):
        self.prices = pd.read_pickle('data/houston-2018-prices.pickle')
        self.nsrdb = NsrdbReader('data/1704559_29.72_-95.35_2018.csv')
        self.zone, self.windows, self.lat, self.long = \
            building.default_building()

        self.env = simulator.SimEnv(
            prices=self.prices, weather=self.nsrdb.weather_hourly,
            agent=None, coords=(self.lat, self.long), zone=self.zone,
            windows=self.windows, config=simulator.SimConfig()
        )

    def array_eq(self, *args, **kwargs):
        assert np.allclose(*args, **kwargs)

    @property
    def action_default(self):
        return [20, 26]

    def test_timestepping(self):
        states = []
        states.append(self.env.reset(0))
        state2, _, _, _ = self.env.step(self.action_default)
        state3, _, _, _ = self.env.step(self.action_default)
        timestamps = self.env.results['timestamp']

        # First timestamp should be hour after year starts
        self.assertEqual(
            timestamps[0],
            dt.datetime(timestamps[0].year, 1, 1, 1, 0,
                        tzinfo=timestamps[0].tzinfo)
        )

        self.array_eq(
            self.env.results['t_outside'],
            self.nsrdb.weather_hourly.iloc[1:3]['Temperature'].values)

        self.array_eq(
            self.env.results['electricity_cost'],
            [0, 0], atol=0.001)
