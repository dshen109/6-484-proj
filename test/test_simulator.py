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

    def test_discrete_action_to_setpoints(self):
        self.env.t_high = 10
        self.env.t_low = 8
        self.assertRaises(
            ValueError, lambda: self.env.discrete_action_to_setpoints(-1))
        self.assertEqual(
            self.env.discrete_action_to_setpoints(0), (8, 8)
        )
        self.assertEqual(
            self.env.discrete_action_to_setpoints(1), (8, 9)
        )
        self.assertEqual(
            self.env.discrete_action_to_setpoints(2), (8, 10)
        )
        self.assertEqual(
            self.env.discrete_action_to_setpoints(3), (9, 8)
        )
        self.assertEqual(
            self.env.discrete_action_to_setpoints(4), (9, 9)
        )
        self.assertEqual(
            self.env.discrete_action_to_setpoints(5), (9, 10)
        )
        self.assertEqual(
            self.env.discrete_action_to_setpoints(6), (10, 8)
        )
        self.assertEqual(
            self.env.discrete_action_to_setpoints(7), (10, 9)
        )
        self.assertEqual(
            self.env.discrete_action_to_setpoints(8), (10, 10)
        )
        self.assertRaises(
            ValueError, lambda: self.env.discrete_action_to_setpoints(9))
