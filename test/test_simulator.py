import datetime as dt

from deep_hvac import simulator
from deep_hvac.util import NsrdbReader
from deep_hvac import building

import numpy as np
import pandas as pd

import mock
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
        for _ in range(10):
            self.env.step(self.action_default)
        timestamps = self.env.results['timestamp']

        # First timestamp should be hour after year starts
        self.assertEqual(
            timestamps[0],
            dt.datetime(timestamps[0].year, 1, 1, 1, 0,
                        tzinfo=timestamps[0].tzinfo)
        )

        self.array_eq(
            self.env.results['t_outside'],
            self.nsrdb.weather_hourly.iloc[1:11]['Temperature'].values)

        expected_cost = [
            1.32181279, 1.33013849, 1.3672246,
            1.41426522, 1.54790073, 1.64943658,
            1.62634529, 1.61891077, 1.90342777,
            1.9263284
        ]

        self.array_eq(
            self.env.results['electricity_cost'],
            expected_cost, atol=0.01)

    def test_discrete_action_to_setpoints(self):
        self.env.t_high = 11
        self.env.t_low = 8
        self.env._map_discrete_actions()
        self.assertRaises(
            IndexError, lambda: self.env.discrete_action_to_setpoints(-1))
        self.assertEqual(
            self.env.discrete_action_to_setpoints(0), (8, 9)
        )
        self.assertEqual(
            self.env.discrete_action_to_setpoints(1), (8, 10)
        )
        self.assertEqual(
            self.env.discrete_action_to_setpoints(2), (8, 11)
        )
        self.assertEqual(
            self.env.discrete_action_to_setpoints(3), (9, 10)
        )
        self.assertEqual(
            self.env.discrete_action_to_setpoints(4), (9, 11)
        )
        self.assertEqual(
            self.env.discrete_action_to_setpoints(5), (10, 11)
        )
        self.assertRaises(
            IndexError, lambda: self.env.discrete_action_to_setpoints(6))

    def test_extreme_discomfort(self):
        self.env.is_occupied = mock.MagicMock(return_value=True)
        self.env.config.terminate_on_discomfort = True
        state1, _, terminate, info = self.env.step([15, 16])
        self.assertTrue(terminate)
        self.assertAlmostEqual(info['discomfort_score'], 5.327, places=2)
        self.assertAlmostEqual(
            info['reward_from_discomfort'], -5.32692, places=2)

    def test_action_change_penalty(self):
        state1, _, _, info = self.env.step([15, 16])
        state2, _, terminate, info = self.env.step([16, 17])
        self.assertEqual(info['action_change_reward'],
                         - self.env.config.action_change_penalty)

    def test_action_invalid(self):
        self.assertFalse(self.env.action_valid(0, 10))
        self.assertTrue(self.env.action_valid(20, 25))
        self.assertFalse(self.env.action_valid(25, 20))
        self.assertFalse(self.env.action_valid(40, 50))
