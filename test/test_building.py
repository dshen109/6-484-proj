import os

from deep_hvac import building
from deep_hvac.naive import SimEnv
from deep_hvac.util import NsrdbReader

import mock
from unittest import TestCase


class TestBuilding(TestCase):

    def setUp(self):
        datadir = 'data'
        nsrdb = NsrdbReader(
            os.path.join(datadir, '1704559_29.72_-95.35_2018.csv'))
        zone, windows, lat, long = building.default_building()
        env = SimEnv(prices=None,
                     weather=nsrdb.weather_hourly,
                     agent=None,
                     coords=[lat, long],
                     zone=zone,
                     windows=windows)
        env.get_avg_hourly_price = mock.MagicMock(return_value=0)

        self.env = env

    def action(self):
        return [21, 26]

    def test_defaults(self):
        # Step through an entire year
        self.env.reset(0)
        for _ in range(24 * 365 - 2):
            self.env.step(self.action())
        results = self.env.results
        import pandas as pd
        pd.to_pickle(results, 'results-default.pickle')
