import datetime
import math

import pandas as pd
from pysolar.solar import get_position


class ErcotPriceReader:

    timezone = 'America/Chicago'

    def __init__(self, fname, settlement_point='HB_HOUSTON'):
        self.filename = fname
        self.settlement_point = settlement_point
        sheets = pd.read_excel(self.filename, sheet_name=None)
        frame = pd.concat(
            [sheet for sheet in sheets.values()]
        )
        frame = frame.query(
            f"`Settlement Point Name` == '{self.settlement_point}'")

        timestamps = (
            pd.to_datetime(frame['Delivery Date']) +
            pd.to_timedelta(frame['Delivery Hour'] - 1, 'H') +
            pd.to_timedelta(frame['Delivery Interval'] - 1, 'T') * 15
        ).dt.tz_localize(
            pytz.timezone(self.timezone), ambiguous='infer'
        )
        frame.index = timestamps
        frame.drop(
            columns=[
                'Delivery Date', 'Delivery Hour', 'Delivery Interval',
                'Repeated Hour Flag'
            ], inplace=True)
        self.prices = frame


class NsrdbReader:
    """Load a CSV file from the NSRDB."""
    temperature_col = 'Temperature'
    dirnorrad_col = 'DNI'  # Direct Normal Irradiation
    difhorrad_col = 'DHI'  # Diffused Horizontal Irradiance

    def __init__(self, fname, tz='America/Chicago'):
        self.filename = fname
        frame = pd.read_csv(self.filename, skiprows=2)
        frame.index = (
            pd.to_datetime(
                frame['Year'].astype(str) + '-' +
                frame['Month'].astype(str) + '-' +
                frame['Day'].astype(str)
            ) +
            pd.to_timedelta(frame['Hour'], 'H') +
            pd.to_timedelta(frame['Minute'], 'T')
        )
        timezone = pytz.timezone(tz)
        frame.index = frame.index.tz_localize(timezone, ambiguous='infer')
        frame.drop(columns=['Year', 'Month', 'Day', 'Hour', 'Minute',
                            'Unnamed: 12'],
                   inplace=True)

        self.weather = frame
        # Create hourly weather.
        timedelta = (
            self.weather.index[1] - self.weather.index[0]
        ).total_seconds() / 3600
        self.weather_hourly = self.weather.resample('1H').mean()
        for col in [self.dirnorrad_col, self.dirnorrad_col]:
            self.weather_hourly[col + '_Whm2'] = \
                self.weather.resample('1H').sum() * timedelta


def hour_of_year(timestamp):
    """Calculate the hour of the year.

    The first hour of the year is 1."""
    beginning_of_year = datetime.datetime(
        timestamp.year, 1, 1, tzinfo=timestamp.tzinfo)
    return (timestamp - beginning_of_year).total_seconds() // 3600


def sun_position(latitude, longitude, timestamp):
    """
    Return tuple of altitude + azimuth

    :param float latitude:
    :param float longitude:
    :param Timestamp timestamp: tz-aware timestamp.
    """
    return get_position(latitude, longitude, timestamp)
