import pandas as pd
import pytz


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
