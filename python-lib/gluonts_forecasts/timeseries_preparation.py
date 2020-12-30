import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype, is_string_dtype
from pandas.tseries.offsets import Tick, BusinessDay, Week, MonthEnd, YearEnd
from pandas.tseries.frequencies import to_offset
import re


def prepare_timeseries_dataframe(dataframe, time_column_name, frequency, timeseries_identifiers_names=[], max_timeseries_length=None):
    """Convert time column to pandas.Datetime without timezones. Truncate timestamps to selected frequency.
    Check that there are no duplicate timestamps and that there are no missing timestamps.
    Sort timeseries. Keep only the most recent timestamps of each timeseries if specified.

    Args:
        dataframe (DataFrame)
        time_column_name (str)
        frequency (str)
        timeseries_identifiers_names (list, optional): List of timeseries identifiers columns. Defaults to [].
        max_timeseries_length (int, optional): Maximum number of timestamps to keep per timeseries. Defaults to None.

    Raises:
        ValueError: If the time column cannot be parsed as a date by pandas.

    Returns:
        Prepared timeseries
    """
    try:
        dataframe[time_column_name] = pd.to_datetime(dataframe[time_column_name]).dt.tz_localize(tz=None)
    except Exception:
        raise ValueError(f"Please parse the date column '{time_column_name}' in a Prepare recipe")

    preparator = TimeseriesPreparator(time_column_name, frequency, timeseries_identifiers_names)

    dataframe_prepared = dataframe.copy()

    preparator.check_data_types(dataframe_prepared)

    dataframe_prepared = preparator.truncate_timestamps(dataframe_prepared)

    dataframe_prepared = preparator.sort(dataframe_prepared)

    preparator.check_regular_frequency(dataframe_prepared)

    if max_timeseries_length:
        dataframe_prepared = preparator.keep_last_timestamps(dataframe_prepared, max_timeseries_length)

    return dataframe_prepared


class TimeseriesPreparator:
    def __init__(self, time_column_name, frequency, timeseries_identifiers_names):
        self.time_column_name = time_column_name
        self.frequency = frequency
        self.timeseries_identifiers_names = timeseries_identifiers_names

    def check_data_types(self, df):
        self._check_timeseries_identifiers_columns_types(df)

    def truncate_timestamps(self, df):
        """Truncate timestamps to selected frequency. For Week/Month/Year, truncate to end of Week/Month/Year.
        Check there are no duplicate timestamps.

        Examples:
            '2020-12-15 12:30:00' becomes '2020-12-15 00:00:00' with frequency 'D'
            '2020-12-15 12:30:00' becomes '2020-12-31 00:00:00' with frequency 'M'
            '2020-12-15 12:30:00' becomes '2021-01-30 00:00:00' with frequency 'A-JAN'

        Args:
            df (DataFrame): Dataframe in wide or long format with a time column.

        Raises:
            ValueError: If there are duplicates timestamps before or after truncation.

        Returns:
            Sorted DataFrame with truncated timestamps.
        """
        df_truncated = df.copy()

        if self._has_duplicates_timestamps(df_truncated):
            error_message = "Input dataset has duplicate timestamps."
            if len(self.timeseries_identifiers_names) == 0:
                error_message += " If the input dataset is in long format, make sure to specify it."
            raise ValueError(error_message)

        frequency_offset = to_offset(self.frequency)
        if isinstance(frequency_offset, Tick):
            df_truncated[self.time_column_name] = df_truncated[self.time_column_name].dt.floor(self.frequency)
        elif isinstance(frequency_offset, BusinessDay):
            df_truncated[self.time_column_name] = df_truncated[self.time_column_name].dt.floor("D")
        else:
            if isinstance(frequency_offset, Week):
                truncation_offset = pd.offsets.Week(weekday=frequency_offset.weekday, n=0)
            elif isinstance(frequency_offset, MonthEnd):
                truncation_offset = pd.offsets.MonthEnd(n=0)
            elif isinstance(frequency_offset, YearEnd):
                truncation_offset = pd.offsets.YearEnd(month=frequency_offset.month, n=0)

            df_truncated[self.time_column_name] = df_truncated[self.time_column_name].dt.floor("D") + truncation_offset

        if self._has_duplicates_timestamps(df_truncated):
            raise ValueError("Input dataset has duplicate timestamps after truncation to selected frequency")

        return df_truncated

    def sort(self, df):
        """Return a DataFrame sorted by timeseries identifiers and time column (both ascending) """
        return df.sort_values(by=self.timeseries_identifiers_names + [self.time_column_name])

    def check_regular_frequency(self, df):
        """Check that time column exactly equals the pandas.dat_range with selected frequency """
        if self.timeseries_identifiers_names:
            for identifiers_values, identifiers_df in df.groupby(self.timeseries_identifiers_names):
                assert_time_column_valid(identifiers_df, self.time_column_name, self.frequency)
        else:
            assert_time_column_valid(df, self.time_column_name, self.frequency)

    def keep_last_timestamps(self, df, max_timeseries_length):
        """Keep only at most the last max_timeseries_length timestamps of each timeseries.

        Args:
            df (DataFrame)
            max_timeseries_length (int): Maximum number of timestamps to keep per timeseries.

        Returns:
            Filtered dataframe
        """
        if len(self.timeseries_identifiers_names) == 0:
            return df.tail(max_timeseries_length)
        else:
            return df.groupby(self.timeseries_identifiers_names).apply(lambda x: x.tail(max_timeseries_length)).reset_index(drop=True)

    def _has_duplicates_timestamps(self, df):
        """Return True if there are no duplicate timestamps within each timeseries """
        return any(df.duplicated(subset=self.timeseries_identifiers_names + [self.time_column_name], keep=False))

    def _check_timeseries_identifiers_columns_types(self, df):
        """ Raises ValueError if a timeseries identifiers column is not numerical or string """
        for column_name in self.timeseries_identifiers_names:
            if not is_numeric_dtype(df[column_name]) and not is_string_dtype(df[column_name]):
                raise ValueError(f"Time series identifier '{column_name}' must be of string or numeric type")


def assert_time_column_valid(dataframe, time_column_name, frequency, start_date=None, periods=None):
    """Assert that the time column has the same values as the pandas.date_range generated with frequency and the first and last row of dataframe[time_column_name]
    (or with start_date and periods if specified).

    Args:
        dataframe (DataFrame)
        time_column_name (str)
        frequency (str): Use as frequency of pandas.date_range.
        start_date (pandas.Timestamp, optional): Use as start_date of pandas.date_range if specified. Defaults to None.
        periods (int, optional): Use as periods of pandas.date_range if specified. Defaults to None.

    Raises:
        ValueError: If the time column doesn't have regular time intervals of the chosen frequency.
    """
    if start_date is None:
        start_date = dataframe[time_column_name].iloc[0]
    if periods:
        date_range_values = pd.date_range(start=start_date, periods=periods, freq=frequency).values
    else:
        end_date = dataframe[time_column_name].iloc[-1]
        date_range_values = pd.date_range(start=start_date, end=end_date, freq=frequency).values

    if not np.array_equal(dataframe[time_column_name].values, date_range_values):
        error_message = f"Time column '{time_column_name}' has missing values with frequency '{frequency}'."
        error_message += " You can use the Time Series Preparation plugin to resample your time column."
        raise ValueError(error_message)