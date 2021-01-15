import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype, is_string_dtype
from pandas.tseries.offsets import Tick, BusinessDay, Week, MonthEnd
from pandas.tseries.frequencies import to_offset
import re
from safe_logger import SafeLogger

logger = SafeLogger("Forecast plugin")


class TimeseriesPreparator:
    """
    Class to check the timeseries has the right data and prepare it to have regular date interval
    """

    def __init__(
        self,
        time_column_name,
        frequency,
        target_columns_names=[],
        timeseries_identifiers_names=[],
        external_features_columns_names=[],
        max_timeseries_length=None,
    ):
        self.time_column_name = time_column_name
        self.frequency = frequency
        self.target_columns_names = target_columns_names
        self.timeseries_identifiers_names = timeseries_identifiers_names
        self.external_features_columns_names = external_features_columns_names
        self.max_timeseries_length = max_timeseries_length

    def prepare_timeseries_dataframe(self, dataframe):
        """Convert time column to pandas.Datetime without timezones. Truncate dates to selected frequency.
        Check that there are no duplicate dates and that there are no missing dates.
        Sort timeseries. Keep only the most recent dates of each timeseries if specified.

        Args:
            dataframe (DataFrame)

        Raises:
            ValueError: If the time column cannot be parsed as a date by pandas.

        Returns:
            Prepared timeseries
        """
        dataframe_prepared = dataframe.copy()

        try:
            dataframe_prepared[self.time_column_name] = pd.to_datetime(dataframe[self.time_column_name]).dt.tz_localize(tz=None)
        except Exception:
            raise ValueError(f"Please parse the date column '{self.time_column_name}' in a Prepare recipe")

        self._check_data(dataframe_prepared)

        dataframe_prepared = self._truncate_dates(dataframe_prepared)

        dataframe_prepared = self._sort(dataframe_prepared)

        self._check_regular_frequency(dataframe_prepared)
        self._log_timeseries_lengths(dataframe_prepared)

        if self.max_timeseries_length:
            dataframe_prepared = self._keep_last_dates(dataframe_prepared)
            self._log_timeseries_lengths(dataframe_prepared, after_sampling=True)

        return dataframe_prepared

    def _check_data(self, df):
        self._check_timeseries_identifiers_columns_types(df)
        self._check_no_missing_values(df)

    def _truncate_dates(self, df):
        """Truncate dates to selected frequency. For Week/Month/Year, truncate to end of Week/Month/Year.
        Check there are no duplicate dates.

        Examples:
            '2020-12-15 12:45:30' becomes '2020-12-15 12:40:00' with frequency '20min'
            '2020-12-15 12:00:00' becomes '2020-12-15 00:00:00' with frequency '24H'
            '2020-12-15 12:30:00' becomes '2020-12-15 00:00:00' with frequency 'D'
            '2020-12-15 12:30:00' becomes '2020-12-31 00:00:00' with frequency 'M'
            '2020-12-15 12:30:00' becomes '2021-01-30 00:00:00' with frequency 'A-JAN'

        Args:
            df (DataFrame): Dataframe in wide or long format with a time column.

        Raises:
            ValueError: If there are duplicates dates before or after truncation.

        Returns:
            Sorted DataFrame with truncated dates.
        """
        df_truncated = df.copy()

        self._check_duplicate_dates(df_truncated)

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

            df_truncated[self.time_column_name] = df_truncated[self.time_column_name].dt.floor("D") + truncation_offset

        self._log_truncation(df_truncated, df)

        self._check_duplicate_dates(df_truncated, after_truncation=True)

        return df_truncated

    def _sort(self, df):
        """Return a DataFrame sorted by timeseries identifiers and time column (both ascending) """
        return df.sort_values(by=self.timeseries_identifiers_names + [self.time_column_name])

    def _check_regular_frequency(self, df):
        """Check that time column exactly equals the pandas.dat_range with selected frequency """
        if self.timeseries_identifiers_names:
            for identifiers_values, identifiers_df in df.groupby(self.timeseries_identifiers_names):
                assert_time_column_valid(identifiers_df, self.time_column_name, self.frequency)
        else:
            assert_time_column_valid(df, self.time_column_name, self.frequency)

    def _keep_last_dates(self, df):
        """Keep only at most the last max_timeseries_length dates of each timeseries.

        Args:
            df (DataFrame)

        Returns:
            Filtered dataframe
        """
        if len(self.timeseries_identifiers_names) == 0:
            return df.tail(self.max_timeseries_length)
        else:
            return df.groupby(self.timeseries_identifiers_names).apply(lambda x: x.tail(self.max_timeseries_length)).reset_index(drop=True)

    def _log_truncation(self, df_truncated, df):
        """Log how many dates were truncated for users to understand how their data were changed

        Args:
            df_truncated (DataFrame): Dataframe after truncation
            df (DataFrame): Original dataframe

        """
        total_dates = len(df_truncated.index)
        truncated_dates = (df_truncated[self.time_column_name] != df[self.time_column_name]).sum()
        if truncated_dates > 0:
            logger.warning(
                f"Dates truncated to {frequency_custom_label(self.frequency)} frequency: {total_dates - truncated_dates} dates kept, {truncated_dates} dates truncated"
            )
            if truncated_dates == total_dates:
                self._check_end_of_frequency(df_truncated, df)
        else:
            logger.info(f"No dates were changed after truncation to {frequency_custom_label(self.frequency)} frequency")

    def _check_end_of_frequency(self, df_truncated, df):
        """Check not all that truncated days are different days"""
        frequency_offset = to_offset(self.frequency)
        if isinstance(frequency_offset, Week):
            if all(df_truncated[self.time_column_name].dt.dayofweek != df[self.time_column_name].dt.dayofweek):
                raise ValueError(f"No weekly dates on {WEEKDAYS[frequency_offset.weekday]}. Please check the 'End of week day' parameter.")

    def _check_duplicate_dates(self, df, after_truncation=False):
        """Check dataframe has no duplicate dates and raise an actionable error message """
        duplicate_dates = self._count_duplicate_dates(df)
        if duplicate_dates > 0:
            error_message = f"Input dataset has {duplicate_dates} duplicate dates"
            if after_truncation:
                error_message += f" after truncation to '{self.frequency}' frequency. Please check the Frequency parameter."
            else:
                error_message += "."
                if len(self.timeseries_identifiers_names) == 0:
                    error_message += " Please check the Long format parameter."
            raise ValueError(error_message)

    def _count_duplicate_dates(self, df):
        """Return total number of duplicates dates within all timeseries """
        return df.duplicated(subset=self.timeseries_identifiers_names + [self.time_column_name], keep=False).sum()

    def _check_timeseries_identifiers_columns_types(self, df):
        """ Raises ValueError if a timeseries identifiers column is not numerical or string """
        for column_name in self.timeseries_identifiers_names:
            if not is_numeric_dtype(df[column_name]) and not is_string_dtype(df[column_name]):
                raise ValueError(f"Time series identifier '{column_name}' must be of string or numeric type. Please change the type in a Prepare recipe.")

    def _check_no_missing_values(self, df):
        for column_name in [self.time_column_name] + self.target_columns_names + self.timeseries_identifiers_names + self.external_features_columns_names:
            if df[column_name].isnull().values.any():
                raise ValueError(f"Column '{column_name}' has missing values. You can use the Time Series Preparation plugin to resample your time series.")

    def _log_timeseries_lengths(self, df, after_sampling=False):
        """Log the number and sizes of time series and whether it's after sampling or not"""
        if len(self.timeseries_identifiers_names) == 0:
            timeseries_lengths = [len(df.index)]
        else:
            timeseries_lengths = list(df.groupby(self.timeseries_identifiers_names).size())
        log_message = f"{len(timeseries_lengths)} time series"
        if all(length == timeseries_lengths[0] for length in timeseries_lengths):
            log_message += f" of {timeseries_lengths[0]} records"
        else:
            log_message += f" of {min(timeseries_lengths)} to {max(timeseries_lengths)} records"
        if after_sampling:
            logger.info(f"Sampled last records: train set contains {log_message}")
        else:
            logger.info(f"Found {log_message}")


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


FREQUENCY_LABEL = {"T": "minute", "H": "hour", "D": "day", "B": "business day"}


WEEKDAYS = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


def frequency_custom_label(frequency):
    frequency_offset = to_offset(frequency)
    if isinstance(frequency_offset, MonthEnd):
        if frequency_offset.n == 3:
            return "quarter"
        elif frequency_offset.n == 6:
            return "semester"
        elif frequency_offset.n == 12:
            return "year"
        elif frequency_offset.n == 1:
            return "end of month"
        else:
            return f"{frequency_offset.n} months"
    elif isinstance(frequency_offset, Week):
        prefix = f"{frequency_offset.n} weeks" if frequency_offset.n > 1 else "week"
        return f"{prefix} ending on {WEEKDAYS[frequency_offset.weekday]}"
    else:
        prefix = f"{frequency_offset.n} " if frequency_offset.n > 1 else ""
        middle = f"{FREQUENCY_LABEL[frequency_offset.name]}"
        suffix = "s" if frequency_offset.n > 1 else ""
        return f"{prefix}{middle}{suffix}"